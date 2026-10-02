# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse MLA attention with split-KV for low-batch decode."""

import functools

import torch

from vllm.triton_utils import LOG2E, LOGE2, tl, triton
from vllm.utils.platform_utils import num_compute_units

# DeepSeek-V3.2 / GLM-5 sparse MLA shape constants.
_BLOCK_DMODEL = 512
_BLOCK_DPE = 64
_BLOCK_DV = 512
_DIM_QK = _BLOCK_DMODEL + _BLOCK_DPE  # 576
# GLM-5.3-Flash (glm5_next) DSA-MLA is rope-free: dim_qk == kv_lora_rank.
_DIM_QK_NOPE = _BLOCK_DMODEL  # 512

_BLOCK_H = 16
# Smallest BLOCK_N the autotune sweep offers; only used for the topk-divisibility
# check at dispatch time.
_MIN_BLOCK_N = 16

# Merge kernel grid is spread across heads and DV tiles to avoid a (1,1)
# launch starving the SMs (pattern from FlashMLA's combine kernel).
_MERGE_BLOCK_H = 1
_MERGE_BLOCK_DV_TILE = 128
assert _BLOCK_DV % _MERGE_BLOCK_DV_TILE == 0
_NUM_MERGE_DV_TILES = _BLOCK_DV // _MERGE_BLOCK_DV_TILE

# Final (prefill) and split (decode) kernels each tune to their own regime.
_FINAL_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 16}, num_warps=nw, num_stages=ns)
    for nw in (2, 4)
    for ns in (2, 4)
]
_SPLIT_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=ns) for ns in (2, 4)
]

# Split-count candidates for `_choose_num_kv_splits`; also the set pre-compiled
# by `_warmup_autotune`.
KV_SPLITS_CANDIDATES = (1, 2, 4, 8, 16)

_MIN_TOPK_PER_SPLIT = 128  # below this, per-split work is too small to amortize
_SPLIT_TARGET_OCCUPANCY = 4  # target this multiple of SM count in total programs


@triton.jit
def _sparse_mla_compute_tile(
    q_buffer,
    k_buffer,  # V is the first BLOCK_DV lanes of each row of k_buffer.
    indices_ptr,
    cur_q,
    cur_head,
    cur_kv_head_id,
    mask_h,
    split_start,
    split_end,
    seq_kv,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_kv_head,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
):
    """Shared stage-1 body: load Q, run the sparse online-softmax loop over
    `[split_start, split_end)` of the topk axis, return accumulators."""
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)

    q = tl.load(
        q_buffer
        + cur_q * stride_q_token
        + cur_head[:, None] * stride_q_head
        + offs_d[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        qpe = tl.load(
            q_buffer
            + cur_q * stride_q_token
            + cur_head[:, None] * stride_q_head
            + offs_dpe[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )

    # Finite sentinel (not -inf) — when an entire BLOCK_N tile is masked,
    # `-inf - -inf = NaN` poisons the softmax; `sentinel - sentinel = 0`
    # gives `exp2(0) = 1` and the matching V rows are already 0.
    NEG_LARGE = -1.0e30
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) + NEG_LARGE
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for start_indice in range(split_start, split_end, BLOCK_N):
        offs_indice = start_indice + tl.arange(0, BLOCK_N)
        mask_indice = offs_indice < split_end
        # int64: workspace-row indices x stride_kv_token exceed 2**31 at
        # chunk 2048 x topk 2048 (4.19M rows x 576).
        indices = tl.load(
            indices_ptr
            + cur_q * stride_indices_token
            + cur_kv_head_id * stride_indices_head
            + offs_indice,
            mask=mask_indice,
            other=-1,
        ).to(tl.int64)
        mask_kv = (indices >= 0) & (indices < seq_kv)

        offs_k = (
            indices[None, :] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_d[:, None]
        )
        k = tl.load(k_buffer + offs_k, mask=mask_kv[None, :], other=0.0)
        qk = tl.dot(q, k.to(q.dtype))

        if BLOCK_DPE > 0:
            offs_kpe = (
                indices[None, :] * stride_kv_token
                + cur_kv_head_id * stride_kv_head
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                k_buffer + offs_kpe,
                mask=mask_kv[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe, kpe.to(q.dtype))

        qk *= sm_scale
        qk = tl.where((mask_h[:, None]) & (mask_kv[None, :]), qk, NEG_LARGE)

        offs_v = (
            indices[:, None] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_dv[None, :]
        )
        v = tl.load(k_buffer + offs_v, mask=mask_kv[:, None], other=0.0)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    return acc, e_max, e_sum


@triton.autotune(configs=_FINAL_AUTOTUNE_CONFIGS, key=["index_topk", "kv_group_num"])
@triton.jit
def _sparse_mla_kernel_final(
    q_buffer,
    k_buffer,
    indices_ptr,
    out_ptr,
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_kv_head,
    stride_out_token,
    stride_out_head,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    index_topk: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
):
    """Single-pass fast path: full topk, write final bf16 output directly."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    acc, e_max, e_sum = _sparse_mla_compute_tile(
        q_buffer,
        k_buffer,
        indices_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        0,
        index_topk,
        seq_kv,
        stride_q_token,
        stride_q_head,
        stride_kv_token,
        stride_kv_head,
        stride_indices_token,
        stride_indices_head,
        sm_scale,
        BLOCK_H,
        BLOCK_N,
        BLOCK_DV,
        BLOCK_DMODEL,
        BLOCK_DPE,
    )

    # Guard against queries with zero valid KV (e_sum == 0 → NaN from 0/0).
    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_dv = tl.arange(0, BLOCK_DV)
    tl.store(
        out_ptr
        + cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :],
        (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        mask=mask_h[:, None],
    )


@triton.autotune(
    configs=_SPLIT_AUTOTUNE_CONFIGS,
    key=["index_topk", "NUM_KV_SPLITS", "kv_group_num"],
)
@triton.jit
def _sparse_mla_kernel_split(
    q_buffer,
    k_buffer,
    indices_ptr,
    mid_out_ptr,
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_kv_head,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    index_topk: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    LOGE2: tl.constexpr,
):
    """Stage 1 of split-KV: process one slice of the topk axis and write
    its `(out_partial, lse_partial)` into the mid buffer."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    split_topk: tl.constexpr = tl.cdiv(index_topk, NUM_KV_SPLITS)
    split_start = split_kv_id * split_topk
    split_end = tl.minimum(split_start + split_topk, index_topk)

    acc, e_max, e_sum = _sparse_mla_compute_tile(
        q_buffer,
        k_buffer,
        indices_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        split_start,
        split_end,
        seq_kv,
        stride_q_token,
        stride_q_head,
        stride_kv_token,
        stride_kv_head,
        stride_indices_token,
        stride_indices_head,
        sm_scale,
        BLOCK_H,
        BLOCK_N,
        BLOCK_DV,
        BLOCK_DMODEL,
        BLOCK_DPE,
    )

    # Partial output and natural-log LSE for stage-2 merge.
    # When a split has no valid KV (`e_sum == 0`), guard the divide so the
    # mid buffer holds 0 instead of NaN; otherwise the `0 * NaN = NaN` term
    # in stage 2 would poison every other split.
    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_dv = tl.arange(0, BLOCK_DV)
    mid_base_2d = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head[:, None] * stride_mid_head
        + split_kv_id * stride_mid_split
    )
    tl.store(
        mid_base_2d + offs_dv[None, :],
        acc / e_sum_safe[:, None],
        mask=mask_h[:, None],
    )
    mid_lse_ptr = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head * stride_mid_head
        + split_kv_id * stride_mid_split
        + BLOCK_DV
    )
    tl.store(mid_lse_ptr, (e_max + tl.log2(e_sum)) * LOGE2, mask=mask_h)


@triton.jit
def _sparse_mla_merge_kernel(
    mid_out_ptr,
    out_ptr,
    h_q,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    stride_out_token,
    stride_out_head,
    NUM_KV_SPLITS: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DV_TILE: tl.constexpr,
):
    """Stage 2: N-way online-softmax merge of per-split `(out, lse)` tiles.

    Grid is `(num_tokens, num_head_groups, num_dv_tiles)`. Each program handles
    `BLOCK_H` heads × `BLOCK_DV_TILE` output-dim lanes. The LSE reduction is
    identical across DV tiles for the same (token, head) — each program
    recomputes it locally, which is cheap (O(NUM_KV_SPLITS) scalars) and
    avoids inter-CTA synchronization.
    """
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_dv_tile = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    offs_dv = cur_dv_tile * BLOCK_DV_TILE + tl.arange(0, BLOCK_DV_TILE)
    mask_dv = offs_dv < BLOCK_DV
    # Finite sentinel — same NaN guard as the split kernel for empty splits.
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV_TILE], dtype=tl.float32)

    mid_base_2d = (
        mid_out_ptr + cur_q * stride_mid_token + cur_head[:, None] * stride_mid_head
    )
    mid_lse_1d = (
        mid_out_ptr + cur_q * stride_mid_token + cur_head * stride_mid_head + BLOCK_DV
    )

    for split_kv_id in range(NUM_KV_SPLITS):
        tv = tl.load(
            mid_base_2d + split_kv_id * stride_mid_split + offs_dv[None, :],
            mask=mask_h[:, None] & mask_dv[None, :],
            other=0.0,
        )
        tlogic = tl.load(
            mid_lse_1d + split_kv_id * stride_mid_split,
            mask=mask_h,
            other=-float("inf"),
        )
        n_e_max = tl.maximum(tlogic, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        exp_logic = tl.exp(tlogic - n_e_max)
        acc = acc * old_scale[:, None] + exp_logic[:, None] * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    tl.store(
        out_ptr
        + cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :],
        (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        mask=mask_h[:, None] & mask_dv[None, :],
    )


@functools.lru_cache(maxsize=256)
def _choose_num_kv_splits(
    num_tokens: int, num_head_groups: int, index_topk: int, sm_count: int
) -> int:
    """Pick a power-of-2 split count so total programs track
    ~_SPLIT_TARGET_OCCUPANCY x SM count, independent of batch size.

    Each program serially scans its slot range, so the grid must not shrink
    as num_tokens grows; always split until each split would hold fewer than
    _MIN_TOPK_PER_SPLIT slots. Emitted values stay within the set pre-compiled
    by `_warmup_autotune` (powers of two dividing index_topk).
    """
    baseline = num_tokens * num_head_groups
    if baseline == 0:
        return 1
    target = max(1, round(_SPLIT_TARGET_OCCUPANCY * sm_count / baseline))
    num_kv_splits = triton.next_power_of_2(target)
    cap = max(1, index_topk // _MIN_TOPK_PER_SPLIT)
    num_kv_splits = min(num_kv_splits, cap)
    while num_kv_splits > 1 and index_topk % num_kv_splits != 0:
        num_kv_splits //= 2
    return num_kv_splits


def triton_mla_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int | None = None,
    sm_count: int | None = None,
) -> torch.Tensor:
    """Sparse MLA attention over topk indices.

    Args:
        q:         [num_tokens, num_heads_q, dim_qk] bf16
        kv:        [seq_kv, num_heads_kv=1, dim_qk] bf16
        indices:   [num_tokens, num_heads_kv=1, topk] int32
        sm_scale:  softmax scale
        num_kv_splits: override auto-heuristic; None/0 = auto, 1 = force single-pass.
        sm_count:  cached device SM count for the split heuristic.

    Returns:
        out:   [num_tokens, num_heads_q, _BLOCK_DV] bf16

    """
    num_tokens, num_heads_q, dim_qk = q.shape
    assert dim_qk in (_DIM_QK, _DIM_QK_NOPE), (
        f"sparse MLA kernel requires dim_qk={_DIM_QK} (DeepSeek-V3.2 / GLM-5) "
        f"or {_DIM_QK_NOPE} (rope-free, kv_lora_rank=512), got {dim_qk}"
    )
    block_dpe = dim_qk - _BLOCK_DMODEL
    assert kv.shape[1] == 1 and kv.shape[2] == dim_qk
    index_topk = indices.shape[2]
    assert index_topk % _MIN_BLOCK_N == 0, (
        f"topk ({index_topk}) must be a multiple of the smallest autotune "
        f"BLOCK_N ({_MIN_BLOCK_N})"
    )

    kv_group_num = num_heads_q
    num_head_groups = triton.cdiv(num_heads_q, min(_BLOCK_H, kv_group_num))

    if num_kv_splits is None or num_kv_splits == 0:
        if sm_count is None:
            sm_count = num_compute_units(q.device.index)
        num_kv_splits = _choose_num_kv_splits(
            num_tokens, num_head_groups, index_topk, sm_count
        )

    out = torch.empty(
        (num_tokens, num_heads_q, _BLOCK_DV),
        dtype=torch.bfloat16,
        device=q.device,
    )

    if num_kv_splits == 1:
        _sparse_mla_kernel_final[(num_tokens, num_head_groups)](
            q_buffer=q,
            k_buffer=kv,
            indices_ptr=indices,
            out_ptr=out,
            seq_kv=kv.shape[0],
            h_q=num_heads_q,
            stride_q_token=q.stride(0),
            stride_q_head=q.stride(1),
            stride_kv_token=kv.stride(0),
            stride_kv_head=kv.stride(1),
            stride_out_token=out.stride(0),
            stride_out_head=out.stride(1),
            stride_indices_token=indices.stride(0),
            stride_indices_head=indices.stride(1),
            sm_scale=sm_scale * LOG2E,
            index_topk=index_topk,
            kv_group_num=kv_group_num,
            BLOCK_H=_BLOCK_H,
            BLOCK_DV=_BLOCK_DV,
            BLOCK_DMODEL=_BLOCK_DMODEL,
            BLOCK_DPE=block_dpe,
        )
        return out

    # Split-KV: partial fp32 output + LSE per (token, head, split).
    mid_out = torch.empty(
        (num_tokens, num_heads_q, num_kv_splits, _BLOCK_DV + 1),
        dtype=torch.float32,
        device=q.device,
    )
    _sparse_mla_kernel_split[(num_tokens, num_head_groups, num_kv_splits)](
        q_buffer=q,
        k_buffer=kv,
        indices_ptr=indices,
        mid_out_ptr=mid_out,
        seq_kv=kv.shape[0],
        h_q=num_heads_q,
        stride_q_token=q.stride(0),
        stride_q_head=q.stride(1),
        stride_kv_token=kv.stride(0),
        stride_kv_head=kv.stride(1),
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_indices_token=indices.stride(0),
        stride_indices_head=indices.stride(1),
        sm_scale=sm_scale * LOG2E,
        index_topk=index_topk,
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        BLOCK_H=_BLOCK_H,
        BLOCK_DV=_BLOCK_DV,
        BLOCK_DMODEL=_BLOCK_DMODEL,
        BLOCK_DPE=block_dpe,
        LOGE2=LOGE2,
    )

    _sparse_mla_merge_kernel[(num_tokens, num_heads_q, _NUM_MERGE_DV_TILES)](
        mid_out_ptr=mid_out,
        out_ptr=out,
        h_q=num_heads_q,
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_out_token=out.stride(0),
        stride_out_head=out.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        BLOCK_H=_MERGE_BLOCK_H,
        BLOCK_DV=_BLOCK_DV,
        BLOCK_DV_TILE=_MERGE_BLOCK_DV_TILE,
        num_warps=2,
    )
    return out


# ---------------------------------------------------------------------------
# fp8_ds_mla dequant-gather (V3.2 layout: 656 bytes → 576 bf16)
# ---------------------------------------------------------------------------

_DS_MLA_NOPE_DIM = 512
_DS_MLA_ROPE_DIM = 64
_DS_MLA_CACHE_BYTES = 656
# NoPE models (qk_rope_head_dim=0) drop the 128B zeroed rope tail from the page.
_DS_MLA_CACHE_BYTES_NOPE = 528
_DS_MLA_DEQUANT_DIM = _DS_MLA_NOPE_DIM + _DS_MLA_ROPE_DIM  # 576
_DS_MLA_QUANT_BLOCK = 128
_DS_MLA_NUM_TILES = _DS_MLA_NOPE_DIM // _DS_MLA_QUANT_BLOCK  # 4


@triton.jit
def _fp8_e4m3_to_f32(x_uint8):
    """Convert fp8_e4m3fn (uint8) to float32 without hardware fp8 support.

    fp8_e4m3fn: 1 sign, 4 exponent (bias 7), 3 mantissa bits.
    Special cases: 0x00 = +0, 0x7F/0xFF = NaN (e4m3fn has no inf).
    """
    sign = (x_uint8 >> 7).to(tl.uint32)  # 0 or 1
    exp_bits = (x_uint8 & 0x78) >> 3  # 0..15
    mant_bits = x_uint8 & 0x07  # 0..7

    # Normal value: (-1)^s * 2^(e-7) * (1 + m/8)
    # fp32 sign bit at position 31
    sign_f32 = sign << 31
    f32_exp = (exp_bits.to(tl.int32) - 7 + 127).to(tl.uint32)
    f32_bits = sign_f32 | (f32_exp << 23) | (mant_bits.to(tl.uint32) << 20)

    # Subnormal (exp=0): value = (-1)^s * 2^(1-7) * (m/8) = (-1)^s * m/512
    subnorm_val = (mant_bits.to(tl.float32) / 512.0) * (tl.where(sign != 0, -1.0, 1.0))

    is_subnormal = (exp_bits == 0) & (mant_bits != 0)
    f32_bits = tl.where(is_subnormal, subnorm_val.to(tl.uint32, bitcast=True), f32_bits)

    # Zero (exp=0, mant=0): result is +0 or -0
    is_zero = (exp_bits == 0) & (mant_bits == 0)
    f32_bits = tl.where(is_zero, sign_f32, f32_bits)

    # NaN (exp=15, mant=7): both 0x7F and 0xFF are NaN in e4m3fn
    is_nan = (exp_bits == 15) & (mant_bits == 7)
    nan_bits = tl.zeros_like(x_uint8).to(tl.uint32) | 0x7FC00000
    f32_bits = tl.where(is_nan, nan_bits, f32_bits)

    return f32_bits.to(tl.float32, bitcast=True)


@triton.jit
def _dequant_ds_mla_slots_kernel(
    out_ptr,  # [total_slots, 576] bf16
    cache_ptr,  # [num_blocks, block_size, 656] uint8
    indices_ptr,  # [total_slots] int32, global slot IDs
    total_slots,
    cache_block_size: tl.constexpr,
    block_stride: tl.int64,
    nope_dim: tl.constexpr,  # 512
    rope_dim: tl.constexpr,  # 64
    quant_block: tl.constexpr,  # 128
    num_tiles: tl.constexpr,  # 4
    dequant_dim: tl.constexpr,  # 576
    cache_bytes: tl.constexpr,  # 656
    SLOTS_PER_PROG: tl.constexpr,
):
    """Dequantize fp8_ds_mla (V3.2) slots into a flat BF16 workspace.

    Cache layout per token (656 bytes):
      [0, 512):   512 float8_e4m3 values (NoPE)
      [512, 528): 4 float32 scale factors (one per 128 fp8 elements)
      [528, 656): 64 bfloat16 values (RoPE, not quantized)

    Output per token (576 bf16 = 1152 bytes):
      [0, 512):   dequantized NoPE
      [512, 576): RoPE (copied directly)

    `SLOTS_PER_PROG` slots per program: one-slot-per-program launches cap at
    ~200-260 GB/s once the workspace exceeds ~1 GB (2.8x slower than at
    small footprints); tiling restores ~900-1000 GB/s at every size.
    """
    pid = tl.program_id(0)
    offs_s = pid * SLOTS_PER_PROG + tl.arange(0, SLOTS_PER_PROG)
    in_bounds = offs_s < total_slots
    slot_idx = tl.load(indices_ptr + offs_s, mask=in_bounds, other=-1).to(tl.int64)
    valid = (slot_idx >= 0) & in_bounds
    slot = tl.maximum(slot_idx, 0)
    token_ptr = (
        cache_ptr
        + (slot // cache_block_size) * block_stride
        + (slot % cache_block_size) * cache_bytes
    )[:, None]
    scale_ptr = (token_ptr + nope_dim).to(tl.pointer_type(tl.float32))
    # int64: at chunk 2048 x topk 2048, row*dequant_dim exceeds 2**31.
    out_row = out_ptr + offs_s[:, None].to(tl.int64) * dequant_dim

    for tile_idx in tl.static_range(num_tiles):
        offsets = tile_idx * quant_block + tl.arange(0, quant_block)
        fp8_uint = tl.load(token_ptr + offsets[None, :], mask=valid[:, None], other=0)
        scale = tl.load(scale_ptr + tile_idx, mask=valid[:, None], other=0.0)
        dequant = _fp8_e4m3_to_f32(fp8_uint) * scale
        tl.store(
            out_row + offsets[None, :],
            dequant.to(tl.bfloat16),
            mask=in_bounds[:, None],
        )

    # RoPE: 64 bf16 values starting at byte offset 528 within each token.
    # rope_dim == 0 for NoPE models (writer zero-fills the 128B tail).
    if rope_dim > 0:
        rope_src = (token_ptr + nope_dim + num_tiles * 4).to(
            tl.pointer_type(tl.bfloat16)
        )
        rope_offs = tl.arange(0, rope_dim)
        rope = tl.load(rope_src + rope_offs[None, :], mask=valid[:, None], other=0.0)
        tl.store(out_row + nope_dim + rope_offs[None, :], rope, mask=in_bounds[:, None])


def dequant_ds_mla_slots(
    out: torch.Tensor,  # [total_slots, 576] bf16, pre-allocated
    cache: torch.Tensor,  # [num_blocks, block_size, 656] uint8
    indices: torch.Tensor,  # [total_slots] int32, global slot IDs
    cache_block_size: int,
    rope_dim: int = _DS_MLA_ROPE_DIM,
) -> None:
    """Dequantize fp8_ds_mla (V3.2) pages at scattered slot indices.

    Args:
        out: Pre-allocated BF16 output tensor [total_slots, 512 + rope_dim].
        cache: FP8 KV cache viewed as uint8 [num_blocks, block_size, 656].
        indices: Global slot IDs [total_slots] int32. Values < 0 are
            written as zeros (padding).
        cache_block_size: Block size (tokens per cache block).
        rope_dim: RoPE elements per token; 0 for NoPE models.

    """
    total_slots = indices.shape[0]
    if total_slots == 0:
        return
    block_stride = cache.stride(0)
    slots_per_prog = 8
    _dequant_ds_mla_slots_kernel[(triton.cdiv(total_slots, slots_per_prog),)](
        out,
        cache,
        indices,
        total_slots,
        cache_block_size=cache_block_size,
        block_stride=block_stride,
        nope_dim=_DS_MLA_NOPE_DIM,
        rope_dim=rope_dim,
        quant_block=_DS_MLA_QUANT_BLOCK,
        num_tiles=_DS_MLA_NUM_TILES,
        dequant_dim=_DS_MLA_NOPE_DIM + rope_dim,
        cache_bytes=_DS_MLA_CACHE_BYTES,
        SLOTS_PER_PROG=slots_per_prog,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# fp8_ds_mla NoPE compact-page writer (528 bytes: 512 fp8 + 4 f32 scales)
# ---------------------------------------------------------------------------


@triton.jit
def _f32_to_e4m3_sw(x):
    """Software f32 -> FP8 E4M3 bits (uint8): RNE, satfinite. Stand-in for
    cvt.rn.satfinite.e4m3x2.f32 on archs without hw FP8 (SM80), where Triton
    cannot compile fp8e4nv casts. Input must be finite with |x| <= FP8_MAX.

    Copy of deepseek_v32/common/kernels.py::_f32_to_e4m3_sw; importing that
    module would run the deepseek_v32 package __init__ (full model chain)
    and cycle back through mla_attention -> this module.
    """
    u = x.to(tl.uint32, bitcast=True)
    s = ((u >> 24) & 0x80).to(tl.uint8)
    a = u & 0x7FFFFFFF
    # Normal path: RNE-round to a 3-bit mantissa (carry folds into exponent).
    r = a + 0x0007FFFF + ((a >> 20) & 1)
    normal = (((r >> 23) - 120) << 3) | ((r >> 20) & 7)
    # Subnormal path: RNE onto the k*2^-9 grid (magic-add rounds to nearest
    # even integer; a rollup to 8 lands exactly on the smallest normal).
    y = tl.abs(x) * 512.0
    sub = (y + 12582912.0) - 12582912.0
    byte = tl.where(r >= 0x3C800000, normal, sub.to(tl.uint32)) & 0xFF
    return (byte | s.to(tl.uint32)).to(tl.uint8)


@triton.jit
def _write_nope_ds_mla_slots_kernel(
    kv_c_ptr,  # [num_tokens, nope_dim] bf16
    cache_ptr,  # [num_blocks, block_size, 528] uint8
    slots_ptr,  # [num_tokens] int64, -1 = padding
    token_stride,
    block_stride,  # bytes per cache block (int64-safe)
    entry_stride,  # bytes per token slot
    block_size,
    nope_dim: tl.constexpr,  # 512
    quant_block: tl.constexpr,  # 128
    num_tiles: tl.constexpr,  # 4
):
    """Bit-compatible with concat_and_cache_ds_mla_kernel's NoPE path:
    per-128 power-of-two scale (max_abs/448 clamped to 1e-4, exp2(ceil(log2)))
    and cvt.rn.satfinite.e4m3 of value/scale. The rope tail is simply absent
    from the 528B page, so nothing reads or writes it."""
    tok = tl.program_id(0)
    slot = tl.load(slots_ptr + tok)
    if slot < 0:
        return
    offs_t = tl.arange(0, num_tiles)
    offs = offs_t[:, None] * quant_block + tl.arange(0, quant_block)[None, :]
    v = tl.load(kv_c_ptr + tok.to(tl.int64) * token_stride + offs).to(tl.float32)
    max_abs = tl.max(tl.abs(v), axis=1)
    scale = tl.math.exp2(tl.ceil(tl.math.log2(tl.maximum(max_abs / 448.0, 1e-4))))
    q = _f32_to_e4m3_sw(v / scale[:, None])
    dst = (
        cache_ptr
        + (slot // block_size) * block_stride
        + (slot % block_size) * entry_stride
    )
    tl.store(dst + offs, q)
    tl.store(dst.to(tl.pointer_type(tl.float32)) + nope_dim // 4 + offs_t, scale)


def write_nope_ds_mla_slots(
    kv_c: torch.Tensor,  # [num_tokens, 512] bf16
    cache: torch.Tensor,  # uint8 [num_blocks, block_size, 528]
    slots: torch.Tensor,  # [num_tokens] int64
) -> None:
    """Write NoPE latents into fp8_ds_mla compact (528B) pages."""
    num_tokens = slots.shape[0]
    if num_tokens == 0:
        return
    _write_nope_ds_mla_slots_kernel[(num_tokens,)](
        kv_c,
        cache,
        slots,
        token_stride=kv_c.stride(0),
        block_stride=cache.stride(0),
        entry_stride=cache.stride(1),
        block_size=cache.shape[1],
        nope_dim=_DS_MLA_NOPE_DIM,
        quant_block=_DS_MLA_QUANT_BLOCK,
        num_tiles=_DS_MLA_NUM_TILES,
        num_warps=1,
    )


# ---------------------------------------------------------------------------
# Fused fp8_ds_mla dequant + sparse attention for prefill chunks
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_mla_prefill_fused_kernel(
    q_ptr,  # [num_tokens, h_q, 576] bf16
    cache_ptr,  # uint8 [num_blocks, block_size, 656]
    indices_ptr,  # int32 [num_tokens, topk] global slot IDs
    out_ptr,  # bf16 [num_tokens, h_q, 512]
    h_q,
    stride_q_token,
    stride_q_head,
    stride_out_token,
    stride_out_head,
    stride_indices_token,
    sm_scale,
    index_topk: tl.constexpr,
    cache_block_size: tl.constexpr,
    block_stride,  # bytes per cache block (int64-safe)
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,  # 512
    BLOCK_DPE: tl.constexpr,  # 64
    BLOCK_DV: tl.constexpr,  # 512
    QUANT_BLOCK: tl.constexpr,  # 128
    CACHE_BYTES: tl.constexpr,  # 656
):
    """One program per (token, full head set): online-softmax over the token's
    topk slots, dequantizing each 656B fp8_ds_mla slot in registers.

    Prefill-only (BLOCK_H covers all h_q heads so each cache slot is read
    exactly once; the 2-pass dequant-to-workspace path is kept for decode).
    """
    cur_q = tl.program_id(0)
    cur_h_tile = tl.program_id(1)
    offs_h = cur_h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dpe = tl.arange(0, BLOCK_DPE)
    offs_tile = tl.arange(0, BLOCK_DMODEL // QUANT_BLOCK)
    offs_q = tl.arange(0, QUANT_BLOCK)
    offs_rope = tl.arange(0, BLOCK_DPE)

    q_base = q_ptr + cur_q * stride_q_token + offs_h[:, None] * stride_q_head
    q_nope = tl.load(q_base + offs_d[None, :], mask=mask_h[:, None], other=0.0)
    q_pe = tl.load(
        q_base + BLOCK_DMODEL + offs_dpe[None, :], mask=mask_h[:, None], other=0.0
    )

    NEG_LARGE = -1.0e30
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) + NEG_LARGE
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    idx_base = indices_ptr + cur_q * stride_indices_token
    for start in range(0, index_topk, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        idx = tl.load(idx_base + offs_n, mask=offs_n < index_topk, other=-1)
        mask_kv = idx >= 0
        slot = tl.maximum(idx, 0).to(tl.int64)
        block = slot // cache_block_size
        pos = slot % cache_block_size
        tok8 = cache_ptr + (block * block_stride + pos * CACHE_BYTES)[:, None]

        # NoPE: 512 fp8_e4m3 bytes as [BN, 4, 128] tiles, scaled by the 4
        # per-128 float32 factors loaded as [BN, 4] — a [BN, 512] scale
        # gather puts 16K outstanding scalar loads per iteration in the
        # register file and collapses the allocator (measured 1328 spills).
        fp8_u = tl.load(
            tok8[:, None, :]
            + (offs_tile[:, None] * QUANT_BLOCK + offs_q[None, :])[None, :, :],
            mask=mask_kv[:, None, None],
            other=0,
        )
        scale_ptrs = (tok8 + BLOCK_DMODEL).to(tl.pointer_type(tl.float32))
        scale = tl.load(
            scale_ptrs + offs_tile[None, :], mask=mask_kv[:, None], other=0.0
        )
        nope = (_fp8_e4m3_to_f32(fp8_u) * scale[:, :, None]).to(tl.bfloat16)
        nope = tl.reshape(nope, (BLOCK_N, BLOCK_DMODEL))

        # RoPE: 64 raw bfloat16 at byte offset 512 + 4*4 = 528.
        rope_ptrs = (tok8 + BLOCK_DMODEL + 4 * 4).to(tl.pointer_type(tl.bfloat16))
        rope = tl.load(rope_ptrs + offs_rope[None, :], mask=mask_kv[:, None], other=0.0)

        qk = tl.dot(q_nope, tl.trans(nope)) + tl.dot(q_pe, tl.trans(rope))
        qk *= sm_scale
        qk = tl.where(mask_h[:, None] & mask_kv[None, :], qk, NEG_LARGE)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(tl.bfloat16), nope)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    tl.store(
        out_ptr
        + cur_q * stride_out_token
        + offs_h[:, None] * stride_out_head
        + tl.arange(0, BLOCK_DV)[None, :],
        (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        mask=mask_h[:, None],
    )


def triton_mla_sparse_attention_fp8_fused(
    q: torch.Tensor,  # [num_tokens, h_q, 576] bf16
    cache: torch.Tensor,  # uint8 [num_blocks, block_size, 656]
    indices: torch.Tensor,  # int32 [num_tokens, topk] global slot IDs
    sm_scale: float,
    block_n: int = 32,
    block_h: int | None = None,
    num_warps: int = 8,
    num_stages: int = 2,
) -> torch.Tensor:
    """Fused fp8_ds_mla dequant + sparse attention for prefill-sized batches.

    Args / Returns match `triton_mla_sparse_attention`, but `cache` is the raw
    fp8_ds_mla uint8 cache and `indices` are global slot IDs. `block_h` sets
    heads per program (slot re-read multiplier = num_heads/block_h); the
    default covers all heads in one program.
    """
    num_tokens, num_heads_q, dim_qk = q.shape
    assert dim_qk == _DIM_QK
    assert cache.shape[-1] == _DS_MLA_CACHE_BYTES
    assert indices.shape[0] == num_tokens
    index_topk = indices.shape[1]
    if block_h is None:
        block_h = triton.next_power_of_2(num_heads_q)

    out = torch.empty(
        (num_tokens, num_heads_q, _BLOCK_DV),
        dtype=torch.bfloat16,
        device=q.device,
    )
    _sparse_mla_prefill_fused_kernel[(num_tokens, triton.cdiv(num_heads_q, block_h))](
        q,
        cache,
        indices,
        out,
        num_heads_q,
        stride_q_token=q.stride(0),
        stride_q_head=q.stride(1),
        stride_out_token=out.stride(0),
        stride_out_head=out.stride(1),
        stride_indices_token=indices.stride(0),
        sm_scale=sm_scale * LOG2E,
        index_topk=index_topk,
        cache_block_size=cache.shape[1],
        block_stride=cache.stride(0),
        BLOCK_H=triton.next_power_of_2(num_heads_q),
        BLOCK_N=block_n,
        BLOCK_DMODEL=_BLOCK_DMODEL,
        BLOCK_DPE=_BLOCK_DPE,
        BLOCK_DV=_BLOCK_DV,
        QUANT_BLOCK=_DS_MLA_QUANT_BLOCK,
        CACHE_BYTES=_DS_MLA_CACHE_BYTES,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
