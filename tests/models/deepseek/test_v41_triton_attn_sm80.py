# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 Triton decode attention composition for DeepSeek V4.1.

Exercises the pieces ``DeepseekV4TritonAttention._forward_decode`` wires
together — 584B legacy-record gathers (software e4m3 dequant) into the
``[T*W, T*W + T*K)`` workspace, the ``_decode_ws_ids`` row mapping, and
``triton_mla_sparse_attention`` with attn_sink — against a plain-torch fp32
reference (formula dequant + full softmax over the masked SWA/topk union with
the sink as an extra denominator term).
"""

import pytest
import torch

from vllm.models.deepseek_v41.nvidia.triton_sparse import (
    _decode_ws_ids,
    _dequant_v41_rows,
)
from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    triton_mla_sparse_attention,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton dequant/attention kernels require CUDA/ROCm",
)

FP8_DIM = 448
ROPE_DIM = 64
TOKEN_DATA_BYTES = FP8_DIM + ROPE_DIM * 2  # 576
NUM_SCALES = FP8_DIM // 64  # 7
SCALE_BYTES = NUM_SCALES + 1  # 8 (1 pad byte)
TOKEN_BYTES = TOKEN_DATA_BYTES + SCALE_BYTES  # 584
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

T = 4
REAL_HEADS = 8
PADDED_HEADS = 16
W = 128
K = 512
RATIO = 2
SWA_BS = 64
# attn_metadata.block_size // compress_ratio == 128 // 2
COMPRESSED_BS = 128 // RATIO
SM_SCALE = 512**-0.5


def _write_v4_record(k: torch.Tensor, block_size: int) -> torch.Tensor:
    """Encode bf16 K rows with the V4 writer into [nb, bs, 584] uint8 pages."""
    n = k.shape[0]
    num_blocks = n // block_size
    tiles = k[:, :FP8_DIM].float().view(n, NUM_SCALES, 64)
    amax = tiles.abs().amax(2).clamp(min=1e-4)
    exponent = torch.ceil(torch.log2(amax / FP8_MAX))
    scale = torch.exp2(exponent)
    fp8 = (tiles / scale.unsqueeze(2)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    scale_bytes = (exponent + 127).clamp(0, 255).to(torch.uint8)

    cache = torch.zeros(
        num_blocks, block_size, TOKEN_BYTES, dtype=torch.uint8, device="cuda"
    )
    for blk in range(num_blocks):
        rows = slice(blk * block_size, (blk + 1) * block_size)
        page = cache[blk].view(-1)
        data = torch.cat(
            [
                fp8[rows].view(block_size, FP8_DIM).view(torch.uint8),
                k[rows, FP8_DIM:].contiguous().view(torch.uint8),
            ],
            dim=1,
        )
        page[: block_size * TOKEN_DATA_BYTES] = data.reshape(-1)
        scales = torch.cat(
            [scale_bytes[rows], torch.zeros(block_size, 1, dtype=torch.uint8)], dim=1
        )
        lo = block_size * TOKEN_DATA_BYTES
        page[lo : lo + block_size * SCALE_BYTES] = scales.reshape(-1)
    return cache


def _dequant_reference(cache, slots, block_size):
    """Plain-torch fp32 dequant of the 584B record by formula (-1 → zeros)."""
    out = torch.zeros(slots.shape[0], 512, dtype=torch.float32, device="cuda")
    valid = slots >= 0
    s = slots[valid].long()
    pages = cache.view(cache.shape[0], -1)[s // block_size]
    data_base = (s % block_size * TOKEN_DATA_BYTES).unsqueeze(1)
    data = pages.gather(1, data_base + torch.arange(TOKEN_DATA_BYTES, device="cuda"))
    fp8 = data[:, :FP8_DIM].view(torch.float8_e4m3fn).float()
    sf_base = (s % block_size) * SCALE_BYTES + block_size * TOKEN_DATA_BYTES
    sf = pages.gather(1, sf_base.unsqueeze(1) + torch.arange(NUM_SCALES, device="cuda"))
    scale = torch.exp2(sf.float() - 127.0).repeat_interleave(64, dim=1)
    row = torch.cat([fp8 * scale, data[:, FP8_DIM:].view(torch.bfloat16).float()], 1)
    # The Triton gather materializes the workspace in bf16; mirror that.
    out[valid] = row.to(torch.bfloat16).float()
    return out


def test_decode_gather_attention_matches_torch_reference():
    torch.manual_seed(0)
    dev = "cuda"

    swa_k = torch.randn(SWA_BS * 4, 512, dtype=torch.bfloat16)
    comp_k = torch.randn(COMPRESSED_BS * 4, 512, dtype=torch.bfloat16)
    swa_cache = _write_v4_record(swa_k, SWA_BS)
    comp_cache = _write_v4_record(comp_k, COMPRESSED_BS)

    # SWA lens at full width and truncated; topk lens at full and truncated.
    swa_lens = torch.tensor([W, W // 2, 1, 0], dtype=torch.int32, device=dev)
    topk_lens = torch.tensor([K, K // 3, K, 2], dtype=torch.int32, device=dev)
    cols = torch.arange(W + K, device=dev, dtype=torch.int32).unsqueeze(0)
    valid = (cols < W) & (cols < swa_lens.unsqueeze(1))
    valid |= (cols >= W) & (cols - W < topk_lens.unsqueeze(1))
    slots = torch.full((T, W + K), -1, dtype=torch.int32, device=dev)
    slots.random_(0, SWA_BS * 4)
    swa_slots = torch.where(valid[:, :W], slots[:, :W], -1).contiguous()
    comp_slots = torch.where(valid[:, W:], slots[:, W:], -1).contiguous()

    ws = torch.zeros(T, W + K, 512, dtype=torch.bfloat16, device=dev)
    ws_rows = ws.view(T * (W + K), 512)
    _dequant_v41_rows(ws_rows[: T * W], swa_cache, swa_slots.view(-1), SWA_BS)
    _dequant_v41_rows(ws_rows[T * W :], comp_cache, comp_slots.view(-1), COMPRESSED_BS)

    ids = _decode_ws_ids(T, W, K, swa_lens, topk_lens, torch.device(dev))
    q = torch.randn(T, PADDED_HEADS, 512, dtype=torch.bfloat16, device=dev)
    sink = torch.full((PADDED_HEADS,), -float("inf"), dtype=torch.float32, device=dev)
    sink[:REAL_HEADS] = torch.tensor(
        [0.5, -2.0, 3.0, -float("inf"), 0.0, 7.5, -1.25, 1.0], device=dev
    )

    out = triton_mla_sparse_attention(
        q, ws.view(-1, 1, 512), ids, SM_SCALE, attn_sink=sink
    )

    # fp32 reference over the same workspace rows and mask.
    rows_swa = _dequant_reference(swa_cache, swa_slots.view(-1), SWA_BS)
    rows_comp = _dequant_reference(comp_cache, comp_slots.view(-1), COMPRESSED_BS)
    rows = torch.cat([rows_swa.view(T, W, 512), rows_comp.view(T, K, 512)], 1)
    s = torch.bmm(q.float(), rows.transpose(1, 2)) * SM_SCALE  # [T, H, W+K]
    s[ids.expand(T, PADDED_HEADS, W + K) < 0] = float("-inf")
    m = s.amax(2, keepdim=True).clamp(min=-1e30)
    p = (s - m).exp()
    den = p.sum(2, keepdim=True) + (sink.view(1, PADDED_HEADS, 1) - m).exp()
    ref = torch.matmul(p, rows) / den

    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
