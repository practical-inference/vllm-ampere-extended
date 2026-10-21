# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Triton sparse MLA backend for SM80 (A100) / SM121 (GB10)."""

from typing import ClassVar

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseImpl,
    XPUMLASparseMetadata,
    XPUMLASparseMetadataBuilder,
)
from vllm.v1.attention.ops.mqa_logits_triton import (
    warmup_fp8_mqa_logits_triton,
    warmup_fp8_paged_mqa_logits_triton,
)
from vllm.v1.attention.ops.triton_merge_attn_states import (
    warmup_mask_empty_context,
)
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    _DIM_QK,
    _DIM_QK_NOPE,
    _DS_MLA_CACHE_BYTES_NOPE,
    _DS_MLA_NOPE_DIM,
    KV_SPLITS_CANDIDATES,
    dequant_ds_mla_slots,
    triton_mla_sparse_attention,
    triton_mla_sparse_attention_fp8_fused,
    write_nope_ds_mla_slots,
)
from vllm.v1.kv_cache_interface import KVCacheSpec

# V3.2 indexers don't expose `n_head`; GLM-5.1-NVFP4 sets index_n_heads=32.
# Autotune key includes (num_heads, head_dim), so a wrong warmup shape forces
# a re-tune on first real request.
_INDEXER_NUM_HEADS = 64
_INDEXER_HEAD_DIM = 128

_DS_MLA_CACHE_BYTES = 656
_DS_MLA_DEQUANT_DIM = 576

# Prefill fused-kernel dispatch: OFF pending SM80 register-feasible design.
# The prototype fused kernel (_sparse_mla_prefill_fused_kernel) is bit-exact
# but spills catastrophically on A100 ([BN,512] fp32 dequant tile +
# [64,512] acc exceed the 128-reg budget: 1120-1328 ptxas spill slots);
# 100-600 ms at 256 tokens vs 9.8 ms for the 2-pass path. Gluon-style explicit
# smem staging is the upgrade path. Until then prefill keeps the 2-pass
# dequant-workspace path and the fix ships on the 2-pass kernels instead.
_PREFILL_FUSED_MIN_TOKENS = 1 << 30

# Dedup dequant-gather dispatch: below this token count the gather is small
# enough that the sort+remap overhead outweighs the duplication it removes
# (decode batches: 133-2048 queries x topk 2048 over a long context dedup
# ~2-5x at ~0.3ms gather cost — not worth it). Prefill chunks (4096-8192
# tokens) dedup ~40x at long context.
_DEDUP_MIN_TOKENS = 4096

logger = init_logger(__name__)

# Persistent dequant-gather workspace, keyed by device, grow-only in slots.
_DS_MLA_DEQUANT_WS: dict[tuple[torch.device, int], torch.Tensor] = {}
# ponytail: every superseded workspace stays alive for the process lifetime.
# Captured cudagraphs bake raw addresses of the workspace they were captured
# with; dropping the old tensor lets the caching allocator reuse that memory
# (eager grow path) and corrupts later replays (device-side index asserts,
# garbage attention). Bounded: one tensor per grow event.
_DS_MLA_DEQUANT_WS_LIVE: list[torch.Tensor] = []


def _get_ds_mla_dequant_workspace(
    device: torch.device, total_slots: int, dequant_dim: int = _DS_MLA_DEQUANT_DIM
):
    ws = _DS_MLA_DEQUANT_WS.get((device, dequant_dim))
    if ws is None or ws.shape[0] < total_slots:
        ws = torch.empty(
            (total_slots, 1, dequant_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        _DS_MLA_DEQUANT_WS_LIVE.append(ws)
        _DS_MLA_DEQUANT_WS[(device, dequant_dim)] = ws
    return ws[:total_slots]


def _dedup_dequant_gather(
    device: torch.device,
    u8_cache: torch.Tensor,  # [num_blocks, block_size, cache_bytes] uint8
    topk_indices: torch.Tensor,  # [num_tokens, topk_w] int32 global slot ids, -1 pad
    dequant_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deduplicating dequant-gather for large sparse-MLA batches.

    The 2-pass path gathers ``num_tokens x topk`` rows per call (16.8M at an
    8192-token chunk, topk 2048) while those queries select from at most
    ``max_seq_len + tail`` distinct slots — at long context the gather is
    ~40x duplicated work (prod trace 2026-10-21: ``_dequant_ds_mla_slots_kernel``
    = 26% of prefill CUDA time, 2x the attention kernel itself).

    Sort + inclusive-cumsum ranks + idempotent scatter: every unique slot is
    dequantized once into the front of the persistent workspace and per-query
    indices are remapped to workspace ranks. All shapes are host-known — zero
    device syncs. Bit-exact vs the 2-pass path: each query reads the same
    dequantized row values in the same topk order.

    Returns (workspace_view, ws_indices) for ``_forward_bf16_kv``.
    """
    num_tokens, topk_w = topk_indices.shape
    flat = topk_indices.reshape(-1)
    valid = flat >= 0
    # Clamp pads to 0 so they dedup with any real slot-0 picks; -1 is
    # restored per position at the end.
    flat_c = torch.where(valid, flat, torch.zeros_like(flat)).to(torch.int32)
    sorted_idx, perm = torch.sort(flat_c)
    # First-occurrence flag: position 0 compares against a -2 sentinel.
    prev = torch.empty_like(sorted_idx)
    prev[0] = -2
    prev[1:] = sorted_idx[:-1]
    isf = (sorted_idx != prev).to(torch.int32)
    # Inclusive-cumsum rank: duplicates inherit their run's first-occurrence
    # rank ([5,5,7] -> isf [1,0,1] -> cumsum [1,1,2] -> ranks [0,0,1]).
    ranks = (torch.cumsum(isf, 0) - 1).to(torch.int32)
    # Host upper bound on unique slots. Airtight: every valid global slot id
    # (block-table context positions and kpool tail slots alike) addresses the
    # same cache, so unique <= min(index count, pool rows). The dequant runs
    # over this bound; rows past the true unique count re-dequant slot 0
    # harmlessly (attention never reads them — ranks < unique count).
    pool_rows = u8_cache.shape[0] * u8_cache.shape[1]
    u_upper = max(min(flat.numel(), pool_rows), 1)
    # Zero-init: rows >= unique-count dequant slot 0 harmlessly (attention
    # never reads them — ranks < unique count by construction).
    uniq = torch.zeros(u_upper, dtype=torch.int32, device=flat.device)
    uniq.scatter_(0, ranks.to(torch.int64), sorted_idx)
    workspace = _get_ds_mla_dequant_workspace(device, u_upper, dequant_dim)
    dequant_ds_mla_slots(
        workspace.reshape(-1, dequant_dim),
        u8_cache,
        uniq,
        cache_block_size=u8_cache.shape[1],
        rope_dim=dequant_dim - _DS_MLA_NOPE_DIM,
    )
    inverse = torch.empty_like(ranks)
    inverse.scatter_(0, perm, ranks)
    ws_indices = torch.where(valid, inverse, torch.full_like(inverse, -1)).view(
        num_tokens, 1, topk_w
    )
    return workspace, ws_indices


class TritonMLASparseMetadataBuilder(XPUMLASparseMetadataBuilder):
    # XPU base keeps NEVER (not validated under cudagraph); this subclass
    # claims UNIFORM_BATCH for the CUDA/Triton path.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    require_uniform_decodes: ClassVar[bool] = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # XPU base never calls AttentionMetadataBuilder.__init__, so the
        # shared helper's config reads would fail; backfill it.
        self.vllm_config = args[2] if len(args) > 2 else kwargs["vllm_config"]
        # Without this, reorder_batch_threshold defaults to None
        # -> decode_threshold=1, so 2-token spec decode is classified as
        # prefill and takes the wrong attention path.
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)


class TritonMLASparseImpl(XPUMLASparseImpl):
    """Triton sparse-MLA impl with split-KV decode (3-7× faster than the
    single-pass XPU base for single-query decode on SM80 / SM121)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sm_count: int | None = None
        if self.topk_indices_buffer is not None:
            self._sm_count = num_compute_units(self.topk_indices_buffer.device.index)
        self._warmup_autotune(kwargs["indexer"])

    def record_logical_topk_ready(self) -> None:
        # This impl shares the top-k indices buffer via SharedTopkIndicesBuffer
        # but does not participate in sparse-MLA index groups.
        pass

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        if kv_cache_dtype == "fp8_ds_mla" and kv_cache.shape[-1] == (
            _DS_MLA_CACHE_BYTES_NOPE
        ):
            # NoPE compact 528B pages: the CUDA ds_mla writer hard-requires
            # the 656B page (it zero-fills the rope tail), so write the
            # NoPE part with the bit-compatible Triton kernel instead.
            write_nope_ds_mla_slots(
                kv_c_normed,
                kv_cache.view(torch.uint8),
                slot_mapping.flatten(),
            )
            return
        super().do_kv_cache_update(
            kv_c_normed,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    def _warmup_autotune(self, indexer) -> None:
        """Prime `@triton.autotune` caches at init so the first request
        doesn't pay the inline config-sweep cost."""
        if self.topk_indices_buffer is None:
            return
        device = self.topk_indices_buffer.device
        topk = self.topk_indices_buffer.shape[-1]
        # Pre-size the persistent scratch to the worst case here, outside any
        # capture: a grow-inside-capture allocation is illegal (pool OOM
        # poisons the capture), and a post-init grow lands the workspace
        # outside the init memory budget (stages sat at ~97% of physical
        # memory once the 1024-token prefill workspace materialized). Dequant
        # slots are driven by prefill: max_num_batched_tokens * topk.
        cfg = get_current_vllm_config_or_none()
        if cfg is not None:
            max_seqs = cfg.scheduler_config.max_num_seqs
            # Spec-decode verify/decode batches carry (1 + K) tokens per
            # request; without the factor the first spec capture grows the
            # workspace (see _DS_MLA_DEQUANT_WS_LIVE for why grows are bad).
            spec_tokens = (
                getattr(cfg.speculative_config, "num_speculative_tokens", 0) or 0
            )
            decode_tokens = max_seqs * (1 + spec_tokens)
            _get_ds_mla_dequant_workspace(
                device,
                max(decode_tokens, cfg.scheduler_config.max_num_batched_tokens) * topk,
                self.head_size,
            )
            from vllm.v1.attention.ops.mqa_logits_triton import (
                _get_paged_mqa_logits_scratch,
            )

            # Paged-mqa logits is decode-only (B * next_n rows, B <=
            # max_num_seqs); sizing it by the prefill budget would waste 2 GiB.
            _get_paged_mqa_logits_scratch(
                device, decode_tokens, cfg.model_config.max_model_len, clean=False
            )
        dim_qk = self.head_size
        q = torch.empty(1, self.num_heads, dim_qk, dtype=torch.bfloat16, device=device)
        kv = torch.empty(64, 1, dim_qk, dtype=torch.bfloat16, device=device)
        indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)
        for splits in KV_SPLITS_CANDIDATES:
            triton_mla_sparse_attention(
                q,
                kv,
                indices,
                sm_scale=self.softmax_scale,
                num_kv_splits=splits,
                sm_count=self._sm_count,
            )
        indexer_num_heads = getattr(indexer, "n_head", None)
        indexer_head_dim = getattr(indexer, "head_dim", None)
        if indexer_num_heads is None or indexer_head_dim is None:
            cfg = get_current_vllm_config_or_none()
            if cfg is not None:
                hf = cfg.model_config.hf_config
                indexer_num_heads = indexer_num_heads or getattr(
                    hf, "index_n_heads", _INDEXER_NUM_HEADS
                )
                indexer_head_dim = indexer_head_dim or getattr(
                    hf, "index_head_dim", _INDEXER_HEAD_DIM
                )
            else:
                indexer_num_heads = indexer_num_heads or _INDEXER_NUM_HEADS
                indexer_head_dim = indexer_head_dim or _INDEXER_HEAD_DIM
        warmup_fp8_mqa_logits_triton(
            num_heads=indexer_num_heads or _INDEXER_NUM_HEADS,
            head_dim=indexer_head_dim or _INDEXER_HEAD_DIM,
            device=device,
        )
        cfg = get_current_vllm_config_or_none()
        if cfg is not None:
            block_size = cfg.cache_config.block_size
        elif indexer.k_cache is not None:
            block_size = indexer.k_cache.shape[1]
        else:
            block_size = 64
        warmup_fp8_paged_mqa_logits_triton(
            num_heads=indexer_num_heads or _INDEXER_NUM_HEADS,
            head_dim=indexer_head_dim or _INDEXER_HEAD_DIM,
            block_size=block_size,
            device=device,
        )
        warmup_mask_empty_context(
            num_heads=self.num_heads, head_size=self.head_size, device=device
        )

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, heads, d_qk]
        topk_indices: torch.Tensor,  # [sq, topk]
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = triton_mla_sparse_attention(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            sm_scale=self.softmax_scale,
            sm_count=self._sm_count,
        )
        return output[:, : self.num_heads, :]

    def _forward_fp8_ds_mla_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, block_size, 656] uint8
        topk_indices: torch.Tensor,  # [sq, topk] global slot IDs
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        """Dequant-gather fp8_ds_mla KV into bf16 workspace, then run
        the existing bf16 sparse MLA attention kernel.

        The workspace is small (num_tokens * topk * 576 bytes) and stays
        in L2 cache for decode, so the extra write+read is nearly free.
        """
        num_tokens = q.shape[0]
        topk = topk_indices.shape[-1]

        if num_tokens >= _PREFILL_FUSED_MIN_TOKENS:
            fused_out = triton_mla_sparse_attention_fp8_fused(
                q,
                kv_c_and_k_pe_cache.view(torch.uint8),
                topk_indices,
                sm_scale=self.softmax_scale,
            )
            return fused_out[:, : self.num_heads, :]

        u8_cache = kv_c_and_k_pe_cache.view(torch.uint8)

        if num_tokens >= _DEDUP_MIN_TOKENS:
            workspace, ws_indices = _dedup_dequant_gather(
                q.device, u8_cache, topk_indices, self.head_size
            )
            return self._forward_bf16_kv(q, workspace, ws_indices, attn_metadata)

        total_slots = num_tokens * topk

        # Flatten topk indices to 1D for the gather kernel
        flat_indices = topk_indices.reshape(-1).to(torch.int32)

        # Persistent bf16 workspace (grow-only, module-cached, shared across
        # layers — layers run sequentially on-stream): [total_slots, 1, d_qk].
        # A per-call allocation of this size lands in every cudagraph's
        # private pool (~150 MiB per captured size at topk=2048).
        dequant_dim = self.head_size
        workspace = _get_ds_mla_dequant_workspace(q.device, total_slots, dequant_dim)

        # Dequant-gather from fp8_ds_mla cache into bf16 workspace
        ws_rows = workspace.reshape(total_slots, dequant_dim)
        dequant_ds_mla_slots(
            ws_rows,
            u8_cache,
            flat_indices,
            cache_block_size=attn_metadata.block_size,
            rope_dim=dequant_dim - _DS_MLA_NOPE_DIM,
        )

        # Remap topk indices into workspace positions:
        # token t, position p -> t * topk + p
        # Preserve -1 (invalid/padding) entries from the original indices
        ws_base = torch.arange(
            num_tokens, device=q.device, dtype=torch.int32
        ).unsqueeze(1) * topk + torch.arange(
            topk, device=q.device, dtype=torch.int32
        ).unsqueeze(0)
        ws_indices = torch.where(
            topk_indices >= 0,
            ws_base,
            torch.full_like(ws_base, -1),
        ).view(num_tokens, 1, -1)

        return self._forward_bf16_kv(q, workspace, ws_indices, attn_metadata)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: XPUMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode

        # Concatenate q if it's a tuple (ql_nope, q_pe)
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        topk_indices_global = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            # Buffer is padded to the kernel tile width (topk + kpool tail,
            # rounded); -1 pad slots stay masked in the attention kernel.
            # Same contract as FLASHMLA_SPARSE._forward_bf16_kv.
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )
        assert isinstance(topk_indices_global, torch.Tensor)

        if self.kv_cache_dtype == "fp8_ds_mla":
            attn_out = self._forward_fp8_ds_mla_kv(
                q, kv_c_and_k_pe_cache, topk_indices_global, attn_metadata
            )
        else:
            attn_out = self._forward_bf16_kv(
                q, kv_c_and_k_pe_cache, topk_indices_global, attn_metadata
            )

        return attn_out, None


class TritonMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: "KVCacheSpec | None" = None,
    ) -> list[int | MultipleOf]:
        # The DSA indexer backend requires block size 64 on CUDA and shares
        # the KV cache group with this backend; the base-class MultipleOf(1)
        # default lets auto-selection settle on 16, which then fails
        # select_common_block_size ("No common block size for 16").
        # MultipleOf(64) (rather than [64]) keeps larger user-specified
        # sizes like 128 usable, which measurably lowers profile-time peak
        # memory for very long contexts.
        return [MultipleOf(64)]

    @staticmethod
    def get_metadata_cls() -> type[XPUMLASparseMetadata]:
        return XPUMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["TritonMLASparseMetadataBuilder"]:
        return TritonMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["TritonMLASparseImpl"]:
        return TritonMLASparseImpl

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            if head_size == _DIM_QK_NOPE:
                return (num_blocks, block_size, _DS_MLA_CACHE_BYTES_NOPE)
            return (num_blocks, block_size, _DS_MLA_CACHE_BYTES)
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [_DIM_QK, _DIM_QK_NOPE]

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if head_size == _DIM_QK_NOPE:
            # 512 is the rope-free NoPE shape (kv_lora_rank=512,
            # qk_rope_head_dim=0). The fp8_ds_mla dequant path runs it with
            # rope_dim=0 (128B rope tail in the 656B page stays zeroed).
            if kv_cache_dtype not in (
                None,
                "auto",
                "bfloat16",
                "float16",
                "fp8_ds_mla",
            ):
                return (
                    "TRITON_MLA_SPARSE supports head_size 512 only with "
                    f"bf16/fp16 kv-cache, got {kv_cache_dtype}"
                )
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
            if vllm_config.model_config is not None:
                hf_text_config = vllm_config.model_config.hf_text_config
                if getattr(hf_text_config, "qk_rope_head_dim", 64) != 0:
                    return (
                        "TRITON_MLA_SPARSE supports head_size 512 only for "
                        "rope-free (NoPE) models"
                    )
        return None

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True
