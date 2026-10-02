# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 sparse MLA attention on the Triton path, for SM80.

SM80 (A100) cannot run FlashMLA (SM90+ TMA/FP8 kernels), and the CuteDSL /
Triton ``dequantize_and_gather_k_cache`` gather either fails to compile or
casts to fp8e4nv. The pure-Triton pieces exist: ``dequant_ds_mla_slots``
index-gathers the legacy 584B fp8 records into a bf16 workspace with a
software e4m3 decode, and ``triton_mla_sparse_attention`` scores that
workspace with attn-sink folding. This layer keeps the FlashMLA layer's
metadata flow (chunk plan, index combine, topk mapping) and replaces every
kernel call with gather-to-workspace + Triton attention.
"""

from typing import TYPE_CHECKING, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v41.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v41.nvidia.flashmla import (
    DeepseekSparseSWAFlashMLABackend,
    DeepseekV4FlashMLAAttention,
)
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import round_up
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    dequant_ds_mla_slots,
    triton_mla_sparse_attention,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
    )
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

# DeepSeek-V4 legacy 584B fp8 record: 448 e4m3 NoPE + 64 raw bf16 RoPE,
# 7 UE8M0 scales per 64 dims plus one pad byte (the segregated-scale
# layout ``dequant_ds_mla_slots`` reads).
_V4_NOPE_DIM = 448
_V4_ROPE_DIM = 64
_V4_QUANT_BLOCK = 64
# Smallest topk width triton_mla_sparse_attention's autotune sweep offers.
_MIN_TOPK_WIDTH = 16


def _paged_slot_ids(
    block_table: torch.Tensor,
    block_size: int,
    positions: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Global cache slot ids (int32) for paged rows at ``positions``.

    Invalid rows are clamped inside the block table before the lookup, then
    masked to -1 (the dequant gather's zero-fill sentinel).
    """
    rows = torch.arange(positions.shape[0], device=positions.device).unsqueeze(1)
    # Advanced indexing requires int64 column ids.
    col = torch.clamp(positions // block_size, max=block_table.shape[1] - 1).long()
    slots = block_table[rows, col] * block_size + positions % block_size
    return torch.where(valid, slots, -1).to(torch.int32)


def _dequant_v41_rows(
    ws_rows: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    cache_block_size: int,
) -> None:
    """Gather the V4.1 legacy 584B fp8 records at ``slots`` into bf16 rows."""
    dequant_ds_mla_slots(
        ws_rows,
        cache.view(torch.uint8),
        slots,
        cache_block_size=cache_block_size,
        rope_dim=_V4_ROPE_DIM,
        nope_dim=_V4_NOPE_DIM,
        quant_block=_V4_QUANT_BLOCK,
        segregated_scale=True,
    )


def _inverse_gptj_rope(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Undo the query's GPT-J rotation on the last ``rope_dim`` dims (fp32).

    Same partner/sign formula as ``fused_inv_rope_fp8_quant`` (which cannot
    compile on SM80), for the bf16 ``_o_proj`` fallback.
    """
    cos_sin = cos_sin_cache[positions].float()
    half = rope_dim // 2
    cos = cos_sin[:, :half].unsqueeze(1)
    sin = cos_sin[:, half : 2 * half].unsqueeze(1)
    nope = o.shape[-1] - rope_dim
    head = o[..., nope : nope + rope_dim].float().unflatten(-1, (-1, 2))
    even = head[..., 0] * cos + head[..., 1] * sin
    odd = head[..., 1] * cos - head[..., 0] * sin
    rotated = torch.stack([even, odd], dim=-1).flatten(-2)
    return torch.cat([o[..., :nope].float(), rotated], dim=-1)


def _decode_ws_ids(
    num_tokens: int,
    width_swa: int,
    width_topk: int,
    swa_lens: torch.Tensor,
    topk_lens: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Attention row ids [T, 1, W+K] int32 into the decode gather workspace.

    Workspace rows ``[0, T*W)`` hold the SWA gather (token-major), rows
    ``[T*W, T*(W+K))`` the compressed topk gather; masked columns are -1
    (the kernel's no-op sentinel), replacing flash_mla's ``topk_length``.
    """
    tok = torch.arange(num_tokens, device=device, dtype=torch.int32).unsqueeze(1)
    cols = torch.arange(width_swa + width_topk, device=device, dtype=torch.int32)
    ids = torch.where(
        cols < swa_lens.unsqueeze(1),
        tok * width_swa + cols,
        torch.full((1, 1), -1, dtype=torch.int32, device=device),
    )
    if width_topk:
        assert topk_lens is not None
        topk_valid = (cols >= width_swa) & (cols - width_swa < topk_lens.unsqueeze(1))
        topk_ids = num_tokens * width_swa + tok * width_topk + (cols - width_swa)
        ids = torch.where(topk_valid, topk_ids, ids)
    return ids.unsqueeze(1)


class DeepseekV4TritonFlashMLABackend(DeepseekV4FlashMLABackend):
    """The FlashMLA compressed-cache backend, extended to SM80.

    Only the capability gate changes: the metadata and block geometry are
    consumed by the Triton kernels here, not by FlashMLA itself.
    """

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekSparseSWATritonFlashMLABackend(DeepseekSparseSWAFlashMLABackend):
    """The FlashMLA-flavoured SWA backend (varlen decode), extended to SM80.

    The builder's tile-scheduler metadata is still allocated; the Triton
    attention path never consumes it.
    """

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV4TritonAttention(DeepseekV4FlashMLAAttention):
    """DeepSeek V4.1 sparse MLA on Triton dequant-gather + Triton attention.

    Same flow as ``DeepseekV4FlashMLAAttention``; the FlashMLA kernels are
    replaced by ``dequant_ds_mla_slots`` gathers into the layer's bf16
    workspace and ``triton_mla_sparse_attention`` on top.
    """

    backend_cls = DeepseekV4TritonFlashMLABackend
    swa_backend_cls = DeepseekSparseSWATritonFlashMLABackend

    def _indexer_cls(self, k_cache: object) -> type:
        # Lazy import: the SM80 indexer module pulls in the Triton FP8
        # logits kernels, which this module does not need at import time.
        from vllm.models.deepseek_v41.nvidia.indexer_sm80 import (
            DeepseekV4TritonIndexer,
        )

        return DeepseekV4TritonIndexer

    def _o_proj(
        self,
        attn_out: "torch.Tensor | QuantizedActivation",
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # SM80 verdict (measured on GPU): the superclass dsv41_o_proj cannot
        # run here -- deep_gemm's fp8_einsum is unavailable (SM90+ only), and
        # fused_inv_rope_fp8_quant fails Triton compilation even with
        # quantize=False (the dead fp8e4nv branch still compiles). Plain-torch
        # bf16 fallback: inverse RoPE + grouped bmm through wo_a.
        assert isinstance(attn_out, torch.Tensor)
        o = attn_out[:, : self.n_local_heads, :]
        o = _inverse_gptj_rope(
            o, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim
        )
        o = o.reshape(o.shape[0], self.n_local_groups, -1).to(torch.bfloat16)
        weight = cast(torch.Tensor, self.wo_a.weight)
        if weight.element_size() < 2:
            raise NotImplementedError(
                "DeepseekV4TritonAttention's bf16 o_proj needs a bf16 wo_a; "
                "no SM80 grouped fp8/nvfp4 einsum kernel is wired up yet."
            )
        grouped = weight.view(self.n_local_groups, self.o_lora_rank, -1).to(o.dtype)
        z = torch.empty(
            (o.shape[0], self.n_local_groups, self.o_lora_rank),
            dtype=torch.bfloat16,
            device=o.device,
        )
        torch.bmm(o.transpose(0, 1), grouped.transpose(1, 2), out=z.transpose(0, 1))
        return self._wo_b_proj(z.flatten(1))

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: reserve the same workspaces the real gathers
            # take (per-chunk prefill, per-token decode), one get_simultaneous
            # per runtime call site so the growth matches. The dequant / topk /
            # attention kernels are skipped this step.
            swa_only = self.compress_ratio == 0
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            if swa_only:
                top_k = 0
            else:
                assert self.topk_indices_buffer is not None
                top_k = self.topk_indices_buffer.shape[-1]
            combined_topk = round_up(top_k + self.window_size, 128)
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            current_workspace_manager().get_simultaneous(
                (
                    (self.max_num_batched_tokens, self.window_size + top_k, 512),
                    torch.bfloat16,
                ),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        # Compressed-cache metadata lives on the kv-source layer's prefix;
        # consumers share that cache and its block table.
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio == 0
        self_kv_cache = None if swa_only else self._compressed_kv_cache()
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _combine_to_ws_ids(
        self,
        combined_indices: torch.Tensor,
        combined_lens: torch.Tensor,
        M: int,
        N: int,
        chunk_size: int,
    ) -> torch.Tensor:
        """Remap combine_topk_swa_indices rows into this workspace's layout.

        The combiner emits ``req * M + local`` ids for an M-wide per-request
        slab; this workspace keeps the compressed region request-major first
        (rows ``[0, chunk_size*N)``), then the SWA gather region
        (``[chunk_size*N, chunk_size*M)``). Entries past ``combined_lens``
        hold workspace leftovers, so mask them to -1: unlike
        flash_mla_sparse_fwd, ``triton_mla_sparse_attention`` has no
        ``topk_length``.
        """
        device = combined_indices.device
        cols = torch.arange(
            combined_indices.shape[1], device=device, dtype=torch.int32
        ).unsqueeze(0)
        valid = cols < combined_lens.unsqueeze(1)
        req = combined_indices // M
        loc = combined_indices - req * M
        gather_base = chunk_size * N
        remapped = torch.where(
            loc < N,
            req * N + loc,
            gather_base + (req * (M - N) + loc - N),
        )
        return torch.where(valid, remapped, torch.full_like(remapped, -1)).unsqueeze(1)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # None for SWA-only layers
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = self.compress_ratio == 0

        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        # Local indices filled by the index source; SWA-only layers pass
        # top_k=0 and never read them.
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[num_decode_tokens:]
        topk_indices = topk_indices[:num_prefill_tokens]
        top_k = 0 if swa_only else topk_indices.shape[-1]
        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
            has_compressed=not swa_only,
        )
        assert chunk_plan, "prefill chunk plan must be non-empty when num_prefills > 0"
        workspace_manager = current_workspace_manager()
        combined_topk = round_up(top_k + self.window_size, 128)
        device = q.device
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
            chunk_size = chunk_end - chunk_start
            workspace = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_M, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            kv, combined_indices_out, combined_lens_out = workspace
            ws_rows = kv.view(chunk_size * chunk_M, q.shape[-1])
            # The gather regions are request-major-contiguous (compressed
            # rows before all SWA rows, not interleaved per request inside M)
            # so each cache gather is one dequant_ds_mla_slots call.
            if not swa_only:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                cbs = attn_metadata.block_size // self.compress_ratio
                bt = attn_metadata.block_table[num_decodes:][chunk_start:chunk_end]
                pos = (
                    torch.arange(chunk_N, device=device, dtype=torch.int32)
                    .unsqueeze(0)
                    .expand(chunk_size, chunk_N)
                )
                comp_lens = seq_lens[chunk_start:chunk_end] // self.compress_ratio
                slots = _paged_slot_ids(bt, cbs, pos, pos < comp_lens.unsqueeze(1))
                _dequant_v41_rows(
                    ws_rows[: chunk_size * chunk_N],
                    compressed_k_cache,
                    slots.view(-1),
                    cbs,
                )

            # Gather SWA KV: per request the window rows
            # [seq_len - gather_len, seq_len), at region offset chunk_N.
            sbs = swa_metadata.block_size
            swa_bt = swa_metadata.block_table[num_decodes:][chunk_start:chunk_end]
            gl = gather_lens[chunk_start:chunk_end]
            gather_width = chunk_M - chunk_N
            pos_s = (seq_lens[chunk_start:chunk_end] - gl).unsqueeze(1) + torch.arange(
                gather_width, device=device, dtype=torch.int32
            )
            cols_g = torch.arange(gather_width, device=device, dtype=torch.int32)
            slots_s = _paged_slot_ids(swa_bt, sbs, pos_s, cols_g < gl.unsqueeze(1))
            _dequant_v41_rows(
                ws_rows[chunk_size * chunk_N :],
                swa_k_cache,
                slots_s.view(-1),
                sbs,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            combined_indices_out = combined_indices_out[: query_end - query_start]
            combined_lens_out = combined_lens_out[: query_end - query_start]

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_M,
                chunk_N,
                out=(combined_indices_out, combined_lens_out),
            )
            ws_indices = self._combine_to_ws_ids(
                combined_indices, combined_lens, chunk_M, chunk_N, chunk_size
            )
            attn = triton_mla_sparse_attention(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=ws_indices,
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
            )
            output[query_start:query_end].copy_(attn)

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # None for SWA-only layers
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_lens = None
        global_indices = None
        if not swa_only:
            # Local indices filled by the index-source layer's indexer.
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            assert swa_metadata.token_to_req_indices is not None
            assert self.topk_indices_buffer is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            global_indices, topk_lens = compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:num_decode_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                block_size,
                is_valid,
            )

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None and swa_lens is not None
        num_tokens = num_decode_tokens
        width_swa = swa_indices.shape[-1]
        width_topk = 0 if global_indices is None else global_indices.shape[-1]
        topk = width_swa + width_topk
        assert topk % _MIN_TOPK_WIDTH == 0, (
            f"decode gather width W+K ({topk}) must be a multiple of "
            f"{_MIN_TOPK_WIDTH} for the Triton sparse attention kernel"
        )

        # Workspace rows [T*W) are the SWA gather (token-major), rows
        # [T*W, T*(W+K)) the compressed gather; the attention indices point
        # into the same flat row space.
        ws = current_workspace_manager().get_simultaneous(
            ((num_tokens, topk, q.shape[-1]), torch.bfloat16),
        )[0]
        ws_rows = ws.view(num_tokens * topk, q.shape[-1])
        device = q.device
        cols_w = torch.arange(width_swa, device=device, dtype=torch.int32)
        swa_slots = torch.where(cols_w < swa_lens.unsqueeze(1), swa_indices, -1).to(
            torch.int32
        )
        _dequant_v41_rows(
            ws_rows[: num_tokens * width_swa],
            self.swa_cache_layer.kv_cache,
            swa_slots.view(-1),
            swa_metadata.block_size,
        )
        if global_indices is not None:
            assert topk_lens is not None
            assert kv_cache is not None
            assert attn_metadata is not None
            cols_k = torch.arange(width_topk, device=device, dtype=torch.int32)
            topk_slots = torch.where(
                cols_k < topk_lens.unsqueeze(1), global_indices, -1
            ).to(torch.int32)
            _dequant_v41_rows(
                ws_rows[num_tokens * width_swa :],
                kv_cache,
                topk_slots.view(-1),
                attn_metadata.block_size // self.compress_ratio,
            )

        # q arrives pre-padded to self.padded_heads by the outer wrapper.
        ids = _decode_ws_ids(
            num_tokens, width_swa, width_topk, swa_lens, topk_lens, device
        )
        # ponytail: two-pass (gather to bf16 workspace, then attend); a
        # single-pass zero-copy kernel dequantizing slots in registers (like
        # _sparse_mla_prefill_fused_kernel, extended with the topk union and
        # sink) is the perf upgrade.
        attn = triton_mla_sparse_attention(
            q=q,
            kv=ws.view(-1, 1, q.shape[-1]),
            indices=ids.unsqueeze(1),
            sm_scale=self.scale,
            attn_sink=self.attn_sink,
        )
        output.copy_(attn)
