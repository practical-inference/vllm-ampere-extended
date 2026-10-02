# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 Lightning Indexer on the Triton FP8 path, for SM80.

SM80 (A100) has neither DeepGEMM nor FP4 hardware: ``dsa_indexer_uses_fp4``
is False there, so the indexer keeps the FP8-132 K pool (128 e4m3 values +
4 B fp32 scale per row, values then scales per page) and scores it with the
Triton MQA-logits fallbacks ``sparse_attn_indexer`` routes to when
``is_deep_gemm_supported()`` is False (``fp8_mqa_logits_triton`` on prefill,
``fp8_paged_mqa_logits_triton`` on decode). The K store writes and both
readers consume that record unchanged; only the candidate-pool scorer needs
a replacement, since ``SparseMQAIndexer`` hard-requires SM100 + DeepGEMM.
"""

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.models.deepseek_v41.attention import DeepseekV4Indexer


class DeepseekV4TritonMQAIndexer(nn.Module):
    """`SparseMQAIndexer` stand-in that scores the FP8-132 candidate pool
    with the dense Triton logits path, masked to the candidate blocks
    inside `sparse_attn_indexer` (O(context), not O(candidates)).

    Built and called as `SparseMQAIndexer` is, so the model can take either.
    """

    weights_dtype = torch.float32
    """The Triton logits kernels take fp32 per-head weights."""

    def __init__(
        self,
        k_cache,
        topk_tokens: int,
        head_dim: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        candidate_blocks: torch.Tensor,
        candidate_block_size: int,
    ):
        super().__init__()
        # Like RocmSparseMQAIndexer: the paged scorer bounds a row by the
        # model length in compressed positions. quant_block_size/scale_fmt
        # are only read by the cache-insert branch (skip_k_cache_insert).
        model_config = get_current_vllm_config().model_config
        self._scorer = SparseAttnIndexer(
            k_cache,
            128,
            "ue8m0",
            topk_tokens,
            head_dim,
            model_config.max_model_len // k_cache.compress_ratio,
            max_total_seq_len,
            topk_indices_buffer,
            skip_k_cache_insert=True,
            compress_ratio=k_cache.compress_ratio,
            candidate_blocks=candidate_blocks,
            candidate_block_size=candidate_block_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        assert k is None, "the model writes the indexer K cache"
        return self._scorer(hidden_states, q_quant, k, weights)


class DeepseekV4TritonIndexer(DeepseekV4Indexer):
    """The V4.1 indexer without DeepGEMM: both scoring layers route through
    the Triton FP8 logits kernels, on the unchanged FP8-132 K pool."""

    mqa_cls = DeepseekV4TritonMQAIndexer
    attn_cls = SparseAttnIndexer
