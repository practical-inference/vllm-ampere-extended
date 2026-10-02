# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for attn_sink in triton_mla_sparse_attention.

Reference: fp32 softmax with the per-head sink appended as an extra
denominator element (DeepGEMM/FlashMLA semantics), on both bf16 and
fp8-dequantized KV inputs.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    _DIM_QK,
    _DS_MLA_CACHE_BYTES,
    _DS_MLA_NOPE_DIM,
    _DS_MLA_ROPE_DIM,
    dequant_ds_mla_slots,
    triton_mla_sparse_attention,
    write_nope_ds_mla_slots,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton sparse MLA kernel requires CUDA/ROCm",
)

SM_SCALE = 0.08


def _torch_ref(q, kv, indices, sink):
    """fp32 reference: softmax over valid slots with sink as extra term."""
    num_tokens, h_q, dim = q.shape
    block_dv = 512
    qf = q.float().cpu()
    kf = kv.float().cpu()
    idx_all = indices.cpu()
    sinkf = sink.float().cpu()
    out = torch.zeros(num_tokens, h_q, block_dv, dtype=torch.float32)
    for t in range(num_tokens):
        idx = idx_all[t, 0].long()
        valid = idx >= 0
        kv_t = torch.zeros(indices.shape[2], dim)
        kv_t[valid] = kf[idx[valid], 0]
        s = qf[t] @ kv_t.T * SM_SCALE  # [h_q, topk]
        s[:, ~valid] = float("-inf")
        m = s.amax(1, keepdim=True).clamp(min=-1e30)
        p = (s - m).exp()
        den = p.sum(1, keepdim=True) + (sinkf - m.squeeze(1)).exp().unsqueeze(1)
        out[t] = ((p @ kv_t) / den)[:, :block_dv]
    return out.to(torch.bfloat16).cuda()


def _fp8_dequant_kv(kv_bf16):
    """Round-trip bf16 KV rows through the V3.2 fp8_ds_mla writer/reader."""
    n = kv_bf16.shape[0]
    bs = 64
    nb = (n + bs - 1) // bs
    cache = torch.zeros(nb, bs, _DS_MLA_CACHE_BYTES, dtype=torch.uint8, device="cuda")
    kv_c = kv_bf16[:, :, :_DS_MLA_NOPE_DIM]
    slots = torch.arange(n, dtype=torch.int64, device="cuda")
    write_nope_ds_mla_slots(kv_c.reshape(n, _DS_MLA_NOPE_DIM), cache, slots)
    out = torch.zeros(
        n, _DS_MLA_NOPE_DIM + _DS_MLA_ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    )
    dequant_ds_mla_slots(
        out,
        cache,
        slots.int(),
        cache_block_size=bs,
        rope_dim=_DS_MLA_ROPE_DIM,
    )
    # Rope tail is unwritten (zeros) in the NoPE writer path; score against
    # the dequantized workspace itself so q/k stay consistent.
    return out[:, :_DS_MLA_NOPE_DIM].reshape(n, 1, _DS_MLA_NOPE_DIM)


@pytest.mark.parametrize("num_kv_splits", [1, 2, 4])
@pytest.mark.parametrize("padded_heads", [16, 32])
def test_attn_sink_matches_torch_reference(num_kv_splits, padded_heads):
    torch.manual_seed(0)
    num_tokens, h_q, topk = 3, 16, 128
    kv = torch.randn(4096, 1, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(num_tokens, h_q, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    indices = torch.randint(0, kv.shape[0], (num_tokens, 1, topk), device="cuda")
    indices[:, :, ::5] = -1
    indices = indices.to(torch.int32)
    sink = torch.full((padded_heads,), float("-inf"), device="cuda")
    sink[:h_q] = torch.tensor(
        [0.5, -2.0, 3.0, -float("inf"), 0.0, 7.5, -1.25, 1.0] * 2, device="cuda"
    )

    out = triton_mla_sparse_attention(
        q, kv, indices, SM_SCALE, num_kv_splits=num_kv_splits, attn_sink=sink
    )
    ref = _torch_ref(q, kv, indices, sink[:h_q]).float()
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_attn_sink_minus_inf_matches_no_sink():
    torch.manual_seed(0)
    num_tokens, h_q, topk = 2, 16, 256
    kv = torch.randn(2048, 1, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(num_tokens, h_q, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    indices = torch.randint(0, kv.shape[0], (num_tokens, 1, topk), device="cuda")
    indices = indices.to(torch.int32)
    sink = torch.full((h_q,), float("-inf"), device="cuda")
    out_sink = triton_mla_sparse_attention(
        q, kv, indices, SM_SCALE, num_kv_splits=2, attn_sink=sink
    )
    out_none = triton_mla_sparse_attention(q, kv, indices, SM_SCALE, num_kv_splits=2)
    assert torch.equal(out_sink.view(torch.int16), out_none.view(torch.int16))


def test_attn_sink_fp8_dequant_inputs():
    torch.manual_seed(0)
    num_tokens, h_q, topk = 2, 16, 128
    kv = _fp8_dequant_kv(
        torch.randn(1024, 1, _DS_MLA_NOPE_DIM, dtype=torch.bfloat16, device="cuda")
    )
    kv = torch.nn.functional.pad(kv, (0, _DS_MLA_ROPE_DIM))  # [n, 1, 576]
    q = torch.randn(num_tokens, h_q, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    indices = torch.randint(0, kv.shape[0], (num_tokens, 1, topk), device="cuda")
    indices = indices.to(torch.int32)
    sink = torch.rand(h_q, device="cuda") * 4.0 - 2.0

    out = triton_mla_sparse_attention(
        q, kv, indices, SM_SCALE, num_kv_splits=4, attn_sink=sink
    )
    ref = _torch_ref(q, kv, indices, sink).float()
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
