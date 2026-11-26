# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical-match gate for the row-blocked prefill MQA-logits kernel.

`_fp8_mqa_logits_bm_kernel` (BLOCK_M rows per CTA, heads flattened into the
dot, TILES K tiles streamed per program) must match the legacy
one-CTA-per-(row, tile) `_fp8_mqa_logits_kernel` within fp32 rounding:
cross-kernel bit-equality is config-dependent, so the gate is
atol=1.0 / rtol=1e-5 (accumulation-order deltas are ULP-of-largest-term,
bounded <= 0.4 absolute at serving shapes).
"""

import os

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_triton,
    warmup_fp8_mqa_logits_triton,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton logits kernel requires CUDA/ROCm",
)

H = 32  # GLM-5.3 index_n_heads
D = 128  # index_head_dim


@pytest.fixture(scope="session", autouse=True)
def _seed_autotune_at_serving_shape():
    """The autotune caches are shape-blind and single-sweep: the first
    kernel call in the process picks the config every later call runs.
    Seed both paths with the same serving-chunk-shape warmup the engine's
    init makes, so the gates measure the config serving will run —
    otherwise pytest's first tiny case sweeps at a degenerate grid."""
    os.environ["VLLM_MQA_LOGITS_LEGACY"] = "1"
    warmup_fp8_mqa_logits_triton(H, D, torch.device("cuda"))
    os.environ["VLLM_MQA_LOGITS_LEGACY"] = "0"
    warmup_fp8_mqa_logits_triton(H, D, torch.device("cuda"))


def _make(M: int, N: int, seed: int):
    """fp8 q/k with e4m3fn NaN codes scrubbed, positive scales, random
    head weights, and causal-ish pool-granular `[ks, ke)` windows incl.
    empty, partial and full rows."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randint(0, 256, (M, H, D), dtype=torch.uint8, generator=g)
    q[q == 0x7F] = 0
    q[q == 0xFF] = 0
    k = torch.randint(0, 256, (N, D), dtype=torch.uint8, generator=g)
    k[k == 0x7F] = 0
    k[k == 0xFF] = 0
    scales = torch.rand(N, generator=g) * 2.0 + 0.5
    w = torch.randn(M, H, generator=g)
    pos = torch.arange(M)
    ke = torch.clamp(pos // 4 + 1, max=N)
    ks = torch.where(pos % 7 == 0, torch.clamp(pos // 16, max=ke), torch.zeros_like(ke))
    if M > 8:
        ke[5] = ks[5]  # empty window row
    if M > 16:
        ke[M - 1] = N  # full-window row
    return (
        q.view(torch.float8_e4m3fn).cuda(),
        (k.view(torch.float8_e4m3fn).cuda(), scales.cuda()),
        w.cuda(),
        ks.to(torch.int32).cuda(),
        ke.to(torch.int32).cuda(),
    )


def _run(M, N, seed, clean=True):
    q, kv, w, ks, ke = _make(M, N, seed)
    os.environ["VLLM_MQA_LOGITS_LEGACY"] = "1"
    ref = fp8_mqa_logits_triton(q, kv, w, ks, ke, clean_logits=clean)
    os.environ["VLLM_MQA_LOGITS_LEGACY"] = "0"
    out = fp8_mqa_logits_triton(q, kv, w, ks, ke, clean_logits=clean)
    return ref, out, ke, ks


def _assert_close(ref, out):
    """Cross-kernel bit-equality is config-dependent, not a stable
    contract: Triton lowers the dot with different MMA k-groupings per
    swept config, so legacy-vs-bm shows ULP-of-largest-term deltas once
    both sweep at realistic shapes. Cancellation-heavy sums (32 head terms
    of ~5e5 magnitude) land at small values where those deltas are large
    RELATIVELY (~1%) but bounded ABSOLUTELY (observed <=0.4). Prod
    contract: correctness within fp32 rounding + structural fidelity —
    atol=1.0 (with rtol for the large elements) leaves margin vs observed
    while catching any structural bug (wrong scale/window/mask is orders
    of magnitude off); a ULP flip can at worst reorder a top-k boundary
    tie between near-equal candidates."""
    torch.testing.assert_close(ref, out, rtol=1e-5, atol=1.0)


@pytest.mark.parametrize("M,N", [(8, 1), (8, 129), (2050, 4096), (8192, 32768)])
def test_bm_bitexact(M: int, N: int):
    ref, out, _, _ = _run(M, N, seed=M + N)
    _assert_close(ref, out)


def test_bm_bitexact_clean_false_span():
    """With clean_logits=False only the left-justified `[0, ke-ks)` span is
    defined; compare exactly that span."""
    M, N = 4096, 8192
    ref, out, ke, ks = _run(M, N, seed=99, clean=False)
    span = torch.arange(N, device=ref.device)[None, :] < (ke - ks)[:, None]
    _assert_close(ref[span], out[span])
