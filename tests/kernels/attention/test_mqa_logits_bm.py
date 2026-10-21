# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exactness + speed gate for the row-blocked prefill MQA-logits kernel.

`_fp8_mqa_logits_bm_kernel` (BLOCK_M rows per CTA, heads flattened into the
dot, TILES K tiles streamed per program) must produce logits bit-identical
to the legacy one-CTA-per-(row, tile) `_fp8_mqa_logits_kernel`.
"""

import os
import time

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
def _seed_autotune_at_prod_shape():
    """The autotune caches are shape-blind and single-sweep: the first
    kernel call in the process picks the config every later call runs.
    Seed both paths with the same prod-chunk-shape warmup the engine's
    init makes, so gate benches measure the config prod will run —
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


@pytest.mark.parametrize("M,N", [(8, 1), (8, 129), (2050, 4096), (8192, 32768)])
def test_bm_bitexact(M: int, N: int):
    ref, out, _, _ = _run(M, N, seed=M + N)
    assert torch.equal(ref, out)


def test_bm_bitexact_clean_false_span():
    """With clean_logits=False only the left-justified `[0, ke-ks)` span is
    defined; compare exactly that span."""
    M, N = 4096, 8192
    ref, out, ke, ks = _run(M, N, seed=99, clean=False)
    span = torch.arange(N, device=ref.device)[None, :] < (ke - ks)[:, None]
    assert torch.equal(ref[span], out[span])


def test_bm_bench():
    """Prod-ish shape: M=8192 chunk, ~111K gathered pool rows (445K ctx,
    kpool=4). Reports ms + effective TFLOPS for legacy vs row-blocked."""
    M, N = 8192, 111_000
    q, kv, w, ks, ke = _make(M, N, seed=7)
    ks = torch.zeros_like(ks)
    ke = torch.full_like(ke, N)

    def timeit(fn, reps=5):
        fn()  # autotune warm
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000

    os.environ["VLLM_MQA_LOGITS_LEGACY"] = "1"
    t_legacy = timeit(
        lambda: fp8_mqa_logits_triton(q, kv, w, ks, ke, clean_logits=False)
    )
    os.environ["VLLM_MQA_LOGITS_LEGACY"] = "0"
    t_bm = timeit(lambda: fp8_mqa_logits_triton(q, kv, w, ks, ke, clean_logits=False))
    tflops = M * N * H * D * 2 / 1e12
    print(
        f"\nM={M} N={N} H={H} D={D}: legacy={t_legacy:.2f}ms "
        f"({tflops / (t_legacy / 1e3):.1f} TFLOPS) "
        f"bm={t_bm:.2f}ms ({tflops / (t_bm / 1e3):.1f} TFLOPS) "
        f"speedup={t_legacy / t_bm:.2f}x"
    )
    assert t_bm < t_legacy


def test_bm_bench_prod_shape():
    """The shape prod actually runs (2026-10-21 trace fit): M=8192 chunk
    rows against the full gathered pool width (N~111K at 445K ctx) with
    ~2.5K-row live windows (top-k pools + the chunk's new pools)."""
    M, N, W = 8192, 111_000, 2560
    q, kv, w, ks, ke = _make(M, N, seed=11)
    ks = torch.zeros_like(ks)
    ke = torch.full_like(ke, W)

    def timeit(fn, reps=10):
        fn()  # autotune warm
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000

    os.environ["VLLM_MQA_LOGITS_LEGACY"] = "1"
    t_legacy = timeit(
        lambda: fp8_mqa_logits_triton(q, kv, w, ks, ke, clean_logits=False)
    )
    os.environ["VLLM_MQA_LOGITS_LEGACY"] = "0"
    t_bm = timeit(lambda: fp8_mqa_logits_triton(q, kv, w, ks, ke, clean_logits=False))
    tflops = M * W * H * D * 2 / 1e12
    print(
        f"\nPROD SHAPE M={M} poolN={N} window={W}: "
        f"legacy={t_legacy:.2f}ms ({tflops / (t_legacy / 1e3):.1f} TFLOPS) "
        f"bm={t_bm:.2f}ms ({tflops / (t_bm / 1e3):.1f} TFLOPS) "
        f"speedup={t_legacy / t_bm:.2f}x"
    )
    assert t_bm < t_legacy
