# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the row-blocked vs legacy prefill fp8_mqa_logits schedules.

Reports ms and effective TFLOPS for the legacy one-CTA-per-(row, tile)
kernel and the row-blocked `_fp8_mqa_logits_bm_kernel` at two shapes:
the full-pool prefill chunk (M=8192, N~111K) and the serving shape with
~2.5K-row live windows.
"""

import argparse
import os
import time

import torch

from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_triton,
    warmup_fp8_mqa_logits_triton,
)

H = 32  # index_n_heads
D = 128  # index_head_dim


def make_inputs(M: int, N: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randint(0, 256, (M, H, D), dtype=torch.uint8, generator=g)
    q[q == 0x7F] = 0
    q[q == 0xFF] = 0
    k = torch.randint(0, 256, (N, D), dtype=torch.uint8, generator=g)
    k[k == 0x7F] = 0
    k[k == 0xFF] = 0
    scales = torch.rand(N, generator=g) * 2.0 + 0.5
    w = torch.randn(M, H, generator=g)
    ke = torch.full((M,), N, dtype=torch.int32)
    ks = torch.zeros((M,), dtype=torch.int32)
    return (
        q.view(torch.float8_e4m3fn).cuda(),
        (k.view(torch.float8_e4m3fn).cuda(), scales.cuda()),
        w.cuda(),
        ks.cuda(),
        ke.cuda(),
    )


def timeit(fn, reps: int) -> float:
    fn()  # autotune warm
    torch.accelerator.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


def bench(label: str, M: int, N: int, window: int, reps: int, seed: int):
    q, kv, w, ks, ke = make_inputs(M, N, seed)
    ke = torch.full_like(ke, window)
    warmup_fp8_mqa_logits_triton(H, D, torch.device("cuda"))

    os.environ["VLLM_MQA_LOGITS_LEGACY"] = "1"
    t_legacy = timeit(
        lambda: fp8_mqa_logits_triton(q, kv, w, ks, ke, clean_logits=False), reps
    )
    os.environ["VLLM_MQA_LOGITS_LEGACY"] = "0"
    t_bm = timeit(
        lambda: fp8_mqa_logits_triton(q, kv, w, ks, ke, clean_logits=False), reps
    )
    tflops = M * window * H * D * 2 / 1e12
    print(
        f"{label}: M={M} poolN={N} window={window} "
        f"legacy={t_legacy:.2f}ms ({tflops / (t_legacy / 1e3):.1f} TFLOPS) "
        f"row-blocked={t_bm:.2f}ms ({tflops / (t_bm / 1e3):.1f} TFLOPS) "
        f"speedup={t_legacy / t_bm:.2f}x"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    bench("full-pool", 8192, 111_000, 111_000, args.reps, seed=7)
    bench("serving-window", 8192, 111_000, 2560, args.reps, seed=11)


if __name__ == "__main__":
    main()
