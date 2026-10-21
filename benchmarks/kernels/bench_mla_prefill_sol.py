# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capped-regime SOL probes for the two uncharacterized GLM prefill buckets.

_dequant_ds_mla_slots_kernel (memory-bound scatter gather; prod trace shows
~840-905 GB/s vs ~1600 GB/s practical HBM) and the fp8 MQA indexer
(compute-bound bf16 dot; prod trace shows ~157 effective TFLOPS vs ~166
capped-card peak, i.e. already at SOL).

Modes:
  full — prod shapes: dequant 17.5M slots (~19 GB) + indexer M=1536 N=40960.
         Empty dev card only.
  ab   — reduced shapes (1M slots; M=1536 N=16384, ~2 GB total) sized for a
         loaded card's headroom; run IDENTICALLY on a 200W and a 400W card:
         equal GB/s => memory-bound (no software headroom), clock-scaled =>
         latency/SM-bound (restructuring pays).

GLM-5.3-Flash is rope-free: nope_dim=512, rope_dim=0, dequant_dim=512,
cache record 656B (512 fp8 + 4 fp32 scales + 128B zero tail).
"""

import statistics

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.v1.attention.ops.mqa_logits_triton import fp8_mqa_logits_triton
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    _dequant_ds_mla_slots_kernel,
)

BLOCK_SIZE = 64


def bench_us(fn) -> float:
    return statistics.median(bench_gpu_time_with_cupti(fn)) * 1e3


def bench_dequant(total_slots: int, arms: list[tuple[int, int]], tag: str) -> None:
    dev = torch.device("cuda")
    cache_tokens = 200_000
    cache = torch.randint(
        0,
        255,
        (cache_tokens // BLOCK_SIZE, BLOCK_SIZE, 656),
        dtype=torch.uint8,
        device=dev,
    )
    cache[cache == 0x7F] = 0  # e4m3fn NaN pattern: keep ref/kernel finite
    indices = torch.randint(
        0, cache_tokens, (total_slots,), dtype=torch.int32, device=dev
    )
    out = torch.empty(total_slots, 512, dtype=torch.bfloat16, device=dev)
    moved = total_slots * (656 + 1024)

    def launch(spp: int, nw: int) -> None:
        _dequant_ds_mla_slots_kernel[(total_slots // spp,)](
            out,
            cache,
            indices,
            total_slots,
            cache_block_size=BLOCK_SIZE,
            block_stride=cache.stride(0),
            nope_dim=512,
            rope_dim=0,
            quant_block=128,
            num_tiles=4,
            dequant_dim=512,
            cache_bytes=656,
            SLOTS_PER_PROG=spp,
            segregated_scale=False,
            num_warps=nw,
        )

    # correctness spot-check vs direct fp8 decode on 64 random rows
    launch(arms[0][0], arms[0][1])
    torch.accelerator.synchronize()
    rows = torch.randint(0, total_slots, (64,), device=dev)
    rec = cache.reshape(-1, 656)[indices[rows].long()]
    fp8 = rec[:, :512].view(torch.float8_e4m3fn).to(torch.float32)
    scale = rec[:, 512:528].view(torch.float32).reshape(-1, 4, 1)
    ref = (fp8.reshape(-1, 4, 128) * scale).reshape(-1, 512).to(torch.bfloat16)
    torch.testing.assert_close(out[rows], ref, atol=2e-2, rtol=2e-2)

    for spp, nw in arms:
        us = bench_us(lambda s=spp, w=nw: launch(s, w))
        print(
            f"{tag} dequant slots={total_slots} SPP={spp:2} w={nw}: "
            f"{us:9.1f} us  {moved / (us * 1e3):7.1f} GB/s"
        )
    torch.accelerator.empty_cache()


def bench_indexer(M: int, N: int, tag: str) -> None:
    dev = torch.device("cuda")
    q = torch.randn(M, 64, 128, device=dev).to(torch.float8_e4m3fn)
    k = torch.randn(N, 128, device=dev).to(torch.float8_e4m3fn)
    scales = (torch.rand(N, device=dev) * 0.01 + 0.005).to(torch.float32)
    weights = torch.rand(M, 64, device=dev)
    ks = torch.zeros(M, dtype=torch.int32, device=dev)
    ke = torch.full((M,), N, dtype=torch.int32, device=dev)

    def run() -> None:
        fp8_mqa_logits_triton(q, (k, scales), weights, ks, ke, clean_logits=False)

    run()
    torch.accelerator.synchronize()
    us = bench_us(run)
    flops = 2 * 64 * 128 * M * N
    print(f"{tag} indexer M={M} N={N}: {us:9.1f} us  {flops / (us * 1e6):7.1f} TFLOPS")
    torch.accelerator.empty_cache()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("full", "ab"), required=True)
    args = ap.parse_args()
    dev_name = torch.cuda.get_device_name(torch.device("cuda"))
    print(f"gpu={dev_name} mode={args.mode}")
    if args.mode == "full":
        bench_dequant(
            17_547_264,
            [(4, 4), (8, 4), (16, 4), (32, 4), (8, 8), (16, 8)],
            "full",
        )
        bench_indexer(1536, 40_960, "full")
    else:
        bench_dequant(393_216, [(8, 4), (16, 4)], "ab")
        bench_indexer(1536, 8_192, "ab")


if __name__ == "__main__":
    main()
