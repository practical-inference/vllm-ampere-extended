# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BLOCK_N autotune-space probe for the 2-pass sparse MLA prefill kernel (SM80).

Question: _sparse_mla_kernel_final owns ~8.6% of prefill wall (prod trace
2026-10-13) and the e4m3-bench finding put it at 675-800 GB/s = 40-50% A100
HBM peak, partially latency-bound, with the autotune space hardcoded to
BLOCK_N=128 inner iterations per query tile. Does widening BLOCK_N reclaim
part of that slice?

Change: launch the kernel directly (the num_kv_splits==1 fast path, rope-free
dim_qk=512 as in GLM-5.3-Flash prod) once per (BLOCK_N, warps, stages) arm.
Correctness first vs the BN=16 reference arm. FlashInfer CUPTI timing, cold
L2. The dequant stage is NOT swept: dequant_ds_mla_slots() is hardcoded
SLOTS_PER_PROG=8/num_warps=4 and its docstring records ~900-1000 GB/s at
full footprint, i.e. already tuned.

Shape caveat: num_heads_q is not pinned by the profile drawer; the autotune
key is (index_topk, kv_group_num=num_heads_q), so arms are swept for
h_q in {32, 64}. Run on a dev card only (prod runs wheels, never test there).
"""

import statistics

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.triton_utils import LOG2E, triton
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    _BLOCK_DMODEL,
    _BLOCK_DV,
    _BLOCK_H,
    _DIM_QK_NOPE,
    _sparse_mla_kernel_final,
)

SEQ_KV = 1 << 17  # 128 MB bf16 KV rows: forces reads cold from HBM
TOPK = 2048
CASES = [(T, h_q) for T in (2048, 8192) for h_q in (32, 64)]
ARMS = [
    (16, 2, 2),
    (16, 2, 4),
    (16, 4, 2),
    (16, 4, 4),
    (32, 2, 2),
    (32, 2, 4),
    (32, 4, 2),
    (32, 4, 4),
    (64, 4, 2),
    (64, 4, 4),
    (128, 4, 2),
    (128, 4, 4),
]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(0)
    print(
        f"gpu={torch.cuda.get_device_name(torch.device('cuda'))} "
        f"seq_kv={SEQ_KV} topk={TOPK}"
    )
    for num_tokens, h_q in CASES:
        run_case(num_tokens, h_q)


def run_case(num_tokens: int, h_q: int) -> None:
    jit = getattr(_sparse_mla_kernel_final, "fn", None)
    if jit is None:  # pragma: no cover
        raise RuntimeError("autotuner .fn missing; Triton version drift")
    dev = torch.device("cuda")
    kv = torch.randn(SEQ_KV, 1, _DIM_QK_NOPE, dtype=torch.bfloat16, device=dev)
    sm_scale = _DIM_QK_NOPE**-0.5 * LOG2E
    n_groups = -(-h_q // min(_BLOCK_H, h_q))
    q = torch.randn(num_tokens, h_q, _DIM_QK_NOPE, dtype=torch.bfloat16, device=dev)
    idx = torch.randint(0, SEQ_KV, (num_tokens, 1, TOPK), dtype=torch.int32, device=dev)
    out = torch.empty(num_tokens, h_q, _BLOCK_DV, dtype=torch.bfloat16, device=dev)

    def launch(bn: int, nw: int, ns: int) -> None:
        jit[(num_tokens, n_groups)](
            q_buffer=q,
            k_buffer=kv,
            indices_ptr=idx,
            out_ptr=out,
            seq_kv=kv.shape[0],
            h_q=h_q,
            stride_q_token=q.stride(0),
            stride_q_head=q.stride(1),
            stride_kv_token=kv.stride(0),
            stride_kv_head=kv.stride(1),
            stride_out_token=out.stride(0),
            stride_out_head=out.stride(1),
            stride_indices_token=idx.stride(0),
            stride_indices_head=idx.stride(1),
            sm_scale=sm_scale,
            index_topk=TOPK,
            kv_group_num=h_q,
            BLOCK_H=_BLOCK_H,
            BLOCK_N=bn,
            BLOCK_DV=_BLOCK_DV,
            BLOCK_DMODEL=_BLOCK_DMODEL,
            BLOCK_DPE=0,
            num_warps=nw,
            num_stages=ns,
        )

    ref = None
    rows = []
    for bn, nw, ns in ARMS:
        try:
            launch(bn, nw, ns)
            torch.accelerator.synchronize()
        except triton.runtime.errors.OutOfResources:
            print(
                f"T={num_tokens:5} h_q={h_q:2} BN={bn:3} w={nw} s={ns}: "
                "pruned (shared memory)"
            )
            continue
        if ref is None:
            ref = out.clone()
        else:
            torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        us = (
            statistics.median(
                bench_gpu_time_with_cupti(
                    lambda bn=bn, nw=nw, ns=ns: launch(bn, nw, ns)
                )
            )
            * 1e3
        )
        kv_bytes = num_tokens * TOPK * _DIM_QK_NOPE * 2
        qo_bytes = 2 * num_tokens * h_q * _DIM_QK_NOPE * 2
        tflops = 4 * num_tokens * h_q * TOPK * _DIM_QK_NOPE / (us * 1e6)
        rows.append((bn, nw, ns, us, (kv_bytes + qo_bytes) / (us * 1e3), tflops))
        print(
            f"T={num_tokens:5} h_q={h_q:2} BN={bn:3} w={nw} s={ns}: "
            f"{us:9.1f} us  {rows[-1][4]:7.1f} GB/s  {rows[-1][5]:5.1f} TF/s"
        )
    base = min(r[3] for r in rows if r[0] == 16)
    best = min(rows, key=lambda r: r[3])
    print(
        f"==> T={num_tokens} h_q={h_q}: winner BN={best[0]} w={best[1]} "
        f"s={best[2]} speedup={base / best[3]:.2f}x "
        f"({base:.0f} -> {best[3]:.0f} us)"
    )


if __name__ == "__main__":
    main()
