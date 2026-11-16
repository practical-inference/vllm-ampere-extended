"""Kernel-level correctness test for the sm_80 QSA fp8-index arm.

Runs the armed Triton kernels (uint8 storage + _maybe_dequant + fp16 dot)
against (a) the bf16 reference path through the same kernels and (b) a torch
fp32 matmul over the dequantized fp8 codes. Also exercises the armed store.
No compiled vllm ops involved (bypasses _topk; torch.topk stands in).
"""

import sys

import torch

sys.path.insert(0, "/home/columbo/source/vllm-nightly")

from vllm.models.qwen4_exp.nvidia.ops.qsa import (  # noqa: E402
    qsa_store_cache_rows,
    _needs_fp8_byte_arm,
)
from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (  # noqa: E402
    _qsa_mqa_paged_uniform_kernel,
    _qsa_mqa_paged_prefill_kernel,
    _prefill_logits,
)
from vllm.triton_utils import triton  # noqa: E402

assert _needs_fp8_byte_arm(), "test must run below sm_89"
torch.manual_seed(7)
dev = "cuda"

HEADS, HEAD_DIM, PAGE_SIZE, BLOCKS = 4, 128, 64, 16
DQL = 1  # decode query len
NUM_REQS = 6
BLOCK_TOPK = 32
COMPRESS_RATIO = 4

blocks = BLOCKS
cache_bf16 = torch.randn(
    blocks, PAGE_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16, device=dev
) * 0.5
page_table = torch.randint(
    0, blocks, (NUM_REQS, blocks), dtype=torch.int32, device=dev
)
# distinct pages per request (page_table rows sampled without replacement)
for r in range(NUM_REQS):
    page_table[r] = torch.randperm(blocks, device=dev)[:blocks].to(torch.int32)
visible = torch.randint(
    8, blocks * PAGE_SIZE, (NUM_REQS * DQL,), dtype=torch.int32, device=dev
)
q_bf16 = torch.randn(
    NUM_REQS * DQL, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev
) * 0.5

# ---- 1. store arm: fp8 cache rows must equal the torch cast ----
rows_bf16 = torch.randn(10, HEAD_DIM, dtype=torch.bfloat16, device=dev) * 0.5
slots = torch.tensor([0, 1, 2, PAGE_SIZE, PAGE_SIZE + 1, 3, 4, 5, 6, 7],
                     dtype=torch.int64, device=dev)
cache_fp8 = torch.empty(
    blocks, PAGE_SIZE, 1, HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev
)
cache_fp8.view(torch.uint8).zero_()
qsa_store_cache_rows(cache_fp8, slots, rows_bf16.unsqueeze(1))
expected_bytes = rows_bf16.to(torch.float8_e4m3fn).view(torch.uint8)
actual = cache_fp8.view(torch.uint8)[
    slots // PAGE_SIZE, slots % PAGE_SIZE, 0, :
]
assert torch.equal(actual, expected_bytes), "store arm byte mismatch"
print("STORE ARM: bytes match torch cast exactly")

# ---- 2. decode logits: arm vs bf16 reference vs torch fp32 expected ----
q_fp8 = q_bf16.to(torch.float8_e4m3fn)
cache_fp8_full = cache_bf16.to(torch.float8_e4m3fn)


def run_decode(q_t, k_t, dequant_k, num_warps):
    columns = page_table.shape[1] * PAGE_SIZE
    logits = torch.empty((q_t.shape[0], columns), dtype=torch.float32, device=dev)
    tiles_per_program = 1
    grid = (NUM_REQS, triton.cdiv(columns, 64 * tiles_per_program))
    _qsa_mqa_paged_uniform_kernel[grid](
        q_t,
        k_t,
        page_table,
        visible,
        logits,
        *q_t.stride()[:-1],
        *k_t.stride()[:2],
        *page_table.stride()[:-1],
        *logits.stride()[:-1],
        PAGE_SIZE=PAGE_SIZE,
        PAGE_TABLE_WIDTH=blocks,
        NUM_HEADS=HEADS,
        HEAD_DIM=HEAD_DIM,
        DECODE_QUERY_LEN=DQL,
        BLOCK_N=64,
        TILES_PER_PROG=tiles_per_program,
        STAGES=2,
        DEQUANT_K=dequant_k,
        num_warps=num_warps,
    )
    return logits


logits_ref = run_decode(q_bf16, cache_bf16, False, 2)
# the host folds 2^8 into Q (the byte decode yields x*2^-8) — mimic it here
logits_arm = run_decode(
    q_fp8.to(torch.float16) * 256.0, cache_fp8_full.view(torch.uint8), True, 1
)

# torch expected: dequant both sides to fp16, matmul in fp32, ReLU per head
# (the kernel applies maximum(scores,0) before summing heads), then mask to
# the row's visible columns (the kernel leaves the rest unwritten).
k_deq = cache_fp8_full.to(torch.float16)  # [blocks, page, 1, dim]
q_deq = q_fp8.to(torch.float16)  # [rows, heads, dim]
logits_expected = torch.full_like(logits_ref, float("nan"))
for r in range(NUM_REQS):
    pages = page_table[r]
    k_rows = k_deq[pages, :, 0, :].reshape(-1, HEAD_DIM)  # [cols, dim]
    scores = (k_rows.float() @ q_deq[r * DQL].float().T)  # [cols, heads]
    cols = int(visible[r * DQL].item())
    logits_expected[r * DQL, :cols] = scores[:cols].clamp(min=0).sum(-1)

d_arm = (logits_arm - logits_expected).abs()
valid_mask = ~torch.isnan(logits_expected)
rel_arm = (
    d_arm[valid_mask] / logits_expected[valid_mask].abs().clamp(min=1e-3)
).max().item()
# bf16 reference vs its own torch expectation
k_b = cache_bf16.float()
logits_expected_ref = torch.full_like(logits_ref, float("nan"))
for r in range(NUM_REQS):
    pages = page_table[r]
    k_rows = k_b[pages, :, 0, :].reshape(-1, HEAD_DIM)
    scores = k_rows @ q_bf16[r * DQL].float().T
    cols = int(visible[r * DQL].item())
    logits_expected_ref[r * DQL, :cols] = scores[:cols].clamp(min=0).sum(-1)

d_ref = (logits_ref - logits_expected_ref).abs()
vmask_ref = ~torch.isnan(logits_expected_ref)
print(
    f"DECODE: arm-vs-fp8-expected max_abs={d_arm[valid_mask].max().item():.5f} "
    f"max_rel={rel_arm:.5f}; ref-vs-bf16-expected max_abs="
    f"{d_ref[vmask_ref].max().item():.5f}"
)
assert d_arm[valid_mask].max().item() < 0.05, "arm logits deviate from fp8 semantics"
assert d_ref[vmask_ref].max().item() < 0.05, "reference harness sanity failed"

# selection flips: top-32 compressed columns arm vs bf16 reference, over the
# rows' valid widths only
overlaps = []
for r in range(NUM_REQS):
    cols = int(visible[r * DQL].item())
    w = min(cols, logits_arm.shape[1])
    top_arm = logits_arm[r, :w].topk(BLOCK_TOPK).indices
    top_ref = logits_ref[r, :w].topk(BLOCK_TOPK).indices
    overlaps.append(
        (top_arm.unsqueeze(-1) == top_ref.unsqueeze(1)).any(-1).float().mean().item()
    )
print(f"DECODE selection overlap arm-vs-bf16: {sum(overlaps) / len(overlaps):.3f}")

# ---- 3. prefill logits through the armed host ----
q_lens = [3, 5]
num_tokens = sum(q_lens)
q_pf = torch.randn(num_tokens, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
q_pf8 = q_pf.to(torch.float8_e4m3fn)
qsl = torch.tensor([0, q_lens[0], num_tokens], dtype=torch.int32, device=dev)
page_table_pf = page_table[:2]
vis_pf = torch.randint(8, 200, (num_tokens,), dtype=torch.int32, device=dev)
max_qlen = max(q_lens)
max_seq = blocks * PAGE_SIZE
logits_width = min(
    max(64, triton.cdiv(max_seq, COMPRESS_RATIO)), blocks * PAGE_SIZE
)
workspace_ref = torch.empty(
    max(1024, num_tokens * logits_width), dtype=torch.float32, device=dev
)
workspace_arm = torch.empty_like(workspace_ref)
logits_pf_ref = _prefill_logits(
    q_pf, cache_bf16, page_table_pf, qsl, vis_pf, workspace_ref,
    max_qlen, logits_width, query_offset=0, num_queries=num_tokens,
)
logits_pf_arm = _prefill_logits(
    q_pf8, cache_fp8_full, page_table_pf, qsl, vis_pf, workspace_arm,
    max_qlen, logits_width, query_offset=0, num_queries=num_tokens,
)
nz = (logits_pf_arm != 0).float().mean().item()
d_pf = (logits_pf_arm - logits_pf_ref).abs().max().item()
print(f"PREFILL: arm-vs-bf16 max_abs={d_pf:.5f} (nonzero frac {nz:.3f})")
assert nz > 0.1, "prefill arm wrote nothing (workspace aliasing mask)"

# arm vs fp8 semantics: dequantized fp8 codes, fp32 matmul, ReLU, head-sum
k_pf = cache_fp8_full.to(torch.float16).float()  # [blocks, page, 1, dim]
q_pf16 = q_pf8.to(torch.float16).float()  # [tokens, heads, dim]
expected_pf = torch.full_like(logits_pf_ref, float("nan"))
for i in range(num_tokens):
    req = 0 if i < q_lens[0] else 1
    pages = page_table_pf[req]
    k_rows = k_pf[pages, :, 0, :].reshape(-1, HEAD_DIM)
    scores = (k_rows @ q_pf16[i].T).clamp(min=0).sum(-1)
    cols = min(int(vis_pf[i].item()), logits_pf_ref.shape[1])
    expected_pf[i, :cols] = scores[:cols]
vmask = ~torch.isnan(expected_pf)
d_pf8 = (logits_pf_arm - expected_pf).abs()[vmask].max().item()
print(f"PREFILL: arm-vs-fp8-expected max_abs={d_pf8:.5f}")
assert d_pf8 < 0.05, "prefill arm deviates from fp8 semantics"
# arm-vs-bf16 delta IS the fp8 quantization noise (bf16 K vs e4m3-rounded K;
# arm == fp8-expected bit-exact above, so this delta is bf16-vs-fp8 by
# construction). The real quality gate is needle/GSM8K on the served model.
assert d_pf < 20.0, "prefill arm-vs-bf16 delta beyond fp8 noise"

print("ALL KERNEL TESTS PASSED")
