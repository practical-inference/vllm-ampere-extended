"""Diagnose the prefill arm mismatch pattern."""
import sys
import torch

sys.path.insert(0, "/home/columbo/source/vllm-nightly")
from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import _prefill_logits  # noqa: E402
from vllm.triton_utils import triton  # noqa: E402

torch.manual_seed(7)
dev = "cuda"
HEADS, HEAD_DIM, PAGE_SIZE, BLOCKS = 4, 128, 64, 16
cache_bf16 = torch.randn(
    BLOCKS, PAGE_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16, device=dev
) * 0.5
cache_fp8 = cache_bf16.to(torch.float8_e4m3fn)
page_table = torch.stack(
    [torch.randperm(BLOCKS, device=dev) for _ in range(2)]
).to(torch.int32)
q_lens = [3, 5]
num_tokens = sum(q_lens)
q_bf = torch.randn(num_tokens, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev) * 0.5
q_fp8 = q_bf.to(torch.float8_e4m3fn)
qsl = torch.tensor([0, q_lens[0], num_tokens], dtype=torch.int32, device=dev)
vis = torch.full((num_tokens,), 200, dtype=torch.int32, device=dev)
max_qlen, max_seq = max(q_lens), BLOCKS * PAGE_SIZE
logits_width = min(max(64, triton.cdiv(max_seq, 4)), BLOCKS * PAGE_SIZE)
w_ref = torch.empty(num_tokens * logits_width + 1024, dtype=torch.float32, device=dev)
w_arm = torch.empty_like(w_ref)
ref = _prefill_logits(q_bf, cache_bf16, page_table, qsl, vis, w_ref,
                      max_qlen, logits_width, query_offset=0, num_queries=num_tokens)
arm = _prefill_logits(q_fp8, cache_fp8, page_table, qsl, vis, w_arm,
                      max_qlen, logits_width, query_offset=0, num_queries=num_tokens)

k8 = cache_fp8.to(torch.float16).float()
q8 = q_fp8.to(torch.float16).float()
for i in (0, 1, 3, 7):
    req = 0 if i < q_lens[0] else 1
    pages = page_table[req]
    k_rows = k8[pages, :, 0, :].reshape(-1, HEAD_DIM)
    scores = (k_rows @ q8[i].T).clamp(min=0).sum(-1)
    cols = min(int(vis[i].item()), logits_width)
    d = (arm[i, :cols] - scores[:cols]).abs()
    dref = (ref[i, :cols] - scores[:cols]).abs()
    print(
        f"row {i} (req {req}): arm-vs-exp max={d.max().item():.4f} "
        f"at col {d.argmax().item()}; ref-vs-exp max={dref.max().item():.4f}; "
        f"arm[0:3]={arm[i,:3].tolist()} exp[0:3]={scores[:3].tolist()}"
    )
