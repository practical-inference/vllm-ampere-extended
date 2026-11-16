"""Minimal _maybe_dequant repro: uint8 tile -> fp16, vs torch dequant."""
import sys
import torch

sys.path.insert(0, "/home/columbo/source/vllm-nightly")
from vllm.models.qwen4_exp.nvidia.ops.qsa import _maybe_dequant  # noqa: E402
from vllm.triton_utils import tl, triton  # noqa: E402


@triton.jit
def _dequant_probe(x_ptr, out_ptr, N: tl.constexpr, D: tl.constexpr):
    n = tl.arange(0, N)
    d = tl.arange(0, D)
    tile = tl.load(x_ptr + n[:, None] * D + d[None, :])
    tile = _maybe_dequant(tile)
    tl.store(out_ptr + n[:, None] * D + d[None, :], tile)


x8 = torch.randint(0, 256, (64, 128), dtype=torch.uint8, device="cuda")
ref = x8.view(torch.float8_e4m3fn).to(torch.float16)
out = torch.empty((64, 128), dtype=torch.float16, device="cuda")
_dequant_probe[(1,)](x8, out, N=64, D=128)
torch.accelerator.synchronize()
d = (out.float() - ref.float() * (1 / 256)).abs()
print(f"probe: max_abs_vs_x*2^-8 = {d.max().item():.6f}; out[0,:4]={out[0,:4].tolist()}")
print(f"ref[0,:4]={[round(v,5) for v in ref[0,:4].float().tolist()]}")
nz_out = (out != 0).float().mean().item()
nz_ref = (ref != 0).float().mean().item()
print(f"nonzero: out={nz_out:.3f} ref={nz_ref:.3f}")
