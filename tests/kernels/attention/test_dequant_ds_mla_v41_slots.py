# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exactness tests for the DeepSeek-V4 legacy 584B fp8 record gather.

`dequant_ds_mla_slots(..., segregated_scale=True)` (index-driven) must match
both a plain-torch dequant of the record and the seq-driven V4.1 oracle
`dequantize_and_gather_k_cache_triton` bit-for-bit.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_mla_sparse_kernel import dequant_ds_mla_slots

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton dequant kernel requires CUDA/ROCm",
)

FP8_DIM = 448  # quantized NoPE dims (bytes)
ROPE_DIM = 64  # raw bf16 rope dims (128 bytes)
TOKEN_DATA_BYTES = FP8_DIM + ROPE_DIM * 2  # 576
NUM_SCALES = FP8_DIM // 64  # 7 UE8M0 scales of 64 dims
SCALE_BYTES = NUM_SCALES + 1  # 8 (1 pad byte)
TOKEN_BYTES = TOKEN_DATA_BYTES + SCALE_BYTES  # 584
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _write_v4_record(k: torch.Tensor, num_blocks: int, block_size: int):
    """Encode bf16 K rows with the V4 writer semantics into 584B pages."""
    n = k.shape[0]
    tiles = k[:, :FP8_DIM].float().view(n, NUM_SCALES, 64)
    amax = tiles.abs().amax(2).clamp(min=1e-4)
    exponent = torch.ceil(torch.log2(amax / FP8_MAX))
    scale = torch.exp2(exponent)
    fp8 = (tiles / scale.unsqueeze(2)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    scale_bytes = (exponent + 127).clamp(0, 255).to(torch.uint8)

    cache = torch.zeros(num_blocks, block_size, TOKEN_BYTES, dtype=torch.uint8)
    for blk in range(num_blocks):
        rows = slice(blk * block_size, (blk + 1) * block_size)
        page = cache[blk].view(-1)
        data = torch.cat(
            [
                fp8[rows].view(block_size, FP8_DIM).view(torch.uint8),
                k[rows, FP8_DIM:].contiguous().view(torch.uint8),
            ],
            dim=1,
        )
        page[: block_size * TOKEN_DATA_BYTES] = data.reshape(-1)
        scales = torch.cat(
            [scale_bytes[rows], torch.zeros(block_size, 1, dtype=torch.uint8)], dim=1
        )
        lo = block_size * TOKEN_DATA_BYTES
        page[lo : lo + block_size * SCALE_BYTES] = scales.reshape(-1)
    return cache.cuda()


@pytest.fixture(scope="module")
def v4_setup():
    torch.manual_seed(0)
    block_size, num_blocks = 64, 3
    k = torch.randn(block_size * num_blocks, 512, dtype=torch.bfloat16)
    cache = _write_v4_record(k, num_blocks, block_size)
    indices = torch.randint(
        0, block_size * num_blocks, (517,), dtype=torch.int32, device="cuda"
    )
    indices[::7] = -1  # padding slots
    return cache, indices, block_size


def _torch_reference(cache, indices, block_size):
    out = torch.zeros(
        indices.shape[0], FP8_DIM + ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    )
    valid = indices >= 0
    slots = indices[valid].long()
    pages = cache.view(cache.shape[0], -1)[slots // block_size]  # [V, bs*584]
    data_base = (slots % block_size * TOKEN_DATA_BYTES).unsqueeze(1)
    data = pages.gather(1, data_base + torch.arange(TOKEN_DATA_BYTES, device="cuda"))
    fp8 = data[:, :FP8_DIM].view(torch.float8_e4m3fn).float()
    sf_base = (slots % block_size) * SCALE_BYTES + block_size * TOKEN_DATA_BYTES
    sf = pages.gather(1, sf_base.unsqueeze(1) + torch.arange(NUM_SCALES, device="cuda"))
    scale = torch.exp2(sf.float() - 127.0).repeat_interleave(64, dim=1)
    out[valid, :FP8_DIM] = (fp8 * scale).to(torch.bfloat16)
    out[valid, FP8_DIM:] = data[:, FP8_DIM:].contiguous().view(torch.bfloat16)
    return out


def _dequant(cache, indices, block_size):
    out = torch.zeros(
        indices.shape[0], FP8_DIM + ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    )
    dequant_ds_mla_slots(
        out,
        cache,
        indices,
        cache_block_size=block_size,
        rope_dim=ROPE_DIM,
        nope_dim=FP8_DIM,
        quant_block=64,
        segregated_scale=True,
    )
    return out


def test_v4_segregated_dequant_bitexact(v4_setup):
    cache, indices, block_size = v4_setup
    out = _dequant(cache, indices, block_size)
    ref = _torch_reference(cache, indices, block_size)
    assert (out.view(torch.int16) == ref.view(torch.int16)).all()


@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="v41 oracle kernel casts to fp8e4nv, unavailable on SM80 "
    "(that's why the index-driven dequant uses the software path)",
)
def test_v4_segregated_dequant_matches_v41_oracle(v4_setup):
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        dequantize_and_gather_k_cache_triton,
    )

    cache, indices, block_size = v4_setup
    num_blocks = cache.shape[0]
    oracle_out = torch.zeros(
        num_blocks, block_size, 512, dtype=torch.bfloat16, device="cuda"
    )
    seq_lens = torch.full((num_blocks,), block_size, dtype=torch.int32, device="cuda")
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").unsqueeze(
        1
    )
    dequantize_and_gather_k_cache_triton(
        oracle_out, cache, seq_lens, None, block_table, block_size, offset=0
    )
    ref = oracle_out.view(num_blocks * block_size, 512)[indices.clamp(min=0).long()]
    ref[indices < 0] = 0

    out = _dequant(cache, indices, block_size)
    assert (out.view(torch.int16) == ref.view(torch.int16)).all()
