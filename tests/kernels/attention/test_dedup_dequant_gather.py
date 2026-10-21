# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exactness test for the dedup dequant-gather fast path.

`_dedup_dequant_gather` (sort + cumsum ranks + idempotent scatter) must
produce workspace rows and per-query indices such that the sparse attention
output is bit-identical to the 2-pass dequant-workspace path.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    _DS_MLA_DEQUANT_WS,
    _DS_MLA_NOPE_DIM,
    _dedup_dequant_gather,
)
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    dequant_ds_mla_slots,
    triton_mla_sparse_attention,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton dequant kernel requires CUDA/ROCm",
)

BLOCK_SIZE = 64
NOPE_DIM = _DS_MLA_NOPE_DIM  # 512
CACHE_BYTES = 656


def _make_cache(num_slots: int, seed: int) -> torch.Tensor:
    """Random fp8_ds_mla pages (NoPE model: 512 fp8 + 16 scale + 128 zero tail)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_blocks = (num_slots + BLOCK_SIZE - 1) // BLOCK_SIZE
    cache = torch.randint(
        0, 255, (num_blocks, BLOCK_SIZE, CACHE_BYTES), dtype=torch.uint8, generator=g
    )
    cache[:, :, 512:528] = 120  # fp32 scale bytes, finite magnitude
    cache[:, :, 528:] = 0  # NoPE zero tail
    cache[cache == 0x7F] = 0  # e4m3fn NaN pattern
    cache[cache == 0xFF] = 0
    return cache.cuda()


def _make_indices(
    num_tokens: int, topk_w: int, num_slots: int, seed: int
) -> torch.Tensor:
    """Heavy-duplication topk indices drawn from a `num_slots` pool, with -1
    padding in the tail positions."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, num_slots, (num_tokens, topk_w), generator=g)
    idx[:, -8:] = -1
    return idx.to(torch.int32).cuda()


def _reference_rows(cache: torch.Tensor, indices: torch.Tensor, dequant_dim: int):
    """The 2-pass path: dequant every (token, pos) row into its own slot."""
    flat = indices.reshape(-1).to(torch.int32)
    ws = torch.zeros(
        flat.numel(), dequant_dim, dtype=torch.bfloat16, device=flat.device
    )
    dequant_ds_mla_slots(
        ws,
        cache.view(torch.uint8),
        flat,
        cache_block_size=BLOCK_SIZE,
        rope_dim=dequant_dim - NOPE_DIM,
    )
    return ws


@pytest.mark.parametrize("num_tokens", [4096, 8192])
@pytest.mark.parametrize("dequant_dim", [512, 576])
def test_dedup_rows_bitexact(num_tokens: int, dequant_dim: int):
    topk_w = 128  # small width keeps the reference gather affordable
    num_slots = 4096
    cache = _make_cache(num_slots, seed=num_tokens + dequant_dim)
    indices = _make_indices(num_tokens, topk_w, num_slots, seed=7)

    ref = _reference_rows(cache, indices, dequant_dim)

    ws, ws_indices = _dedup_dequant_gather(
        cache.device, cache.view(torch.uint8), indices, dequant_dim
    )
    assert ws_indices.shape == (num_tokens, 1, topk_w)
    assert ws_indices.dtype == torch.int32

    # Every row the attention will read must be bitwise identical to the
    # 2-pass row for the same slot; -1 positions stay -1.
    flat = indices.reshape(-1)
    ranks = ws_indices.reshape(-1).long()
    valid = flat >= 0
    assert bool((ranks[~valid] == -1).all())
    assert bool((ranks[valid] >= 0).all())
    assert bool((ranks[valid] < ws.shape[0]).all())
    assert torch.equal(ws.reshape(-1, dequant_dim)[ranks[valid]], ref[valid])


@pytest.mark.parametrize("num_tokens", [4096])
def test_dedup_attention_bitexact(num_tokens: int):
    """End-to-end: sparse attention over (dedup ws, ranks) == (2-pass ws, base)."""
    topk_w = 128
    num_slots = 4096
    heads = 16
    dequant_dim = 512
    cache = _make_cache(num_slots, seed=99)
    indices = _make_indices(num_tokens, topk_w, num_slots, seed=11)

    ref_ws = _reference_rows(cache, indices, dequant_dim)
    base_idx = torch.arange(num_tokens, dtype=torch.int64).unsqueeze(
        1
    ) * topk_w + torch.arange(topk_w, dtype=torch.int64).unsqueeze(0)
    base_idx = torch.where(indices >= 0, base_idx, -1).view(num_tokens, 1, topk_w)
    base_idx = base_idx.to(torch.int32).cuda()

    ws, ws_indices = _dedup_dequant_gather(
        cache.device, cache.view(torch.uint8), indices, dequant_dim
    )

    q = torch.randn(num_tokens, heads, dequant_dim, dtype=torch.bfloat16).cuda()
    out_ref = triton_mla_sparse_attention(
        q, ref_ws.view(-1, 1, dequant_dim), base_idx, sm_scale=0.1
    )
    out_dedup = triton_mla_sparse_attention(
        q, ws.view(-1, 1, dequant_dim), ws_indices, sm_scale=0.1
    )
    assert torch.equal(out_ref[:, :heads, :], out_dedup[:, :heads, :])


def test_dedup_all_padded():
    """All -1 indices: ranks valid, rows masked, no crash."""
    num_tokens, topk_w = 4096, 64
    dequant_dim = 512
    cache = _make_cache(1024, seed=3)
    indices = torch.full((num_tokens, topk_w), -1, dtype=torch.int32).cuda()
    ws, ws_indices = _dedup_dequant_gather(
        cache.device, cache.view(torch.uint8), indices, dequant_dim
    )
    assert bool((ws_indices == -1).all())
    assert ws.shape[0] >= 1


def test_dedup_workspace_reuse():
    """Grow-only workspace: a smaller u_upper views the front, same storage."""
    dequant_dim = 512
    cache = _make_cache(8192, seed=5)
    big = _make_indices(4096, 64, 8192, seed=1)
    ws_big, _ = _dedup_dequant_gather(
        cache.device, cache.view(torch.uint8), big, dequant_dim
    )
    small = _make_indices(4096, 64, 512, seed=2)
    ws_small, _ = _dedup_dequant_gather(
        cache.device, cache.view(torch.uint8), small, dequant_dim
    )
    # u_upper = min(N=262144, pool_rows=8192) for both — identical size here;
    # the assertion pins the grow-only contract (never shrink, front views).
    assert ws_small.shape[0] == ws_big.shape[0]
    assert ws_small.data_ptr() == ws_big.data_ptr()
    assert (cache.device, dequant_dim) in _DS_MLA_DEQUANT_WS


def test_dedup_u_upper_pool_bound():
    """u_upper = min(index count, pool rows): pool binds when it is smaller."""
    dequant_dim = 512
    num_slots = 1024
    cache = _make_cache(num_slots, seed=6)
    indices = _make_indices(4096, 64, num_slots, seed=8)
    ws, ws_indices = _dedup_dequant_gather(
        cache.device, cache.view(torch.uint8), indices, dequant_dim
    )
    assert ws.shape[0] == num_slots
    ranks = ws_indices.reshape(-1)
    assert bool((ranks[indices.reshape(-1) >= 0] < ws.shape[0]).all())


def _smoke_bench(num_tokens: int = 8192, topk_w: int = 2112, num_slots: int = 450_000):
    """Rough per-call cost, dedup vs 2-pass, at prod-ish prefill shapes."""
    import time

    dequant_dim = 512
    cache = _make_cache(num_slots, seed=42)
    indices = _make_indices(num_tokens, topk_w, num_slots, seed=43)

    def timeit(fn, reps=10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000

    t_dedup = timeit(
        lambda: _dedup_dequant_gather(
            cache.device, cache.view(torch.uint8), indices, dequant_dim
        )
    )
    t_ref = timeit(lambda: _reference_rows(cache, indices, dequant_dim))
    print(
        f"\nnum_tokens={num_tokens} topk={topk_w} slots={num_slots}: "
        f"2-pass gather={t_ref:.2f}ms dedup total={t_dedup:.2f}ms"
    )


if __name__ == "__main__":
    _smoke_bench()
