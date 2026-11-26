# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Self-check for the video-encoder micro-batch chunker.
# Run from the repo root: PYTHONPATH=. python -m pytest tests/test_video_chunk.py
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner


def test_slices_cover_and_respect_budget():
    budgets = [(379, 2560, 16384), (100, 100, 999), (1, 5, 10), (430, 1000, 16384)]
    for t, hw, budget in budgets:
        s = EncoderRunner._video_chunk_slices(t, hw, budget)
        assert sum(s) == t, (t, hw, budget, s)
        for units in s:
            assert units * hw <= budget or hw > budget, (t, hw, budget, s)
            assert 1 <= units <= max(1, budget // hw) or hw > budget, (t, hw, budget, s)


def test_degenerate_falls_back_to_single_slice():
    assert EncoderRunner._video_chunk_slices(7, 0, 100) == [7]
    assert EncoderRunner._video_chunk_slices(7, 5, 0) == [7]


def test_over_budget_unit_stays_one_per_slice():
    # h*w larger than the whole budget: each temporal unit is its own slice
    # (cannot chunk below one unit) and none are merged.
    s = EncoderRunner._video_chunk_slices(4, 50000, 16384)
    assert s == [1, 1, 1, 1]
