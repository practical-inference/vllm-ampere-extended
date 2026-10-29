# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the Dynamic SD batch-size schedule helpers."""

import logging

import pytest
import torch

import vllm.envs as envs
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.structured_output import StructuredOutputManager


def _make_lookup(
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]],
    *,
    max_batch_size: int = 256,
    runtime_num_speculative_tokens: int = 3,
) -> list[int]:
    return build_dynamic_sd_schedule_lookup(
        num_speculative_tokens_per_batch_size=num_speculative_tokens_per_batch_size,
        vllm_max_batch_size=max_batch_size,
        vllm_num_speculative_tokens=runtime_num_speculative_tokens,
    )


def _make_scheduler_with_dynamic_sd(
    schedule: list[tuple[int, int, int]],
    *,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    runtime_num_speculative_tokens: int = 3,
) -> Scheduler:
    base_scheduler = create_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_speculative_tokens=runtime_num_speculative_tokens,
    )

    speculative_config = base_scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    speculative_config.num_speculative_tokens_per_batch_size = schedule

    return Scheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )


def _add_requests_and_schedule(
    scheduler: Scheduler, num_requests: int, *, num_tokens: int = 10
):
    requests = create_requests(num_requests=num_requests, num_tokens=num_tokens)
    for request in requests:
        scheduler.add_request(request)
    return scheduler.schedule()


def test_dynamic_sd_uses_batch_size_schedule():
    dynamic_sd_lookup = _make_lookup(
        [
            (1, 16, 3),
            (32, 128, 2),
            (256, 2048, 0),
        ]
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[16] == 3
    assert dynamic_sd_lookup[17] == 3
    assert dynamic_sd_lookup[31] == 3
    assert dynamic_sd_lookup[32] == 2
    assert dynamic_sd_lookup[128] == 2
    assert dynamic_sd_lookup[129] == 2
    assert dynamic_sd_lookup[255] == 2
    assert dynamic_sd_lookup[256] == 0


def test_dynamic_sd_requires_schedule_starting_at_batch_size_one():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_lookup([(2, 16, 3)])


def test_dynamic_sd_clamps_k_to_runtime_max():
    dynamic_sd_lookup = _make_lookup(
        [(1, 256, 4)],
        runtime_num_speculative_tokens=3,
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[256] == 3


def test_dynamic_sd_rejects_invalid_schedule_entry():
    with pytest.raises(ValueError, match="3-item sequence"):
        _make_lookup([(1, 16, 3), (32, 64)])  # type: ignore[list-item]


def test_dynamic_sd_rejects_overlapping_ranges():
    with pytest.raises(ValueError, match="non-overlapping and sorted"):
        _make_lookup([(1, 16, 3), (16, 32, 2)])


def test_dynamic_sd_rejects_negative_k():
    with pytest.raises(ValueError, match="values must be >= 0"):
        _make_lookup([(1, 16, -1)])


def test_dynamic_sd_rejects_empty_schedule():
    with pytest.raises(ValueError, match="must not be empty"):
        _make_lookup([])


def test_dynamic_sd_requires_schedule_config():
    with pytest.raises(
        ValueError, match="num_speculative_tokens_per_batch_size is required"
    ):
        build_dynamic_sd_schedule_lookup(
            None,
            vllm_max_batch_size=256,
            vllm_num_speculative_tokens=3,
        )


def test_dynamic_sd_lookup_rejects_invalid_batch_size_queries():
    dynamic_sd_lookup = _make_lookup([(1, 256, 3)])

    assert dynamic_sd_lookup[0] == 0
    with pytest.raises(IndexError):
        _ = dynamic_sd_lookup[257]


def test_scheduler_initializes_dynamic_sd_lookup_from_speculative_config():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )

    assert scheduler.dynamic_sd_lookup is not None
    assert scheduler.num_spec_tokens == 3


def test_scheduler_uses_dsd_k_based_on_number_of_scheduled_requests():
    test_cases = [
        (4, 3),
        (64, 2),
        (256, 0),
    ]

    for num_requests, expected_k in test_cases:
        scheduler = _make_scheduler_with_dynamic_sd(
            [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
            max_num_seqs=num_requests,
            max_num_batched_tokens=num_requests * 10,
            runtime_num_speculative_tokens=3,
        )
        output = _add_requests_and_schedule(scheduler, num_requests)

        assert len(output.num_scheduled_tokens) == num_requests
        assert output.num_spec_tokens_to_schedule == expected_k


def test_scheduler_clamps_dsd_k_to_runtime_num_speculative_tokens():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 256, 5)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_falls_back_to_static_k_when_dsd_not_configured():
    scheduler = create_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=40,
        num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 4)

    assert scheduler.dynamic_sd_lookup is None
    assert output.num_spec_tokens_to_schedule == 3


def test_dynamic_sd_is_disabled_with_data_parallel(caplog_vllm):
    with caplog_vllm.at_level(logging.WARNING, logger="vllm"):
        scheduler = create_scheduler(
            max_num_seqs=256,
            max_num_batched_tokens=2560,
            num_speculative_tokens=3,
            num_speculative_tokens_per_batch_size=[
                (1, 16, 3),
                (64, 128, 2),
                (256, 4096, 0),
            ],
            data_parallel_size=2,
        )

    speculative_config = scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    assert speculative_config.num_speculative_tokens_per_batch_size is None
    assert scheduler.dynamic_sd_lookup is None
    assert "Dynamic speculative decoding is not supported with data parallelism" in (
        caplog_vllm.text
    )

    output = _add_requests_and_schedule(scheduler, 256)
    assert len(output.num_scheduled_tokens) == 256
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_uses_static_k_when_no_requests_are_scheduled():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )
    output = scheduler.schedule()

    assert len(output.num_scheduled_tokens) == 0
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_rejects_bad_dsd_config_at_construction():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_scheduler_with_dynamic_sd([(2, 16, 3)])


def test_scheduler_passes_max_num_seqs_as_dsd_runtime_batch_limit():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert scheduler.dynamic_sd_lookup is not None
    assert len(scheduler.dynamic_sd_lookup) == 17
    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3


def _make_scheduler_with_k0_lifecycle(
    schedule: list[tuple[int, int, int]],
    *,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 512,
    num_spec: int = 2,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> Scheduler:
    if monkeypatch is not None:
        monkeypatch.setattr(envs, "VLLM_EAGLE_K0_LIFECYCLE", True)
        # The lifecycle is V1-runner scoped; force the V1 selection the same
        # way the env would.
        monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    base = create_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_speculative_tokens=num_spec,
    )
    # The lifecycle gate keys on the eagle-family method; the scheduler-side
    # mechanics are drafter-agnostic, so re-label the validated ngram config.
    object.__setattr__(base.vllm_config.speculative_config, "method", "mtp")
    spec_config = base.vllm_config.speculative_config
    assert spec_config is not None
    spec_config.num_speculative_tokens_per_batch_size = schedule
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer"], spec),
            KVCacheGroupSpec(["draft_layer"], spec, is_eagle_group=True),
        ],
    )
    return Scheduler(
        vllm_config=base.vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=16,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base.vllm_config),
    )


def _step(scheduler: Scheduler):
    output = scheduler.schedule()
    scheduled = output.num_scheduled_tokens
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=list(scheduled.keys()),
            req_id_to_index={rid: i for i, rid in enumerate(scheduled)},
            sampled_token_ids=[[0] for _ in scheduled],
        ),
    )
    return output


def test_eagle_k0_lifecycle_disabled_by_default(monkeypatch):
    scheduler = _make_scheduler_with_k0_lifecycle(
        [(1, 4, 2), (5, 16, 0)], monkeypatch=None
    )
    assert not scheduler.eagle_k0_lifecycle

    for request in create_requests(num_requests=5, num_tokens=20):
        scheduler.add_request(request)
    output = _step(scheduler)
    assert output.num_spec_tokens_to_schedule == 0
    assert not output.eagle_k0_hold
    assert not output.eagle_catchup_req_ids
    assert not scheduler._eagle_hold


def test_eagle_k0_lifecycle_hold_resume_cycle(monkeypatch):
    scheduler = _make_scheduler_with_k0_lifecycle(
        [(1, 4, 2), (5, 16, 0)], monkeypatch=monkeypatch
    )
    assert scheduler.eagle_k0_lifecycle
    coordinator = scheduler.kv_cache_manager.coordinator
    requests = create_requests(num_requests=4, num_tokens=20)
    for request in requests:
        scheduler.add_request(request)

    # K=2 tier: normal speculation, no hold; the written-coverage tracker
    # is refreshed to the post-rejection frontier.
    output = _step(scheduler)
    assert output.num_spec_tokens_to_schedule == 2
    assert not output.eagle_k0_hold
    assert not scheduler._eagle_hold
    for request in requests:
        assert scheduler._eagle_sync_points[request.request_id] == (
            request.num_computed_tokens
        )

    # Admitting a 5th request moves the batch to the K=0 tier: the edge
    # engages the hold; the newly admitted request has no tracked write.
    fifth = create_requests(num_requests=5, num_tokens=20)[-1]
    scheduler.add_request(fifth)
    output = _step(scheduler)
    assert output.num_spec_tokens_to_schedule == 0
    assert output.eagle_k0_hold
    assert scheduler._eagle_hold
    assert scheduler._eagle_sync_points[fifth.request_id] == 0

    # Held steps: the entry trim releases the edge step's transient draft
    # growth and the allocation cap keeps the tables at the frozen sync
    # points while num_computed advances.
    output = _step(scheduler)
    assert output.num_spec_tokens_to_schedule == 0
    assert output.eagle_k0_hold
    held_capacities = {
        rid: coordinator.eagle_synced_tokens(rid)
        for rid in scheduler._eagle_sync_points
    }
    for _ in range(2):
        output = _step(scheduler)
        assert output.num_spec_tokens_to_schedule == 0
        assert output.eagle_k0_hold
    for rid, sync_point in scheduler._eagle_sync_points.items():
        capacity = coordinator.eagle_synced_tokens(rid)
        assert sync_point <= capacity < sync_point + 16
    assert {
        rid: coordinator.eagle_synced_tokens(rid)
        for rid in scheduler._eagle_sync_points
    } == held_capacities

    # Dropping back under the tier resumes: one-step pause, catch-up gaps
    # equal to the held-back range, hold lifted.
    scheduler.finish_requests([requests[0].request_id], RequestStatus.FINISHED_STOPPED)
    fifth_nc = scheduler.requests[fifth.request_id].num_computed_tokens
    output = _step(scheduler)
    assert output.num_spec_tokens_to_schedule == 0
    assert not output.eagle_k0_hold
    assert not scheduler._eagle_hold
    assert set(output.eagle_catchup_req_ids) == {r.request_id for r in requests[1:]} | {
        fifth.request_id
    }
    for request in requests[1:]:
        gap = output.eagle_catchup_req_ids[request.request_id]
        assert 0 < gap <= request.num_computed_tokens
    # The re-grown eagle blocks must reach the runner with an aligned row:
    # the resume step routes the catch-up requests through the resumed
    # channel (remove + re-add with the FULL post-regrowth table) because
    # the runner's rows are append-only and carry a stale tail from the
    # hold-edge trim (the 2026-10-29 delta-review a1/a3 findings).
    cached = output.scheduled_cached_reqs
    assert fifth.request_id in cached.resumed_req_ids
    idx = cached.req_ids.index(fifth.request_id)
    block_ids = cached.new_block_ids[idx]
    assert block_ids is not None and block_ids[1], (
        "re-grown eagle blocks missing from the resume step's replace table"
    )
    # The mid-hold admission catches up over its full context (every token
    # computed before the resume step; the pause step's own token is not
    # part of the gap).
    assert output.eagle_catchup_req_ids[fifth.request_id] == fifth_nc

    # The step after the pause speculates normally again.
    output = _step(scheduler)
    assert output.num_spec_tokens_to_schedule == 2
    assert not output.eagle_k0_hold


def test_eagle_k0_lifecycle_midhold_admission(monkeypatch):
    # 2026-10-28 exp2 regression: admissions stalled for the whole K=0 hold
    # (c32 waves collapsed 3/12, admission queue depth 16 at max concurrency).
    # A same-prefix request admitted mid-hold must be scheduled, must NOT
    # adopt eagle-group blocks (its sync-point-0 hold trim would free
    # adopted blocks its source still reads), and must not disturb the
    # sources' written eagle coverage.
    scheduler = _make_scheduler_with_k0_lifecycle(
        [(1, 4, 2), (5, 16, 0)], monkeypatch=monkeypatch
    )
    coordinator = scheduler.kv_cache_manager.coordinator
    requests = create_requests(num_requests=4, num_tokens=20, same_prompt=True)
    for request in requests:
        scheduler.add_request(request)
    output = _step(scheduler)
    assert output.num_spec_tokens_to_schedule == 2

    # The 5th same-prompt request tips the batch into the K=0 tier.
    fifth = create_requests(num_requests=5, num_tokens=20, same_prompt=True)[-1]
    scheduler.add_request(fifth)
    output = _step(scheduler)
    assert output.eagle_k0_hold
    assert scheduler._eagle_sync_points[fifth.request_id] == 0
    # One held step: the entry trim releases the edge step's transient draft
    # growth (the 5th's adopted hit included) and the tables sit at their
    # sync points.
    output = _step(scheduler)
    assert output.eagle_k0_hold

    # Mid-hold admission of a 6th same-prompt request: scheduled (no
    # deferral), eagle table empty (hit stripped + allocation capped at
    # the sync point), sources' written eagle coverage untouched.
    sixth = create_requests(num_requests=6, num_tokens=20, same_prompt=True)[-1]
    before = {
        rid: coordinator.eagle_synced_tokens(rid)
        for rid in scheduler._eagle_sync_points
    }
    scheduler.add_request(sixth)
    output = _step(scheduler)
    assert output.eagle_k0_hold
    assert sixth.request_id in output.num_scheduled_tokens
    assert scheduler._eagle_sync_points[sixth.request_id] == 0
    assert coordinator.eagle_synced_tokens(sixth.request_id) == 0
    after = {
        rid: coordinator.eagle_synced_tokens(rid)
        for rid in scheduler._eagle_sync_points
    }
    assert {rid: after[rid] for rid in before} == before

    # Held steps keep the admitted request's eagle table empty.
    for _ in range(2):
        output = _step(scheduler)
        assert output.eagle_k0_hold
        assert coordinator.eagle_synced_tokens(sixth.request_id) == 0

    # Dropping under the tier resumes: the mid-hold admission catches up
    # over its full computed context (sync point 0), hold lifts. Two
    # finishes take the batch from 6 to 4 (the K=2 tier).
    scheduler.finish_requests(
        [requests[0].request_id, requests[1].request_id],
        RequestStatus.FINISHED_STOPPED,
    )
    sixth_nc = scheduler.requests[sixth.request_id].num_computed_tokens
    output = _step(scheduler)
    assert not scheduler._eagle_hold
    assert output.eagle_catchup_req_ids[sixth.request_id] == sixth_nc
    # The sync-0 admission rides the same resumed channel: the full
    # post-regrowth table replaces its (empty) runner row.
    cached = output.scheduled_cached_reqs
    assert sixth.request_id in cached.resumed_req_ids
    idx = cached.req_ids.index(sixth.request_id)
    block_ids = cached.new_block_ids[idx]
    assert block_ids is not None and block_ids[1]
    # The step after the pause speculates normally again.
    output = _step(scheduler)
    assert output.num_spec_tokens_to_schedule == 2
    assert not output.eagle_k0_hold
