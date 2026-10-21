<!-- markdownlint-disable MD060 -->

# KV offloading on Ampere: measured net-negative for long-context serving

**Verdict: avoid KV offloading on this workload.** With the
`OffloadingConnector` enabled, GPU-side overhead shrank the effective KV pool
enough to cause the very preemptions it exists to rescue, and
capture-then-resume did not win the loss back.

## Setup

- GPU: NVIDIA PG506-230 (SM80, Ampere), single card
- Workload: 40 longest doc-summary strips, 30 concurrent requests
- vLLM: nightly build (0.32.x), PP=1
- Two arms, same card, same pool, same request set, run back-to-back:
    - **Arm A** — engine configured with `OffloadingConnector` (KV offload to CPU)
    - **Arm C** — identical config with the KV offload connector removed

## Data

| Metric | Arm A (offload on) | Arm C (offload off) | Delta |
|---|---|---|---|
| Wall time | 269 s | 234 s | **−13% with offload off** |
| p50 request latency | 120 s | 110 s | −8% |
| p90 request latency | 184 s | 160 s | −13% |
| Preemptions | 10 | 0 | — |
| KV captured during probe | 2.17% | — | — |

## Why it loses

Consistent with the measurements: the connector holds GPU memory for its own
bookkeeping and staging that would otherwise hold KV blocks. The effective
pool shrinks, the scheduler hits the pool ceiling, and preemptions begin.
Capture worked (2.17% of tier traffic was captured during the probe window),
but the rescued recomputation was far smaller than the cost of the
preemptions the connector induced. We observed the correlation
(overhead → smaller effective pool → preemptions) rather than tracing the
connector's internals; the numbers above are the evidence.

## When might offload still win?

Not on this workload. Offloading pays when capture rate is high relative to
its overhead — e.g., very large context-to-working-set ratios or bursty
multi-session reuse. On a single-card Ampere box serving concurrent long
contexts, the overhead dominates. Re-test if you change the concurrency
profile or pool size materially.

## Reproduce

Run the same workload twice against the same card and pool: once with
`OffloadingConnector` in the engine config, once without it. Measure wall
time, p50/p90 request latency, and the preemption count from the vLLM
metrics endpoint.

Measured 2026-10-18 on the validation fleet.
