# WORKORDER: GLM night window — lever verdicts (2026-11-14 morning)

Window: GLM :8080 exclusive to columbo 2026-11-13 21:00 -> 2026-11-14 03:45 MST.
Runbook: glm-night-window-runbook-20261113.md. Outcome: PROMOTED
(serve-script defaults folded; all night drop-ins deleted; :8006 untouched all night).

## Promoted config (serve script defaults, backup .pre-promotion-20261114)

| Knob | Old | New | Why |
|------|-----|-----|-----|
| MAX_NUM_SEQS | 128 | 96 | owner-approved; frees ~9.4 GiB mamba static; ramp-verified c<=88 unharmed |
| SPEC_SCHEDULE | [[1,1,2],[2,128,3]] | [[1,1,2],[2,128,2]] | K-ceiling 2: acceptance preserved (run-mean 2.93 vs 2.87), per-position unchanged |
| LONG_PREFILL_TOKENS | 0 (off) | 2048 | new knob in serve script; caps long-prompt share of the per-step token budget |
| LONG_PREFILL_ADAPTIVE | 0 (off) | 1 | fair-share floor MBT/(queued+running); storm canary gaps p90 -33%, max -34% at median ~baseline |

## Full lever table (measured, repro: populate 6x200k -> multiround 2x -> storm 4x300k + looping canary)

| ID | Lever | Verdict | Evidence (collapse windows / storm-wall / canary p50/p90/max) |
|----|-------|---------|---------------------------------------------------------------|
| B0 | baseline re-verify | storm REPRODUCES | 2w/330s, wall 290.6s, 14.8ms/1.88s/2.41s |
| E1 | RESERVE_ISL=0 | doesn't work (no benefit) | 2w/590s, wall 292.2s, 14.8/1.97/2.38 — backlog becomes Waiting, same starvation |
| E1b | RESERVE_ISL=0 + c=88 ramp | safe, no death | 88/88 sessions ok, kv peak 98.7%, 'Cannot get 2 free blocks' never fired at batch saturation |
| E4 | MBT=4096 | doesn't work (worse) | 3w/730s, wall 364.5s, 892ms/1.10s/1.61s — throttle without smoothing |
| E5 | PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,gc_threshold:0.8 | doesn't work (net negative) | 3w/750s, wall 395.8s (+24%), storm p50 990ms (~55x), max 6.65s; no GC stall >2s found |
| L1 | LPT=2048 flat | doesn't work (median regresses) | 3w/700s, wall 315.9s, storm p50 627ms |
| L2 | LPT=2048 adaptive | WORKS | 3w/650s, wall 320.2s, 17.4ms/1.26s/1.60s — tail -33/34%, median ~= baseline |
| E7 | L2 + ladder K=2 | WORKS | wall 320.1s, 16.1/1.26/1.61; acceptance run-mean 2.93, final 2.77 (drop 0.10 < 0.15) |
| E6 | E7 + seqs=96 | WORKS (stack verified) | wall 320.3s, 18.0/1.26/1.58 — identical profile, ~9.4 GiB freed |
| E8 | MBT=16384 | NON-BOOTABLE at 1M ctx/0.965 | KV pool lands at 2.78 GiB < 6.67 GiB floor; bootable only at max-model-len <= ~428k or fewer seqs |
| GA1 | guarded admission cap=1/step (max_new_admissions_per_step) | doesn't work — REJECTED | identical staircase (311-320 vs 312-320s), wall 319.9 vs 319.8s, collapse 650s/3w both, wait/defer 0% both (door never binds here); canary max worse (3.67s vs 1.59s, 1 run). The ~320s wall is prefill-throughput-bound, not admission-concurrency-bound. Feature left INERT (default None) in fork: 1588ddc3e6 |
| FIN | promoted boot | GREEN | repro matches L2/E7/E6; 2h15m soak: 120/120 samples 200, 0 dead, storm ramp alive=True through soak |

## Known limits / what remains open

- The defer->burst->storm cycle is NOT eliminated by any flag-level lever: 2-3 collapse
  windows of ~300s persist under full storm load by construction (1.2M tokens of
  simultaneous cold prefill on TP4 lockstep). L2-adaptive cuts the starvation TAIL
  (1.88s -> 1.26s p90) and preserves the median; it does not flatten aggregate collapse.
- The morning's "Deferred 10 at KV 11-30%" signature never fired under the repro
  (0 defer events) — L2 was validated against Waiting-backpressure starvation only.
- Guarded admission WAS developed in the fork (max_new_admissions_per_step, 1588ddc3e6,
  env-cap armed at 05:14, GA1 above) and measured REJECTED: the collapse is
  prefill-throughput-bound, so no admission pacing helps. Feature stays INERT.
  (b) dequant-ws lane EV (+6.4% prefill question) remains a port-days decision,
  not measurable at MBT>=16384 on the 1M recipe (E8 non-bootable).
- Onset history (full-log bucketing, 54K heartbeats): Nov 1-2 storms were WORSE
  (161 windows) than the Nov 13 spike (101) under the same serving recipe, and
  predate the current wheel (Nov 8 build). The storms are traffic-shaped —
  group arrivals of big cold prefills — not introduced by any config change.
- SSM int8/int4, mamba_page_size_padded, expandable_segments: dead per pin forensics
  (glm_pin_forensics_verdict_20261113.md) — unchanged.

## Cleanup

- Night drop-ins: ALL DELETED (dir empty; audit clean).
- OOM hook: fix committed to fork + prod venv (4aecdb8693) — armed-line verified,
  INERT unless VLLM_OOM_SNAPSHOT_DIR set. /data/models/oom_snaps empty (no OOM).
- PR #22 (main <- lane/oom-snapshot-tooling) now includes: OOM tooling + forensics
  verdict + repro/analyzer/soak runner + date-aware analyzer --since + torch-compat fix.
