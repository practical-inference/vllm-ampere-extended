# WORKORDER: GLM night window runbook — prefill-storm elimination + lever development (2026-11-13)

Owner directive (2026-11-13 ~21:00 MST): GLM :8080 is exclusive to columbo all
night; qwen :8006 serves the owner's new session — DO NOT TOUCH :8006.
P1 = stop the prefill storms with minimum sacrifice to the rest of the
performance picture. P2 = the prioritized lever list. Run every experiment
until each lever is KNOWN working or not. Morning state: either a validated
improvement promoted, or the proven baseline restored.

## The measured problem (baseline, 12:11 boot)

- defer->burst->storm cycle, 58 collapse windows today. Worst: Deferred 10 at
  KV 11-30%, then a 39,431 tok/s prefill burst, gen <20 tok/s for 150s.
- Running max 12 / mean 2.9 (interactive); owner's batch use case hits max
  concurrency at will. Owner approved max-num-seqs 96 (loses efficiency >88).
- Analyzer: /tmp/opencode/glm_storm_analyzer.py <log> --since HH:MM:SS
- Repro: /tmp/opencode/glm_storm_repro.py (populate -> multiround -> storm,
  canary stream = the starvation gauge)

## Infrastructure staged (no bounce yet)

- Serve-script knobs (default = proven behavior; backup
  serve_flash_mtp_fp8_draftq_mtile.sh.pre-leverknobs-20261113):
  MAX_NUM_SEQS (128), MAX_NUM_BATCHED_TOKENS (8192), RESERVE_ISL (1).
- Prod venv OOM hook re-patched (backup gpu_worker.py.pre-oomsnap-20261113):
  VLLM_OOM_SNAPSHOT_DIR arms stacks=all history + per-rank JSON dump on OOM.
- KNOWN HISTORY: the --no-reserve-isl arm already killed the engine once
  ('Cannot get 2 free blocks', scheduler.py:988 — unguarded over-admission).
  K0 slot release fires ONLY at K=0 step-tier -> INERT under the prod ladder
  (min K=2); the deferral door-shut is reserve-isl or mamba admission, NOT K0.

## Boot matrix (sequential; each boot = drop-in env + restart + repro + analyzer)

| ID | Config (drop-in env) | Question | Decision gate |
|----|----------------------|----------|---------------|
| B0 | (none — baseline re-verify) | Does the repro reproduce the storm? | repro shows Deferred>=3 then burst >=10x prompt rate; else fix repro first |
| E1 | RESERVE_ISL=0 | Does the door-shut vanish? Death mode fire at interactive load? | storms gone + engine alive 30min -> candidate; death -> revert, develop guarded admission |
| E1b | RESERVE_ISL=0 + batch ramp (probe_ramp to 88 conc) | Does over-admission death fire at batch concurrency? | death -> E1 is interactive-only; guarded admission required for batch |
| E4 | MAX_NUM_BATCHED_TOKENS=4096 (+ best of E1) | Storm smoothing: decode interleave during bursts | collapse-window seconds -50% vs E1-best, prefill wall +<20% |
| E5 | PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8 | Fragmentation cure; GC-pause cost | allocator_settings shows both; saturation within 5%; no >2s GC stalls |
| E6 | MAX_NUM_SEQS=96 (+ best-of) | ~9.4GiB freed; batch case unharmed? | ramp to 88-96 conc: tok/s within 5% of 128 at c<=88 |
| E7 | SPEC_SCHEDULE=[[1,1,2],[2,128,2]] | K ceiling 2: acceptance vs throughput trade | mean acceptance len drop <0.15 AND gen tok/s >= baseline |
| E8 | (b) measure: MAX_NUM_BATCHED_TOKENS=16384 + VLLM_OOM_SNAPSHOT_DIR set | transient slope at MBT 16384 (the +6.4% prefill question) | transient GiB measured; bootability at 0.965 util yes/no |
| FIN | best-of promotion boot + 2h stability soak | morning state | promoted if all gates green; else baseline restored |

Drop-in filename per experiment: exp-20261113-night-<id>.conf (no
MTP_TOKENS/SPEC_SCHEDULE key collisions with any future sweep drop-in).
All drop-ins DELETED at morning promotion or rollback — audit clean by day.

## Metrics per experiment (the minimum-sacrifice gauge)

1. Collapse windows: count + total seconds at gen<20 tok/s (analyzer).
2. Canary: p50/p90/max inter-token gaps during storms (repro output).
3. TTFT per request (repro JSON lines; per-request metrics also on server).
4. Aggregate gen tok/s mean+p50 (heartbeat).
5. E5 adds: allocator_settings verification + OOM-repro attempt (Nov 11 shape).
6. E8 adds: transient GiB at MBT 16384 (allocator history slope).

## Known risks / fallbacks

- E1 death signature: 'Cannot get 2 free blocks' scheduler.py:988 -> immediate
  revert to RESERVE_ISL=1; the lever answer is then "needs guarded admission"
  (code lever: cap admissions/step — develop only if time remains after E4-E8).
- GC stalls (E5): >2s heartbeat dips -> drop garbage_collection_threshold
  to 0.9 or drop the knob, keep max_split_size.
- Any engine death: restart with the proven config (drop-in removed) before
  the next experiment; every boot verified with /data/models/bin/fleet_state.sh.
- :8006/:8017 untouched all night. OOM snapshots land in VLLM_OOM_SNAPSHOT_DIR.

## Morning deliverable

The lever table with KNOWN verdicts (works / doesn't / needs-code), the
storm-metrics before/after, and either the promoted config (with the soak
evidence) or the restored baseline.
