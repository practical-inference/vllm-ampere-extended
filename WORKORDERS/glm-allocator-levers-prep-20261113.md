# WORKORDER: GLM allocator-lever window prep — gc_threshold + max_split_size (2026-11-13)

Prep for the next GLM :8080 window. Env-only knobs, one boot, no code.
Based on the pin forensics (/data/models/glm_pin_forensics_verdict_20261113.md):
the OOM was fragmentation (0.8 GiB free-in-pool, no hole >= the failed 127 MiB
chunk transient) under a ~82.5 GiB static footprint. expandable_segments is
DOUBLY unavailable (TP4 custom-AR IPC + KV-connector check) — these two knobs
are the remaining allocator levers that work under those constraints.

## The knobs (ONE env var — torch resets absent options, so both live in a

## single string; see vllm/device_allocator/alloc_conf.py docstring)

    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.8"

- `max_split_size_mb:256` — blocks LARGER than 256 MiB are never split, so
  the >=256 MiB holes stay intact for the big transient placements; the
  127-235 MiB chunk transients split only smaller blocks. Rationale: the
  observed transients are 127-235 MiB (forensics trace); 256 keeps
  headroom-class holes whole.
- `garbage_collection_threshold:0.8` — the allocator proactively releases
  cached (inactive) blocks once reserved/capacity crosses 80%, reclaiming
  fragmented pool space before the OOM point. RISK: GC takes the allocator
  lock — pause spikes under pressure are the thing the soak measures.

Deployment surface: the serve script env (preferred — the sweep's drop-in
keys MTP_TOKENS/SPEC_SCHEDULE are unaffected; if a drop-in is used instead,
it MUST use a different filename per workorder rule 3, e.g.
`exp-YYYYMMDD-allocconf.conf`) + rides the promoted pin+vision config.

## Boot plan (one window)

1. Boot with the env; verify allocator_settings in the first memory snapshot
   or via `torch.cuda.memory._snapshot()['allocator_settings']` shows BOTH
   options (parseArgs resets absent options — a typo silently drops the
   other knob).
2. Saturation wave A/B vs the current config (paired-saturation methodology
   from glm_k_release): same prompt mix, 3 waves each.
3. OOM-repro attempt: the Nov 11 OOM'd boot's shape (MBT/pin as banked) with
   the new env — the direct does-it-fix-the-127MiB-placement test.
4. OOM tooling: set VLLM_OOM_SNAPSHOT_DIR=/data/models (+
   VLLM_OOM_SNAPSHOT_MAX_ENTRIES=200000) — the fork's env-gated
   snapshot-on-OOM hook (lane/oom-snapshot-tooling) then captures fully
   attributable frames (stacks=all) if anything OOMs anyway.

## Success criteria

- Boot clean, health 200, allocator_settings shows both knobs.
- 3 saturation waves: aggregate gen tok/s within 5% of the current config.
- No OOM on the Nov 11 repro shape.
- No GC-pause stalls visible as >2s throughput dips in the 10s engine
  heartbeats during saturation.

## Rollback

Unset PYTORCH_CUDA_ALLOC_CONF (serve script default is unset) + restart.
No code to revert (env-only).

## Optional same-window A/B (OWNER PERMISSION REQUIRED — prod behavior change)

The 11-13 production finding: `VLLM_MAMBA_K0_SLOT_RELEASE=1` is STILL ARMED
on prod (survived the believed rollback) and `--scheduler-reserve-full-isl`
defers big-context requests at the admission door (the dashboard "waits";
sessions emit 0 tok/s while deferred — NOT preemptions, nothing recomputed).
A/B: one boot with VLLM_MAMBA_K0_SLOT_RELEASE unset, measure the
Waiting/Deferred flicker + per-session TTFT under the same load. Decides
whether the brittleness report is the release mechanism or the ISL reserve.
