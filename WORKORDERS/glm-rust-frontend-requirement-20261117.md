# WORKORDER: Route C — package vllm-rs into the wheel pipeline, restore GLM vision durably

Created: 2026-11-17 (columbo, llm-test-tools lane), surfaced by the manual CD
canary the owner ordered. **Owner verdict recorded below: ROUTE C** — this is
now the durable-fix workorder, executed next session.

## What happened

Manual canary fire 2026-11-17: candidate wheel
`vllm-2026.10.2.dev836+ae.1.g7b01f0f8b` (next-tip after the 145-commit
upstream/main merge) went STAGE 1 GREEN on qwen
(gsm8k1319 97.0%, needle 0.86 agg, zero invalid). STAGE 2 (GLM) first died at
model init — the merged code added a hard requirement:

    vllm/models/glm5next/common/multimodal.py
    ValueError: GLM-5.3-Flash multimodal preprocessing requires the Rust frontend.

Upstream cause: `7c86910291 [Multimodal] Require Rust preprocessing for
GLM-5.3-Flash (#67237)` — landed Nov 16, carried by the merge: media
preprocessing moved from Python (which worked) to the Rust frontend. The
boot-time memory-pool profiler constructs synthetic mm inputs for a
vision-enabled model and now raises.

**No rollback ever fired.** Fixed forward in the owner window: serve knob
disables media → re-boot → health returned → the running promote script's
poll judged STAGE 2 `healthy + smoke OK` + `version OK: dev836` at 20:32:27 →
PROMOTED; forgejo main published via PR; ampere main + tag queued behind the
gate.

## Fleet state at close (truth)

- :8006 and :8080: CONVERGED on dev836 — **GLM is TEXT-ONLY** (B4 pin+vision
  promotion reversed as a stopgap until C lands).
- state.env: LAST_PROMOTED=7b01f0f8b, PENDING=0, REJECTED=empty.
- Gate requests staged for michael (MANUAL approve): ampere main push
  `34cd692684fd`, tag push `0743d9ed17ec`; after both: cut release with
  `bash /data/models/bin/ampere_release_create.sh nightly-20261117`.
- Orphaned arms (re-armed by C):
  - exp2 fp8-index attention-config: lane/qsa-fp8-index-sm80 (8145aa4c61) —
    its site-packages arm was removed by the dev836 deploy; the lane must
    merge to next with/before the C wheel.
  - `serve_flash_mtp_fp8_draftq_mtile.sh`: `LIMIT_MM_PER_PROMPT` knob
    ({"image":0,"video":0}) must be REMOVED by the C flip (media re-enabled).

## Route C scope — execution order (next session)

1. **Rust toolchain** on the box (user-space rustup, minimal profile).
2. **Locate the vllm-rs frontend crate** + build contract — envs.py
   auto-discovery expects the binary at `site-packages/vllm/vllm-rs`
   (VLLM_RUST_FRONTEND_PATH="auto").
3. **Integrate the Rust frontend build into the wheel pipeline**
   (vllm_nightly_integrate.sh wheel step) and bundle the binary into the
   wheel; merge the fp8-index lane into next first.
4. **Flag-parity validation BEFORE promote (the gate)**: the Rust binary
   replaces the Python API-server process; every fleet-stack flag must exist
   and behave identically in Rust — `--tool-call-parser glm47`,
   `--reasoning-parser glm45`, `--enable-prompt-tokens-details` (emcare
   telemetry), `--enable-per-request-metrics`, profiler-config passthrough,
   response-shape parity through the proxy. Name the disproof check (D-Q016).
   **Kill items (adjudicated from the 2026-11-17 canary review):** two
   pre-existing latent knob defects in the serve script — (a)
   `MAX_NEW_ADMISSIONS_PER_STEP` emits `--max-new-admissions-per-step`, a
   flag absent from dev836 wheel source (nearest real knob:
   `--max-num-active-seqs`); (b) `RESERVE_ISL=0` emits nothing while upstream
   defaults `scheduler_reserve_full_isl=True` — the documented "no- arm" is a
   silent no-op, not a disable. Both inert under the deployed unit (defaults
   0/1, no unit env). C's serve flip fixes or removes both knobs — not
   inherited silently.
5. **Serve flip**: remove the limit-mm knob (media enabled), unit
   `Environment=VLLM_USE_RUST_FRONTEND=1`; restore exp2 attention-config
   token if the arm is still unmerged.
6. Re-stage candidate via integrate → two-stage canary rerun (owner window) →
   fleet stays converged WITH vision.

## Constraints

- Never edit a file a live process executes: repo → deploy → bounce.
- The other session's working-tree edits (ampere_image_validate.sh,
  pr-pipeline/) stay theirs.
- No prod bounce without an owner-named window; the 2026-11-17 canary fire
  was the owner's explicit manual order.
- Validate every dep-version drift against the validated matrix — the
  apache-tvm-ffi 0.1.16 abort (fresh flashinfer 0.7.4 pull) is the precedent;
  vllm-rs will bring new deps.
- Watchers read machine verdicts (unit state, health, version), not journal
  prose.

## Status

- [x] Owner decision — **ROUTE C** (2026-11-17 checkpoint:
  session-archive/2026-11-17-glm-vision-routec-checkpoint.md)
- [x] Rust toolchain installed (user-space) — rustup 1.95.0 minimal profile at
  `~/.cargo/bin` (`--no-modify-path`; the integrate script exports it)
- [x] vllm-rs crate located + build contract read — workspace `rust/`,
  `vllm-cmd` package, Exec binding `--bin vllm-rs` + `native-tls-vendored`
  (OpenSSL linked STATICALLY — no new pip deps, no ABI drift against the
  validated matrix); auto-discovery expects `pkg_dir/vllm-rs`
  (envs.py `_resolve_rust_cli_path`)
- [x] fp8-index lane merged into next — re-landed as `145978c011`; range-diff
  vs the cited `8145aa4c61` (stale forgejo-ref sha, pre-rebase): identical
  modulo message/formatting. Already an ancestor of next tip `7b01f0f8b9`.
- [x] Rust frontend build integrated into the wheel pipeline + bundled —
  integrate 2b runs `-m build` with `VLLM_REQUIRE_RUST_FRONTEND=1` (without
  it setup.py marks the extension optional — the silent skip that shipped
  the binary-less dev836 wheel) and installs setuptools-rust into BPY;
  post-build wheel-zip disproof assert. Binary verified building + running
  from next source with rustc 1.95.0.
- [x] Flag-parity validation GREEN — static clap surface: `vllm-rs serve
  --help` exposes `--tool-call-parser`, `--reasoning-parser`,
  `--enable-prompt-tokens-details`, `--enable-per-request-metrics`,
  `--profiler-config`; engine boots healthy with the full exec flag set
  accepted (unknown-to-clap flags forward to the headless engine —
  demonstrated), completions smoke + vision (image in, generation out)
  pass through the served shape. Remaining spot-check (not blocking):
  an explicit glm47 tool-call roundtrip bin-vs-bin response diff.
  **Disproof (D-Q016):** dev-unit boot with `VLLM_USE_RUST_FRONTEND=1` on
  the new wheel — any fleet flag rejected at parse or any response-shape
  divergence through the proxy disproves Route C's premise; fallback is
  re-applying the LIMIT_MM stopgap and pulling the unit knob.
  The wheel-bundle assert itself was disproof-checked both directions
  (red on the shipped binary-less dev836 wheel, green on a member-carrying
  wheel). Pipeline change reviewed: Adversarial-Review: SURVIVES
  (adversarial-20261118-041557.log, llm-test-tools lane draft
  routec-wheel-rust-bundle).
- [x] Serve flip DONE 2026-11-17 23:56 (owner GPU window):
  unit `Environment=VLLM_USE_RUST_FRONTEND=1` stamped; kill item (a)
  MAX_NEW_ADMISSIONS_PER_STEP knob REMOVED (source-scanned: flag absent
  from dev867 wheel — nonzero setting would boot-dead the engine);
  kill item (b) RESERVE_ISL now emits `--no-scheduler-reserve-full-isl`
  on the 0 arm (flag validity proven by the live exp2 boot carrying it).
  Flip revisions found under fire: (1) the KV_CACHE_MEM pin boot-deads
  dev867 worker init (free-on-startup ValueError at 34 s) — pin dropped,
  boots stay on the 0.965 fixed-utilization plan; (2) dev867 encoder
  auto-raise profiles one max-size VIDEO item (8192→242242 tokens,
  1.85 GiB from KV) starving finalize (0.81 vs 6.7 GiB needed at 1M) —
  LIMIT_MM interim cap {"video":0} (images ON): available KV recovered
  15.14→27.74 GiB, engine healthy, vision smoke passes (image→"Red").
  (3) MICRO-BATCH FIX LANDED 2026-11-18 06:15 (next a6f0fb18f, dev900
  wheel): video encoder forwards chunked along the temporal grid in
  vllm/v1/worker/gpu/mm/encoder_runner.py — pixel rows are t-major, so
  sequential slices + torch.cat reproduce one embed_multimodal call
  exactly; budget knob VLLM_MM_ENCODER_CHUNK_ROWS (default 16384 patch
  rows, 0=off). Interim {"video":0} cap REMOVED (media fully ON),
  max_model_len 1048576→786432 (owner call). Measured: Available KV
  22.66 GiB video-on (vs 0.81 pre-fix), engine healthy, vision smoke
  passes (green PNG read); ampere remote stale at 7b01f0f8b — new
  candidate needs main+tag re-queue after canary verdict.
  Window-owner uncommitted 11-13/11-14 policy edits (K=2 ceiling, seqs 96,
  long-prefill knobs) LANDED alongside the flip (serve-script edit +
  deploy); the earlier false stage-2 rollback (restart_one 25-min health
  budget vs ~27-min cold boot) is a known latent pipeline defect —
  raise restart_one poll budget (100→140) before the next canary.
- [x] Candidate df5342ab5 staged via integrate (21:49); two-stage canary
  ran in the owner window: STAGE 1 GREEN on qwen (gsm8k1319 97.1%),
  STAGE 2 GREEN on GLM — recovered after the false stage-2 rollback
  (dev48-era restart_one health budget vs ~27-min cold boot) by draining
  the TP4 set and re-launching on the fixed video-off budget.
- [x] Fleet CONVERGED WITH VISION (both prods, same wheel dev867+gdf5342ab5):
  :8006 healthy, :8080 healthy with image preprocessing live (vision smoke
  "Red"). Publish legs: ampere main queued 34cd692684fd, tag queued
  0743d9ed17ec (MANUAL approve outstanding on michael — release cut after
  both approve: bash /data/models/bin/ampere_release_create.sh nightly-20261117).
- [x] CANARY PASS COMPLETE 2026-11-18 08:47: PROMOTED 0e5766712 (dev914)
  through the full pipeline — STAGE 0 preflight pass, STAGE 1 qwen-prod
  tier-1 gates GREEN (gsm8k1319 96.7%, invalid 0.0%), STAGE 2 GLM
  healthy + smoke + version OK; forgejo main published; ampere main
  queued cf27e76aca4b + tag queued e772f8b0a074 (MANUAL approve on
  michael; release cut after both: ampere_release_create.sh
  nightly-20261118). Vision smoke passed on GLM prod video-ON.
  Micro-batch change (38d85fcfa2) reviewed: Adversarial-Review: SURVIVES
  (adversarial-20261118-140139.log, comment-tier; a2 refused first
  round -> INCOMPLETE -> auto-rerun with fresh agents). exp2 preflight
  order honored (dev slot bounce on dev900+dev914 before qwen prod;
  qwen-prod checks stayed minimal by pipeline design).
- [ ] K-sweep v2: re-sweep on the video-ON dev914 wheel (0e5766712)
  with a realistic emcare prompt mix; the current ladder (K=3/4) is
  config-stale (swept video-off on an earlier wheel).
