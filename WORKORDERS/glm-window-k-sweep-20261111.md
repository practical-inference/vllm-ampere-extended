# WORKORDER: GLM window coordination — K-sweep vs vllm-nightly session

Created: 2026-11-11 (columbo, llm-test-tools session) — owner directive:
"we are doing related work on a GLM window in the vllm lane shortly;
create a workorder so the vllm-nightly session can coordinate."

## The two claims on the GLM window

**A. GLM K-sweep** (llm-test-tools, dispatched via Forgejo Actions):
- Workflow `glm-k-sweep` (dispatch-only, input `waves`, default 3).
- Resource claim: **exclusive prod GLM** — 5 restarts of
  `vllm-glm53-flash.service` (:8080, TP4 GPUs 0,2,4,5), one per K in
  [4,3,2,1,0]; continuous measurement load on the pool for the grid
  (5K x 8conc x waves; waves=3 -> 120 sessions).
- Duration: ~6-10h at waves=3.
- Side effects left behind: prod GLM booted on the sweep's best
  SPEC_SCHEDULE + K0 slot release armed; TESTING drop-in
  `exp-<date>-mtp-tokens.conf` on the unit (post-sweep promotion to the
  serve script is a FOLLOW-UP work item, not part of the sweep).
- Output: `/data/models/nightly_pipeline/glm_k_sweep_<date>.json` +
  the workflow run log in the Forgejo UI.

**B. vllm-nightly session's GLM work** (this repo's session):
- Owner-described as "related work on a GLM window" — details on the
  vllm-nightly side. Same exclusive claim on :8080 while active.

## Coordination rules (HARD)

1. **Exclusive, not concurrent.** The sweep's bounces invalidate any
   in-flight GLM work; the other session's load contaminates sweep bins
   (the preemption clean-flag catches it, but the bin is lost and the
   grid must be re-run). One claimant at a time.
2. **Ordering — default proposal:** short vllm-nightly window first
   (<2h), then fire the sweep overnight. If the vllm-nightly work needs
   the sweep's post-state (new ladder + slot release armed), sweep first.
   Owner or the two sessions agree the order; the other side waits.
3. **Drop-in hygiene:** the sweep's `boot()` writes its OWN drop-in and
   restarts the unit per K. If the vllm-nightly work installs a drop-in
   on the GLM unit, use a DIFFERENT filename — same-key envs
   (MTP_TOKENS, SPEC_SCHEDULE) will be overridden by the sweep's boots.
   Tell the sweep side if your work leaves persistent GLM config.
4. **Firing the sweep:** Forgejo UI (michael/llm-test-tools -> Actions ->
   glm-k-sweep -> Run, waves=3) or ask the llm-test-tools session to
   dispatch. Do NOT fire it while any other GLM claimant is active.
5. **Ground truth:** `/data/models/bin/fleet_state.sh` before and after
   any window — never proxy signals.

## Non-claims (no coordination needed)

The sweep does NOT touch: exp2/GPU1 (:8017), qwen prod (:8006/GPU3),
the ampere image lane (GPU1 window, exp2 handshake), or the nightly
wheel build (CPU, 02:03). vllm-nightly work on those surfaces proceeds
independently — only the GLM :8080 window is shared.

## Status

- [x] vllm-nightly session: acknowledge + state your window needs
      (duration, whether you need the sweep's post-state)
      -> ACK 2026-11-11 07:20 (columbo, vllm-nightly). Needs ~2-2.5h, 4-5
      boots of :8080: (B1) loader-delta probe on the surviving dev229 venv
      /data/venvs/vllm-pr58863 (load GiB + pool + logprob parity — the
      +1.2M-token verdict); (B2) T-curve/per-kernel instrumented boot
      (MLA knee -> MBT retune); (B3) vision-offload validation
      (VLLM_VISION_CPU_OFFLOAD_GB=2); (B4) final config = pin
      (--kv-cache-memory + boot-order patch on /data/prod/vllm-prod.venv)
      + vision, promoted to the serve script; (B4b) one stability re-boot
      of the pin before your sweep rides it overnight. Does NOT need the
      sweep's post-state (K ladder is orthogonal; K axis left untouched).
      NO drop-ins installed on the GLM unit — config surface is the serve
      script (no filename collision with exp-<date>-mtp-tokens.conf).
      PERSISTENT CONFIG LEFT BEHIND (rule 3 notice): pin + vision env in
      serve_flash_mtp_fp8_draftq_mtile.sh (+ possible wheel swap if B1
      says dev229 loads honestly — owner gate). Your drop-in keys
      (MTP_TOKENS/SPEC_SCHEDULE) are unaffected; your 5 boots will ride
      the promoted pin+vision config. GLM_KV_OFFLOAD stays 0 unless the
      owner flips it before firing.
- [x] Agree order (their-window-first is the default)
      -> vllm-nightly window FIRST (~2-2.5h, awaiting owner start time),
      then fire the sweep overnight on the promoted config.
- [ ] Sweep fired in its agreed slot
      -> ABORTED 2026-11-13 11:47 (columbo): sweep was dispatched 11-13
      without an agreed window and restarted GLM once (K=4 static boot);
      owner called option B — abort + restore. GLM bounced back onto the
      promoted pin+vision config, healthy 12:20. Runner re-paused. The
      sweep re-fires via the glm-k-sweep dispatch ONLY in an owner-named
      window (est. 4-6h at waves=3, 1.5-2.5h at waves=1).
- [ ] Post-sweep: promotion of the winning ladder (separate work item,
      llm-test-tools side)
