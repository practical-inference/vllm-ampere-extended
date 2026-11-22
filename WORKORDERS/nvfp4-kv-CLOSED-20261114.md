# Project Closure: NVFP4 (4-bit) KV Cache for vllm-ampere-extended

Status: **CLOSED — documented rejection** (owner decision, 2026-11-14). Reopenable under the conditions in §4.
Supersedes: the GO framing in `nvfp4-kv-requirements-20261114.md` / `nvfp4-kv-dev-plan-20261114.md` (their §1/§6 capacity numbers are corrected by the Phase 0 audit).

## 1. Verdict

The capacity-lever value proposition is refuted by the fleet's own boot
captures: the 656→352 B record shrink is a per-record ratio, not a pool
ratio. The MLA record is ~9-10% of per-token pool cost on GLM-Flash and ~24%
on DS41F (hybrid state pages and the fp8 SWA cache own the rest), so fp4 buys
**≤ +10% / ≤ +13.6% pool** — not ×1.86. The 80 GB-fit headline is already
true at fp8 under the promoted config (75.31 GiB/rank occupancy). Full
arithmetic, gap matrix, and open-question closures:
`nvfp4-kv-phase0-audit-20261114.md` (PR #25).

## 2. What exists (so a reopening starts from truth)

- **Upstream (zero fork work):** the entire SM100/SM120 fp4-KV machinery
  (dtype registration, CUDA writer/reader, FlashMLA/FlashInfer/trtllm-gen
  backends) and the generic GQA fp4 path on sm_80 via FlashInfer
  (`_nvfp4_kv_on_fa2` admits capability family 80 — unverified on our cards).
- **Fork-only gap (the 4-6 dev-days a reopening would spend):** G1 —
  nvfp4_ds_mla admission + 352 B shape + Triton gather arm in
  TRITON_MLA_SPARSE (fp8 template: `dequant_ds_mla_slots`,
  `triton_mla_sparse_kernel.py:720`; dispatch `triton_mla_sparse.py:346-415`);
  G2 — sm_80 Triton writer (software e2m1 ties-to-even + e4m3 scales + SF
  permutation `nvfp4_sf_byte = 8*(s&3)+(s>>2)`; precedents:
  `round_to_nearest_e2m1` nvfp4_kv_cache_kernels.cu:59,
  `_f32_to_e4m3_sw` triton_mla_sparse_kernel.py).
- **Record format (stable, in-tree):** 352 B = 256 B e2m1 NoPE + 64 B
  unscaled-e4m3 RoPE + 32 B permuted e4m3 SFs; `sf = e4m3(max(amax/6, 2^-9))`
  (`nvfp4_ds_mla_cache_kernels.cu:36-69`).

## 3. Recorded tests

### 3a. In-tree tests that must stay green (SM100 machinery — upstream-owned)

- `tests/v1/attention/test_mla_backends.py` — nvfp4_ds_mla cache building
  (:657-704: e2m1-packed NoPE + unscaled e4m3 RoPE + per-16 e4m3 SFs, padded
  to 16B) and backend dtype matrix (:847).
- `tests/v1/attention/test_flashinfer_mla_sparse_prims_ts.py` — prims_ts
  reader for nvfp4_ds_mla; `ROW_BYTES = {"fp8": 576, "nvfp4_ds_mla": 352}`
  (:46); parametrized fp8/nvfp4_ds_mla (:207-303).
- `tests/config/test_kv_cache_dtype.py` — handler registry / quant-mode
  mapping.
- `tests/test_config.py:480` — nvfp4_ds_mla rejects decode-context
  parallelism.
- `tests/kernels/attention/test_attention_selector.py:845` — selector
  admission precedent (Gemma4, TRITON_ATTN).

### 3b. Fleet gates a reopened project must pass (requirements §7, unchanged)

Measured on the dev slot, greedy/seeded:
- **Q1** needle recall 140K + 300K, exact-match cold-prefill; **round-2
  restate is its own hard gate (hit=True required)** — no fp8 pass precedent
  (`longctx_perf9.logcut.txt:3`: round2 needle_hit=False; the night report's
  contrary line is an adjudicated overstatement).
- **Q2** GSM8K full set: drop ≤ 0.5 pp vs bf16, invalid < 0.5%.
- **Q3** MTP acceptance-length drop ≤ 0.15 vs fp8 (models with a head; N/A
  for DS41F checkpoint).
- **Q4** storm repro: collapse windows ≤ fp8 baseline, collapse seconds
  ≤ +20%, canary p50 within +50% (reference: FIN boot wall 319.8s).
- **Q5** per-stream decode (TPOT): no regression > 10% vs fp8.
- Instruments: `WORKORDERS/tools/` on lane/oom-snapshot-tooling (merged PR
  #22): `glm_storm_repro.py`, analyzer (`--since` is DATE-aware),
  `night_run.sh`, `night_soak.sh`, `night_e1b_ramp.py`. Long-ctx benches were
  `/tmp/opencode/bench_{decode,longctx}.py` (volatile — re-pin if reused).
- Death-signature probe: log grep `Cannot get \d+ free blocks from the pool`
  = 0 (raise site `vllm/v1/core/block_pool.py:694`, runtime-formatted — grep
  logs, never source literals).

### 3c. Phase 0 verification method (how the numbers here were produced)

- Anchor checks: `git grep <pattern> ampere/main -- vllm/ csrc/` against the
  production fork (ampere/main @ 5ede3e2eec; the serving wheel is
  `2026.10.2.dev48+ae.1.g04da26725`).
- Pool composition: boot-log greps of `Available KV cache memory`
  (gpu_worker.py), `Extended KV cache to N blocks` (extensible_kv_cache.py),
  `GPU KV cache size` (kv_cache_utils.py), `kv cache group sizes`.
- Captures: `WORKORDERS/evidence-nvfp4-kv-20261114/` (16 files, incl. the
  pool-composition pins added in PR #25).

## 4. Re-entry conditions (come back if any of these)

1. **DS41F long-ctx deployment becomes strategic** enough that ≤ +200K pool
   tokens (~4-6 dev-days, G1+G2, DS41F slice only) justifies the spend.
2. **Upstream lands an sm_80 sparse-MLA path** — the arm then becomes
   upstreamable and the ROI calculus changes.
3. **State-cache quantization becomes viable** (forensics currently dead:
   recurrent error compounding, no kernels) — that is the only true 2×
   lever on the hybrids, since the states own ~76-90% of the pool.
4. **Blackwell hardware enters the fleet** — the upstream SM100 nvfp4 path
   works there today; per-model quality gates (§3b) still apply.

## 5. Artifact index

- PR #23 (requirements + dev plan, SURVIVES-stamped), #24 (review hygiene:
  Evidence citations, anchor path fixes, attempt-9 truncation repair), #25
  (Phase 0 audit) — all merged to forgejo/main.
- Evidence: `WORKORDERS/evidence-nvfp4-kv-20261114/` — 16 captures.
- MemPalace: `vllm-nightly/session-checkpoint` 57d38de4, `decisions`
  d92aa91f, diary 390ed7ed (Phase 0 session); this closure filed on close.
- Adversarial-review artifacts: `.pr-drafts/evidence/nvfp4-kv-{requirements,dev-plan}-20261114/`
  (SURVIVES 2026-11-14).
