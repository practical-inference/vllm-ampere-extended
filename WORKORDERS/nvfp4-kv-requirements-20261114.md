# Requirements Specification: NVFP4 (4-bit) KV Cache for vllm-ampere-extended

Status: DRAFT for spec review — 2026-11-14
Owner: columbo (vllm-nightly fork)
Target repo: vllm-project/vllm fork (`vllm-ampere-extended`)

## 1. Purpose and value proposition

Halve KV-cache bytes-per-token via NVFP4 storage (packed e2m1 + fp8/fp32
micro-block scales) for all four serving models. Measured byte geometry:
`fp8_ds_mla` record = 656 B/token, `nvfp4_ds_mla` = 352 B/token — a **1.86×
KV-pool expansion** for MLA-family models with identical VRAM.

Consequences this spec makes achievable:
- **Fewer cards / cheaper cards:** GLM-5.3-Flash today occupies ~80 GiB per
  rank of 96 GiB (weights 25.3 + mamba/KDA statics ~37.7 + pinned KV 17).
  fp4 KV cuts the KV slice ~9 GiB/rank, leaving ~72 GiB — the hypothesis that
  the same service fits 80 GB Ampere (A100-80G) hardware must be verified in
  Phase 0, and is the headline value claim.
- **Viability, not just speed:** DS41F never had enough pool to be worth
  running (1,414,344 tokens best-measured, util ceiling 0.9594; 0.967 OOMs at
  300K). 1.86× takes it to ~2.63-2.83M — enough for ~8 simultaneous 300K
  multi-round deep-dives per TP4, if fp4 retrieval quality holds.
- **Broader applicability of the fork** beyond "make big models fit": 4-bit
  KV is a capacity lever for any Ampere deployment.

## 2. Fleet facts (evidence basis)

| fact | value | source |
|---|---|---|
| cards | 6× PG506-230, 96 GB, sm_80 (8.0), 400 W default | nvidia-smi |
| power cap | 200 W firmware cap on some boards, out-of-band; thermal never binds (89 C throttle) | unit header + owner correction ×N |
| GLM-Flash pool (fp8, util .965, 1M ctx, TP4) | 3,751,723 tok | service log, 2026-11-13/14 boots |
| DS41F champion pool (fp8, util .9594, 524K, TP4 0/1/2/6) | 1,414,344 tok | dsv41_night_report.md |
| DS41F prefill / decode | ~8.5K tok/s / 68 tok/s/stream (GLM: ~110-120) | dsv41_night_report.md |
| fp8 KV quality on DS41F | GSM8K 0.895/0.001-inv (bf16 0.897); needle round1@140K True, **round2 restate False**, round3@300K True — fp8 has no round-2 pass precedent | longctx_perf9.log:1-5 |
| over-admission raise site | `ValueError(f"Cannot get {num_blocks} free blocks from the pool")` — runtime-formatted; source-literal greps never match, log greps need the regex | vllm/v1/core/block_pool.py:694 |
| fp4 primitives in tree | reshape_and_cache fp4 dispatch (both dtypes); triton MLA whitelists nvfp4_ds_mla warmup; selector arm precedent (Gemma4, TRITON_ATTN) | csrc/cache_kernels.cu:954, triton_mla.py:299, test_attention_selector.py:845 |
| CUDA fp4 backends | flashmla_sparse: major==10 only; flashinfer: prims_ts kernel only | flashmla_sparse.py:261, flashinfer_mla_sparse.py:152 |
| recurrent state gates | SSM int8/int4 dead: recurrent error compounding, no kernels | glm_pin_forensics_verdict_20261113.md |

## 3. Goals / Non-goals

**Goals**
- G1: `--kv-cache-dtype nvfp4|nvfp4_4over6|nvfp4_ds_mla` serves all four
  models on sm_80 through arch-portable readers, default-inert.
- G2: fp4 KV-pool ≥ 1.8× fp8 at identical VRAM (MLA family), by group-sizing
  and eviction accounting.
- G3: per-model quality gates (§7) met or the model is *documented rejected* —
  never shipped below gate.
**Non-goals**
- N1: no quantization of recurrent state caches (mamba/KDA, GDN) — forensics
  dead (§9 O1).
- N2: no model retrain/finetune — quantization is training-free dynamic scale
  by design (§5).
- N3: no fp4 tensor-core compute emulation — on sm_80 fp4 is storage only.
- N4: no new KV-offload/connector work — existing interaction is preserved
  and documented only.

## 4. Functional requirements

- **FR-1 (dtype registration):** CacheConfig accepts and validates the three
  dtype strings; storage dtype maps to uint8; unsupported backend×arch
  combinations fail at boot with an explicit diagnostic naming the
  backend and the gate (pattern: flashmla_sparse major!=10 error).
- **FR-2 (store path):** `reshape_and_cache` fp4 dispatch live on sm_80
  (software cvt, ties-to-even matching `cvt.rn.satfinite.e2m1x2.f32`,
  per nvfp4_kv_cache_kernels.cu convention).
- **FR-3 (read path — MLA family):** Triton sparse-MLA readers gain an
  fp4 arm (unpack e2m1 → dequant by micro-block scale → bf16 in-register).
  Applies to DS41F, Big GLM 5.3, and the ds_mla attention slice of GLM-5.3-Flash.
- **FR-4 (read path — GQA):** qwen3_next full-attention pages (GQA geometry,
  not latents) gain the same fp4 dequant arm via TRITON_ATTN selector.
- **FR-5 (indexer/engram interaction):** DS v4.1 indexer (W=128, topk=512,
  ratio-1/2 writes) and `DSV41_ENGRAM_DTYPE=int4` coexist with fp4 KV — the
  indexer reads pre-insert latents; KV bytes change post-verdict. Verify
  no dtype-assumption coupling.
- **FR-6 (admission parity):** reserve-full-isl, watermark, K-scaled mamba
  admission pages, max-num-seqs semantics unchanged; fp4 must not shift
  over-admission death-mode boundaries — re-verify at saturation by log
  grep `Cannot get \d+ free blocks from the pool` (raise site:
  vllm/v1/core/block_pool.py:694; the message is runtime-formatted, so
  source-literal greps can never match — grep the logs, not the code).
- **FR-7 (graphs/async):** CUDA-graph capture (FULL_AND_PIECEWISE + breakable
  + max_cudagraph_capture_size) captures fp4 strides correctly; async
  scheduling path unaffected.
- **FR-8 (selector):** backend selection admits (model, fp4-dtype) pairs only
  where a reader arm exists; rejection message actionable (naming dtype +
  missing arm), mirroring the Gemma4 selector-test precedent.
- **FR-9 (docs):** rendered docs updated for the new dtype values and the
  quality-gate matrix (per docs/contributing user-visible-changes rule).

## 5. Non-functional requirements

- **NFR-1 (zero-impact default):** with no fp4 dtype set, byte-identical
  behavior to today (all changes env/dtype-gated).
- **NFR-2 (training-free):** dynamic per-block/per-token scales; no
  calibration data requirement. Optional affine-calibration hook may exist
  but must not be required for G1.
- **NFR-3 (dequant budget):** in-attention dequant ALU cost must keep the
  fp4 aggregate prefill wall within +15% of fp8 on the storm repro
  (today's reference: FIN boot wall 319.8s).
- **NFR-4 (memory):** dummy-run at Phase 2 must not exceed fp8 peak; fp4's
  VRP benefit lands in the pool, not in new activations.

## 6. Model-specific scope

| model | cache slice fp4-eligible | expected pool | special verification |
|---|---|---|---|
| GLM-5.3-Flash | ds_mla attention KV (state cache stays bf16) | 3.75M → ~6.97M tok TP4 | E7 ladder acceptance re-measure; 80 GB fit (§1) |
| DeepSeek v4.1 Flash | full latent pool | 1.41M → ~2.63M tok TP4 | FR-5 indexer coupling; acceptance N/A (no MTP head in ckpt); this is the *viability* case |
| Big GLM 5.3 | latent pool (TP2xPP3 fallback recipe) | ×1.86 | recipe is least-tested path in fleet — soak longer (§8 Phase 5) |
| Qwen 3.8 Flash Next | full-attention minority of layers (GDN bulk excluded) | ×~2 on slice only | lowest value — may be dropped per Phase-0 footprint audit |

## 7. Quality and performance gates (acceptance)

Per model, measured on the dev slot with greedy/seeded sampling:
- Q1 needle recall at 140K and 300K ctx with fp4 KV: exact-match on
  cold-prefill rounds. Round-2 restate is its own hard gate
  (exact-match, hit=True required): the fp8 measured run shows
  round2 needle_hit=False (longctx_perf9.log:3), i.e. this gate has no
  fp8 pass precedent — fp4 is being asked to clear a bar fp8 itself
  missed, and a miss is a documented rejection, not a ship.
- Q2 GSM8K full set: drop ≤ 0.5 pp vs bf16 reference and invalid < 0.5 %.
- Q3 MTP acceptance length (where a head exists): drop ≤ 0.15 vs fp8 (E7 gate).
- Q4 storm-repro metrics: collapse windows count ≤ fp8 baseline, total
  collapse seconds ≤ +20%, canary p50 gap within +50% of fp8.
- Q5 per-stream decode (TPOT): no regression > 10% vs fp8.
Gate failure on a model = documented rejection in the lever table; other
models may still promote.

## 8. Constraints

- C1 **"no big dev projects" rule:** every phase is an independently
  shippable increment with its own gate and rollback (see dev plan).
- C2 power/firmware: 200 W out-of-band caps are environmental; thermal
  never binds (89 C). Plans must not re-litigate this.
- C3 lane discipline: work lands on fork lanes (forgejo direct), never on
  the posted #66647 head checkout; DCO `-s` commits with Co-authored-by.
- C4 single fork, no upstream public writes in this project without gate
  approval (out of scope anyway: G1-G3 are fork-scope).
- C5 **tree scope:** all source anchors in this spec are verified against
  the production fork tree (dsd-tier head — the posted #66647 head
  checkout), which is where the fp4 machinery lives. `forgejo/main` at
  time of writing does NOT carry it; Phase 0 must land or rebase the
  anchor set onto the working lane, and every later phase re-verifies
  its own anchors in the tree it runs against.

## 9. Open questions (Phase 0 answers them)

- O1 confirmed: recurrent state stays bf16 on hybrids; the 2× decode deficit
  on hybrids and DS41F decode (68 tok/s) is NOT improved by fp4 — expected
  effect on decode is ±small (traffic halved, but hybrid pools are
  state-dominated). Accept?
- O2 exact fp4 record geometry vs the trtllm-gen swizzle (does the Triton
  reader need the SM100 scale-swizzle for parity, or can it use linear
  scales?) — resolve in read-path spike.
- O3 quantization-error sensitivity of *RoPE* dims: tree's fp4_ds_mla keeps
  RoPE fp8 — is fp8 RoPE (not fp4) the design answer for sm80 parity, or is
  bf16 RoPE needed on Ampere (no fp8 HW, software cvt)?

## 10. Evidence manifest

Load-bearing numbers, each pinned as a capture in
`WORKORDERS/evidence-nvfp4-kv-20261114/` (copied 2026-11-14 from the
live instrument output of this session — see the review tooling note in
the dev plan about tree scope):

| number | pinned capture |
|---|---|
| fp8_ds_mla=656 B, nvfp4_ds_mla=352 B | source of truth: `vllm/v1/kv_cache_interface.py:32` (production tree); cross-ref night report §"Remaining 2× gap" |
| GLM-Flash pool 3,751,723 tok; storm-wall 319.8s; collapse 650s/3w; canary 16.5ms/1.26s/1.59s | `night-FIN-repro.jsonl`, `night-FIN-runner.log` (ANALYZER block) |
| E7 ladder gate: acceptance 2.93 (run-mean) vs 2.87, wall 320.1s | `night-E7-runner.log` |
| GA1 admission-cap rejection (identical profile, canary max worse) | `night-GA1-runner.log` |
| B0 defer staircase 72-91s/index (storm shape reference) | `night-B0-runner.log` |
| DS41F: 1,414,344 tok pool; 68 tok/s decode; TTFT 16.4/41.7/0.4s; needle round1 True / **round2 False** / round3 True | `longctx_perf9.log`, `dsv41_night_report.md` |
| util ceiling .9594 / 0.967 OOM at 300K; pin_landing +6.1% pool | `glm_window_verdicts_20261111.json` |
