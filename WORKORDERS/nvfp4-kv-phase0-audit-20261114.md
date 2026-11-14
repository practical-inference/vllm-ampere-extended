# Phase 0 Audit: NVFP4 (4-bit) KV Cache — gap matrix, open-question closures, and a material correction

Status: COMPLETE — 2026-11-14 (columbo)
Companions: nvfp4-kv-requirements-20261114.md, nvfp4-kv-dev-plan-20261114.md
Audit tree: `ampere/main` @ 5ede3e2eec (vllm-ampere-extended — the production
fork the serving wheel is built from: prod venv `2026.10.2.dev48+ae.1.g04da26725`).

## 0. Audit-tree correction (C5 refinement)

The requirements' C5 named the "dsd-tier head" as the anchor tree. Phase 0
found the fp4-serving machinery is split across two trees:

- The **dtype/spec layer** (kv_cache_interface, CacheConfig, cache_kernels
  dispatch, mla_attention plumbing) exists in BOTH `dsd-tier-update`
  (vllm-nightly) and `ampere/main`.
- The **production serving backend** `TRITON_MLA_SPARSE`
  (`vllm/v1/attention/backends/mla/triton_mla_sparse.py`) exists ONLY in
  `ampere/main` — absent from the dsd-tier tree. Ground truth: the prod
  :8080 boot log names backends `DEEPSEEK_V32_INDEXER, KPOOL_TAIL,
  TRITON_MLA_SPARSE` (service log, 2026-10-01 boot).

All Phase 1+ code lands against `ampere/main`. Anchors below are verified in
`ampere/main` @ 5ede3e2eec.

## 1. Gap matrix (machinery present vs missing)

Legend: ✓ = in tree, G# = gap (the actual Phase 1-2 work).

| FR | component | status | evidence (ampere/main) |
|---|---|---|---|
| FR-1 | dtype registration (nvfp4, nvfp4_4over6, nvfp4_ds_mla in CacheDType Literal; KVQuantMode.NVFP4=5 / NVFP4_DS_MLA=10; handler registry) | ✓ | vllm/config/cache.py:44-67, vllm/v1/kv_cache_interface.py:34,110-137 |
| FR-1 | page-size accounting keyed on the 352 B record | ✓ | vllm/v1/kv_cache_interface.py:34 (DS_MLA_STATE_CONTENT_BYTES {656,352}); vllm/model_executor/layers/attention/mla_attention.py:1525; platforms/interface equivalent consumed via spec state_content_bytes |
| FR-1 | model-layer nvfp4_ds_mla arms | ✓ | mla_attention.py:370,514,834,1044,3141 |
| FR-2 | record format + reference quant math (352 B: 256 B e2m1 NoPE + 64 B unscaled-e4m3 RoPE + 32 B permuted e4m3 SF; sf = e4m3(max(amax/6, 2^-9))) | ✓ | csrc/nvfp4_ds_mla_cache_kernels.cu:36-53 |
| FR-2 | **sm80 store path** | **G2** | the CUDA writer TU is compiled only for SM10x/11x + SM12x gencodes (CMakeLists.txt:1021,1049-1059, ENABLE_NVFP4_SM100/SM120); on sm_80 neither TU builds and host entries reject the format (nvfp4_ds_mla_cache_kernels.cu:24-32 "non-native paths trap"). Software-cvt building blocks EXIST: `round_to_nearest_e2m1` (csrc/nvfp4_kv_cache_kernels.cu:59, ties-to-even matching cvt.rn.satfinite.e2m1x2.f32) and `_f32_to_e4m3_sw` (triton_mla_sparse_kernel.py, SM80 Triton e4m3 RNE-satfinite). Need: a Triton writer for the 352 B record on sm_80. |
| FR-3 | **TRITON_MLA_SPARSE nvfp4_ds_mla reader arm (the core gap)** | **G1** | supported_kv_cache_dtypes = [auto, fp16, bf16, fp8_ds_mla, fp8-alias] — no nvfp4_ds_mla (triton_mla_sparse.py:465-471); get_kv_cache_shape keys on fp8_ds_mla only (:511-522); forward dispatch :448 fp8-only; NoPE-512 supports_combination excludes it (:541-555). The fp8 template to extend: `dequant_ds_mla_slots` (vllm/v1/attention/ops/triton_mla_sparse_kernel.py:720 — parameterized dequant-gather into a persistent bf16 workspace, triton_mla_sparse.py:346-415). NOTE: the fused fp8 prefill kernel exists but is DISABLED in production (`_PREFILL_FUSED_MIN_TOKENS = 1 << 30`, triton_mla_sparse_kernel.py:64) — the fp4 arm needs only the gather path. |
| FR-3 | other MLA backends' nvfp4 admission | ✓ (non-sm80) | FLASHMLA_SPARSE: SM100-only gate (flashmla_sparse.py:261-265); FLASHINFER_MLA_SPARSE (TRTLLM): nvfp4_ds_mla "prims_ts kernel only", SM100 (flashinfer_mla_sparse.py:99-106,155-160); FLASHMLA_MEGA_ATTN_DSV41: nvfp4_ds_mla, SM100-only "no SM90 instantiation" (deepseek_v41/sparse_mla.py:289-311) |
| FR-4 | generic GQA nvfp4 (qwen case) | ✓ exists / unverified on sm80 | flashinfer writer `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping` (flashinfer.py:18, from the flashinfer package) + `_nvfp4_kv_on_fa2()` admits capability family **80** (flashinfer.py:116-117) + FA2 native decode below SM90 (:120-122). BUT the fleet qwen runs the QSA custom backend (supported: auto/bf16/fp8/fp8_e4m3 only) — see §4 drop decision. |
| FR-5 | indexer/engram coupling | ✓ (design-level) | DS41F indexer reads pre-insert latents; DSV41_ENGRAM_DTYPE=int4 orthogonal (night report boot flags). Phase 2 verifies no dtype-assumption coupling at runtime. |
| FR-6 | admission parity | ✓ (no fp4-specific accounting found) | over-admission death-signature unchanged: block_pool raise site is dtype-agnostic (grep of 656-hardcodes shows only record-format asserts: csrc/cache_kernels.cu:1073,1082,1856,1947-1948) |
| FR-7 | graphs/async | ✓ low-risk | fp8_ds_mla already captures opaque-uint8 pages through FULL_AND_PIECEWISE+breakable in production (both prod units); nvfp4_ds_mla is the same tensor family (uint8 [blocks, block_size, 352]). Phase 2 re-verifies. |
| FR-8 | selector rejection messages | ✓ precedent | flashmla_sparse.py:261-265 names dtype + gate; mla_attention.py:365-375 normalizes FLASHMLA_SPARSE+quantized → fp8_ds_mla unless nvfp4_ds_mla |
| — | eviction bytes-per-token | ✓ | spec-driven (page_size_bytes from state_content_bytes); no 656-hardcode outside record asserts |

## 2. Open questions closed (exit gate)

- **O1 (recurrent state stays bf16; decode ±small) — CONFIRMED, and sharpened:**
  the state caches are not merely un-quantized, they DOMINATE the pool on the
  hybrids (§3). fp4's decode effect on GLM-Flash/DS41F is bounded by the MLA
  slice's share of attention traffic — ±small stands.
- **O2 (swizzle coupling) — CLOSED:** there is no SM100 tensor-swizzle
  requirement for a Triton reader. Two scale placements exist, both plain
  index arithmetic: (a) ds_mla record: the SF byte permutation
  `nvfp4_sf_byte(s) = 8*(s&3)+(s>>2)` (nvfp4_ds_mla_cache_kernels.cu:49-69,
  static_assert vs FlashMLA sm100 config) — a Triton reader/writer reproduces
  it directly; (b) generic GQA layout: `swizzle_scale_offset`
  (nvfp4_kv_cache_kernels.cu:34-44, trtllm-gen [T//4,4,4,S//4] permute) —
  only needed if reading the flashinfer-written GQA layout; a reader for that
  layout must reproduce it, but it is byte arithmetic, not a hardware
  dependency.
- **O3 (RoPE dtype) — CLOSED:** the fp4 record keeps RoPE **unscaled e4m3**
  ("e4m3's 4 exponent bits span the RoPE range unaided",
  nvfp4_ds_mla_cache_kernels.cu:38-39, writer :164-177, reader :232-240).
  fp8 RoPE IS the design answer; on sm_80 the e4m3 store is software
  (`_f32_to_e4m3_sw` precedent). No bf16 RoPE needed.

## 3. MATERIAL CORRECTION — the ×1.86 pool-expansion claim is refuted by our own boot captures

The spec's §1/§6 treat the 656→352 B record shrink as a ×1.86 pool expansion.
It is a **per-record** ratio only. Pool expansion is
`per_token_total / (per_token_total − 304 B)`, where per_token_total includes
ALL cache groups (hybrid state pages, SWA caches) and TP replication. The
fleet's own boot profiles:

- **GLM-5.3-Flash (prod :8080, promoted config, 2026-11-14 boots):**
  KV budget 27.49 GiB/rank (gpu_worker.py:837) → 3407 blocks = 24.61 GiB/rank
  (extensible_kv_cache.py:625) → pool 3,866,340 tok
  (kv_cache_utils.py:1168) ⇒ **~6.8-7.6 KiB per token per rank**. The 656 B
  MLA record is **~9-10%** of it — the hybrid state pages dominate.
  fp4 pool gain ∈ **[+0%, +10%]** (0 if the state group binds without
  reallocation; ≤10% with full proportional reallocation) — **not ×1.86**,
  and the "3.75M → ~6.97M" §6 figure is off by ~8×. Captures:
  `glm-pool-size.logcut.txt` (pool), this doc pins the 27.49/24.61 GiB lines
  (service log 05:16-05:31 boots).
- **DS41F (champion :8210 boot, 2026-10-04):** KV budget 3.57 GiB/rank
  (gpu_worker.py:719, dsv41_final.log) → 1,414,344 tok ⇒ **2.71 KiB/token/rank**;
  the 656 B latent record is **~24%**. fp4 gain ≤ **+13.6%** (→ ≤1.61M tok) —
  not ×1.86 / 2.63M. The SM100 nvfp4 mode itself keeps the SWA cache fp8
  ("V4.1 fp8 SWA cache + NVFP4 compressed cache", sparse_mla.py:292) — the
  record split is inherent to the dtype, not an sm80 limitation.
- **Qwen 3.8-Flash-Next:** state-dominated to a greater extreme (k0-thread
  capacity drawer: 10.24 GiB / 446 blocks ⇒ ~23.5 MiB/block of state pages).
  Full-attn slice is a minority of a minority. **DROP — confirmed.**
- **Big GLM 5.3 (TP2xPP3):** no boot-profile capture on file; given the
  family's hybrid state dominance, expect single-digit gains. One boot-profile
  check (existing logs or a dev-slot boot) is the entry condition if pursued.

**80 GB-fit (§1 headline): the arithmetic says fp8 ALREADY fits.** Today's
promoted boot: per-rank occupancy = 96 − 20.69 free = 75.31 GiB at fp8
(weights+statics+graphs 48.6 + KV 24.61 + transient 2.09). An 80 GiB card
runs the same config with ~4.7 GiB slack (pool scales down proportionally).
fp4 is not the 80 GB enabler under the current config; the spec's premise
predates the promotion. Live verification on an 80 GiB-capped run remains an
owner-ordered item (needs an exp2 window; GPU1 currently hosts exp2 at 94 GiB).

## 4. Go/no-go per model (Phase 0 deliverable)

| model | verdict | basis |
|---|---|---|
| DeepSeek v4.1 Flash | **MARGINAL GO** — the only model with a real (≤+13.6%) pool gain; viability case intact but resized: 1.41M → ≤1.61M tok (+~0.2M ≈ +0.6 × 300K sessions), not 2.63M | §3 DS41F arithmetic; G1+G2 are the only code gaps |
| GLM-5.3-Flash | **NO-GO for capacity** (document): ≤+10% pool, 80 GB fit already true at fp8. Revisit only if a future config makes the MLA slice binding | §3 GLM arithmetic |
| Big GLM 5.3 | **PENDING** one boot-profile check; expected no-go | §3 |
| Qwen 3.8-Flash-Next | **DROP** (footprint audit: state-dominated, QSA backend has no fp4 arm, lowest value) | §3; dev plan Phase 5 entry condition not met |

## 5. Revised project shape (owner decision required)

The capacity-lever framing (×1.86, cheaper cards, 8×300K deep-dives) is
refuted. What survives:

1. **DS41F +0.2M tokens** for ~4-6 dev-days (G1 gather arm + G2 Triton
   writer + gates Q1-Q5) — an owner call on whether that margin justifies it.
2. **Halved MLA-slice KV traffic** (decode TPOT) — O1 bounds it at ±small on
   the hybrids; not a lever by itself.
3. The **machinery** (G1/G2) is smaller than the spec assumed: one Triton
   gather kernel + admission/shape plumbing, mirroring the existing fp8 arm
   at triton_mla_sparse_kernel.py:720 / triton_mla_sparse.py:346-415.

Recommendation: re-scope the workorder to the DS41F slice with corrected
numbers (amend §1/§6 of the requirements + Phase 2-5 of the dev plan), or
close the project with this audit as the documented rejection of the
capacity-lever framing. Either way the ×1.86 claims must not survive in any
upstream-facing text.

## 6. Evidence manifest (new pins this phase)

| number | capture |
|---|---|
| GLM-Flash 27.49 GiB/rank KV budget; 3407 blocks = 24.61 GiB; 3,866,340 tok | `/data/models/vllm_glm53_flash_service.log` boots 01:12/03:43/05:16/05:30 (gpu_worker.py:837, extensible_kv_cache.py:625, kv_cache_utils.py:1168) — pin: `evidence-nvfp4-kv-20261114/glm-pool-composition.logcut.txt` |
| DS41F 3.57 GiB/rank KV budget; 1,414,344 tok; group sizes [32,32,32,32,128,8] | `/data/models/dsv41_final.log` 01:17 boots (gpu_worker.py:719, kv_cache_utils.py:2476,786) — pin: `evidence-nvfp4-kv-20261114/dsv41-pool-composition.logcut.txt` |
| prod backends TRITON_MLA_SPARSE etc. | service log 2026-10-01 boot, speculator.py:116 line — pin: `evidence-nvfp4-kv-20261114/prod-backend-names.logcut.txt` |
| all file:line anchors | `git grep`/reads against `ampere/main` @ 5ede3e2eec (commands in session log) |
