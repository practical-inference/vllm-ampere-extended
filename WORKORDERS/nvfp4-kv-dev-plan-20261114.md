# Development Plan: NVFP4 (4-bit) KV Cache for vllm-ampere-extended

Status: DRAFT for spec review — 2026-11-14
Companion: nvfp4-kv-requirements-20261114.md (FR/NFR/gates referenced by ID)

## 0. Incremental strategy

Each phase ends shippable, gated, and rollback-able (requirement C1 —
"no big dev projects"). Order is by payoff-per-day, not by dependency
convenience: DS41F is the *viability* case (whole feature exists for it),
GLM-Flash is the *cost* case (80 GB fit), Big GLM is the recipe-risk case,
Qwen is the lowest-value slice and goes last — Phase 0 footprint audit
may drop it entirely.

## Phase 0 — Inventory & gap audit (1 day)

**Goal:** establish what fp4 machinery already exists before writing any.
- Audit per (model × backend × dtype): store dispatch, Triton reader arms,
  selector admission, group-sizing accounting (DS_MLA_STATE_CONTENT_BYTES
  vs 352 B actual), eviction bytes-per-token, graph-capture strides.
  Every anchor checked for presence in the tree being audited (requirement
  C5): an anchor that does not resolve is a finding, not an assumption.
- Confirm FR-1/FR-2 status; measure DS41F and GLM-Flash pool footprints on
  dev slot to validate the 80 GB-fit hypothesis (§1, O1) — record whether
  activations+graphs+margin truly fit 80 GiB.
- Answer O2 (swizzle coupling) and O3 (RoPE dtype) by reading the kernel
  contract, not guessing.
**Deliverable:** gap matrix + go/no-go per model (Qwen drop decision).
**Exit gate:** every §9 open question closed with file:line evidence.

## Phase 1 — Central fp4 primitives (2-3 days)

**Scope (arch-portable, model-agnostic):**
- Triton dequant-on-load primitive: uint8 packed-e2m1 unpack + micro-block
  scale multiply → bf16 in-register (the §FR-3/FR-4 shared machinery).
- Store-path verification on sm_80 (software cvt parity: ties-to-even
  vs reference LUT — golden round-trip test quant→dequant vs Blackwell
  semantics fixture).
- CacheConfig/validation plumbing per FR-1 (error messages name backend +
  gate per flashmla_sparse precedent).
**Tests:** round-trip equality vs fixture; rejection diagnostics; dummy-run
VRAM ≤ fp8 (NFR-4).
**Exit gate:** primitives pass on sm_80 with fp8 dtype still default-inert
(NFR-1: today's boots byte-identical).

## Phase 2 — DS41F slice (2-3 days) — the viability case

**Scope:** nvfp4_ds_mla reader arm in the Triton sparse-MLA path;
indexer/engram coupling verified per FR-5; admission/death-signature
re-check per FR-6 (saturation probe + 'Cannot get 2 free blocks' grep).
**Tests:** Q1 (needle 140K/300K), Q2 (GSM8K), Q4 (storm repro), Q5; pool
count ≥ 1.8× fp8 on same TP4 (G2 acceptance); boot at fp4 + batch-ramp
88/96 (GA1-protocol).
**Exit gate:** DS41F viability verdict — *viable* (all gates) or
*documented rejected with evidence*.
**Note:** acceptance-length N/A here (no MTP head in checkpoint) — if
Phase 0 shows an ngram/V2-runner path cheap to enable, it is a *separate
follow-on workorder*, never folded into this one. Q1's round-2 restate
gate is new work, not re-verification: the one measured fp8 run
*missed* it (longctx_perf9.log:3 needle_hit=False), so the pass/fail
baseline must come from Q1's own definition, not a "precedent" claim.

## Phase 3 — GLM-5.3-Flash slice (1-2 days) — the cost case

**Scope:** same reader family as Phase 2 (ds_mla readers are one module);
state cache untouched (bf16, K-scaled pages); E7 ladder acceptance
re-measure on fp4 KV (Q3 — the E7 gate is this model's currency).
**Tests:** Q1-Q5 on dev slot; explicit 80 GB fit check (launch with the
fp4 pool on an 80 GiB-capped run if firmware allows, else compute
headroom vs fp8 peak and record the assumption).
**Exit gate:** fp4 GLM-Flash promotion-ready, or Q3 regression documented.

## Phase 4 — Big GLM 5.3 + TP2xPP3 recipe (1-2 days)

**Scope:** latent-pool arm carried from Phases 2-3; validated on the
PP3 fallback recipe (least-tested path in the fleet).
**Tests:** Q1-Q5; recipe soak doubled (4h) on promotion boot (risk-weighted).
**Exit gate:** Big-GLM verdict.

## Phase 5 — Qwen 3.8 Flash Next (2-3 days, OPTIONAL)

**Entry condition:** Phase 0 footprint audit shows the full-attention slice
is big enough for fp4 to matter on this model. Otherwise: close the work
here with the footprint evidence (the slice is a minority; GDN bulk was
never eligible).
**Scope:** GQA-geometry fp4 arm via TRITON_ATTN (FR-4), selector admission
per Gemma4 precedent (FR-8).
**Exit gate:** Q1-Q5 or explicit drop.

## Cross-cutting

**Test matrix (every phase touching a model):**
- correctness: needle recall exact-match, GSM8K, greedy parity vs fp8
  reference (`check_logprobs_close` per tests/AGENTS.md)
- capacity: pool token count vs G2 on the boot log
- stability: storm repro (populate 6x200k → multiround 2x → storm 4x300k,
  looping canary) + batch ramp to approved max conc
- death-signature: saturation probe; log grep `Cannot get \d+ free blocks
    from the pool` = 0 (raise site vllm/v1/core/block_pool.py:694 — the
    message is runtime-formatted, so grep logs, never source literals)
- graphs: capture/replay of all configured graph modes

**Rollback:** per phase — fp4 dtype env removed = byte-identical default
(NFR-1). No data-format migrations; fp4 and fp8 pools are runtime choices.

**Promotion:** per-model independent. Gate-green model promotes into the
serve script's default (house pattern); gate-failed model is *documented
rejected* in the lever table with its failing measurement — never shipped.

**Effort summary:** Phases 0-4 ≈ 9-13 dev-days (all payoff models decided);
Phase 5 adds 2-3 only if justified. Each phase independently shippable;
project can stop after any phase with a complete verdict per decided model.

**Documentation (FR-9):** rendered docs for new dtype values + the gate
matrix, landed with each promotion PR, not at the end.

**Risks & mitigations:**
- R1 fp4 retrieval quality below Q1/Q2 on some checkpoint → per-model gate,
  documented rejection, others unaffected (by design). Sharpened by
  review: even fp8 missed round-2 restate in the one measured run, so
  retrieval quality is the project's dominant risk, not a solved base —
  Q1/Q3/Q5 are gates with teeth, and a single miss is a *rejection*.
- R2 dequant ALU cost breaches NFR-3 (+15% wall) → measured every storm
  repro; if it binds, ship fp4 as an opt-in capacity mode, not a default.
- R3 indexer/engram dtype coupling surprises (FR-5) → Phase 0 audit reads
  the contract before code; coupling found = scope fix, not re-plan.
- R4 80 GB fit fails (activations/graphs exceed margin) → headline claim
  becomes "1.86× pool on same hardware", still full value of G2/G3.
- R5 least-tested PP3 recipe breaks under fp4 (Phase 4) → doubled soak.
