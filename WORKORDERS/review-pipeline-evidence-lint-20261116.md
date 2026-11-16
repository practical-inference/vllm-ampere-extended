# WORKORDER: review-pipeline evidence lint + smell-tool fixes (post-mortem 2026-11-16)

Status: FILED 2026-11-16 (columbo). Owner review requested. No GPU needed —
all items are tooling/process work. Supersedes nothing; complements the
MECHANICS lessons in SESSION_CHECKPOINT_2026-11-15_night_ampere-released.md.

## Context (what happened)

The qsa fp8-index arm (next @ 32d144e153) went through adversarial review on
2026-11-15/16. First round: a2+a3 DIED (~40 min agent time) on evidence gaps;
the fix cycle cost a GPU bounce + kernel-test re-run + captures + history
rewrite (~30 min); the rerun cost another ~51 min (a2 771s, a3 2287s).
maintainer_smell.sh hung twice (~20 min each, watchdog exit 124) and never
produced a verdict. Total avoidable waste ≈ 1.5-2h. All of it was
foreseeable: the evidence-discipline rules were already written down
(SESSION_CHECKPOINT_2026-11-14_nvfp4-kv-spec.md §MECHANICS: "pin EVERY
load-bearing number to a capture with provenance — reviewers check pins
against captures"; the nvfp4 evidence dir pattern in
WORKORDERS/nvfp4-kv-CLOSED-20261114.md §3c).

## Failure modes

- **FM-1 — Evidence produced but not recorded at generation time.** The
  kernel test ran interactively (bash, foreground); its output existed only
  in the session transcript. The review requires committed artifacts. Fix
  candidate: a `record_run.sh <evidence-dir> <name> -- <cmd>` wrapper (tee +
  rc + git sha + timestamp) and a convention that EVERY test/bench invocation
  during a review-bound experiment goes through it (or through run_job with
  log_path pointed INTO the evidence dir — the primitive exists; it wasn't
  used).
- **FM-2 — Argued claims written as measured claims.** "Bit-exact vs the
  native fp8 dot" was a mathematical argument stated as a measurement; the
  test actually compares against a torch fp32 reference of the semantics.
  a3 killed it. Fix candidate: draft Claims bullets must tag
  [measured]/[argued]; the lint rejects untagged numerics.
- **FM-3 — Load-bearing lines left in a 255MB append log.** The boot-dead
  CompilationError and the fp8-engagement proof lived in
  /data/models/leoncca_exp2.log but not in the evidence dir. The nvfp4
  project extracted exactly such pins (glm-pool-composition.logcut.txt); the
  pattern was skipped. Fix candidate: the lint requires every claimed
  symptom/engagement fact to cite a capture IN the evidence dir.
- **FM-4 — A 0-byte file committed as evidence** (fp8-arm-benches.logcut.txt
  — the killed mislabeled job's log). Fix candidate: lint rejects 0-byte
  evidence files and files whose name promises content the file doesn't
  contain.
- **FM-5 — Unfavorable A/B deltas unreported.** The draft's Claims cited
  parity/positive numbers only; the short-context regression (c1/c4 −8-16%)
  was in the session's data but not the draft. a3 caught it. Fix candidate:
  lint requires every A/B cited in the draft to include ALL shapes/arms from
  the cited artifact (cross-check draft numbers vs the JSON/logcut).
- **FM-6 — maintainer_smell.sh hangs (twice, watchdog 124).** ROOT CAUSE
  (from the stderr/stdout artifacts
  `.pr-drafts/evidence/qsa-fp8-sm80-arm/smell-20261115-{214354,222023}.log*`):
  the smell agent's repo retrievals (`rg` into the repo) are blocked by
  ENF-001 (lexical bypass guard); the adversarial packets carry the
  workaround guidance (a2's log: "Repo grep is gated; the guard explicitly
  allows a python scan") and the adversarial pipeline pre-builds site-maps —
  the smell packet has NEITHER, so the agent burns its entire watchdog
  fighting blocked calls and dies without a verdict. Fixes: (a) add the
  ENF-001 guidance + site-maps to the smell packet template (parity with
  adversarial packets); (b) make the smell tool acquire the same draft lock
  the adversarial tool uses and QUEUE instead of running concurrently
  (both hangs overlapped an active adversarial run); (c) on watchdog kill,
  dump the agent's last N actions into the artifact for diagnosability (the
  stdout tail is what made this diagnosis possible — make it systematic).
- **FM-7 — validate_pr.sh G5c range walk.** It walked back to cbd81ff0a (the
  last promoted candidate) and failed on ~11 pre-existing integrate commits
  that were never going to get trailers — a PR-shaped range derivation applied
  to a lane landing. Fix candidate: an explicit `--base <ref>` (or
  merge-base-with-next detection) so lane landings validate only their own
  commits.
- **FM-8 — Rerun launched while the original run held the draft lock** (one
  wasted job; the error message was clear). Fix candidate: the adversarial
  tool's lock error should print the holding run's PID/log path (it printed
  the dir; adding "still-running since <ts>, log <path>" makes the wait
  decision trivial).

## Proposed fixes — development pipeline (pr-pipeline)

1. **G0 evidence lint (new, mechanical, seconds — run BEFORE the expensive
   agents):** draft-citation cross-check. Every backticked path in the draft
   must exist in-tree, be non-empty, and (for claimed symptoms/numbers)
   contain the claimed string/number; every Claims bullet tagged
   [measured]/[argued]; no 0-byte files under the cited evidence dirs; A/B
   tables cross-checked against the cited artifacts. This lint would have
   caught FM-1/2/3/4/5 pre-flight for ~zero cost — the a2/a3 deaths were all
   lint-detectable.
2. **`record_run.sh`** (FM-1): tee-wrapper that stamps git sha + rc + ts into
   the evidence dir; referenced from the draft template's Validation section.
3. **validate_pr.sh `--base`** (FM-7).
4. **Smell packet parity** (FM-6a): ENF-001 guidance + site-maps in the smell
   packet; smell queued behind/locked against adversarial runs (FM-6b);
   watchdog dumps last actions (FM-6c).

## Proposed fixes — semantic router

1. **run_job evidence convention:** an optional `evidence_dir`/`artifact_name`
   pair that (a) writes the log INTO the evidence dir and (b) appends a
   one-line provenance record (sha, rc, ts) — makes FM-1 structurally hard.
2. **D-Q016-style pre-flight for review-bound work:** the fix-gate warning
   already fires on uncommitted changes; add a sibling warning when a draft
   exists whose Claims cite paths that don't exist/are empty (reuses the G0
   lint logic at warn level).
3. **Agent-packet standard:** every tool that spawns repo-attached agents
   (smell, adversarial, future ones) embeds the ENF-001 retrieval guidance +
   site-maps — one shared packet preamble, sourced from the MECHANICS
   lessons rather than re-derived per tool.
4. **Watchdog diagnosability:** agent-spawning tools dump the agent's last
   actions on watchdog kill (the smell diagnosis was only possible because
   stdout happened to survive).

## Facts needed to action (all verified this session)

- Adversarial run artifacts:
  `.pr-drafts/evidence/qsa-fp8-sm80-arm/run-20261116-044400/` (a1 SURVIVES
  2552s, a2 DIES→SURVIVES on rerun 771s, a3 DIES→SURVIVES on rerun 2287s);
  rerun log `/data/models/WORKORDERS/evidence-index-shrink-20261115/
  adversarial-rerun2.logcut.txt`; smell hangs:
  `.pr-drafts/evidence/qsa-fp8-sm80-arm/smell-20261115-{214354,222023}.log*`.
- The pipeline tools: `~/source/llm-test-tools/pr-pipeline/`
  (validate_pr.sh, adversarial_review.py/sh, maintainer_smell.sh,
  new_draft.sh, forgejo-pre-receive.sh).
- The evidence-discipline precedents: 11-14 checkpoint MECHANICS section;
  nvfp4-kv-CLOSED-20261114.md §3c; the evidence convention in AGENTS.md
  ("Evidence convention (fork)").
- The landing this post-mortem came from: next @ 32d144e153 (arm 145978c011),
  draft `.pr-drafts/qsa-fp8-sm80-arm.md`, evidence dir
  `WORKORDERS/evidence-index-shrink-20261115/` (in-tree).
- ENF-001 guard semantics: repo-tree lexical retrieval blocked;
  investigate_code literal or SR_OK=1 override; python scans pass for
  non-repo/derived targets (the adversarial a2 agent used python scans
  successfully).
- No GPU required for any item here. (GPU fleet on case data until
  2026-11-16 evening per owner.)
