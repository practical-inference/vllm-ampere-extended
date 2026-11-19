<!-- markdownlint-disable MD060 -->
# Delta vs upstream vLLM

`main` = upstream `main` (nightly merge) + the commits below: 62 source commits (72 files changed, 5764 insertions(+), 439 deletions(-)).

Diff base: upstream commit `79897d9f47` — the tip this tree last merged from origin. Reproduce at this commit with:
`git log --no-merges 79897d9f47..a9cfb7052f -- vllm/ csrc/` and `git diff --stat 79897d9f47..a9cfb7052f -- vllm/ csrc/`.

Areas upstream does not ship for this hardware are what this distribution exists to carry; general fixes are contributed upstream first (see README). Regenerated automatically by the nightly integrate.

| Area | Commits |
|---|---|
| general fixes | 27 |
| spec-decode fixes | 14 |
| model bring-up (GLM5Next / Qwen4Exp / DSV41) | 7 |
| sparse attention (TRITON_MLA_SPARSE) | 5 |
| KV-cache / offload fixes | 4 |
| fork delta (squash: spec-decode fixes + KV-cache / offload fixes) | 2 |
| fork delta (squash: sparse attention (TRITON_MLA_SPARSE) + KV-cache / offload fixes) | 1 |
| fork delta (squash: sparse attention (TRITON_MLA_SPARSE) + spec-decode fixes) | 1 |
| fork delta (squash: sparse attention (TRITON_MLA_SPARSE) + SM80 / pre-SM89 fp8 fallbacks + model bring-up (GLM5Next / Qwen4Exp / DSV41) + spec-decode fixes + KV-cache / offload fixes) | 1 |

<details><summary>Full commit list</summary>

- [Core] Fix mypy static-gate drift in video chunker returns (92f71ba38a)
- [core] mm: micro-batch video encoder forwards to bound the KV transient (38d85fcfa2)
- [core] qsa: sm_80 fp8-indexer arm — uint8 storage + software e4m3 dequant (145978c011)
- config: wire the draft rope propagation in SpeculativeConfig.__post_init__ (708c44bb71)
- fix: demand recording must not sit under the K0 diagnostics guard (31bc038e67)
- fix: torch.accelerator APIs in fork-delta worker/model code (785ec31b15)
- fix: port fork test/offloader call sites onto the upstream kv-cache interface rename (7894bf8633)
- fix(v1) fork: restore the copy-event wait + the MAMBA env registrations on main (1519c55404)
- fix(v1): keep spec-decode draft padding out of input_ids under async scheduling (c27c4f4b30)
- feat: token-embedding UVA offload for Qwen4Exp (VLLM_EMBED_UVA=1) (4dc2d4dbd4)
- fix: warm the compile cache before the pinned-path memory profile (5647cb9445)
- feat: vision tower CPU offload via UVAOffloader bulk-copy (VLLM_VISION_CPU_OFFLOAD_GB) (eb067aa792)
- worker: profile transient + cudagraph claims in the pinned-KV path (4de178aa35)
- bench: memprofile via allocator history (torch.profiler inert under graph capture) (3933a8d246)
- bench: memprofile filter on device_memory_usage + empty-table diagnostic (111f6fb1d3)
- bench: env-gated per-op device-memory attribution for forward bench (VLLM_BENCH_MEMPROFILE) (2a6b776fd8)
- bench: emit per-case peak-memory in forward bench output-json (local measurement tool) (9c185b8a81)
- feat(sched): KV valve demand trigger — pool-size-insensitive arming (a61560efe6)
- fix(sched): KV valve arms only when requests are waiting (d130dcef04)
- fix(sched): key dynamic-SD tier on the running batch, not per-step scheduled count (9d8ce24a64)
- correction: f3e27e6ef2 'whitespace only' claim false — remove resurrected dead single_group_spec (d2c1a7fde9)
- fix: sparse-indexer SM80 fallback validation fixes (781 passed on SM80) (22714165ba)
- [Feature][Hybrid] K0 spec-window slot release for hybrid mamba models (b51baae430)
- fix: align MTP draft QSA geometry with the shared top-k buffer (ef7d022c42)
- fix: drop upstream's duplicate process_weights_after_loading in qsa.py (d0a2f67817)
- [Bugfix][Frontend] Thread excludes/max_whitespace_cnt into every or-branch leaf (9c2a2bd005)
- [Bugfix][Frontend] Re-tighten any_order on the GLM-4.7 strict path (f424de6748)
- fix: restore the decode-path SM80 triton fallback in the sparse indexer (aa7e5f011b)
- fix: route the k=64/128 persistent-topk arms through the dtype wrapper after the upstream launcher split (0f03b25c08)
- fix: restore the SM80 triton fallback + DeepGEMM gate in the sparse indexer after the upstream restructure (aeec9d9df2)
- Revert "Merge branch 'perf/eagle-k0-lifecycle' into next" (9107e1fba5)
- Revert "fix(spec): eagle K=0 resume re-growth must reach the runner's block table" (183ef23a1a)
- Revert "fix(spec): eagle K=0 resume rebuilds runner rows via the resumed channel" (9b31b8d4e8)
- Revert "perf(sched): eagle K=0 hold engages only on all-decode steps" (b794cff8a5)
- Revert "perf(runner): batch the eagle K=0 stash capture to one clone set per step" (5d4cfa746a)
- Revert "perf(runner): prune the eagle K=0 stash only when the live set changes" (008d530292)
- Revert "docs: correct claims in 7e67d27777/9570e9e0f5/9b97d10d0b (the stash-clone chase)" (ca70f82b4f)
- docs: correct claims in 7e67d27777/9570e9e0f5/9b97d10d0b (the stash-clone chase) (91c751dbac)
- docs: regenerate DELTA.md for the current delta (9d2030103a)
- perf(runner): prune the eagle K=0 stash only when the live set changes (9b97d10d0b)
- perf(runner): batch the eagle K=0 stash capture to one clone set per step (9570e9e0f5)
- perf(sched): eagle K=0 hold engages only on all-decode steps (7e67d27777)
- fix(spec): eagle K=0 resume rebuilds runner rows via the resumed channel (72e2f45c01)
- fix(spec): eagle K=0 resume re-growth must reach the runner's block table (6d6c7e8f2e)
- docs: regenerate DELTA.md for the current delta (b017bd1ce9)
- fix(spec): mypy narrowing on the K=0 catch-up propose + test helper (6c1e67f8c1)
- style: ruff format scheduler.py eagle-hit strip (d187cac631)
- fix(spec): eagle K=0 hold admits requests — eagle hits stripped, table capped (38f18ce0f1)
- [Bugfix][Frontend] Compile GLM-4.7 strict tool grammars with any_order=True (3b2cb3575a)
- fix(spec): eagle K=0 catch-up next tokens must be row-aligned (df4ba64557)
- fix(spec): eagle K=0 resume re-growth must precede the block-id snapshot (add5d58351)
- feat(spec): eagle K=0 draft-resource lifecycle state machine + runner catch-up (A') (ae774cb74a)
- perf(kv): eagle-scoped allocation extras + synced-token introspection (A' groundwork) (643481013f)
- fix: bump pinned ruff to v0.16.4 (6ff6eddf7d)
- fix: fold the K=0 accepted-token offset in align preprocess (draft-less decode rows take the regular path) (6ea82baffd)
- fix: track upstream refactor in fork delta (4d1b910d91)
- fix: sync amd kda signature with nvidia + torch.accelerator in dedup test (5c1b5e6aa1)
- (no subject) (6d1e2280ee)
- [Bugfix][SpecDecode] Propagate rope overrides to same-checkpoint draft configs (f4545481f8)
- (no subject) (929f2f5fac)
- [Core] Pre-compute non-spec chunk indices for the glm5next KDA chunk path (2d94403b5c)
- chore: re-seed next onto upstream lineage — reapply the fork delta (6968d526b1)

</details>
