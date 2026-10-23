<!-- markdownlint-disable MD060 -->
# Delta vs upstream vLLM

`main` = upstream `main` (nightly merge) + the commits below: 5 source commits (58 files changed, 4173 insertions(+), 212 deletions(-)).

Diff base: upstream commit `8fae0ff58b` — the tip this tree last merged from origin. Reproduce at this commit with:
`git log --no-merges 8fae0ff58b..f4d131a574 -- vllm/ csrc/` and `git diff --stat 8fae0ff58b..f4d131a574 -- vllm/ csrc/`.

Areas upstream does not ship for this hardware are what this distribution exists to carry; general fixes are contributed upstream first (see README). Regenerated automatically by the nightly integrate.

| Area | Commits |
|---|---|
| general fixes | 3 |
| spec-decode fixes | 1 |
| sparse attention (TRITON_MLA_SPARSE) | 1 |

<details><summary>Full commit list</summary>

- docs: correct the non-spec chunk-indices sync mechanism comment
- [Bugfix][SpecDecode] Propagate rope overrides to same-checkpoint draft configs
- [Kernel] Dedup fp8_ds_mla dequant-gather for large sparse-MLA batches
- [Core] Pre-compute non-spec chunk indices to remove per-layer GPU sync in glm5next KDA chunk path
- chore: re-seed next onto upstream lineage — reapply the fork delta

</details>
