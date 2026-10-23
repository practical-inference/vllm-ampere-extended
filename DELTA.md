<!-- markdownlint-disable MD060 -->
# Delta vs upstream vLLM

`main` = upstream `main` (nightly merge) + the commits below: 4 source commits (58 files changed, 4170 insertions(+), 212 deletions(-)).

Diff base: upstream commit `5c3d7ac9b6` — the tip this tree last merged from origin. Reproduce at this commit with:
`git log --no-merges 5c3d7ac9b6..d345e28523 -- vllm/ csrc/` and `git diff --stat 5c3d7ac9b6..d345e28523 -- vllm/ csrc/`.

Areas upstream does not ship for this hardware are what this distribution exists to carry; general fixes are contributed upstream first (see README). Regenerated automatically by the nightly integrate.

| Area | Commits |
|---|---|
| general fixes | 2 |
| spec-decode fixes | 1 |
| sparse attention (TRITON_MLA_SPARSE) | 1 |

<details><summary>Full commit list</summary>

- [Bugfix][SpecDecode] Propagate rope overrides to same-checkpoint draft configs
- [Kernel] Dedup fp8_ds_mla dequant-gather for large sparse-MLA batches
- [Core] Pre-compute non-spec chunk indices to remove per-layer GPU sync in glm5next KDA chunk path
- chore: re-seed next onto upstream lineage — reapply the fork delta

</details>
