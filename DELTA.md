<!-- markdownlint-disable MD060 -->
# Delta vs upstream vLLM

`main` = upstream `main` (nightly merge) + the commits below: 3 source commits (58 files changed, 4134 insertions(+), 210 deletions(-)).

Diff base: upstream commit `b689feb9f2` — the tip this tree last merged from origin. Reproduce at this commit with:
`git log --no-merges b689feb9f2..8e35a25cd7 -- vllm/ csrc/` and `git diff --stat b689feb9f2..8e35a25cd7 -- vllm/ csrc/`.

Areas upstream does not ship for this hardware are what this distribution exists to carry; general fixes are contributed upstream first (see README). Regenerated automatically by the nightly integrate.

| Area | Commits |
|---|---|
| general fixes | 2 |
| sparse attention (TRITON_MLA_SPARSE) | 1 |

<details><summary>Full commit list</summary>

- [Kernel] Dedup fp8_ds_mla dequant-gather for large sparse-MLA batches
- [Core] Pre-compute non-spec chunk indices to remove per-layer GPU sync in glm5next KDA chunk path
- chore: re-seed next onto upstream lineage — reapply the fork delta

</details>
