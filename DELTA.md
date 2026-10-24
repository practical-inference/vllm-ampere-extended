<!-- markdownlint-disable MD060 -->
# Delta vs upstream vLLM

`main` = upstream `main` (nightly merge) + the commits below: 6 source commits (59 files changed, 4178 insertions(+), 217 deletions(-)).

Diff base: upstream commit `7a64fb8fc9` — the tip this tree last merged from origin. Reproduce at this commit with:
`git log --no-merges 7a64fb8fc9..53c37be4ff -- vllm/ csrc/` and `git diff --stat 7a64fb8fc9..53c37be4ff -- vllm/ csrc/`.

Areas upstream does not ship for this hardware are what this distribution exists to carry; general fixes are contributed upstream first (see README). Regenerated automatically by the nightly integrate.

| Area | Commits |
|---|---|
| general fixes | 5 |
| spec-decode fixes | 1 |

<details><summary>Full commit list</summary>

- fix: sync amd kda signature with nvidia + torch.accelerator in dedup test
- 
- [Bugfix][SpecDecode] Propagate rope overrides to same-checkpoint draft configs
- 
- [Core] Pre-compute non-spec chunk indices for the glm5next KDA chunk path
- chore: re-seed next onto upstream lineage — reapply the fork delta

</details>
