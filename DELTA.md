<!-- markdownlint-disable MD060 -->
# Delta vs upstream vLLM

`main` = upstream `main` (nightly merge) + the commits below: 4 source commits (58 files changed, 4173 insertions(+), 212 deletions(-)).

Diff base: upstream commit `ccd4e4030f` — the tip this tree last merged from origin. Reproduce at this commit with:
`git log --no-merges ccd4e4030f..6353d43454 -- vllm/ csrc/` and `git diff --stat ccd4e4030f..6353d43454 -- vllm/ csrc/`.

Areas upstream does not ship for this hardware are what this distribution exists to carry; general fixes are contributed upstream first (see README). Regenerated automatically by the nightly integrate.

| Area | Commits |
|---|---|
| general fixes | 3 |
| spec-decode fixes | 1 |

<details><summary>Full commit list</summary>

- [Bugfix][SpecDecode] Propagate rope overrides to same-checkpoint draft configs
- 
- [Core] Pre-compute non-spec chunk indices for the glm5next KDA chunk path
- chore: re-seed next onto upstream lineage — reapply the fork delta

</details>
