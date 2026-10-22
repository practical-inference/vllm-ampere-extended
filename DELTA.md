<!-- markdownlint-disable MD060 -->
# Delta vs upstream vLLM

`main` = upstream `main` (nightly merge) + the commits below: 1 source commits (55 files changed, 4025 insertions(+), 204 deletions(-)).

Diff base: upstream commit `c723f6d3ee` — the tip this tree last merged from origin. Reproduce at this commit with:
`git log --no-merges c723f6d3ee..687adc104a -- vllm/ csrc/` and `git diff --stat c723f6d3ee..687adc104a -- vllm/ csrc/`.

Areas upstream does not ship for this hardware are what this distribution exists to carry; general fixes are contributed upstream first (see README). Regenerated automatically by the nightly integrate.

| Area | Commits |
|---|---|
| general fixes | 1 |

<details><summary>Full commit list</summary>

- chore: re-seed next onto upstream lineage — reapply the fork delta

</details>
