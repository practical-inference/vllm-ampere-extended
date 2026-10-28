<!-- markdownlint-disable MD060 -->
# Delta vs upstream vLLM

`main` = upstream `main` (nightly merge) + the commits below: 7 source commits (59 files changed, 4183 insertions(+), 219 deletions(-)).

Diff base: upstream commit `7e3e3f6ab6` — the tip this tree last merged from origin. Reproduce at this commit with:
`git log --no-merges 7e3e3f6ab6..5172a0ee9d -- vllm/ csrc/` and `git diff --stat 7e3e3f6ab6..5172a0ee9d -- vllm/ csrc/`.

Areas upstream does not ship for this hardware are what this distribution exists to carry; general fixes are contributed upstream first (see README). Regenerated automatically by the nightly integrate.

| Area | Commits |
|---|---|
| model bring-up (GLM5Next / Qwen4Exp / DSV41) | 2 |
| fork delta (squash: sparse attention (TRITON_MLA_SPARSE) + spec-decode fixes) | 1 |
| general fixes | 1 |
| spec-decode fixes | 1 |
| sparse attention (TRITON_MLA_SPARSE) | 1 |
| fork delta (squash: sparse attention (TRITON_MLA_SPARSE) + SM80 / pre-SM89 fp8 fallbacks + model bring-up (GLM5Next / Qwen4Exp / DSV41) + spec-decode fixes + KV-cache / offload fixes) | 1 |

<details><summary>Full commit list</summary>

- fix: track upstream refactor in fork delta (4d1b910d91)
- fix: sync amd kda signature with nvidia + torch.accelerator in dedup test (5c1b5e6aa1)
- (no subject) (6d1e2280ee)
- [Bugfix][SpecDecode] Propagate rope overrides to same-checkpoint draft configs (f4545481f8)
- (no subject) (929f2f5fac)
- [Core] Pre-compute non-spec chunk indices for the glm5next KDA chunk path (2d94403b5c)
- chore: re-seed next onto upstream lineage — reapply the fork delta (6968d526b1)

</details>
