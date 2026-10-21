---
name: revert-forgejo-main
description: Revert or repair commits on forgejo mirrors whose main branch is protected (michael/vllm, vllm-minimax, vllm-nightly, vllm-quant). Use when a push to forgejo main is rejected with "Not allowed to push to protected branch main", when reverting a bad commit on the mirrors, or when an emergency direct push (including force push) to forgejo main is required.
---

# Reverting pushes on forgejo-protected main

Forgejo (192.168.2.31:3001, API base `http://192.168.2.31:3001/api/v1`) has
branch protection on `main` for `michael/{vllm,vllm-minimax,vllm-nightly,vllm-quant}`:
`enable_push=false`, `apply_to_admins=true`, `required_approvals=0`.
Direct pushes to main are rejected **including for admin**. Token: `~/.config/forgejo/token`.

Two verified paths. Default to Path A; Path B is the emergency hatch.

## Path A — PR route (verified end-to-end 2026-10-10)

```bash
TOKEN=$(cat ~/.config/forgejo/token)
BASE=http://192.168.2.31:3001/api/v1
REPO=michael/vllm-nightly          # adjust per mirror

git checkout -b revert/<topic> forgejo/main
git revert -s <sha>                # -s: global commit-msg hook enforces DCO
git push forgejo revert/<topic>    # non-protected branches push fine

PR=$(curl -s -X POST -H "Authorization: token $TOKEN" -H "Content-Type: application/json" \
  -d '{"title":"[Revert] <topic>","head":"revert/<topic>","base":"main"}' \
  $BASE/repos/$REPO/pulls | jq -r .number)
curl -s -X POST -H "Authorization: token $TOKEN" -H "Content-Type: application/json" \
  -d '{"Do":"merge"}' $BASE/repos/$REPO/pulls/$PR/merge
```

`required_approvals=0` → self-merge is allowed. Web UI equivalent:
`http://mothership:3001/michael/<repo>/compare/main...<branch>`.

## Path B — emergency direct push (API flip; verified 2026-10-10)

The only way to force-push main (protection cannot be configured to allow
force push on this Forgejo build — no `allow_force_push` field). Lift → push → restore:

```bash
TOKEN=$(cat ~/.config/forgejo/token)
BASE=http://192.168.2.31:3001/api/v1
REPO=michael/vllm-nightly

# 1. lift protection (DELETE by branch name, returns 204)
curl -s -X DELETE -H "Authorization: token $TOKEN" $BASE/repos/$REPO/branch_protections/main

# 2. push (fast-forward or --force; pre-receive adversarial gate still runs)
git push forgejo <sha>:main        # or --force

# 3. restore protection — ALWAYS do this, even if the push failed
curl -s -X POST -H "Authorization: token $TOKEN" -H "Content-Type: application/json" \
  -d '{"branch_name":"main","enable_push":false,"apply_to_admins":true,"required_approvals":0}' \
  $BASE/repos/$REPO/branch_protections
```

## Notes

- **Adversarial gate**: pushes to the vllm repos may run the pre-receive
  validation gate (minutes, not seconds). Do not kill the push — wait for the
  verdict (ENF-009).
- **Commit-msg hook**: global client-side hook enforces `Signed-off-by` (DCO)
  and AI-attribution trailer on all forgejo commits. `git revert -s` covers
  DCO; if the hook rejects, follow its fix instructions. Synthetic test
  commits can bypass with `--no-verify`.
- **Revert everywhere**: a mirror revert is not done until the same revert
  lands on ampere (`git push ampere <sha>:main` — GitHub, unprotected) and
  prod is synced (pull on prod hosts). Established workflow: main, mirrors, prod.
- **Tag pushes are unaffected** by branch protection — the promote workflow
  (`git push forgejo <tag> && git push ampere <tag>`) needs none of this.
- **vllm-rebase is unprotected** (default branch is `glm53-rebase-step1`, WIP lane).
- Scratch-repo validation history: `apply_to_admins` enforced; force-push
  blocking unavailable on Forgejo 16.0.5 (legacy gitea-1.22 API, `/rules` → 404);
  PR-only was the only enforced mode, hence this setup.
