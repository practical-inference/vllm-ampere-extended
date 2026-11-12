# Upgrading between tags

How releases are cut, what a tag guarantees, and how to move a production
deployment between tags without surprises.

## What a tag guarantees

Every build passes a two-stage canary against live serving traffic before
it is tagged — a tag here means the exact build we run in production. The
nightly integrate is tagged only when the canary passes (cadence: see the
[support matrix](../README.md#support-matrix); the canary story:
[README](../README.md#what-we-do)).

| Tag | Use it for | Mutability |
| --- | --- | --- |
| `vX.Y.Z-ae.N` | stable serving; matches a GitHub release | immutable |
| `nightly-YYYYMMDD` | tracking the nightly integrate | immutable |
| `sha-<7>` (or `@sha256:` digest) | reproducing an exact build | immutable |
| `latest` | convenience | **floats** — never pin production to it |

Spelling note: git release tags and GHCR container tags can spell the same
version differently (`v2026.10.1+ae.1` as a git tag, `v2026.10.1-ae.1` on
GHCR) — always copy the tag string from the registry you are pulling from.

## How a release is cut

`main` follows upstream vLLM `main` nightly. Each night the integrate
builds, the two-stage canary runs it against live serving traffic, and on a
pass the release automation (outside this tree) cuts the git tag, builds
the image and wheel, publishes the image to GHCR and attaches the wheel to
the release, and regenerates the feature-availability page onto `main`.
Each release's notes list the config changes, newly validated manifest
entries, and user-facing commits;
[docs/ampere/availability.md](../docs/ampere/availability.md) maps features
to the tag that first carries them.

## Upgrade procedure

1. Pick the target: a `vX.Y.Z-ae.N` release tag for stable serving, a
   `nightly-YYYYMMDD` to track the integrate, or `sha-<7>`/digest to
   reproduce an exact build.
2. Check what the new tag carries:
   [docs/ampere/availability.md](../docs/ampere/availability.md) and the
   release notes.
3. Record the digest you are running now, so rollback is copy-paste:

   ```bash
   docker inspect --format '{{range .RepoDigests}}{{.}}{{"\n"}}{{end}}' \
     ghcr.io/practical-inference/vllm-ampere-extended:v2026.10.1-ae.1
   ```

4. Edit the tag in your service definition (systemd unit or shell wrapper:
   [runbooks/run-container.md](run-container.md#systemd-unit)) and restart.
5. Verify alive: [runbooks/run-container.md](run-container.md#verify-it-is-alive).

## Rollback

Redeploy the previous immutable tag — or the exact digest recorded in step
3, which reproduces the running build regardless of tags:

```bash
docker run --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  -v /data/models:/models:ro \
  ghcr.io/practical-inference/vllm-ampere-extended@sha256:<digest> \
  --model /models/<model> --api-key "$VLLM_API_KEY"
```

Pass the same model flags you were running — the digest reproduces the
build, not your launch configuration.

Rolling back the container does not roll back anything the new build wrote
on volumes — mounts are read-only in the documented lines, so there is
nothing to undo.
