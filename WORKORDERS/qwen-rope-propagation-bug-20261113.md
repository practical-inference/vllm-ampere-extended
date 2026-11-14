# WORKORDER: qwen rope-propagation bug — root cause, runtime fix, durable-landing handoff

Created: 2026-11-13 (columbo, llm-test-tools session). Owner-directed
root-cause of the qwen cold-load crashes ("both qwen servers crash when I
load them cold with 323K context"). **The vllm-nightly session owns the
durable landing** — the runtime fix below is ephemeral by design.

## The bug (confirmed at source level)

- Symptom: both qwen servers (:8006, :8017) died on cold ~323K-token
  requests — `torch.AcceleratorError: CUDA device-side assert` surfacing
  in model-state `preprocess_state → .to(device)` (async; the origin is an
  ATen `vectorized_gather_kernel` index-OOB in the drafter).
- Root cause: wheel dev48+ae.1.g04da26725 ships
  `_propagate_rope_to_draft` in `vllm/config/speculative.py` **defined but
  never called** (1540d07a15's call-site hunk never landed). The drafter
  built its rotary cos/sin cache at the model's NATIVE
  `max_position_embeddings` = **262144**, while YARN 2x (the promoted
  config) lets the target run to 524288. Any request past 262144 tokens
  indexes past the drafter's cache → OOB → assert → engine death.
- Differential proven: 262K cold **passes**; 337K cold **dies**. Boundary
  = the native limit, exactly as the owner hypothesised ("anything over
  262K (needs the rope 2x) will kill it").
- Siblings in the same regime: the fork TRIPWIRE index-OOB (11-11 16:59),
  Triton OOM (11-12 05:13).

## Runtime fix (APPLIED — ephemeral, see the trap)

`/data/venvs/vllm-nightly/lib/python3.12/site-packages/vllm/config/
speculative.py` — `__post_init__` now calls
`_propagate_rope_to_draft(self.target_model_config.hf_config,
self.draft_model_config.hf_config)` before `return self`. Both qwen units
bounced and verified: 310K cold → HTTP 200, finish stop, content "OK".
The B4 **boot-order patch** (commit `4de178aa35` on main — "worker:
profile transient + cudagraph claims in the pinned-KV path") was also
wiped by the wheel deploys and restored the same way (without it the
negative `--kv-cache-memory` pin reserves -0.93 GiB → "No available
memory for cache blocks" at boot).

**THE TRAP**: site-packages patches evaporate on every wheel deploy
(vllm_wheel_deploy.sh rm -rf's the package) — and the prod venv's copies
evaporate on the hourly promote sync. 1540d07a15 was lost exactly this
way. The patches above WILL vanish on the next deploy.

## Durable landing (vllm-nightly session — the ask)

1. **Wire the call on `next`**: commit the `__post_init__` propagation
   (the exact patch above) to `vllm/config/speculative.py` on `next`
   (currently 04da26725), with a regression test — extend
   `tests/config/test_speculative_draft_hf_overrides.py` (it already has
   the yarn fixtures; add: a target hf_config carrying `rope_parameters`
   + a draft config → assert the draft's text config inherits them after
   `__post_init__`). Push next → mgfork + the forgejo mirror.
2. **Land the boot-order patch on `next` too**: `4de178aa35` sits on
   `main` only — the nightly wheel (built from next) ships without it, so
   every wheel deploy reverts it. Cherry-pick or merge to next.
3. **next↔main reconciliation**: next (04da26725) and main have diverged
   to the point where the running qwen wheel is not on ampere/main's
   history and vice versa. The patch split (rope fix on main, wheel from
   next) is a symptom. Reconcile at the next integrate cycle.
4. **dev135 exp2 memory-tuning item** (separate): the k0-v2 wheel
   (dev135+ae.1.g5ba077f11 — already in /data/wheels/vllm-ampere, carries
   the TESTED dict-walk rope propagation + the k0-v2 valve) fails the
   exp2/prod config at boot: "No available memory for cache blocks" — the
   B4 pin tuning (-1e9 + boot-order) was done for dev48. Tuning pass
   needed before the qwen servers can ride the k0-v2 wheel.
5. Note: ampere-extended main (767978ed36) already carries the wired
   dict-walk + tests — no ampere push needed for the rope fix; the dev135
   tuning item (4) is the ampere-side gap.

## Verification bar

Cold boot → single ~337K-token request (the repro:
/tmp/opencode/cold323k_repro.py pattern — 1.35M chars, max_tokens 200) →
HTTP 200, finish stop, engine alive. 262K must keep passing.

## Status

- [ ] vllm-nightly session: acknowledge + schedule the wiring commit
- [ ] Wiring commit on next + regression test
- [ ] Boot-order patch (4de178aa35) on next
- [ ] next↔main reconciliation
- [ ] dev135 exp2 memory-tuning pass
- [ ] Next nightly wheel: verify the repro passes WITHOUT site-packages
      patches (the durability proof)
