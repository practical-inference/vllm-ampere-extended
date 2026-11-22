# GLM pin forensics — verdict (2026-11-13, columbo)

Source: 8 torch allocator snapshots (/data/models/glm_oom_snapshot_*.pickle,
banked 2026-11-11 15:21, two OOM events ~3s apart, all 4 TP ranks).
Tool: /tmp/opencode/pin_forensics.py + _v3.py; reports in
/tmp/opencode/pin_forensics_report.txt + _v3_report.txt.

## The OOM
- Trigger: a **127 MiB** chunk-transient alloc failed (pool (0,0), default
  stream) — event 673 once; event 675 twice (133,345,280 B then 134,024,192 B).
- At OOM: reserved 91.28 GiB, active 90.48 GiB, **free-in-pool only
  0.74-0.80 GiB** and fragmented (no hole >= the 127 MiB request; segment
  expansion also failed).
- The failed size is not exotic: the SAME 127 MiB alloc succeeded 2x earlier
  in the trace window (repeated cycle table: 127 MiB x3, 235 MiB x2,
  62 MiB x5, 117 MiB x2, 58 MiB x4 — the per-chunk prefill transients,
  ~1.1 GiB stacked at the OOM moment).

## Static census at OOM (identical across ranks/events)
- 25.338 GiB x1 — model weights (W4A16 per rank).
- **37.7 GiB in per-layer blocks: 43 x 0.5625 GiB (24.19) + 43 x 0.28125 GiB
  (12.09) + 43 x 0.0176 + 46 x 0.0156** — per-layer mamba/SSM + conv caches
  (the x43 = layer count; mamba_ssm_cache_dtype bfloat16, align mode).
- 17.000 GiB x1 — the pinned KV pool (--kv-cache-memory).
- ~2.5 GiB misc (1.5 + 1.0 GiB workspaces/graph pools, small blocks).
- allocator_settings: PYTORCH_CUDA_ALLOC_CONF EMPTY — expandable_segments
  OFF, no GC threshold.

## Verdict
1. The OOM is NOT the (b)-plan's 2.4 GiB dequant workspace: **no alloc
   >=256 MiB appears anywhere in the 27-31 s trace windows**. The transients
   in play are <=235 MiB chunk cycles.
2. The OOM is a **static-footprint squeeze + fragmentation**: weights 25.3 +
   per-layer mamba/SSM caches ~37.7 + pinned KV 17 + misc ~2.5 = ~82.5 GiB
   hard-static, leaving <1 GiB usable headroom under the 0.965 util cap;
   a ~1.1 GiB stack of ordinary chunk transients then cannot place 127 MiB.
3. The pin itself (17 GiB KV) is not the culprit — the fat static item is the
   **per-layer mamba/SSM cache family (~37.7 GiB)**, which the pin math does
   not account for.

## Fix levers (ranked)
1. Shrink/bound the per-layer mamba+conv caches (max-num-seqs, mamba cache
   mode/dtype, K0 slot release frees slots) — biggest static lever.
2. ~~expandable_segments~~ **RETRACTED 2026-11-13: not available on GLM TP4.**
   The live serve script documents why (serve_flash_mtp_fp8_draftq_mtile.sh
   ~line 75): expandable segments are not IPC-exportable and TP4
   custom_all_reduce dies at boot (custom_all_reduce.cuh:164 'invalid
   argument'). Re-enable only with --disable-custom-all-reduce (perf
   tradeoff, not taken). Corroborating: the KV-offload connector also
   rejects expandable_segments:True; TP1 units (leoncca, qwen35, canonical
   GLM KV_OFFLOAD=0) DO use it — the ban is TP4-custom-AR-specific. The
   fragmentation OOM found above is the known price of that constraint.
   Candidate replacement knob: garbage_collection_threshold (allocator GC
   releases cached blocks under pressure — works without expandable
   segments; carries its own GC-pause risk, needs a rig pass).
3. The (b) mixed-granularity port (dequant ws + MLA staging) remains valid
   for the LONG-prefill slope but is NOT what killed this boot; its EV drops
   (plan's own risk note anticipated this).

## Lever investigation (2026-11-13, code-confirmed)
- **Mamba pool IS K-scaled**: MambaManager.max_allocated_blocks_per_request
  = `2 + num_speculative_blocks + num_prefill_checkpoint_blocks` (align mode;
  single_type_kv_cache_manager.py:1711). Pool sizing includes the adaptive-K
  ceiling (3) + checkpoint blocks -> lowering the K ladder ceiling or ckpt
  blocks shrinks the boot-time pool. K0 release (live) frees at runtime.
- **mamba_page_size_padded: dead end** — prod log: "Padding mamba page size
  by 0.70%" (~0.26 GiB of 37.7). Side fact: attention block size raised to
  1152 tokens/page so attention page >= mamba page.
- **max_split_size: mechanism already in-tree** —
  Worker._scoped_allocator_max_split (gpu_worker.py:430) wraps load_model
  (line 611) only; serving-phase fragmentation is uncovered. Global
  `max_split_size_mb` via PYTORCH_CUDA_ALLOC_CONF is the lever (torch
  parseArgs resets absent options — set the full string; see
  vllm/device_allocator/alloc_conf.py docstring).
- **expandable_segments: DOUBLY dead** — second leg code-confirmed
  (vllm/config/vllm.py:1262): any KV connector + expandable_segments:True
  raises unless cumem allocator (VMM remap invalidates registered KV memory).
  GLM runs OffloadingConnector -> even --disable-custom-all-reduce would not
  unlock it. Option C fully closed.
- **gc_threshold: the remaining clean allocator knob** (no remapping, no
  connector interaction). Rig-pass candidate #1; max_split_size global is #2.
- Prod actuals (12:14 boot): attn block 1152 tok; groups (1152,1152,1152,
  1152,8); KV pool 3.75M tok = 3.58x concurrency @1M.

## Tooling gap
Frames in these dumps are unusable (C++-only stacks collapse to the gather
path; user_metadata empty). Next capture should record with python stacks
(_record_memory_history stack mode) so sites are attributable.
