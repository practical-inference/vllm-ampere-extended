# DSv4.1-Flash overnight report (2026-10-04, ~01:30)

## Live now on :8210 — final champion config
`deepseek-v4.1-flash`, TP4 (GPUs 0/1/2/6), CUDA graphs (FULL_AND_PIECEWISE +
breakable), fp8_ds_mla KV, 524,288 ctx, max-num-seqs 32, util 0.9594,
capture cap 32, no custom all-reduce.
Log: /data/models/dsv41_final.log · full trail: /data/models/dsv41_nightlog.md

Boot command (also in /data/models/bin/dsv41_window.sh, now updated):
```
cd /home/michael/source/vllm-nightly && \
DSV41_ENGRAM_DTYPE=int4 NCCL_NVLS_ENABLE=0 \
CUDA_VISIBLE_DEVICES=<4 GPU UUIDs 0,1,2,6> VLLM_USE_FLASHINFER_SAMPLER=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid nohup .venv/bin/vllm serve /data/models/DeepSeek-V4.1-Flash-W4A16-Engram-AutoRound \
  --tensor-parallel-size 4 --language-model-only --max-model-len 524288 \
  --max-num-seqs 32 --gpu-memory-utilization 0.9594 --kv-cache-dtype fp8_ds_mla \
  --compilation-config '{"max_cudagraph_capture_size":32}' \
  --disable-custom-all-reduce --served-model-name deepseek-v4.1-flash \
  --trust-remote-code --enable-auto-tool-choice \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --port 8210
```
(DSV41_ENGRAM_DTYPE=int4 is MANDATORY — missing it = engram shape assert at load.)

## Impact ladder (each step measured)
| config | c1 tok/s | c16 agg | KV pool | quality |
|---|---|---|---|---|
| eager bf16 (this afternoon) | 9 | 27 | 618K @131K | GSM8K 0.897 full |
| + CUDA graphs (perf8) | 58 | 525 | 515K @131K | 0.900/100q |
| + fp8_ds_mla + 524K + cap32 + 0.9594 (FINAL) | 53–58 | ~520 | **1,414,344 @524K (2.70×)** | **0.895 full / 0.001 invalid** |

## Your use-case (100–300K multi-round), measured on final config
- 140K prompt TTFT: 16.4 s · 300K: 41.7 s (prefill ~8.5K tok/s)
- multi-round follow-up TTFT: **0.4 s** (prefix cache hit)
- decode at 140K ctx: **68 tok/s/stream** (better than short-ctx!)
- needle recall at 140K & 300K with fp8 KV: exact-match hits, incl. restate in round 2
- KV headroom: one 512K session + ~2 more 300K, or ~4× 300K sub-agents

## Tried and rejected (all booted & measured)
- ngram spec-decode: forces V1 model runner → CUDA graph capture dies
  ("capturing stream has unjoined work"). Needs fork work for V2 runner. PARKED.
- max-num-batched-tokens 16384: OOM at dummy-run (10 GiB chunk, no headroom). PARKED.
- --async-scheduling: decode identical (GPU-bound). REVERTED.
- custom all-reduce: crashes on graph replay → --disable-custom-all-reduce required.

## Remaining 2× gap vs GLM-5.3-Flash — honest status
Decode at long ctx 68 vs GLM ~110–120. Cause: SM80 Triton sparse-MLA/indexer
kernel time (launch overhead is gone post-graphs; this is real GPU time) and
DS v4.1's heavier per-token KV records (1.41M vs 3.8M pool on same TP4).
Paths: vendor MTP head (not in checkpoint), ngram spec via V2-runner fork work,
kernel optimization of triton_sparse decode. All = separate projects, per your
"no big dev projects" rule.

## Files
- Night log: /data/models/dsv41_nightlog.md
- Final eval: /data/models/gsm8k_final_full.{json,log} (0.895/0.001, 1319q, 4.9 q/s)
- fp8 100q: /data/models/gsm8k_fp8kv100.{json,log} (0.920)
- Longctx: /data/models/longctx_perf9.log · benches: /tmp/opencode/bench_{decode,longctx}.py
- Server logs kept: dsv41_perf{6..12}.log, dsv41_final.log in /data/models
- GPU4 note: vllm-leoncca-exp.service (:8016, qwen A/B) is a legit systemd unit — left alone.
