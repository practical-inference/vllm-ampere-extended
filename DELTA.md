<!-- markdownlint-disable MD060 -->
# Delta vs upstream vLLM

`main` = upstream `main` (nightly merge) + the commits below: 65 source commits (55 files changed, 4025 insertions(+), 204 deletions(-)).

Diff base: upstream commit `07fa88c0d1` — the tip this tree last merged from origin. Reproduce at this commit with:
`git log --no-merges 07fa88c0d1..f0adffb23c -- vllm/ csrc/` and `git diff --stat 07fa88c0d1..f0adffb23c -- vllm/ csrc/`.

Areas upstream does not ship for this hardware are what this distribution exists to carry; general fixes are contributed upstream first (see README). Regenerated automatically by the nightly integrate.

| Area | Commits |
|---|---|
| sparse attention (TRITON_MLA_SPARSE) | 22 |
| SM80 / pre-SM89 fp8 fallbacks | 12 |
| general fixes | 10 |
| spec-decode fixes | 9 |
| model bring-up (GLM5Next / Qwen4Exp / DSV41) | 7 |
| KV-cache / offload fixes | 5 |

<details><summary>Full commit list</summary>

- [Bugfix] lint: bring the delta through the fork pre-commit gate
- [Build] Remove orphaned generated marlin stable-ABI sources
- checkpoint: MLA prefill SOL benchmarks + SM80 marlin kernel variants
- [Scheduler] Proactive KV cache valve to prevent full-pool churn
- [Instrumentation] Log KV admission failures with full block accounting
- Revert always-transfer hunk in DraftTokensHandler; restore fast path
- [Bugfix] Clamp spec-decode sample index (PP>1); permit fused AR-norm in graph capture
- [Bugfix][Qwen4Exp] Disable fused QSA prepare for fp8 KV on SM80
- [Qwen4Exp] Add k=64 persistent_topk dispatch and config gate
- [Qwen4Exp] Read 512 QSA tokens via block k=128
- fix: remove unmatched draft_tokens broadcast in PPHandler.receive
- [Bugfix] Do not lazily create the FlashInfer AR workspace inside cudagraph capture
- [Hardware] Pack fp8 indexer Q as uint8 bytes on pre-SM89
- Fix draft model PP and drafter None guard for spec decode with PP
- Add TRITON_MLA_SPARSE as BF16-KV sparse fallback on SM12x
- [Perf][DSV41] Chunk prefill attention calls to bound split-KV scratch
- [Bugfix][DSV41] Flatten decode SWA indices before building the gather mask
- [Perf][DSV41] Add int4 engram table path via DSV41_ENGRAM_DTYPE=int4
- [Core] Add draft_vocab_mask to restrict which tokens a draft may propose
- [Bugfix][DSpark] Resolve draft weight names against AutoRound quant metadata
- [Perf][DSA] Route high-row long-context decode top-k to per_row
- [Bugfix][Qwen4Exp] Make pinned PLE fp8 lookup work on sm80
- [Bugfix] Bound CoW block-copy transient to avoid OOM on mamba state rows
- feat: skip_draft_when_k0 for DFlash + fixed-width tier validation
- feat: opt-in skip_draft_when_k0 for AutoRegressiveSpeculator
- [Bugfix][MRV2] Respect dynamic K=0 from scheduler in AutoRegressiveSpeculator
- [Bugfix] dynsd: fix V2 speculator decode cudagraph capture with per-batch-size K
- [Core] QSA: support SimpleCPUOffloadConnector KV offloading
- [Bugfix][QSA] fp8 KV on sub-SM89: Triton cannot load fp8e4nv on A100
- [Bugfix][KV Cache] Software-encode e4m3 in plain-row rope insert for SM80
- [Core] TRITON_MLA_SPARSE attention backend for DeepSeek V4.1 on SM80
- [Core] SM80 Lightning Indexer for DeepSeek V4.1
- [Core] V4.1 584B record dequant + attn_sink in Triton sparse MLA
- [Core] Software e4m3 encode for DeepSeek V4 compressor writes on SM80
- [GLM5Next] NoPE compact 528B fp8_ds_mla pages
- [Spec Decode] Fix ZeroDivisionError capturing dynamic-SD speculator graphs
- [GLM5Next] NoPE (rope-free) sparse MLA on TRITON_MLA_SPARSE, bf16 + fp8_ds_mla
- [GLM5Next] Software e4m3 encode for SM80 kpool quant kernels
- [Bugfix][Spec Decode] Unblock MTP under pipeline parallelism
- [GLM5Next] ponytail marker at SM80 cache-insert split
- [GLM5Next] Software e4m3 encode for SM80 indexer-q in fused_q
- [GLM5Next] getattr-guard prefill in _use_sparse_mha
- [GLM5Next] SM80 fallback for fused fp8 cache writes
- [GLM5Next] Triton FP8 fallback for kpool indexer on SM80
- [Misc] No-op record_logical_topk_ready for TritonMLASparseImpl
- [Misc] Make rebased fork delta mypy-clean
- [Misc] Satisfy pydocstyle D-rules in Triton kernel files
- [Bugfix][DSA] Arch-gate deep_gemm paged-MQA logits metadata
- [Core] Skip fp8 pack-seq jit warmup on sub-SM89 archs
- [Bugfix][KV Offload] Stamp cross-stage KV block geometry for PP
- [Core] Backfill vllm_config in TritonMLASparseMetadataBuilder
- [Core] Gate fused indexer Q-quant to SM89+ on the rebased tree
- [Core] Port persistent paged-MQA logits scratch to rebased tree
- [Bugfix][KV Offload] Fix shared-region races and pinning under PP
- [Core] Restore mask_empty_context Triton kernel (TRITON_MLA_SPARSE backend dep)
- [Core] Serve GlmMoeDsa via the deepseek_v2 implementation on the rebased tree
- [Core] Make PP sampled-token echo async on side streams
- [Bugfix] int64 logits-row offsets in paged fp8 MQA logits stores
- [Bugfix] int64 logits-row offsets in fp8 MQA logits prefill stores
- [Bugfix] int64 workspace-row offsets in sparse MLA 2-pass prefill
- [Core] Port SM80 sparse-MLA dequant tiling, fused-proto dispatch-off, workspace prealloc
- [Bugfix] Fix int32 overflow in paged MQA logits block addressing
- [Bugfix] Fix paged MQA logits tail store
- [Core] Declare MultipleOf(64) block sizes for TRITON_MLA_SPARSE
- [Core] Add TRITON_MLA_SPARSE backend for SM80/SM121 sparse MLA support

</details>
