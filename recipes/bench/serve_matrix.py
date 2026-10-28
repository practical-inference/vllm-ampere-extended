#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving benchmark matrix for vllm-ampere-extended recipes.

Measures, with fixed sampling and fully echoed configuration:
  single   — sequential requests: per-stream decode tok/s + TTFT
  batch64  — N concurrent requests: aggregate tok/s + TTFT
  longctx  — long prompts: prefill TTFT + decode tok/s
  demo     — one request with per-token arrival timestamps (demo replay input)

Temperature 0, fixed I/O lengths, warmup excluded. Output is a JSON file
with the numbers AND the exact invocation, so every published table can be
reproduced. Compatibility-creating backends report absolute numbers (there
is no dense fallback to compare against for the sparse-attention classes).
"""

import argparse
import json
import sys
import time

from openai import OpenAI

PARA = (
    "The warehouse hums with the low frequency of conveyor bearings while "
    "pallet jacks trace their chalk lines between the racking bays. "
    "Inventory clerks reconcile manifests against the scanner logs, and "
    "the loading dock doors open on their pneumatic arms to let the "
    "afternoon light across the concrete. "
)


def build_prompt(target_tokens):
    # ~1.3 tokens per word for this prose; good enough, usage.prompt_tokens
    # in the response records the exact count.
    words = target_tokens / 1.3
    para_words = len(PARA.split())
    reps = max(1, int(words / para_words))
    return PARA * reps


def one_request(client, model, prompt, out_tokens, seed, record_arrivals=False):
    start = time.monotonic()
    arrivals = []
    usage = None
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=out_tokens,
        temperature=0.0,
        seed=seed,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            arrivals.append(time.monotonic() - start)
        if getattr(chunk, "usage", None):
            usage = chunk.usage
    total = time.monotonic() - start
    ttft = arrivals[0] if arrivals else total
    # Per-stream decode rate is only meaningful when chunks streamed steadily;
    # under load chunks can arrive buffered at the end (span << decode window),
    # which would inflate the rate — guard and mark those invalid.
    decode_tps = None
    if usage and len(arrivals) > 1:
        span = arrivals[-1] - arrivals[0]
        window = total - ttft
        if span > 0.5 * window and span > 0:
            decode_tps = (usage.completion_tokens - 1) / span
    return {
        "ttft_s": round(ttft, 3),
        "total_s": round(total, 3),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "decode_tok_s": round(decode_tps, 1) if decode_tps else None,
        "arrivals_ms": [round(a * 1000) for a in arrivals] if record_arrivals else None,
    }


def scenario(
    client, model, name, prompt, out_tokens, concurrency, seed, arrivals=False
):
    import concurrent.futures

    def run(i):
        return one_request(
            client,
            model,
            prompt,
            out_tokens,
            seed + i,
            record_arrivals=arrivals and i == 0,
        )

    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(run, range(concurrency)))
    wall = time.monotonic() - t0
    ttfts = sorted(r["ttft_s"] for r in results)
    decode = [r["decode_tok_s"] for r in results if r["decode_tok_s"]]
    total_tokens = sum(r["completion_tokens"] or 0 for r in results)
    out = {
        "concurrency": concurrency,
        "wall_s": round(wall, 2),
        "ttft_p50_s": round(ttfts[len(ttfts) // 2], 3),
        "ttft_p90_s": round(ttfts[int(len(ttfts) * 0.9)], 3),
        "decode_tok_s_mean": round(sum(decode) / len(decode), 1) if decode else None,
        "aggregate_tok_s": round(total_tokens / wall, 1),
        "prompt_tokens": results[0]["prompt_tokens"],
        "completion_tokens_each": results[0]["completion_tokens"],
    }
    if arrivals:
        out["demo_arrivals_ms"] = results[0]["arrivals_ms"]
    return out, results


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--in-tokens", type=int, default=512)
    p.add_argument("--out-tokens", type=int, default=256)
    p.add_argument("--long-in-tokens", type=int, default=32768)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--single-requests", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600.0)
    prompt = build_prompt(args.in_tokens)
    long_prompt = build_prompt(args.long_in_tokens)

    for _ in range(args.warmup):
        one_request(client, args.model, build_prompt(128), 32, args.seed)

    # single-stream = truly sequential (concurrency 1); per-stream decode
    # rates at concurrency >1 are unreliable (chunk buffering under load).
    single, single_raw = scenario(
        client, args.model, "single", prompt, args.out_tokens, 1, args.seed
    )
    for _ in range(args.single_requests - 1):
        one_request(client, args.model, prompt, args.out_tokens, args.seed)
    batch, _ = scenario(
        client, args.model, "batch", prompt, args.out_tokens, args.batch, args.seed
    )
    longctx, _ = scenario(
        client, args.model, "longctx", long_prompt, args.out_tokens, 4, args.seed
    )
    demo, _ = scenario(
        client, args.model, "demo", prompt, args.out_tokens, 1, args.seed, arrivals=True
    )

    result = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "in_tokens": args.in_tokens,
            "out_tokens": args.out_tokens,
            "long_in_tokens": args.long_in_tokens,
            "batch": args.batch,
            "single_requests": args.single_requests,
            "warmup": args.warmup,
            "seed": args.seed,
            "temperature": 0.0,
            "invocation": " ".join(sys.argv),
        },
        "single": single,
        "batch": batch,
        "longctx": longctx,
        "demo": demo,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
