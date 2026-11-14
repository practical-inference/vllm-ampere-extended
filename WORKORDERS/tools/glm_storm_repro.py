#!/usr/bin/env python3
"""GLM prefill-storm repro: reproduce the defer->burst->storm cycle.

Phases:
  populate  — N sessions each cold-prefill a distinct big context (prefix cache + state)
  multiround— each session sends short follow-up turns (prefix-cache hits)
  storm     — K NEW big prompts land in a tight window while canary decodes
  canary    — one continuous light decode stream throughout; its inter-token
              timeline IS the starvation metric (the 'minimum sacrifice' gauge)

Emits one JSON line per event: {phase, session, ttft_s, wall_s, tok_s, ...}
Canary emits: {phase: "canary", t_s, chunk_i, gap_s, tok_s_window}
"""
import argparse, json, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import requests

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform "
         "victor whiskey xray yankee zulu one two three four five six seven "
         "eight nine").split()
TOK_PER_WORD = 1.513


def build_prompt(approx_tokens: int, seed: int, suffix: str = "") -> str:
    rng = random.Random(seed)
    n_words = int(approx_tokens / TOK_PER_WORD)
    return ("Transcript follows. End with ACK: " + " ".join(
        rng.choice(WORDS) for _ in range(n_words)) + suffix)


def stream_one(base, model, prompt, max_tokens, timeout, label, out,
               stop=None, record_gaps=False):
    t0 = time.perf_counter()
    ttft = None
    gaps = []
    last = None
    n_chunks = 0
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0, "stream": True}
    try:
        with requests.post(f"{base}/v1/chat/completions", json=payload,
                           stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if stop is not None and stop.is_set():
                    break
                if not line.startswith(b"data:") or b"[DONE]" in line:
                    continue
                chunk = json.loads(line[5:])
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    if last is not None and record_gaps:
                        gaps.append(now - last)
                    last = now
                    n_chunks += 1
    except Exception as e:
        out.put(json.dumps({"phase": label, "error": str(e)[:200]}))
        return
    wall = time.perf_counter() - t0
    rec = {"phase": label, "ttft_s": round(ttft, 3) if ttft else None,
           "wall_s": round(wall, 3), "n_chunks": n_chunks,
           "tok_s": round(n_chunks / (wall - (ttft or 0)), 1) if ttft and wall > ttft else None}
    out.put(json.dumps(rec))
    if record_gaps and gaps:
        gaps.sort()
        out.put(json.dumps({"phase": label + "-gaps", "n": len(gaps),
                            "p50_ms": round(gaps[len(gaps)//2]*1000, 1),
                            "p90_ms": round(gaps[int(len(gaps)*0.9)]*1000, 1),
                            "max_s": round(gaps[-1], 2)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--prefix-tokens", type=int, default=200000)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--storm-requests", type=int, default=4)
    ap.add_argument("--storm-tokens", type=int, default=300000)
    ap.add_argument("--canary-max-tokens", type=int, default=4000)
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    out = []
    emit = lambda rec: (out.append(json.dumps(rec)), print(json.dumps(rec), flush=True))

    stop = threading.Event()
    canary_out = []
    def canary():
        # continuous light decode: short prompt, long generation, gap recording
        stream_one(args.base_url, args.model,
                   "Count from 1 to 999 slowly, one number per line.",
                   args.canary_max_tokens, args.timeout, "canary",
                   canary_out, stop=stop, record_gaps=True)
    canary_t = threading.Thread(target=canary, daemon=True)
    canary_t.start()
    time.sleep(3)

    # Phase 1: populate — cold prefills, sequential-ish (2 at a time)
    print(f"PHASE populate: {args.sessions} x {args.prefix_tokens}tok", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(stream_one, args.base_url, args.model,
                          build_prompt(args.prefix_tokens, 100 + i, f" session {i}"),
                          8, args.timeout, f"populate-{i}", out)
                for i in range(args.sessions)]
        [f.result() for f in futs]

    # Phase 2: multiround — prefix-cache-hit turns
    print(f"PHASE multiround: {args.rounds} rounds", file=sys.stderr)
    for rnd in range(args.rounds):
        with ThreadPoolExecutor(max_workers=args.sessions) as ex:
            futs = [ex.submit(stream_one, args.base_url, args.model,
                              build_prompt(args.prefix_tokens, 100 + i,
                                           f" round {rnd}: summarize in 5 words."),
                              32, args.timeout, f"round{rnd}-{i}", out)
                    for i in range(args.sessions)]
            [f.result() for f in futs]

    # Phase 3: storm — K new big prompts in a tight window
    print(f"PHASE storm: {args.storm_requests} x {args.storm_tokens}tok", file=sys.stderr)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.storm_requests) as ex:
        futs = [ex.submit(stream_one, args.base_url, args.model,
                          build_prompt(args.storm_tokens, 500 + i, f" storm {i}"),
                          8, args.timeout, f"storm-{i}", out)
                for i in range(args.storm_requests)]
        [f.result() for f in futs]
    emit({"phase": "storm-wall", "wall_s": round(time.perf_counter() - t0, 1)})

    time.sleep(5)
    stop.set()
    canary_t.join(timeout=30)
    for rec in canary_out:
        print(rec, flush=True)
    print("STORM-DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
