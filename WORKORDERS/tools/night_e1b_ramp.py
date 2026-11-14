#!/usr/bin/env python
"""E1b batch ramp: hold c=88 concurrent deep-dive sessions on :8080 with
KV demand (~65K tok each, ~5.7M total) beyond the 5.33M pool, monitor
'Cannot get 2 free blocks' death signature. Canary = chat stream, gaps =
the starvation gauge. Mirrors ab_ramp.py shape."""
import json, queue, random, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
MODEL = "glm-5.3-flash"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 88
CTX = 65000
GEN = 1536
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform "
         "victor whiskey xray yankee zulu one two three four five six seven "
         "eight nine").split()


def kv_usage():
    try:
        m = urllib.request.urlopen(BASE + "/metrics", timeout=5).read().decode()
        v = [l for l in m.splitlines() if "kv_cache_usage_perc" in l and not l.startswith("#")]
        return float(v[-1].split()[-1]) if v else -1
    except Exception:
        return -1


def alive():
    try:
        return urllib.request.urlopen(BASE + "/health", timeout=2).status == 200
    except Exception:
        return False


def session(i, out):
    rng = random.Random(900 + i)
    prompt = " ".join(rng.choice(WORDS) for _ in range(int(CTX / 1.513)))
    t0 = time.perf_counter()
    ttft = None
    n = 0
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": GEN, "temperature": 0}
    try:
        with requests.post(f"{BASE}/v1/completions", json=payload,
                           stream=True, timeout=3600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith(b"data:") or b"[DONE]" in line:
                    continue
                c = json.loads(line[5:])
                if c.get("choices", [{}])[0].get("text"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n += 1
        out.put(json.dumps({"phase": f"sess-{i}", "ttft_s": round(ttft or 0, 3),
                            "wall_s": round(time.perf_counter() - t0, 1), "toks": n}))
    except Exception as e:
        out.put(json.dumps({"phase": f"sess-{i}", "error": str(e)[:200]}))


def canary(out, stop):
    while not stop.is_set():
        t0 = time.perf_counter()
        gaps = []
        last = None
        ttft = None
        payload = {"model": MODEL,
                   "messages": [{"role": "user", "content": "Count from 1 to 999 slowly."}],
                   "max_tokens": 2000, "temperature": 0, "stream": True}
        try:
            with requests.post(f"{BASE}/v1/chat/completions", json=payload,
                               stream=True, timeout=3600) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if stop.is_set():
                        break
                    if not line.startswith(b"data:") or b"[DONE]" in line:
                        continue
                    d = json.loads(line[5:]).get("choices", [{}])[0].get("delta", {})
                    if d.get("content") or d.get("reasoning"):
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        if last is not None:
                            gaps.append(now - last)
                        last = now
        except Exception as e:
            out.put(json.dumps({"phase": "canary", "error": str(e)[:120]}))
            time.sleep(2)
        if gaps:
            s = sorted(gaps)
            out.put(json.dumps({"phase": "canary-gaps", "n": len(s),
                                "p50_ms": round(s[len(s) // 2] * 1000, 1),
                                "p90_ms": round(s[int(len(s) * .9)] * 1000, 1),
                                "max_s": round(s[-1], 2),
                                "ttft_s": round(ttft or 0, 3)}))


def main():
    out = queue.Queue()

    def pump():
        while True:
            print(out.get(), flush=True)

    threading.Thread(target=pump, daemon=True).start()
    stop = threading.Event()
    threading.Thread(target=canary, args=(out, stop), daemon=True).start()
    time.sleep(3)

    print(f"E1b: c={N} x {CTX}-tok sessions ({N * CTX / 1e6:.1f}M tok demand)", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N) as ex:
        futs = [ex.submit(session, i, out) for i in range(N)]
    # with-block exit waits for all sessions to finish
    print(f"all {N} sessions complete t+{time.time() - t0:.0f}s; holding 300s", flush=True)
    for _ in range(60):
        time.sleep(5)
        print(f"  [t+{time.time() - t0:6.1f}s] kv={kv_usage() * 100:.1f}% alive={alive()}", flush=True)
    print(f"E1b FINAL: alive={alive()} kv={kv_usage() * 100:.1f}%", flush=True)
    stop.set()
    print("E1b RUN-DONE", flush=True)


if __name__ == "__main__":
    main()
