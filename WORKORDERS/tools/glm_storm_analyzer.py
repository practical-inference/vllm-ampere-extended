#!/usr/bin/env python3
"""GLM storm/collapse analyzer: parse the service log heartbeats.

Usage: glm_storm_analyzer.py <service.log> [--since HH:MM:SS] [--boot HH:MM:SS]
Outputs: collapse windows (gen<20 tok/s, >=3 consecutive), deferral stats,
storm intensity (max prompt tok/s per window), Running distribution.
"""
import re, sys, collections

GIB = 1024 ** 3

def main():
    log = sys.argv[1]
    since = "00:00:00"
    if "--since" in sys.argv:
        since = sys.argv[sys.argv.index("--since") + 1]
    txt = open(log, errors="replace").read()
    hb = []
    for m in re.finditer(
        r"(11-1\d) (\d\d:\d\d:\d\d).*?Avg prompt throughput: ([\d.]+) tokens/s, "
        r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, "
        r"Waiting: (\d+) reqs(?:, Deferred: (\d+) reqs)?, GPU KV cache usage: ([\d.]+)%", txt):
        # --since accepts "HH:MM:SS" (time-of-day only) or "MM-DD HH:MM:SS"
        if len(since) > 8:
            key = m.group(1) + " " + m.group(2)
            if key < since:
                continue
            t = m.group(2)
        else:
            t = m.group(2)
            if t < since:
                continue
        hb.append((t, float(m.group(3)), float(m.group(4)), int(m.group(5)),
                   int(m.group(6)), int(m.group(7) or 0), float(m.group(8))))
    if not hb:
        print("no heartbeats in range"); return
    print(f"heartbeats: {len(hb)}  span {hb[0][0]} -> {hb[-1][0]}")
    runs = [h[3] for h in hb]
    print(f"Running: max={max(runs)} mean={sum(runs)/len(runs):.1f}")
    # collapse windows
    wins, cur = [], []
    for h in hb:
        if h[2] < 20:
            cur.append(h)
        else:
            if len(cur) >= 3:
                wins.append(cur)
            cur = []
    if len(cur) >= 3:
        wins.append(cur)
    real = [w for w in wins if any(x[3] > 0 or x[4] > 0 or x[1] > 100 for x in w)]
    print(f"collapse windows (>=30s at gen<20 tok/s): {len(wins)} total, {len(real)} with traffic")
    for w in real[-8:]:
        p_max = max(x[1] for x in w)
        kv = (min(x[6] for x in w), max(x[6] for x in w))
        print(f"  {w[0][0]}->{w[-1][0]} ({len(w)*10}s): prompt_max={p_max:.0f} "
              f"KV={kv[0]:.0f}-{kv[1]:.0f}% Run={min(x[3] for x in w)}-{max(x[3] for x in w)} "
              f"Wait_max={max(x[4] for x in w)} Def_max={max(x[5] for x in w)}")
    # deferral stats
    d = [(h[0], h[4] + h[5]) for h in hb]
    nz = [x for x in d if x[1]]
    print(f"wait/defer non-zero: {len(nz)}/{len(d)} ({100*len(nz)/len(d):.1f}%), max={max((x[1] for x in nz), default=0)}")
    # gen throughput picture
    gens = [h[2] for h in hb]
    print(f"gen tok/s: mean={sum(gens)/len(gens):.0f} p50={sorted(gens)[len(gens)//2]:.0f} max={max(gens):.0f}")

if __name__ == "__main__":
    main()
