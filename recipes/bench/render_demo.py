#!/usr/bin/env python3
"""Render a token-arrival replay (GIF) from serve_matrix.py demo output.

Input: one or two serve_matrix JSON files (--fork, --stock). Each panel
replays that stack's per-token arrivals in real time from its own TTFT, so
the viewer sees relative responsiveness side by side. A stock panel that
never booted renders as an empty track labeled with the failure.

Output: an animated GIF (pillow writer — no ffmpeg dependency).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def load_trace(path, label):
    with open(path) as f:
        d = json.load(f)
    arrivals = d["demo"]["demo_arrivals_ms"]
    if not arrivals:
        return {"label": label, "booted": False, "arrivals": []}
    # Normalize to the stack's own first token (TTFT shown as offset text).
    t0 = arrivals[0]
    return {
        "label": label,
        "booted": True,
        "arrivals": [a - t0 for a in arrivals],
        "ttft_ms": t0,
        "tok_s": d["demo"]["decode_tok_s_mean"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fork", required=True, help="serve_matrix JSON (fork)")
    p.add_argument("--stock", help="serve_matrix JSON (stock; omit if it never booted)")
    p.add_argument("--stock-failure", help="text file with the stock failure excerpt")
    p.add_argument("--out", required=True, help="output .gif path")
    p.add_argument("--title", default="qwen3.8-flash-next on one A100 — token arrival replay")
    args = p.parse_args()

    traces = [load_trace(args.fork, "vllm-ampere-extended")]
    if args.stock:
        traces.append(load_trace(args.stock, "stock vLLM (upstream)"))
    elif args.stock_failure:
        traces.append({"label": "stock vLLM (upstream)", "booted": False,
                       "arrivals": [],
                       "note": open(args.stock_failure).read(200)})

    n_panels = len(traces)
    duration_ms = max((t["arrivals"] or [0])[-1] for t in traces) + 1500
    fps = 20
    frames = int(duration_ms / 1000 * fps)

    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 1.9 * n_panels), squeeze=False)
    fig.suptitle(args.title, fontsize=11)
    artists = []
    for ax, t in zip(axes[:, 0], traces):
        ax.set_xlim(0, duration_ms / 1000)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("seconds (replayed in real time)" if ax is axes[-1, 0] else "")
        if t["booted"]:
            n = len(t["arrivals"])
            ax.set_ylabel(f"{t['label']}\n{n} tokens", rotation=0, ha="right", va="center",
                          fontsize=9)
            ax.set_xlim(-0.3, duration_ms / 1000)
            dots = ax.plot([], [], "o", ms=3, color="#2a72d4")[0]
            artists.append((ax, dots, t, None))
        else:
            ax.set_ylabel(t["label"], rotation=0, ha="right", va="center", fontsize=9)
            note = t.get("note", "did not boot")
            ax.text(0.02, 0.5, note, fontsize=8, family="monospace", va="center")
            ax.set_xticks([])
            artists.append((ax, None, t, note))

    def update(frame):
        now_s = frame / fps
        for ax, dots, t, note in artists:
            if not t["booted"]:
                continue
            visible = [i for i, a in enumerate(t["arrivals"]) if a / 1000 <= now_s]
            if visible:
                xs = [t["arrivals"][i] / 1000 for i in visible]
                dots.set_data(xs, [0.5] * len(xs))
        return []

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(args.out, writer=PillowWriter(fps=fps))
    print(f"wrote {args.out} ({frames} frames, {duration_ms/1000:.1f}s replay)")


if __name__ == "__main__":
    main()
