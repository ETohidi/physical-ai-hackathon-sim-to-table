#!/usr/bin/env python3
"""3x3 coverage heatmap: policy success per grid spot (sim eval), main vs ablation.

Input: one or more eval JSONL files, one line per episode: {"cube_xy": [x, y], "success": true/false}
(eval_in_sim.py is patched to write this; see patches/eval_log_cube.patch). Grid spots are inferred by
binning x and y into 3 bins each over the observed range (or pass --xlim/--ylim to fix the bins).

Usage: python scripts/coverage_heatmap.py --main results/eval_main.jsonl --ablation results/eval_ablation.jsonl --out results/coverage_heatmap.png
"""
import argparse, json
import numpy as np

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    xy = np.array([r["cube_xy"] for r in rows], dtype=float); ok = np.array([bool(r["success"]) for r in rows])
    return xy, ok

def grid(xy, ok, xlim, ylim):
    xs = np.linspace(xlim[0], xlim[1], 4); ys = np.linspace(ylim[0], ylim[1], 4)
    rate = np.full((3, 3), np.nan); n = np.zeros((3, 3), int)
    for (x, y), s in zip(xy, ok):
        i = min(int(np.searchsorted(xs, x, side="right") - 1), 2); j = min(int(np.searchsorted(ys, y, side="right") - 1), 2)
        i, j = max(i, 0), max(j, 0)
        n[j, i] += 1; rate[j, i] = (0 if np.isnan(rate[j, i]) else rate[j, i] * (n[j, i] - 1)) / n[j, i] + s / n[j, i]
    return rate, n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True); ap.add_argument("--ablation")
    ap.add_argument("--xlim", nargs=2, type=float); ap.add_argument("--ylim", nargs=2, type=float)
    ap.add_argument("--out", default="results/coverage_heatmap.png")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sets = [("main (sim+cosmos+real)", load(a.main))]
    if a.ablation: sets.append(("ablation (no cosmos)", load(a.ablation)))
    allxy = np.vstack([s[1][0] for s in sets])
    xlim = a.xlim or (allxy[:, 0].min(), allxy[:, 0].max()); ylim = a.ylim or (allxy[:, 1].min(), allxy[:, 1].max())
    fig, axes = plt.subplots(1, len(sets), figsize=(4.5 * len(sets), 4), squeeze=False)
    for ax, (name, (xy, ok)) in zip(axes[0], sets):
        rate, n = grid(xy, ok, xlim, ylim)
        im = ax.imshow(rate, vmin=0, vmax=1, cmap="viridis", origin="lower")
        for j in range(3):
            for i in range(3):
                ax.text(i, j, "-" if n[j, i] == 0 else f"{rate[j, i]:.0%}\n(n={n[j, i]})", ha="center", va="center", color="w", fontsize=9)
        ax.set_title(f"{name}\noverall {ok.mean():.0%} (n={len(ok)})"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[0].tolist(), label="success rate"); fig.suptitle("Coverage map: success per grid spot (sim eval)")
    fig.savefig(a.out, dpi=150, bbox_inches="tight"); print("wrote", a.out)
    md = ["| set | overall | per-spot (row-major, bottom-left first) |", "|---|---|---|"]
    for name, (xy, ok) in sets:
        rate, n = grid(xy, ok, xlim, ylim)
        md.append(f"| {name} | {ok.mean():.0%} (n={len(ok)}) | " + ", ".join("-" if np.isnan(v) else f"{v:.0%}" for v in rate.flatten()) + " |")
    open(a.out.rsplit(".", 1)[0] + ".md", "w").write("\n".join(md) + "\n")

if __name__ == "__main__":
    main()
