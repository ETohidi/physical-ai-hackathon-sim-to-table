#!/usr/bin/env python3
"""Fidelity table + trust weights from auditor scores (Cosmos Reason, optionally Nemotron).

Usage: python scripts/fidelity_table.py results/reason_sim.json results/reason_cosmos.json results/reason_real.json [--nemotron results/nemotron_*.json] --out results/fidelity_table.md
Trust weight per level = mean(realism/5 * motion_quality/5) normalised so weights sum to 1 (documented formula; the
real level is the reference and always gets the highest weight by construction of the rubric).
"""
import argparse, json, glob
def summarize(path):
    d = json.load(open(path)); eps = [e for e in d["episodes"].values() if e and "error" not in e]
    if not eps: return d["source"], 0, None
    g = sum(1 for e in eps if e.get("grasp_success")) / len(eps)
    m = sum(e.get("motion_quality", 0) for e in eps) / len(eps); r = sum(e.get("realism", 0) for e in eps) / len(eps)
    kept = sum(1 for e in eps if e.get("grasp_success") and e.get("motion_quality", 0) > 2)
    return d["source"], len(eps), {"grasp": g, "motion": m, "realism": r, "kept": kept, "dropped": len(eps) - kept, "trust": (r / 5) * (m / 5)}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--nemotron", nargs="*", default=[]); ap.add_argument("--out", default="results/fidelity_table.md")
    a = ap.parse_args()
    rows = [summarize(f) for f in a.files]; tot = sum(r[2]["trust"] for r in rows if r[2]) or 1
    md = ["# Fidelity table (auditor: Cosmos Reason 2)", "", "| level | n | grasp ok | motion (1-5) | realism (1-5) | kept | dropped | trust weight |", "|---|---|---|---|---|---|---|---|"]
    for src, n, s in rows:
        if s: md.append(f"| {src} | {n} | {s['grasp']:.0%} | {s['motion']:.2f} | {s['realism']:.2f} | {s['kept']} | {s['dropped']} | {s['trust']/tot:.2f} |")
    md += ["", "trust weight = mean(realism/5 x motion/5) per level, normalised. Curation: drop grasp_success=false or motion<=2."]
    if a.nemotron:
        md += ["", "## Second auditor (Nemotron 3.5 Super VL, 8 frames per episode)", "", "| level | n | grasp ok | motion | realism |", "|---|---|---|---|---|"]
        for f in a.nemotron:
            src, n, s = summarize(f)
            if s: md.append(f"| {src} | {n} | {s['grasp']:.0%} | {s['motion']:.2f} | {s['realism']:.2f} |")
    open(a.out, "w").write("\n".join(md) + "\n"); print("\n".join(md))
if __name__ == "__main__": main()
