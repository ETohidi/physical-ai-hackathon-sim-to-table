#!/usr/bin/env python3
"""Build results/manifest.json: episode counts per source (sim / cosmos / real) and curation summary.

Usage: python scripts/make_manifest.py --sim datasets/sim --cosmos datasets/sim_cosmos --real datasets/real \
         --reason results/reason_sim.json results/reason_real.json --out results/manifest.json
--sim/--cosmos/--real each accept one or more directories; counts are summed across them.
Counts episodes from LeRobot v2.1 meta/episodes.jsonl (falls back to meta/info.json's total_episodes,
which also covers LeRobot v3.0's per-file chunked layout, then to counting mp4s as a last resort).
"""
import argparse, json, os, glob
from pathlib import Path

def count_episodes_one(root):
    if not root or not os.path.isdir(root):
        return 0
    ej = Path(root) / "meta" / "episodes.jsonl"
    if ej.exists():
        return sum(1 for line in open(ej) if line.strip())
    info = Path(root) / "meta" / "info.json"
    if info.exists():
        n = json.load(open(info)).get("total_episodes")
        if n is not None:
            return n
    return len(glob.glob(str(Path(root) / "videos" / "**" / "*.mp4"), recursive=True))

def count_episodes(roots):
    if not roots:
        return 0
    if isinstance(roots, str):
        roots = [roots]
    return sum(count_episodes_one(r) for r in roots)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", nargs="+"); ap.add_argument("--cosmos", nargs="+"); ap.add_argument("--real", nargs="+")
    ap.add_argument("--reason", nargs="*", default=[])
    ap.add_argument("--dropped", type=int, default=0, help="episodes removed by curation")
    ap.add_argument("--out", default="results/manifest.json")
    a = ap.parse_args()
    m = {
        "team": "Abstraction is All You Need",
        "track": "Physical AI: Sim to Real Pick and Place",
        "policy": "GR00T N1.7-3B (post-trained)",
        "real_data_source": "Real teleop episodes recorded at the event by team Natural Intelligence (shared with permission; 33 recorded, 3 dropped per their labels). Phone reference clips by us.",
        "episodes": {"sim": count_episodes(a.sim), "cosmos": count_episodes(a.cosmos), "real": count_episodes(a.real)},
        "curation": {"tool": "nvidia/cosmos-reason2-8b", "dropped_episodes": a.dropped, "rule": "drop if grasp_success=false or motion_quality<=2", "files": a.reason},
    }
    m["episodes"]["total"] = sum(v for k, v in m["episodes"].items())
    assert m["episodes"]["real"] <= 50, "real teleop cap is 50"
    assert m["episodes"]["total"] <= 300, "total dataset cap is 300"
    assert m["episodes"]["sim"] + m["episodes"]["cosmos"] > 0, "main checkpoint must include sim-derived data"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(m, open(a.out, "w"), indent=2)
    print(json.dumps(m, indent=2))

if __name__ == "__main__":
    main()
