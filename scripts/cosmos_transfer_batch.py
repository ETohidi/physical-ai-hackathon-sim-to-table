#!/usr/bin/env python3
"""Re-render sim episode videos toward the real station look with Cosmos Transfer 2.5 (NIM on the VM).

Run ON THE VM (session A) after the NIM container is up:
  docker login nvcr.io -u '$oauthtoken' -p "$NVIDIA_API_KEY"
  docker run -d --name cosmos-transfer --runtime=nvidia --gpus all --ulimit nofile=65536:65536 \
     -e NGC_API_KEY=$NVIDIA_API_KEY -v ~/.cache/nim:/opt/nim/.cache -u $(id -u) -p 8001:8000 \
     nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:latest        # check tag with: ngc registry image list 'nvcr.io/nim/nvidia/cosmos**'
  curl -s localhost:8001/v1/health/ready   # wait for "ready" (first start downloads weights, several minutes)

  # serve the sim videos over HTTP so the NIM can fetch them by URL
  (cd datasets/sim && python3 -m http.server 8080 --bind 0.0.0.0 &)
  python3 scripts/cosmos_transfer_batch.py --sim datasets/sim --out datasets/sim_cosmos \
      --prompt "$(cat results/station_prompt.txt)" --limit 24

Edge conditioning: control_weight 1.0 with no explicit control video lets the NIM derive edges from the input
(structure-preserving: geometry and motion unchanged, so recorded actions stay valid). If the server insists on a
control video, pass --edge-dir with precomputed Canny mp4s (make_edges() below writes them with OpenCV).
Output: datasets/sim_cosmos mirrors datasets/sim (parquet + meta copied, front-camera videos replaced).
"""
import argparse, base64, json, os, shutil, sys, time, glob
from pathlib import Path
import urllib.request

def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def make_edges(src, dst):
    import cv2  # pip install opencv-python-headless
    cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ok, f = cap.read()
        if not ok: break
        e = cv2.Canny(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 100, 200)
        out.write(cv2.cvtColor(e, cv2.COLOR_GRAY2BGR))
    cap.release(); out.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True, help="describe the REAL station: table, lighting, cube, pad, camera")
    ap.add_argument("--nim", default="http://localhost:8001/v1/infer")
    ap.add_argument("--http", default="http://localhost:8080", help="where datasets/sim is served")
    ap.add_argument("--camera-key", default="observation.images.front")
    ap.add_argument("--resolution", default="480")
    ap.add_argument("--control-weight", type=float, default=1.0)
    ap.add_argument("--edge-dir", default=None, help="precomputed edge videos (same relative paths); omit to let NIM derive edges")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    sim, out = Path(a.sim), Path(a.out)
    if not out.exists():
        shutil.copytree(sim, out, ignore=shutil.ignore_patterns("*.mp4"))
    vids = sorted(glob.glob(str(sim / "videos" / "**" / a.camera_key / "*.mp4"), recursive=True))[: a.limit]
    log_path = out / "cosmos_transfer_log.json"
    log = json.load(open(log_path)) if log_path.exists() else {}
    for i, v in enumerate(vids):
        rel = os.path.relpath(v, sim)
        dst = out / rel
        if dst.exists() and rel in log:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        body = {"prompt": a.prompt, "video": f"{a.http}/{rel}", "resolution": a.resolution, "seed": a.seed,
                "edge": {"control_weight": a.control_weight}}
        if a.edge_dir:
            body["edge"]["control"] = f"{a.http.rstrip('/')}/../{a.edge_dir}/{rel}"
        t0 = time.time()
        try:
            r = post(a.nim, body)
            open(dst, "wb").write(base64.b64decode(r["b64_video"]))
            log[rel] = {"seconds": round(time.time() - t0, 1), "seed": r.get("seed", a.seed)}
            print(f"[{i+1}/{len(vids)}] {rel} ok in {log[rel]['seconds']} s")
        except Exception as e:
            log[rel] = {"error": str(e)[:300]}
            print(f"[{i+1}/{len(vids)}] {rel} FAILED: {e}", file=sys.stderr)
        json.dump(log, open(log_path, "w"), indent=2)
    # episodes whose front video was not transferred keep no video -> drop them from the cosmos dataset
    kept = [k for k, v in log.items() if "error" not in v]
    print(f"transferred {len(kept)}/{len(vids)}; NOTE: other camera keys (wrist) are copied unchanged by merge step")

if __name__ == "__main__":
    main()
