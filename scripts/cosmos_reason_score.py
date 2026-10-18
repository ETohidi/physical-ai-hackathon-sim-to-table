#!/usr/bin/env python3
"""Score robot episode videos with NVIDIA Cosmos Reason2 (hosted on build.nvidia.com).

Usage:
  python scripts/cosmos_reason_score.py --videos datasets/sim/videos --source sim --out results/reason_sim.json
  python scripts/cosmos_reason_score.py --videos datasets/real/videos --source real --out results/reason_real.json

Needs NVIDIA_API_KEY in the environment.
Second auditor (Nemotron 3.5 Super VL on Nebius Token Factory), same rubric, for an independent opinion:
  NVIDIA_BASE_URL=$NEBIUS_BASE_URL NVIDIA_API_KEY=$NEBIUS_API_KEY COSMOS_REASON_MODEL=nvidia/Nemotron-3_5-Super-VL \
  python scripts/cosmos_reason_score.py --videos datasets/sim/videos --source sim --out results/nemotron_sim.json --frames 8
(--frames N sends N evenly spaced JPEG frames as image_url parts instead of a video_url, for APIs without video input.) Set NVIDIA_BASE_URL=http://<vm>:8000/v1 to use a local NIM container instead of build.nvidia.com (load private/secrets/.env first).
Each episode mp4 is sent once; the model returns JSON with grasp_success, placed_on_pad,
motion_quality (1-5), realism (1-5) and a short reason. Results are merged into one JSON per source.
Curation decision (see results/curation.md): drop episodes with grasp_success=false or motion_quality<=2.
"""
import argparse, base64, json, os, sys, time, glob
from pathlib import Path
import urllib.request

API = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1") + "/chat/completions"
MODEL = os.environ.get("COSMOS_REASON_MODEL", "nvidia/cosmos-reason2-8b")

PROMPT = (
    "You are auditing a robot-arm demonstration video (SO-101 arm, 30 mm cube, green pad). "
    "Answer ONLY with a JSON object with keys: grasp_success (true/false), placed_on_pad (true/false), "
    "motion_quality (integer 1-5, 5 = smooth and direct), realism (integer 1-5, 5 = indistinguishable from a real camera), "
    "source_guess ('sim' or 'real'), reason (one sentence)."
)

def encode_video(path, max_mb=20):
    size = os.path.getsize(path) / 1e6
    if size > max_mb:
        print(f"skip {path}: {size:.1f} MB > {max_mb} MB (downscale first: ffmpeg -i in.mp4 -vf scale=320:-2 -r 10 out.mp4)", file=sys.stderr)
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def encode_frames(path, n):
    """n evenly spaced frames as base64 JPEG (needs opencv-python-headless)."""
    import cv2
    cap = cv2.VideoCapture(path); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * (total - 1) / max(n - 1, 1)))
        ok, f = cap.read()
        if not ok: continue
        f = cv2.resize(f, (320, 240)); ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append(base64.b64encode(buf.tobytes()).decode())
    cap.release(); return out

def score(path, key, frames=0):
    if frames:
        parts = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in encode_frames(path, frames)]
        if not parts: return None
        content = [{"type": "text", "text": PROMPT + f" The {len(parts)} images are frames of one video in time order."}] + parts
        body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": 300, "temperature": 0.0}
    else:
        b64 = encode_video(path)
        if b64 is None:
            return None
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
            ]}],
            "max_tokens": 300, "temperature": 0.0,
            "media_io_kwargs": {"video": {"fps": 4.0}},
        }
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                txt = json.load(r)["choices"][0]["message"]["content"]
            start, end = txt.find("{"), txt.rfind("}")
            return json.loads(txt[start:end + 1])
        except Exception as e:  # noqa
            print(f"attempt {attempt+1} failed for {path}: {e}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    return {"error": "failed"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="dir with episode mp4s (searched recursively)")
    ap.add_argument("--source", required=True, choices=["sim", "cosmos", "real"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", default="**/*.mp4")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0, help="send N frames as images instead of the video (for Token Factory VLMs)")
    a = ap.parse_args()
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        sys.exit("NVIDIA_API_KEY not set")
    files = sorted(glob.glob(str(Path(a.videos) / a.glob), recursive=True))
    if a.limit:
        files = files[: a.limit]
    out = {"source": a.source, "model": MODEL, "episodes": {}}
    if os.path.exists(a.out):
        out = json.load(open(a.out))
    for i, f in enumerate(files):
        k = os.path.relpath(f, a.videos)
        if k in out["episodes"]:
            continue
        print(f"[{i+1}/{len(files)}] {k}")
        out["episodes"][k] = score(f, key, a.frames)
        json.dump(out, open(a.out, "w"), indent=2)
    eps = [e for e in out["episodes"].values() if e and "error" not in e]
    if eps:
        gs = sum(1 for e in eps if e.get("grasp_success")) / len(eps)
        mq = sum(e.get("motion_quality", 0) for e in eps) / len(eps)
        rl = sum(e.get("realism", 0) for e in eps) / len(eps)
        print(f"{a.source}: n={len(eps)} grasp_success={gs:.2f} motion_quality={mq:.2f} realism={rl:.2f}")

if __name__ == "__main__":
    main()
