#!/usr/bin/env python3
"""Closed-loop GR00T eval on the real SO-101 follower (Phase 6, Path A).

Same wire format as so101/scripts/eval_in_sim.py (gr00t_client.Gr00tClient), but the observation
comes from the real arm via LeRobot: front camera 640x480 RGB + observation.state in
use_degrees=False units (arm -100..100, gripper 0..100), which is the dataset convention, so no
unit conversion. Each query returns a 16-step chunk; the first --action_horizon steps run at --fps.
A supervisor (ReleaseSupervisor) stops the policy once a grasp -> release -> retreat is seen and holds the
pose ("DONE: release detected"); --max_seconds (default 100, judges cap at 120) ends the run otherwise.
Ctrl-C stops, moves the arm back to its start (rest) pose and disconnects.

Start the server first (VM: ./start_server.sh, Mac: ssh -L 5555:localhost:5555 ubuntu@$VM_IP), then:

  # message-contract check, no robot, no camera
  python repo/scripts/real_eval_client.py --dry-run
  # real arm (ports/camera index from the station file)
  python repo/scripts/real_eval_client.py --station private/assets/real_demos_share/station.yaml \
      --record_dir private/assets/eval
"""
import argparse, importlib.util, os, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LO = np.array([-100.0] * 5 + [0.0], dtype=np.float32)
HI = np.array([100.0] * 6, dtype=np.float32)
# observation.state gripper min/max of the real demos (station_v3/meta/stats.json). In those demos the
# gripper reads ~2 empty-closed, ~17 closed on a 30 mm cube, ~26 on the 40 mm one, 38..99 open (one release
# only opened to ~39), so "closed" is the bottom 30 % of the range and "open" anything above 35 % (hysteresis).
GRIP_MIN, GRIP_MAX = 1.42, 98.71
GRIP_CLOSED_BELOW = GRIP_MIN + 0.30 * (GRIP_MAX - GRIP_MIN)  # 30.6
GRIP_OPEN_ABOVE = GRIP_MIN + 0.35 * (GRIP_MAX - GRIP_MIN)  # 35.5


class ReleaseSupervisor:
    """Detects grasp -> release -> retreat on the measured state and says when to stop the policy.

    Phases: wait_open (the rest pose has the gripper closed, so a grasp only counts after it has opened)
    -> wait_grasp (closed >= closed_s) -> grasped -> released (open again) -> done once shoulder_lift or
    elbow_flex has moved >= lift_delta back toward the start pose for >= lifted_s (on the way home one of the
    two often moves away first). Closing again after a release counts as a re-grasp only before the retreat
    has started (the demos close the gripper while returning to rest).
    """

    def __init__(self, start, closed_below, open_above, closed_s, lift_delta, lifted_s):
        self.start, self.closed_below, self.open_above = start, closed_below, open_above
        self.closed_s, self.lift_delta, self.lifted_s = closed_s, lift_delta, lifted_s
        self.phase, self.since, self.release_dist = "wait_open", None, np.zeros(2)

    def _dist(self, q):
        return np.abs(np.array([q[1] - self.start[1], q[2] - self.start[2]]))

    def _go(self, phase, t, q):
        print(f"[sup] {self.phase} -> {phase} (gripper {q[5]:.1f})", flush=True)
        self.phase, self.since = phase, None

    def update(self, q, t) -> bool:
        g = q[5]
        if self.phase == "wait_open":
            if g > self.open_above:
                self._go("wait_grasp", t, q)
        elif self.phase == "wait_grasp":
            if g < self.closed_below:
                self.since = self.since if self.since is not None else t
                if t - self.since >= self.closed_s:
                    self._go("grasped", t, q)
            else:
                self.since = None
        elif self.phase == "grasped":
            if g > self.open_above:
                self._go("released", t, q)
                self.release_dist = self._dist(q)
        elif self.phase == "released":
            if g < self.closed_below and self.since is None:
                self._go("wait_grasp", t, q)
                self.since = t
            elif np.max(self.release_dist - self._dist(q)) >= self.lift_delta:
                self.since = self.since if self.since is not None else t
                if t - self.since >= self.lifted_s:
                    return True
            else:
                self.since = None
        return False


def load_gr00t_client():
    # Load the module by path: importing the so101_pickplace package pulls in gymnasium/Isaac Lab.
    path = HERE.parent / "so101" / "so101_pickplace" / "gr00t_client.py"
    spec = importlib.util.spec_from_file_location("gr00t_client", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", help="station.yaml; fills --robot_port / --camera_index if not given")
    ap.add_argument("--robot_port")
    ap.add_argument("--robot_id", default="station_follower")
    ap.add_argument("--camera_index", type=int)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--policy_host", default="localhost")
    ap.add_argument("--policy_port", type=int, default=5555)
    ap.add_argument("--task_text", default="pick up the cube and place it on the green pad")
    ap.add_argument("--action_horizon", type=int, default=8, help="Predicted steps executed per policy query.")
    ap.add_argument("--max_seconds", type=float, default=100, help="Stop after this long (judges cap at 120 s; 0 = until Ctrl-C).")
    ap.add_argument("--grip_closed_below", type=float, default=GRIP_CLOSED_BELOW, help="Gripper state below this = closed.")
    ap.add_argument("--grip_open_above", type=float, default=GRIP_OPEN_ABOVE, help="Gripper state above this = open.")
    ap.add_argument("--grasp_s", type=float, default=0.5, help="Closed this long counts as a grasp.")
    ap.add_argument("--lift_delta", type=float, default=15.0,
                    help="After release, shoulder_lift or elbow_flex must come this much closer to the start pose ...")
    ap.add_argument("--lifted_s", type=float, default=1.0, help="... for this long -> DONE, hold pose.")
    ap.add_argument("--no_supervisor", action="store_true", help="Disable the release-detection stop.")
    ap.add_argument("--max_relative_target", type=float, default=10.0,
                    help="Per-step joint clamp in normalised units (LeRobot safety); 0 disables.")
    ap.add_argument("--record_dir", default="", help="Save the front-camera stream as an mp4 here.")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="No robot/camera: fake state + gray image, checks the server message contract.")
    ap.add_argument("--dry_run_queries", type=int, default=5)
    a = ap.parse_args()
    if a.station:
        import yaml
        st = yaml.safe_load(open(a.station))
        a.robot_port = a.robot_port or st["robot"]["port"]
        if a.camera_index is None:
            a.camera_index = int(st["cameras"]["front"]["index_or_path"])
    if not a.dry_run and (not a.robot_port or a.camera_index is None):
        ap.error("need --robot_port and --camera_index (or --station)")
    return a


def check_modality(client):
    mc = client.get_modality_config()
    got = {k: list(mc[k]["modality_keys"]) for k in ("video", "state", "action") if k in mc}
    print("[real] server modality config:", got, flush=True)
    if "front" not in got.get("video", []):
        raise SystemExit(f"checkpoint has no 'front' camera: {got}")
    extra = [k for k in got["video"] if k != "front"]
    if extra:
        raise SystemExit(f"checkpoint also expects cameras {extra}; this client sends front only")
    for k in ("state", "action"):
        if got.get(k) != ["single_arm", "gripper"]:
            raise SystemExit(f"checkpoint {k} keys {got.get(k)} != ['single_arm', 'gripper']")


def build_obs(add_dims, front, state, task):
    obs = {
        "video": {"front": np.ascontiguousarray(front, dtype=np.uint8)},
        "state": {"single_arm": state[:5].astype(np.float32), "gripper": state[5:6].astype(np.float32)},
        "language": {"annotation.human.task_description": task},
    }
    return add_dims(obs)


def get_chunk(client, obs):
    action, _ = client.get_action(obs)
    chunk = np.concatenate([action["single_arm"][0], action["gripper"][0]], axis=-1)  # (H, 6) normalised
    return np.clip(chunk, LO, HI)


def dry_run(a, gc, client):
    state = np.zeros(6, dtype=np.float32)
    front = np.full((a.height, a.width, 3), 128, dtype=np.uint8)
    lat = []
    for i in range(a.dry_run_queries):
        t0 = time.perf_counter()
        action, _ = client.get_action(build_obs(gc.add_batch_time_dims, front, state, a.task_text))
        lat.append(1e3 * (time.perf_counter() - t0))
        if i == 0:
            print("[dry] action keys/shapes:", {k: np.asarray(v).shape for k, v in action.items()}, flush=True)
        chunk = np.concatenate([action["single_arm"][0], action["gripper"][0]], axis=-1)
        print(f"[dry] query {i + 1}: chunk {chunk.shape}, step0 {np.round(chunk[0], 1).tolist()}, "
              f"{lat[-1]:.0f} ms", flush=True)
    assert chunk.ndim == 2 and chunk.shape[1] == 6, f"unexpected chunk shape {chunk.shape}"
    if chunk.shape[0] < a.action_horizon:
        print(f"[dry] WARNING: chunk has {chunk.shape[0]} steps < action_horizon {a.action_horizon}")
    print(f"[dry] contract OK, inference avg {np.mean(lat):.0f} ms", flush=True)


def move_to(robot, target, seconds=3.0, fps=30):
    obs = robot.get_observation()
    start = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
    n = max(1, int(seconds * fps))
    for i in range(1, n + 1):
        q = start + (target - start) * i / n
        robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})
        time.sleep(1.0 / fps)


def run_robot(a, gc, client):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cfg = SO101FollowerConfig(
        port=a.robot_port, id=a.robot_id, use_degrees=False,
        max_relative_target=a.max_relative_target or None,
        cameras={"front": OpenCVCameraConfig(index_or_path=a.camera_index, width=a.width, height=a.height, fps=a.fps)},
    )
    robot = SO101Follower(cfg)
    robot.connect()  # if the motors disagree with the calibration file: press ENTER (never 'c')
    rec, rest = None, None
    try:
        obs = robot.get_observation()
        rest = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
        print("[real] rest pose:", np.round(rest, 1).tolist(), flush=True)
        if a.record_dir:
            os.makedirs(a.record_dir, exist_ok=True)
            path = os.path.join(a.record_dir, time.strftime("real_eval_%H%M%S.mp4"))
            rec = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}",
                 "-r", str(a.fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", path],
                stdin=subprocess.PIPE,
            )
            print(f"[real] recording {path}", flush=True)
        sup = None if a.no_supervisor else ReleaseSupervisor(
            rest, a.grip_closed_below, a.grip_open_above, a.grasp_s, a.lift_delta, a.lifted_s)
        client.reset()
        dt, t_start, steps, lat, done = 1.0 / a.fps, time.perf_counter(), 0, [], False
        print("[real] running, Ctrl-C to stop", flush=True)
        while not done and (not a.max_seconds or time.perf_counter() - t_start < a.max_seconds):
            state = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
            t0 = time.perf_counter()
            chunk = get_chunk(client, build_obs(gc.add_batch_time_dims, obs["front"], state, a.task_text))
            lat.append(1e3 * (time.perf_counter() - t0))
            for h in range(min(a.action_horizon, chunk.shape[0])):
                tick = time.perf_counter()
                robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, chunk[h])})
                obs = robot.get_observation()
                if rec is not None:
                    rec.stdin.write(np.ascontiguousarray(obs["front"]).tobytes())
                steps += 1
                q = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
                if sup is not None and sup.update(q, time.perf_counter()):
                    done = True
                    break
                time.sleep(max(0.0, dt - (time.perf_counter() - tick)))
            if steps % (a.fps * 5) < a.action_horizon:
                print(f"[real] {steps} steps, {time.perf_counter() - t_start:.0f} s, "
                      f"inference avg {np.mean(lat):.0f} ms", flush=True)
        if done:
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})
            print(f"DONE: release detected after {time.perf_counter() - t_start:.1f} s, holding pose "
                  "(Ctrl-C returns to rest)", flush=True)
            while True:
                time.sleep(dt)
                if rec is not None:
                    rec.stdin.write(np.ascontiguousarray(robot.cameras["front"].read_latest()).tobytes())
        else:
            print(f"[real] max_seconds={a.max_seconds:.0f} reached", flush=True)
    except KeyboardInterrupt:
        print("\n[real] stop requested", flush=True)
    finally:
        if rec is not None:
            rec.stdin.close()
            rec.wait()
        try:
            if rest is not None:
                print("[real] returning to rest pose", flush=True)
                move_to(robot, rest, fps=a.fps)
        finally:
            robot.disconnect()


def main():
    a = parse_args()
    gc = load_gr00t_client()
    client = gc.Gr00tClient(a.policy_host, a.policy_port)
    try:
        if not client.ping():
            raise SystemExit(f"policy server not reachable at {a.policy_host}:{a.policy_port} "
                             "(start_server.sh on the VM + ssh -L 5555:localhost:5555)")
        check_modality(client)
        dry_run(a, gc, client) if a.dry_run else run_robot(a, gc, client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
