# Sim to Table with a multi-level digital twin

**Team "Abstraction is All You Need" — Nebius Build Week Hackathon with NVIDIA, Berlin, 18 Oct 2026 — Physical AI track.**

Task: an SO-101 arm picks up a 30 mm cube and places it on the pad, judged on real hardware.

## The idea: levels of detail, like a digital twin

We treat the data pipeline as a twin with levels of detail, the same method we use for network digital twins
(coarse models for scale, fine models for truth, a bridge and a feedback loop between them):

| Level | Role (company metaphor) | What it is | Episodes |
|---|---|---|---|
| Coarse | *Manager* — whole workspace, cheap, schematic look | Isaac Sim 6.1 + Isaac Lab 3.0, scripted expert, cube anywhere on the table, red and blue cube | 83 + 60 |
| Bridge | *Team lead* — dresses the plans in the real look | NVIDIA Cosmos Transfer 2.5 (NIM on our VM), edge-conditioned, motion unchanged, prompt = the real light-box station | 12 |
| Fine | *Engineer* — true but scarce, narrow scope | Real teleop on the event station (recorded by team **Natural Intelligence**, shared with us; 33 episodes, 3 dropped by their labels) | 30 |
| Auditor | *Status report going up* | NVIDIA Cosmos Reason 2 (NIM on our VM) scores every episode of every level with one rubric: grasp ok, motion 1–5, realism 1–5 → curation decision + per-level fidelity table | all |

Feedback up: the auditor's per-level scores become trust weights for the training mix (`results/fidelity_table.md`),
the ablation (no bridge) shows what the bridge adds, and the weakest grid spots are where the next real demos go.

## What we built today

- **Starter port to Isaac Lab 3.0**: the starter (`so101/`, vendored from ddiddi/so101-sim-pickplace-gr00t, Apache-2.0)
  targets an Isaac Lab tag that does not exist; `patches/isaaclab3_api.patch` ports the observation API (`image` → `image_rgb`).
- **Sim data at scale on Nebius**: 83 red-cube + 60 blue-cube episodes (the event cube is blue) on one RTX PRO 6000, front + wrist cameras, LeRobot v2.1.
- **Cosmos on the team GPU**: the hosted API is geo-blocked in Germany, so both NIM containers run on the VM
  (`scripts/cosmos_reason_score.py`, `scripts/cosmos_transfer_batch.py`, `results/station_prompt.txt`).
- **Two policies**: GR00T N1.7-3B post-trained on sim + Cosmos (main, front camera only; `checkpoints/main/checkpoint-1000`) and
  ACT trained on the real episodes (front + side; `checkpoints/act_real`). GR00T's loader rejects LeRobot v3.0 datasets, so the real
  level went to ACT; this is documented in the manifest.
- **Real-arm client with a release supervisor**: `scripts/real_eval_client.py` drives the follower from the GR00T server over ZeroMQ
  and stops once a grasp → release → lift sequence is detected (validated on 29/30 real demos), so a success is not undone.
- **Evidence**: `results/manifest.json` (episode counts per source), `results/reason_*.json`, `results/curation.md`,
  `results/fidelity_table.md`, `results/camera_overlay.png`, `scripts/coverage_heatmap.py` (per-grid-spot success map).

## What we observed on the real arm

The sim-only GR00T checkpoint produced erratic joint targets on the real arm (clamped by the safety limiter): the coarse level
without the fine level, i.e. the sim-to-real gap made visible. The ACT policy trained on the real level moves purposefully but
mislocalises the cube on several spots. The combination (coarse coverage + fine fidelity) is the next turn of the loop:
convert the real v3.0 dataset for GR00T, re-weight by the auditor's trust scores, and re-collect sim with the measured camera pose.

## Reproduce

```
# VM (Nebius, RTX PRO 6000): see so101/README.md and so101/nebius/setup_vm.sh (IsaacLab branch release/3.0.0 + patches/)
cd so101 && source env.sh && NUM_EPISODES=80 ./collect.sh && ./prepare_dataset.sh        # sim level
CUBE_COLOR=blue NUM_EPISODES=60 ./collect.sh                                              # blue cube
bash cosmos_on_vm.sh run                                                                  # Cosmos NIMs on :8001 / :8002
python scripts/cosmos_reason_score.py --videos datasets/<set>/videos --source sim --out results/reason_sim.json
python scripts/cosmos_transfer_batch.py --sim datasets/<set> --out datasets/sim_cosmos --prompt "$(cat results/station_prompt.txt)" --limit 6
MAX_STEPS=1000 GLOBAL_BATCH_SIZE=16 ./finetune.sh                                         # GR00T main
lerobot-train --policy.type=act --dataset.root=datasets/real_share/station_v3 ...          # ACT on the real level
python scripts/fidelity_table.py results/reason_*.json && python scripts/make_manifest.py ...
# Station: ./start_server.sh --port 5556 on the VM, ssh -L 5555:localhost:5556, python scripts/real_eval_client.py --station station.yaml
```

## Credits

Real teleop episodes, station calibration and station notes: team **Natural Intelligence** (same event, same station), shared with permission.
Starter environment: ddiddi/so101-sim-pickplace-gr00t (Apache-2.0). Models: NVIDIA GR00T N1.7, Cosmos Reason 2, Cosmos Transfer 2.5; Isaac Sim / Isaac Lab; Hugging Face LeRobot. Compute: Nebius AI Cloud.
