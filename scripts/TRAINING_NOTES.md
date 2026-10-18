# Training with multiple datasets (no merge needed)

GR00T's `gr00t finetune` accepts several `--dataset-path` entries (the HF blog recipe for SO-101 shows
multiple paths balanced by `--balance-dataset-weights`). So instead of physically merging LeRobot datasets:

Main checkpoint (sim + cosmos + real):
  DATASET_PATHS="datasets/sim datasets/sim_cosmos datasets/real" ./finetune.sh      # check how repo/so101/finetune.sh passes --dataset-path; edit it to accept a list if needed
Ablation (no cosmos):
  DATASET_PATHS="datasets/sim datasets/real" MAX_STEPS=500 ./finetune.sh

All three datasets must share the same modality config (same camera keys, same state/action names).
datasets/sim_cosmos is a copy of datasets/sim with the front-camera mp4s replaced, so it is automatically consistent.
If the station has only ONE camera, collect sim with the wrist camera removed from the modality config (or drop the
wrist key from all three datasets) BEFORE training — mismatched camera keys = immediate failure.
