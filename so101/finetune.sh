#!/bin/bash
# Post-train GR00T N1.7-3B on the sim dataset (single GPU, DGX Spark).
#   ./finetune.sh                          # 4000 steps @ batch 16, checkpoint every 1000
#   MAX_STEPS=10000 GLOBAL_BATCH_SIZE=12 ./finetune.sh
#   ./finetune.sh --resume-from-checkpoint
# Progress: tail -f $FINETUNE_OUTPUT/train.log
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# DATASET_PATHS="rel/or/abs/path1 rel/or/abs/path2 ..." trains on a mix of datasets: GR00T's
# launch_finetune.py takes a single --dataset_path string and splits it on os.pathsep (":" on
# Linux), so multiple paths are passed as one colon-joined --dataset-path value here.
if [ -n "${DATASET_PATHS:-}" ]; then
    read -r -a _DS_IN <<< "$DATASET_PATHS"
    _DS_ABS=()
    for p in "${_DS_IN[@]}"; do
        case "$p" in
            /*) _DS_ABS+=("$p") ;;
            *) _DS_ABS+=("$PROJECT_DIR/$p") ;;
        esac
    done
    DS=$(IFS=:; echo "${_DS_ABS[*]}")
else
    DS="${TRAIN_DATASET:-$DATASET_ROOT}"
fi
IFS=: read -r -a _DS_CHECK <<< "$DS"
for p in "${_DS_CHECK[@]}"; do
    [ -f "$p/meta/modality.json" ] && [ -f "$p/meta/stats.json" ] || { echo "dataset not prepared at $p - run prepare_dataset.sh" >&2; exit 1; }
done
[ -f "$BASE_MODEL/config.json" ] || { echo "base model missing at $BASE_MODEL" >&2; exit 1; }
mkdir -p "$FINETUNE_OUTPUT"
gr00t_activate

export NUM_GPUS=1
export USE_WANDB=0
export MAX_STEPS="${MAX_STEPS:-4000}"            # ~2 s/step at batch 16 on GB10
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

echo "Fine-tuning on $DS -> $FINETUNE_OUTPUT (steps=$MAX_STEPS batch=$GLOBAL_BATCH_SIZE)"
# NVIDIA e2e sim workflow recipe: tune visual backbone + projector + diffusion head, keep the LLM frozen.
exec bash examples/finetune.sh \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DS" \
    --modality-config-path "$MODALITY_CONFIG" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --output-dir "$FINETUNE_OUTPUT" \
    --save-only-model \
    "$@" \
    -- --no-tune-llm --tune-visual --tune-projector --tune-diffusion-model \
    2>&1 | tee -a "$FINETUNE_OUTPUT/train.log"
