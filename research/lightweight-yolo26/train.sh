#!/usr/bin/env bash
# Lightweight YOLO26 Research — Launcher Script (Ultralytics Pipeline)
# Trains all 7 experiments using train_ultralytics.py + Ultralytics dataset config.
#
# Usage:
#   bash train.sh               # Run all experiments sequentially
#   bash train.sh --profile     # Only run profiling
#   bash train.sh E1            # Run single experiment E1
#   bash train.sh E1,E2,E3      # Run specific experiments
#
# Assumptions:
#   - COCO 2017 data at $COCO_DIR (default set in cfg/coco.yaml)
#   - Conda/Python env with torch, ultralytics installed
#   - GPU with ≥24 GB VRAM (for batch 32); reduce if OOM

set -euo pipefail

# ============================================================
# CONFIG — EDIT THESE PATHS
# ============================================================
COCO_DIR="${COCO_DIR:-/media/ntnuvip/HardDisk/user01/datasets/coco}"        # Path to COCO 2017
OUTPUT_DIR="${OUTPUT_DIR:-./runs}"             # Output directory for checkpoints + logs
CONDA_ENV="${CONDA_ENV:-py311}"                # Conda environment name
NUM_GPUS="${NUM_GPUS:-1}"                      # Number of GPUs for DDP (≥2 uses torchrun)

# ============================================================
# EXPERIMENT → YAML MAPPING
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CFG_DIR="cfg"

EXPERIMENTS=(
    "E0_baseline:baseline:yolo26n_baseline"
    "E1_reparam:reparam:yolo26n_reparam"
    "E2_gate:gate:yolo26n_gate"
    "E3_combined:combined:yolo26n_combined"
    "E4_neck_ghost:neck-ghost:yolo26n_neck_ghost"
    "E5_neck_light:neck-light:yolo26n_neck_light"
    "E6_neck_wide:neck-wide:yolo26n_neck_wide"
)

declare -A E_MAP
for e in "${EXPERIMENTS[@]}"; do
    key="${e%%:*}"
    rest="${e#*:}"
    val="${rest%%:*}"
    yaml_name="${rest##*:}"
    E_MAP["$key"]="$val|$yaml_name"
done

# ============================================================
# STEP 1: PROFILING
# ============================================================
echo ""
echo "============================================"
echo "  STEP 1: Profiling all models"
echo "============================================"
echo ""

python profile_models.py --device cuda --img-size 640

echo ""
echo "Profiling complete. Results in profiling_results.json"
echo ""

# If --profile only, exit here
if [[ "${1:-}" == "--profile" ]]; then
    echo "Profile-only mode. Exiting."
    exit 0
fi

# ============================================================
# STEP 2: TRAINING
# ============================================================
SELECTED=()
if [[ $# -gt 0 ]]; then
    IFS=',' read -ra INPUT_EXPS <<< "$1"
    for e in "${INPUT_EXPS[@]}"; do
        if [[ -v E_MAP["$e"] ]]; then
            SELECTED+=("$e")
        else
            echo "Unknown experiment: $e. Available: ${!E_MAP[*]}"
            exit 1
        fi
    done
else
    SELECTED=("${!E_MAP[@]}")
fi

echo ""
echo "============================================"
echo "  STEP 2: Training selected experiments"
echo "============================================"
echo ""

for exp_name in "${SELECTED[@]}"; do
    entry="${E_MAP[$exp_name]}"
    model_name="${entry%%|*}"
    yaml_name="${entry##*|}"
    yaml_path="$CFG_DIR/${yaml_name}.yaml"

    echo ""
    echo "============================================"
    echo "  Running: $exp_name ($model_name) — $yaml_path"
    echo "============================================"
    echo ""

    TRAIN_CMD="python train_ultralytics.py \
        --cfg $yaml_path \
        --data $CFG_DIR/coco.yaml \
        --batch 32 \
        --epochs 245 \
        --imgsz 640 \
        --lr 0.001 \
        --device 0 \
        --project $OUTPUT_DIR \
        --name $model_name"

    echo "  Command: $TRAIN_CMD"
    echo ""

    if [[ $NUM_GPUS -gt 1 ]]; then
        torchrun --nproc_per_node=$NUM_GPUS \
            train_ultralytics.py \
            --cfg "$yaml_path" \
            --data "$CFG_DIR/coco.yaml" \
            --batch $((32 / NUM_GPUS)) \
            --epochs 245 \
            --imgsz 640 \
            --lr 0.001 \
            --device "$(seq -s, 0 $((NUM_GPUS - 1)))" \
            --project "$OUTPUT_DIR" \
            --name "$model_name"
    else
        python train_ultralytics.py \
            --cfg "$yaml_path" \
            --data "$CFG_DIR/coco.yaml" \
            --batch 32 \
            --epochs 245 \
            --imgsz 640 \
            --lr 0.001 \
            --device 0 \
            --project "$OUTPUT_DIR" \
            --name "$model_name"
    fi

    echo ""
    echo "--------------------------------------------"
    echo "  Completed: $exp_name"
    echo "--------------------------------------------"
done

echo ""
echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================"
echo ""

# Generate summary from Ultralytics results.csv files
python -c "
import glob, csv, os
results = []
for csv_file in sorted(glob.glob('$OUTPUT_DIR/*/results.csv')):
    model_dir = os.path.dirname(csv_file)
    model_name = os.path.basename(model_dir)
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        last_row = None
        for row in reader:
            last_row = row
    if last_row:
        results.append({
            'model': model_name,
            'mAP50-95': float(last_row.get('metrics/mAP50-95(B)', 0)),
            'mAP50': float(last_row.get('metrics/mAP50(B)', 0)),
            'precision': float(last_row.get('metrics/precision(B)', 0)),
            'recall': float(last_row.get('metrics/recall(B)', 0)),
            'epochs': int(last_row.get('epoch', 0)),
        })
        print(f\"{model_name:25s} | mAP50-95: {results[-1]['mAP50-95']:.3f} | mAP50: {results[-1]['mAP50']:.3f} | epochs: {results[-1]['epochs']}\")
import json
with open('$OUTPUT_DIR/experiment_summary.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSummary saved to \$OUTPUT_DIR/experiment_summary.json')
"