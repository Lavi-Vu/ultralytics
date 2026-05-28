#!/usr/bin/env bash
# Lightweight YOLO26 Research — Launcher Script
# Runs profiling first, then full training for all 7 experiments.
# Configure paths below before running.
#
# Usage:
#   bash train.sh               # Run all experiments sequentially
#   bash train.sh --profile     # Only run profiling
#   bash train.sh E1            # Run single experiment E1
#   bash train.sh E1,E2,E3      # Run specific experiments
#
# Assumptions:
#   - COCO 2017 data at $COCO_DIR
#   - Conda/Python env with torch, thop, pycocotools installed
#   - GPU with ≥24 GB VRAM (for batch 256); reduce to 128 if OOM

set -euo pipefail

# ============================================================
# CONFIG — EDIT THESE PATHS
# ============================================================
COCO_DIR="${COCO_DIR:-/datasets/coco}"        # Path to COCO 2017 (images + annotations)
OUTPUT_DIR="${OUTPUT_DIR:-./runs}"             # Output directory for checkpoints + logs
CONDA_ENV="${CONDA_ENV:-lightweight-yolo26}"   # Conda environment name
NUM_GPUS="${NUM_GPUS:-4}"                      # Number of GPUs for DDP

# ============================================================
# RESOLVE EXPERIMENTS
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

EXPERIMENTS=(
    "E0_baseline:baseline"
    "E1_reparam:reparam"
    "E2_gate:gate"
    "E3_combined:combined"
    "E4_neck_ghost:neck-ghost"
    "E5_neck_light:neck-light"
    "E6_neck_wide:neck-wide"
)

declare -A E_MAP
for e in "${EXPERIMENTS[@]}"; do
    key="${e%%:*}"
    val="${e##*:}"
    E_MAP["$key"]="$val"
done

# ============================================================
# STEP 1: PROFILING
# ============================================================
echo ""
echo "============================================"
echo "  STEP 1: Profiling all models"
echo "============================================"
echo ""

python profile.py --device cuda --img-size 640

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
    model_name="${E_MAP[$exp_name]}"

    echo ""
    echo "============================================"
    echo "  Running: $exp_name ($model_name)"
    echo "============================================"
    echo ""

    # Use torchrun for multi-GPU training
    if [[ $NUM_GPUS -gt 1 ]]; then
        torchrun --nproc_per_node=$NUM_GPUS \
            train.py \
            --model "$model_name" \
            --batch $((256 / NUM_GPUS)) \
            --epochs 300 \
            --data-dir "$COCO_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --img-size 640
    else
        python train.py \
            --model "$model_name" \
            --batch 256 \
            --epochs 300 \
            --data-dir "$COCO_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --img-size 640
    fi

    # Validate best checkpoint
    best_ckpt="$OUTPUT_DIR/$model_name/best.pt"
    if [[ -f "$best_ckpt" ]]; then
        echo ""
        echo "Validating $exp_name best checkpoint..."
        python val.py \
            --model "$model_name" \
            --weights "$best_ckpt" \
            --data-dir "$COCO_DIR" \
            --img-size 640
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

# Generate summary
python -c "
import json, glob
results = []
for model_dir in sorted(glob.glob('$OUTPUT_DIR/*/best.pt')):
    import torch
    ckpt = torch.load(model_dir, map_location='cpu')
    results.append({
        'model': model_dir.split('/')[-2],
        'mAP': ckpt.get('mAP', 0.0),
        'epoch': ckpt.get('epoch', -1),
    })
    print(f\"{results[-1]['model']:20s} | mAP: {results[-1]['mAP']:.3f} @ epoch {results[-1]['epoch']}\")
with open('$OUTPUT_DIR/experiment_summary.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSummary saved to $OUTPUT_DIR/experiment_summary.json')
"