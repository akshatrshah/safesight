#!/usr/bin/env bash
# Overnight run: download check/fix, build a 1500-image forklift+pallet
# dataset, two-stage training, all with deliberately low resource usage
# so this doesn't overheat the laptop or peg every core while you sleep.
#
# Thermal/resource choices explained:
#   - OMP_NUM_THREADS / MKL_NUM_THREADS capped to 2: this is the actual
#     lever that limits how many CPU cores get hammered by the underlying
#     math libraries. `nice` alone only helps when something else is
#     competing for CPU, it does NOT cap total usage when nothing else
#     is running, which is exactly the overnight-alone scenario.
#   - --workers 2, --batch 6: smaller, gentler load per step instead of
#     large bursty spikes.
#   - caffeinate -s: stops the Mac from going to sleep while this runs,
#     WITHOUT this, macOS can sleep the whole system overnight and
#     silently pause training for hours with no error, no warning.

set -e
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "=== Step 1: checking the LOCO download is actually complete ==="
ZIP_PATH="datasets/loco_raw/loco_images.zip"
MIN_SIZE_BYTES=700000000

if [ -f "$ZIP_PATH" ]; then
    ACTUAL_SIZE=$(stat -f%z "$ZIP_PATH" 2>/dev/null || stat -c%s "$ZIP_PATH" 2>/dev/null)
    if [ "$ACTUAL_SIZE" -lt "$MIN_SIZE_BYTES" ]; then
        echo "Existing zip is only $ACTUAL_SIZE bytes, incomplete, removing it and the partial extraction"
        rm -f "$ZIP_PATH"
        rm -rf datasets/loco_raw/images
    else
        echo "Existing zip looks complete ($ACTUAL_SIZE bytes)"
    fi
fi

python3 scripts/download_loco_full.py

echo ""
echo "=== Step 2: building a 1500-image forklift + pallet dataset (1200 train / 300 val) ==="
python3 scripts/prepare_poc_subset.py --train-cap 1200 --val-cap 300

echo ""
echo "=== Step 3: stage 1, backbone frozen, low resource settings ==="
mkdir -p experiments
python3 scripts/finetune_detector.py \
  --data datasets/loco_poc/data.yaml \
  --base-model yolov8s.pt \
  --epochs 15 --freeze 10 \
  --imgsz 640 --erasing 0.5 --scale 0.7 --mixup 0.1 \
  --workers 2 --batch 6 \
  --device mps \
  --out experiments/stage1_results.json

STAGE1_BEST=$(python3 -c "import json; print(json.load(open('experiments/stage1_results.json'))['best_weights_path'])")
echo "Stage 1 complete. Best weights: $STAGE1_BEST"

echo ""
echo "=== Step 4: stage 2, full network unfrozen, low resource settings ==="
echo "Epoch ceiling is high (150) since it's overnight, patience=25 lets it stop itself"
echo "when it's genuinely done improving rather than running the full ceiling regardless."
python3 scripts/finetune_detector.py \
  --data datasets/loco_poc/data.yaml \
  --base-model "$STAGE1_BEST" \
  --epochs 150 --patience 25 \
  --imgsz 640 --erasing 0.5 --scale 0.7 --mixup 0.1 \
  --tta \
  --workers 2 --batch 6 \
  --device mps \
  --out experiments/stage2_results.json

echo ""
echo "=== Full pipeline complete ==="
python3 -c "import json; print(json.dumps(json.load(open('experiments/stage2_results.json')), indent=2))"
