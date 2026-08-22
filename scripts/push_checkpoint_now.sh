#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

echo "Scanning all training runs for their best-ever mAP50..."

BEST_OVERALL_MAP=0
BEST_OVERALL_PT=""

for results_csv in runs/detect/*/results.csv; do
    [ -f "$results_csv" ] || continue
    run_dir=$(dirname "$results_csv")
    weights_path="$run_dir/weights/best.pt"
    [ -f "$weights_path" ] || continue

    run_best_map=$(python3 -c "
import csv
best = 0.0
with open('$results_csv') as f:
    for row in csv.DictReader(f):
        val = row.get('               metrics/mAP50(B)', row.get('metrics/mAP50(B)', '0'))
        try:
            best = max(best, float(val))
        except ValueError:
            pass
print(best)
")
    echo "  $run_dir: best mAP50 so far = $run_best_map"

    is_better=$(python3 -c "print('yes' if $run_best_map > $BEST_OVERALL_MAP else 'no')")
    if [ "$is_better" = "yes" ]; then
        BEST_OVERALL_MAP=$run_best_map
        BEST_OVERALL_PT=$weights_path
    fi
done

if [ -z "$BEST_OVERALL_PT" ]; then
    echo "No completed training runs with results found yet, nothing to push."
    exit 1
fi

echo ""
echo "Best model overall: $BEST_OVERALL_PT, mAP50=$BEST_OVERALL_MAP"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
git add -f "$BEST_OVERALL_PT"

LAST_PT="$(dirname "$BEST_OVERALL_PT")/last.pt"
if [ -f "$LAST_PT" ]; then
    git add -f "$LAST_PT"
fi

RESULTS_CSV="$(dirname "$BEST_OVERALL_PT")/../results.csv"
if [ -f "$RESULTS_CSV" ]; then
    git add -f "$RESULTS_CSV"
fi

git commit -m "checkpoint push at $TIMESTAMP, best mAP50 across all runs = $BEST_OVERALL_MAP"
git push

echo ""
echo "Pushed the genuinely best-scoring model found, not just the newest file."
