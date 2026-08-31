#!/bin/bash
# Parallel LAV inference across 6 GPUs, 2 cases per GPU.
#
# Usage:
#   bash infer_all.sh <input_dir> <output_dir>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 2 ]; then
    echo "Usage: bash infer_all.sh <input_dir> <output_dir>"
    echo "  input_dir   directory of .mp4 clips to augment"
    echo "  output_dir  where per-clip results are written"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

NUM_GPUS=6
CASES_PER_GPU=2

mkdir -p "$OUTPUT_DIR"

# ── Collect pending videos ──────────────────────────────────────────
pending=()
for f in "$INPUT_DIR"/*.mp4; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .mp4)
    out_path="$OUTPUT_DIR/$name"
    if [ -d "$out_path" ] && [ "$(ls -A "$out_path" 2>/dev/null)" ]; then
        echo "SKIP (exists): $name"
        continue
    fi
    pending+=("$f")
done

total=${#pending[@]}
echo ""
echo "=== LAV parallel inference ==="
echo "  Input:  $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Pending: $total videos"
echo "  GPUs: $NUM_GPUS  |  Cases/GPU: $CASES_PER_GPU"
echo ""

if [ "$total" -eq 0 ]; then
    echo "Nothing to do."
    exit 0
fi

# ── Worker function: runs assigned videos on one GPU ────────────────
run_worker() {
    local gpu_id=$1
    shift
    for video_file in "$@"; do
        local name=$(basename "$video_file" .mp4)
        local out_path="$OUTPUT_DIR/$name"
        local seed=$((RANDOM % 9000 + 1000))
        echo "[GPU $gpu_id] START: $name"
        CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT_DIR/lav_wan_sidewalk.py" \
            --video_path "$video_file" \
            --save_dir "$out_path" \
            --seed "$seed" \
            --n_lights 4 \
            > "$OUTPUT_DIR/${name}_gpu${gpu_id}.log" 2>&1
        local rc=$?
        if [ $rc -eq 0 ]; then
            echo "[GPU $gpu_id] DONE:  $name"
        else
            echo "[GPU $gpu_id] FAIL:  $name (exit $rc)"
        fi
    done
}

# ── Dispatch: assign 2 videos per GPU, launch in parallel ──────────
pids=()
idx=0
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    # Gather up to CASES_PER_GPU videos for this GPU
    batch=()
    for _ in $(seq 1 $CASES_PER_GPU); do
        if [ $idx -lt $total ]; then
            batch+=("${pending[$idx]}")
            idx=$((idx + 1))
        fi
    done
    [ ${#batch[@]} -eq 0 ] && continue

    run_worker "$gpu_id" "${batch[@]}" &
    pids+=($!)
done

echo "Launched ${#pids[@]} workers, waiting..."
echo ""

# ── Wait for all workers ───────────────────────────────────────────
fail_count=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail_count=$((fail_count + 1))
done

echo ""
echo "=== All done! ($fail_count worker(s) had failures) ==="
