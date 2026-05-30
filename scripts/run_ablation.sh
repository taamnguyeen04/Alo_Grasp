set -e

QUICK_MODE=""
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE="train.epochs=5 data.max_train_samples=5000 data.max_val_samples=2000"
    echo "QUICK MODE: 5 epochs, 5k train samples, 2k val samples"
fi

VARIANTS=(
    "v1_clip_concat"
    "v2_clip_film_last"
    "v3_dinov2_film_last"
    "v4_dinov2_film_multiscale"
    "v5_full_method"
)

RESULTS_FILE="logs/ablation/results.txt"
mkdir -p logs/ablation
echo "Ablation results — $(date)" > "$RESULTS_FILE"
echo "===================================================" >> "$RESULTS_FILE"

for variant in "${VARIANTS[@]}"; do
    echo ""
    echo "============================================"
    echo "  Variant: $variant"
    echo "============================================"
    CONFIG="configs/ablation/${variant}.yaml"

    # Train
    echo "[Training] $variant"
    python train.py --config "$CONFIG" $QUICK_MODE

    # Find the latest checkpoint for this variant
    LATEST_RUN=$(ls -td logs/ablation/${variant}_* 2>/dev/null | head -1)
    CKPT="$LATEST_RUN/best.ckpt"
    if [[ ! -f "$CKPT" ]]; then
        echo "WARNING: no checkpoint found at $CKPT — skipping eval"
        continue
    fi

    # Evaluate with Base/New/H
    echo "[Evaluating] $variant"
    RESULT=$(python evaluate.py --checkpoint "$CKPT" --split val --base-new --batch-size 64 \
             2>&1 | tail -20)
    echo "$RESULT" | tee -a "$RESULTS_FILE"
    echo "--- end $variant ---" >> "$RESULTS_FILE"
done

echo ""
echo "============================================"
echo "  ABLATION SUMMARY (see $RESULTS_FILE)"
echo "============================================"
grep -E "Variant|Base \(|New |Harmonic" "$RESULTS_FILE"
