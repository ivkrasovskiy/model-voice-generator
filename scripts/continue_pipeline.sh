#!/usr/bin/env bash
# Wait for transcription to complete (transcripts.csv has 1917 lines = header + 1916 clips),
# then run manifest → prepare → finetune.

cd "$(dirname "$0")/.."
mkdir -p logs

TRANSCRIPTS=dataset/segments/casanova/transcripts.csv
TARGET_LINES=1917

echo "[continue] waiting for $TRANSCRIPTS to reach $TARGET_LINES lines..."
while true; do
    if [ -f "$TRANSCRIPTS" ]; then
        current=$(wc -l < "$TRANSCRIPTS")
        echo "[continue] progress: $current/$TARGET_LINES"
        if [ "$current" -ge "$TARGET_LINES" ]; then
            break
        fi
    fi
    sleep 60
done

echo ""
echo "[continue] transcription done; building manifest..."
PYTHONHASHSEED=random uv run python scripts/build_manifest.py --book casanova 2>&1 | tee logs/manifest.log

echo ""
echo "[continue] preparing F5-TTS dataset..."
PYTHONHASHSEED=random uv run python scripts/prepare_f5_dataset.py --book casanova 2>&1 | tee logs/prepare.log

echo ""
echo "[continue] launching fine-tune (200 steps)..."
PYTHONHASHSEED=random uv run python scripts/finetune_f5.py \
    --dataset cumberbatch_casanova \
    --run-name finetune_casanova \
    --max-steps 200 \
    --lr 1e-5 \
    --grad-accum 4 \
    --train-last-n 4 2>&1 | tee logs/finetune.log

echo ""
echo "[continue] pipeline COMPLETE"
echo "  Loss log: runs/finetune_casanova/loss_log.csv"
echo "  To compare voices, run:"
echo "    PYTHONHASHSEED=random uv run python scripts/compare_voices.py"
