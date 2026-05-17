#!/usr/bin/env bash
# Pipeline status snapshot — run after waking up to see what completed.

cd "$(dirname "$0")/.."

echo "=== Pipeline Status ==="
echo ""

# 1. Segmentation
if [ -f dataset/segments/casanova/manifest.csv ]; then
    n=$(($(wc -l < dataset/segments/casanova/manifest.csv) - 1))
    echo "✓ Segmentation: $n clips"
else
    echo "✗ Segmentation: not started"
fi

# 2. Transcription
if [ -f dataset/segments/casanova/transcripts.csv ]; then
    n=$(($(wc -l < dataset/segments/casanova/transcripts.csv) - 1))
    total=1916
    pct=$((n * 100 / total))
    echo "✓ Transcription: $n / $total clips ($pct%)"
else
    echo "✗ Transcription: not started"
fi

# 3. Training manifest
if [ -f dataset/segments/casanova/training_manifest.csv ]; then
    n=$(($(wc -l < dataset/segments/casanova/training_manifest.csv) - 1))
    echo "✓ Training manifest: $n clips"
else
    echo "✗ Training manifest: not built"
fi

# 4. F5-TTS dataset prepared
if [ -f data/cumberbatch_casanova/metadata.csv ]; then
    n=$(($(wc -l < data/cumberbatch_casanova/metadata.csv) - 1))
    echo "✓ F5-TTS dataset prepared: $n entries"
else
    echo "✗ F5-TTS dataset: not prepared"
fi

# 5. Fine-tune
if [ -f runs/finetune_casanova/loss_log.csv ]; then
    steps=$(($(wc -l < runs/finetune_casanova/loss_log.csv) - 1))
    first=$(sed -n '2p' runs/finetune_casanova/loss_log.csv | cut -d, -f2)
    last=$(tail -1 runs/finetune_casanova/loss_log.csv | cut -d, -f2)
    echo "✓ Fine-tune: $steps logged steps"
    echo "    first loss: $first"
    echo "    last loss:  $last"
    if [ -f runs/finetune_casanova/checkpoints/final.pt ]; then
        echo "    ✓ final checkpoint exists"
    fi
else
    echo "✗ Fine-tune: not started"
fi

# 6. Comparison samples
echo ""
if [ -d tts_output/compare ]; then
    n=$(ls tts_output/compare/*.wav 2>/dev/null | wc -l)
    echo "✓ Comparison samples: $n files in tts_output/compare/"
else
    echo "✗ Comparison samples: not generated (run: PYTHONHASHSEED=random uv run python scripts/compare_voices.py)"
fi

echo ""
echo "=== Running processes ==="
ps aux | grep -E "(transcribe_v3|finetune_f5|continue_pipeline|prepare_f5|build_manifest)" | grep -v grep | head -5 || echo "(none)"
