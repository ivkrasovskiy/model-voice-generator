#!/usr/bin/env bash
# End-to-end pipeline: segment Casanova → transcribe → build manifest → fine-tune
#
# Each step writes a status marker so subsequent runs can resume.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs runs

step() {
    local name=$1; shift
    local marker="logs/.${name}.done"
    if [ -f "$marker" ]; then
        echo "[skip] $name (marker exists at $marker)"
        return 0
    fi
    echo ""
    echo "=== $name ==="
    "$@" 2>&1 | tee "logs/${name}.log"
    touch "$marker"
    echo "[done] $name"
}

# Step 1: VAD segmentation
step segment uv run python scripts/segment_vad.py --book casanova

# Step 2: Whisper large-v3 transcription
step transcribe uv run python scripts/transcribe_v3.py --book casanova

# Step 3: Filter and build training manifest
step manifest uv run python scripts/build_manifest.py --book casanova

# Step 4: Prepare F5-TTS dataset format
step prepare uv run python scripts/prepare_f5_dataset.py --book casanova

# Step 5: Validate manifest quality before fine-tuning
echo ""
echo "=== validating manifest ==="
uv run python - <<'EOF'
import csv
from pathlib import Path
m = Path("dataset/segments/casanova/training_manifest.csv")
rows = list(csv.DictReader(m.open()))
durs = [float(r["duration_sec"]) for r in rows]
total_hrs = sum(durs) / 3600
print(f"  manifest: {len(rows)} clips, {total_hrs:.2f} hours")
print(f"  duration range: {min(durs):.2f}s – {max(durs):.2f}s, avg {sum(durs)/len(durs):.2f}s")
texts = [r["text"] for r in rows if r["text"]]
print(f"  text lengths: min={min(len(t) for t in texts)} max={max(len(t) for t in texts)} avg={sum(len(t) for t in texts)//len(texts)} chars")
if total_hrs < 0.5:
    print("WARNING: less than 30 min — fine-tuning may not converge well")
    exit(1)
EOF

# Step 6: Fine-tune F5-TTS (200 steps to verify loss decreases)
step finetune uv run python scripts/finetune_f5.py \
    --dataset cumberbatch_casanova \
    --run-name finetune_casanova \
    --max-steps 200 \
    --lr 1e-5 \
    --grad-accum 4

echo ""
echo "=== pipeline complete ==="
echo "Loss log: runs/finetune_casanova/loss_log.csv"
echo "Checkpoints: runs/finetune_casanova/checkpoints/"
