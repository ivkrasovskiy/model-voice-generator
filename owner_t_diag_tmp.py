import random
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path("/Users/ivkrasovskii/model-voice-generator")
sys.path.insert(0, str(REPO_ROOT))

from accent_coach.pipeline.alignment import align_audio, filter_aspirating_stops
from accent_coach.pipeline.audio_io import load_standard_audio
from accent_coach.pipeline.vot import extract_stop_features, extract_vot
from scripts.lib.manifest import load_manifest, resolve_path

random.seed(42)
manifest = REPO_ROOT / "tts_output/owner_cal_50/manifest.json"
entries = load_manifest(manifest)
entries = sorted(entries, key=lambda e: (e.get("end_s", 3) - e.get("start_s", 0)), reverse=True)
sample = random.sample(entries[:max(20*3, 40)], min(20, len(entries)))

print(f"sample size: {len(sample)}")
t_results = []
for i, e in enumerate(sample):
    wav = resolve_path(e, manifest, REPO_ROOT)
    tr = e.get("transcript") or e.get("prompt", "")
    if wav is None or not tr:
        continue
    loaded = load_standard_audio(wav)
    if loaded is None:
        continue
    audio, sr = loaded
    try:
        phon = align_audio(wav, tr, i, "rp")
    except Exception as ex:
        print(f"  [{i}] align failed: {ex}")
        continue
    stops = filter_aspirating_stops(phon)
    for stop in stops:
        if stop.phoneme != "t":
            continue
        vot = extract_vot(audio, sr, stop)
        t_results.append((wav.name, stop.word, stop.start_time, vot))

print(f"\n/t/ tokens found: {len(t_results)}")
for r in t_results:
    print(f"  {r}")
