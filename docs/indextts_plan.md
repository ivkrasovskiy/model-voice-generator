# IndexTTS-2 evaluation plan

Self-contained plan for a fresh Sonnet session to test whether IndexTTS-2 (Sept 2025) beats both F5-TTS (current identity leader) and XTTS-v2 (current WER leader) on our 15-phrase eval set.

Read this file end-to-end before starting. Do not skip the feasibility gate.

---

## Why IndexTTS-2 (and not F5R-TTS)

**F5R-TTS was investigated and ruled out.** See [literature_notes.md §4](./literature_notes.md). Killer constraints: their gain comes from re-pretraining F5-TTS from scratch on 7,226 h with an architectural change to the output head (Gaussian μ/σ instead of point estimate). That custom checkpoint is not released. RL phase alone needs 8× A100 40 GB, batch 6,400. Not doable on M3 Pro 18 GB.

**IndexTTS-2** is the right next move because:
- Released Sept 2025 — newer than F5-TTS and XTTS-v2.
- Audio-only reference (no `ref_text` input) — architecturally eliminates ref-text leakage, the root cause of F5-TTS's WER 1.4.
- Paper claims it beats both F5-TTS and CosyVoice 2 on WER **and** SIM in zero-shot.
- Open weights on HuggingFace (`IndexTeam/IndexTTS-2`).

**Risk**: docs assume CUDA 12.8. No documented CPU or MPS path. The plan starts with a feasibility gate; if CPU inference is broken or unusably slow, abort to a fallback (§Fallbacks).

---

## Current eval baselines (what to beat)

15-phrase eval, per-clip ECAPA vs real Cumberbatch recording. From [../CLAUDE.md](../CLAUDE.md):

| Model | WER ↓ | ECAPA ↑ | DNSMOS OVR ↑ |
|---|---|---|---|
| F5-TTS baseline (sel-CFG, ref_narrator.wav) | 1.388 | **0.827** | 3.86 |
| F5-TTS `finetune_kd_combined_lam5_v2/best.pt` | 1.404 | 0.828 | 3.91 |
| F5-TTS `finetune_kd_combined_lam5_v2/ema_best.pt` | 1.369 | 0.826 | 3.87 |
| XTTS-v2 zero-shot | **0.108** | 0.689 | 2.58 |

**Win criteria**: WER < 0.5 AND ECAPA > 0.80. Anything below either threshold is "interesting but not a winner."

---

## Step 0 — feasibility gate (~30 min)

**Decision point: do NOT proceed past this step until smoke test passes.**

```bash
# Workspace
cd /Users/ivkrasovskii/model-voice-generator

# Clone + create venv (NOT in main .venv — IndexTTS pins are likely to fight F5-TTS deps)
git clone https://github.com/index-tts/index-tts.git /tmp/index-tts
uv venv .venv_indextts --python 3.10
```

Try the repo's recommended install first:

```bash
cd /tmp/index-tts
VIRTUAL_ENV=/Users/ivkrasovskii/model-voice-generator/.venv_indextts uv sync --all-extras 2>&1 | tee /tmp/indextts_install.log
```

**Likely failure modes and fixes** (we hit similar ones with XTTS):

| Error | Fix |
|---|---|
| `BeamSearchScorer` / similar removed | `uv pip install --python .venv_indextts/bin/python "transformers<4.44"` |
| `torch.load` weights_only error | `uv pip install --python .venv_indextts/bin/python "torch<2.6" "torchaudio<2.6"` |
| Missing CUDA wheel for `flash-attn` / `deepspeed` | `uv pip uninstall flash-attn deepspeed` (skip them; they're optional) |
| Missing pynini / nemo_text_processing | Install via `conda` is recommended upstream; on Mac, try `uv pip install pynini==2.1.6` |

Download weights to repo-local checkpoints dir (~2-3 GB):

```bash
cd /tmp/index-tts
/Users/ivkrasovskii/model-voice-generator/.venv_indextts/bin/huggingface-cli \
  download IndexTeam/IndexTTS-2 --local-dir checkpoints
```

Smoke test — generate ONE phrase on CPU:

```bash
cd /tmp/index-tts
/Users/ivkrasovskii/model-voice-generator/.venv_indextts/bin/python -c "
from indextts.infer_v2 import IndexTTS2
import time
tts = IndexTTS2(cfg_path='checkpoints/config.yaml', model_dir='checkpoints', device='cpu')
t = time.time()
tts.infer(
    spk_audio_prompt='/Users/ivkrasovskii/model-voice-generator/tts_output/ref_narrator.wav',
    text='Stella was a cute and intelligent woman.',
    output_path='/tmp/indextts_smoke.wav',
)
print(f'OK, {time.time()-t:.1f}s')
"
```

**Pass criteria (ALL must hold)**:
- ✓ No import errors / no missing CUDA ops
- ✓ `/tmp/indextts_smoke.wav` exists, file size > 50 KB
- ✓ Generation time < 120 s on CPU (extrapolates to ~30 min for full 15-phrase run; tolerable)
- ✓ Audio is intelligible — open in Finder, hit space, you should hear "Stella was a cute and intelligent woman" in roughly the right voice

**If any fail**: jump to §Fallbacks. Update [../CLAUDE.md](../CLAUDE.md) with the failure mode in "Next steps." Do not attempt to fix CUDA-only ops by patching the code.

---

## Step 1 — generation script (~30 min)

Create `scripts/indextts_gen.py`. Use [`scripts/xtts_gen.py`](../scripts/xtts_gen.py) as the template — same I/O contract.

```python
"""
Generate eval phrases with IndexTTS-2 zero-shot voice cloning.

Architecture: audio-only reference (no ref_text). Eliminates F5-TTS-style leakage.

Usage (run with .venv_indextts, NOT .venv):
    .venv_indextts/bin/python scripts/indextts_gen.py
"""

import csv, json, os, sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
PROJECT_ROOT = Path(__file__).parent.parent

# IndexTTS-2 must be importable; add the cloned repo to sys.path
sys.path.insert(0, "/tmp/index-tts")

import soundfile as sf
from indextts.infer_v2 import IndexTTS2

REF_AUDIO = str(PROJECT_ROOT / "tts_output/ref_narrator.wav")
PHRASES_CSV = PROJECT_ROOT / "tts_output/cross_eval_50/eval_short.csv"
OUT_DIR = PROJECT_ROOT / "tts_output" / "eval_indextts_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"   # update to "mps" only if smoke test passes on MPS; default is CPU
CFG_PATH = "/tmp/index-tts/checkpoints/config.yaml"
MODEL_DIR = "/tmp/index-tts/checkpoints"

print(f"Loading IndexTTS-2 on {DEVICE}...")
tts = IndexTTS2(cfg_path=CFG_PATH, model_dir=MODEL_DIR, device=DEVICE)
print(f"  Model loaded. Reference: {Path(REF_AUDIO).name}")

phrases = []
with open(PHRASES_CSV) as f:
    for row in csv.DictReader(f):
        phrases.append(row)

manifest = []
for row in phrases:
    slug, prompt = row["slug"], row["prompt"]
    out_path = OUT_DIR / f"indextts_{slug}.wav"
    if not out_path.exists():
        tts.infer(spk_audio_prompt=REF_AUDIO, text=prompt, output_path=str(out_path))
    info = sf.info(str(out_path))
    print(f"  → {out_path.name} ({info.duration:.1f}s)")
    manifest.append({
        "label": "indextts_v2", "step": 0, "slug": slug, "prompt": prompt,
        "wav_path": str(out_path), "sr": int(info.samplerate),
    })

(OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\n✓ {len(manifest)} clips → {OUT_DIR}")
```

Run it (in background — 15 × ~30 s = ~8 min on CPU; could be longer):

```bash
nohup .venv_indextts/bin/python scripts/indextts_gen.py > runs/indextts_gen.log 2>&1 &
```

**Manifest schema** (must match exactly — `posthoc_eval.py --score-only` reads these keys):

| Key | Type | Source |
|---|---|---|
| `label` | str | `"indextts_v2"` |
| `step` | int | `0` |
| `slug` | str | from `eval_short.csv` (e.g. `"cas_01"`) |
| `prompt` | str | from `eval_short.csv` |
| `wav_path` | str | absolute path to generated WAV |
| `sr` | int | from `soundfile.info()` |

---

## Step 2 — score (~5 min)

```bash
.venv/bin/python scripts/posthoc_eval.py \
  --score-only \
  --phrases-csv tts_output/cross_eval_50/eval_short.csv \
  --out-dir tts_output/eval_indextts_v2
```

This is identical to the XTTS-v2 scoring path. The summary table prints WER / ECAPA / DNSMOS at the end.

---

## Step 3 — decision

Add the new row to the comparison table in [../CLAUDE.md](../CLAUDE.md) "Eval results". Then:

| Outcome | Action |
|---|---|
| WER < 0.5 **AND** ECAPA > 0.80 | **Winner.** Switch IndexTTS-2 in as the production path. Plan fine-tune next. |
| WER < 0.5 but ECAPA ∈ [0.75, 0.80] | Marginal. Try fine-tuning IndexTTS-2 on our 354-clip subset — speaker adaptation usually adds 0.05-0.10 ECAPA. |
| WER < 0.5 but ECAPA < 0.75 | Same trade-off as XTTS-v2. Move to CosyVoice 2 (§Fallbacks). |
| WER ≥ 0.5 | Surprising. Re-listen — IndexTTS-2 shouldn't leak. Investigate before discarding. |

Update [CLAUDE.md "Next steps"](../CLAUDE.md) based on outcome.

---

## Fallbacks

If Step 0 fails or Step 3 says "move on":

### Fallback A — CosyVoice 2
- HuggingFace: `FunAudioLLM/CosyVoice2-0.5B`
- Repo: `https://github.com/FunAudioLLM/CosyVoice` (git clone + `pip install -r requirements.txt`)
- Architecture: supervised semantic tokens + chunk-aware flow matching. Audio-only reference.
- Apply this same plan structure: feasibility gate → `cosyvoice_gen.py` → score.

### Fallback B — Fish Speech v1.5+
- HuggingFace: `fishaudio/s2-pro` (or `fish-speech-1.5`)
- Repo: `https://github.com/fishaudio/fish-speech`
- Most mature pip-installable option; lower risk of broken Mac install.

### Fallback C — accept the trade-off
If none of the modern models work on M3 Pro CPU:
- F5-TTS owns identity (ECAPA 0.83), XTTS-v2 owns intelligibility (WER 0.11).
- **Hybrid post-processing**: generate with XTTS-v2 (clean words), then apply a voice-conversion step using a fine-tuned BC encoder. This was considered architecturally hard but might be the only path on this hardware.
- Or rent a 24 GB cloud GPU and run IndexTTS-2 / CosyVoice 2 properly. [docs/finetune_layers.md](./finetune_layers.md) §"When to migrate off F5-TTS" has the cost numbers.

---

## Pitfalls we already hit (so you don't waste time)

From the XTTS-v2 install in this repo:

1. **PyTorch 2.6 changed `weights_only` default to `True`** — breaks every model that pickle-stores configs. Pin `torch<2.6`.
2. **`transformers>=4.44` removed `BeamSearchScorer`** — breaks XTTS streaming generation and many older AR models. Pin `transformers<4.44`.
3. **Use a dedicated venv** (`.venv_indextts/`) — never install into `.venv/`. The F5-TTS env has tight pins that will fight.
4. **MPS often fails on autoregressive decoders** — XTTS-v2 needed CPU. Assume IndexTTS-2 will too unless smoke test on MPS works.
5. **First-run weight download is slow** — `huggingface-cli download` is ~2-3 GB. Cache it once, point both smoke test and gen script at the same dir.
6. **Always use `uv` for package management** — never raw `pip` or `python -m venv`. Project convention.

---

## Acceptance / done-when

This plan is complete when:
1. Step 0 either passes (proceed to gen) or fails (escalate with documented reason).
2. `tts_output/eval_indextts_v2/` exists with 15 WAVs + `manifest.json` + `scores.csv`.
3. [../CLAUDE.md](../CLAUDE.md) "Eval results" table has the new IndexTTS-2 row.
4. [../CLAUDE.md](../CLAUDE.md) "Next steps" reflects what to do given the result.
