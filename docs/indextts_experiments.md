# IndexTTS-2 follow-up experiments

Self-contained plan for a fresh Sonnet session. The initial IndexTTS-2 evaluation succeeded — [indextts_plan.md](./indextts_plan.md) is complete. Voice is recognizable, no gibberish, ECAPA 0.784, WER 0.037. Per-clip ECAPA range: 0.747 (`cas_07`) to 0.851 (`cas_06`).

The goal of this plan is to push average ECAPA from 0.784 → 0.80+ via two independent levers, ordered low-risk to high-risk.

Read end-to-end before starting.

---

## Pre-step — extend the eval set with longer phrases (already done)

Current short eval is 15 phrases (17-60 chars, ~3-5s each). Real-world use needs longer-form quality.

`tts_output/cross_eval_50/eval_long.csv` already exists with **8 real BC transcripts** (5 Casanova + 3 Sherlock), each 9-12 seconds long. **Same leakage protections as `eval_short.csv`**: every clip is in `data/cumberbatch_casanova_clean/` or `data/cumberbatch_sherlock_narrator/` but **excluded from `cumberbatch_casanova_train_354/` and `cumberbatch_sherlock_train_354/`** (the future-fine-tune training sets), and disjoint from `eval_short.csv` ref clips. Each row has `ref_audio_path` pointing to the real recording for per-clip ECAPA.

**For every experiment below, run both eval sets**:
- `eval_short.csv` — 15 phrases — primary metric, comparable to all prior runs
- `eval_long.csv` — 8 phrases — secondary, tests register/breath stability and longer-context identity

Run both via `posthoc_eval.py --phrases-csv <csv>` separately, append both rows to the comparison table in CLAUDE.md.

**Critical for any fine-tuning** (Experiment B): the training script MUST exclude every clip referenced in `eval_long.csv` and `eval_short.csv` from its dataset. The `data/cumberbatch_casanova_train_354/` subset was already built with this exclusion — verify before training by intersecting `ref_audio_path` filenames with the training metadata.

---

## Environment recap

| Path | What |
|---|---|
| `vendor/index-tts/` | IndexTTS-2 git clone (commit `830f6f8f`, gitignored) |
| `vendor/index-tts/.venv/` | Working venv (`uv sync --no-dev` rebuilds from `uv.lock`) |
| `vendor/index-tts/checkpoints/` | Model weights, HF rev `740dcaff` (~8.3 GB) |
| `scripts/indextts_gen.py` | Current zero-shot generator (uses `vendor/` paths) |
| `scripts/indextts_smoke_test.py` | Regression check — runs in ~2 min, asserts baseline thresholds |
| `tts_output/eval_indextts_v2/` | Current results to beat |
| `tts_output/eval_indextts_v2/scores.regression_baseline.csv` | Locked baseline — never overwrite |

If `vendor/index-tts/` is missing, follow the "Rebuild from scratch" recipe in [../CLAUDE.md](../CLAUDE.md) "IndexTTS-2 install pins" section.

## Regression protection (REQUIRED FOR EVERY EXPERIMENT)

Before starting AND after finishing each experiment below, run:

```bash
vendor/index-tts/.venv/bin/python scripts/indextts_smoke_test.py
```

It generates 2 baseline phrases and asserts WER + ECAPA thresholds against the locked baseline. **If it fails after your changes, the experiment broke baseline capability — stop and fix before deploying.**

Additional rules to keep baseline safe:
- **Never modify `vendor/index-tts/` source files.** Write wrappers in `scripts/` instead. If you absolutely must patch (e.g. monkeypatch IndexTTS-2 for centroid), do it at import time in your script.
- **Never overwrite `tts_output/eval_indextts_v2/`** — that directory IS the baseline artifact set. New experiments write to new dirs:
  - Experiment A: `tts_output/eval_indextts_centroid_short/` + `_long/`
  - Experiment B: `tts_output/eval_indextts_lora_short/` + `_long/`
- **Never overwrite `.regression_baseline.csv` files** — they're the locked truth.

---

## Experiment A — Speaker embedding centroid (low risk, half-day)

**Hypothesis**: A single 12s reference clip is noisy. Averaging embeddings from N high-quality BC clips should reduce that noise and lock in identity more reliably. This is the same idea behind speaker-verification centroids — well-established in the literature.

**What changes in IndexTTS-2's pipeline**:
Looking at [`/tmp/index-tts/indextts/infer_v2.py`](file:///tmp/index-tts/indextts/infer_v2.py) lines 428-470, the speaker conditioning has two pieces per reference:
1. `spk_cond_emb` — W2V-BERT features → `self.get_emb(...)` (line 444)
2. `style` — CAMPPlus 192-dim global style vector (line 454)

Plus three derived tensors cached together: `S_ref` (quantized tokens), `ref_mel` (mel for the s2mel decoder), `prompt_condition`.

The cleanest patch: average `spk_cond_emb` and `style` across N reference clips. The derived tensors (S_ref, ref_mel, prompt_condition) come from a *single* representative clip — averaging mels makes no acoustic sense.

### Steps

**A.1 — Pick reference clips (15 min)**

Use the 20 highest-quality Casanova clips. Pre-filtered candidates exist in `data/cumberbatch_casanova/audit.csv` (ECAPA scores per clip). Take the top 20 by ECAPA from clips of duration 8-12 seconds.

```bash
.venv/bin/python -c "
import csv
from pathlib import Path

audit = Path('data/cumberbatch_casanova/audit.csv')
with open(audit) as f:
    rows = [r for r in csv.DictReader(f) if 8.0 <= float(r.get('duration', 0)) <= 12.0]
rows.sort(key=lambda r: -float(r['ecapa_sim']))
top20 = rows[:20]
out = Path('tts_output/centroid_refs.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['wav_path', 'duration', 'ecapa_sim'])
    w.writeheader()
    for r in top20:
        w.writerow({'wav_path': r['audio_file'], 'duration': r['duration'], 'ecapa_sim': r['ecapa_sim']})
print(f'wrote {out}')
"
```

If `audit.csv` doesn't have `ecapa_sim`, fall back to picking any 20 random clips of 10-12s and trust the audit was already done.

Also keep a "representative" clip for the derived tensors — use clip #1 from the centroid set, OR `tts_output/ref_narrator.wav` if you want the existing baseline.

**A.2 — Patch IndexTTS-2 for centroid input (45 min)**

Do NOT modify the cloned `/tmp/index-tts/` source directly. Instead, write `scripts/indextts_centroid_gen.py` that subclasses or monkey-patches the speaker-conditioning step. Approach:

```python
# Pseudocode — actual implementation will read .get_emb() and CAMPPlus interfaces
import torch
from indextts.infer_v2 import IndexTTS2

class IndexTTS2Centroid(IndexTTS2):
    def set_centroid_refs(self, ref_paths: list[str], representative_ref: str):
        """Pre-compute averaged speaker conditioning from N refs."""
        embs, styles = [], []
        for p in ref_paths:
            # mirror lines 435-454 of infer_v2.py, but collect not cache
            audio, sr = self._load_and_cut_audio(p, 15, verbose=False)
            # ... compute spk_cond_emb and style for this clip ...
            embs.append(spk_cond_emb)
            styles.append(style)
        # Average (mean reduces noise; consider trimmed mean if outliers)
        self._centroid_spk_cond_emb = torch.stack(embs).mean(dim=0)
        self._centroid_style = torch.stack(styles).mean(dim=0)
        # Derived tensors from the representative clip only
        # ... compute S_ref, ref_mel, prompt_condition from representative_ref ...
        self._centroid_S_ref = ...
        self._centroid_ref_mel = ...
        self._centroid_prompt_condition = ...

    def infer(self, *args, **kwargs):
        # Bypass the per-call cache by pre-seeding the cache fields
        self.cache_spk_cond = self._centroid_spk_cond_emb
        self.cache_s2mel_style = self._centroid_style
        self.cache_s2mel_prompt = self._centroid_prompt_condition
        self.cache_mel = self._centroid_ref_mel
        self.cache_spk_audio_prompt = "__centroid__"  # any non-None sentinel
        return super().infer(*args, **kwargs)
```

Key constraint: the cache invalidation logic in `infer_v2.py:428` is `if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt`. Set `cache_spk_audio_prompt` to a fixed sentinel string and pass that same sentinel as `spk_audio_prompt` on every `.infer()` call so the cache hits.

Pitfalls:
- `spk_cond_emb` shapes may vary by audio length. Truncate or pad to a common length (the model cuts inputs to 15s anyway, so use the longest dim).
- `style` (CAMPPlus 1×192) is fixed-shape — straightforward to average.
- Mean averaging is the baseline. If results are poor, try median or trimmed mean (drop top/bottom 10%).

**A.3 — Generate + score (~20 min)**

```bash
# Generate short eval
/tmp/index-tts/.venv/bin/python scripts/indextts_centroid_gen.py \
    --phrases-csv tts_output/cross_eval_50/eval_short.csv \
    --out-dir tts_output/eval_indextts_centroid_short

# Score
.venv/bin/python scripts/posthoc_eval.py --score-only \
    --phrases-csv tts_output/cross_eval_50/eval_short.csv \
    --out-dir tts_output/eval_indextts_centroid_short

# Generate long eval
/tmp/index-tts/.venv/bin/python scripts/indextts_centroid_gen.py \
    --phrases-csv tts_output/cross_eval_50/eval_long.csv \
    --out-dir tts_output/eval_indextts_centroid_long

# Score
.venv/bin/python scripts/posthoc_eval.py --score-only \
    --phrases-csv tts_output/cross_eval_50/eval_long.csv \
    --out-dir tts_output/eval_indextts_centroid_long
```

**A.4 — Decision (5 min)**

| ECAPA short | Action |
|---|---|
| ≥ 0.82 | **Winner.** Stop here, ship centroid version. Skip Experiment B. |
| 0.79-0.82 | Modest gain. Run Experiment B for the final push. |
| ≤ 0.78 (no improvement) | Centroid isn't the lever. Investigate: are we averaging the right thing? Try `style` only, or `spk_cond_emb` only, to isolate. If still no gain, skip to Experiment B. |

---

## Experiment B — LoRA fine-tune of the GPT submodule (high effort, ~1 week)

**Hypothesis**: The IndexTTS-2 GPT autoregressive model generates audio tokens conditioned on (text, spk_emb). LoRA-adapting just the GPT for BC's voice patterns should improve identity without touching the s2mel decoder or BigVGAN.

**Risk** (be honest about this before starting):
- IndexTTS-2 has **no released training script**. Writing one from scratch.
- The loss formulation is undocumented — we'll have to infer it from the inference code.
- 1-2 week project with uncertain outcome.
- Only start this after Experiment A is exhausted.

### Architecture refresher

Read [`/tmp/index-tts/indextts/gpt/model.py`](file:///tmp/index-tts/indextts/gpt/model.py) and [`/tmp/index-tts/indextts/gpt/model_v2.py`](file:///tmp/index-tts/indextts/gpt/model_v2.py) to confirm. Expected pipeline:

```
target_audio  → W2V-BERT features → spk_cond_emb_target → semantic_codec.quantize → target_audio_tokens
ref_audio     → W2V-BERT features → spk_cond_emb_ref
target_text   → tokenizer → text_tokens

LOSS = cross_entropy(GPT(text_tokens, spk_cond_emb_ref), target_audio_tokens)
```

Verify by reading the inference call in `infer_v2.py:540-570` — see how GPT consumes `spk_cond_emb` and what it emits.

### Steps

**B.1 — Verify the loss path (1-2 days)**

Before LoRA, confirm we can compute a meaningful loss on the existing checkpoint without training. Write `scripts/indextts_loss_check.py`:
1. Pick 5 (text, target_audio) pairs from `data/cumberbatch_casanova_train_354/`
2. Use the target_audio as `ref` (sanity check: model should predict its own tokens with low loss)
3. Run forward pass, compute cross-entropy on target tokens
4. Print per-pair loss; expect <2.0 for the self-ref case (this is the "teacher" loss)

If this loss is sensible (small numbers, decreasing for easier pairs), the pipeline works. If it's nonsensical, debug before continuing.

**B.2 — LoRA setup (1 day)**

Install `peft`:
```bash
VIRTUAL_ENV=/tmp/index-tts/.venv uv pip install --python /tmp/index-tts/.venv/bin/python peft
```

LoRA config to start:
- `rank=16, alpha=32, dropout=0.05`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj` in GPT attention layers (read GPT module to find exact names — IndexTTS GPT may use different names)
- Trainable params expected: ~3-10 M (0.1-0.3% of the 3.3 GB GPT)

Memory budget on M3 Pro 18 GB:
- GPT weights (frozen): ~3.3 GB
- LoRA params: ~50 MB
- AdamW state (8 bytes/param × LoRA params): ~80 MB
- Activations for ~10s audio batch=1: ~3 GB
- Headroom: ~10 GB — should fit. If it doesn't, drop rank to 8 or freeze more layers.

**B.3 — Training loop (~3-5 days)**

Write `scripts/finetune_indextts.py` inspired by `scripts/finetune_f5.py`:
- Dataset: `data/cumberbatch_casanova_train_354/` (354 clips, 37.8 min). Already manifest-formatted.
- Validation: hold out 25 clips (use `lib/dataset.deterministic_split`)
- Optimizer: AdamW, lr=1e-4, warmup 100 steps, cosine decay
- Steps: 500-2000 (early stop on eval loss, window 200 — copy logic from `finetune_f5.py`)
- Save best.pt + ema_best.pt (use `EMATracker` from finetune_f5)
- Log to TensorBoard in `runs/finetune_indextts_lora_v1/`
- Generate samples every 100 steps using EMA weights — same idea as planned-but-not-done ECAPA hook for F5
- MPS cache flush every 25 steps (`torch.mps.empty_cache()`)

Distinct run names (project convention from CLAUDE.md):
- `finetune_indextts_lora_r16_v1` — first attempt
- `finetune_indextts_lora_r32_v1` — if rank=16 underfits
- `finetune_indextts_lora_emo_v1` — if emotion conditioning needs adjustment

**B.4 — Eval cadence**

Every 100 steps:
- Compute eval_loss on held-out 25 clips
- Generate 2 short phrases using EMA weights
- Log ECAPA vs centroid (no Whisper during training — keep MPS clean)

At the end, generate the full 15-phrase short eval + 5-phrase long eval. Compare to baseline IndexTTS-2 results.

**B.5 — Decision**

| ECAPA short | Action |
|---|---|
| ≥ 0.83 | **Winner.** Switch to LoRA-adapted IndexTTS-2 for production. |
| 0.80-0.83 | Significant gain. Iterate on hyperparameters (rank, lr, modules). Diminishing returns soon. |
| 0.77-0.80 | LoRA helped but didn't move it enough. Try larger rank (32-64) or training all 22 GPT blocks. |
| < 0.77 | Regression. Investigate — likely a training bug or wrong loss. Don't deploy. |

---

## What "done" looks like

After both experiments:

1. `tts_output/eval_indextts_centroid_short/` + `_long/` — Exp A results
2. `tts_output/eval_indextts_lora_short/` + `_long/` — Exp B results (if B was run)
3. `runs/finetune_indextts_lora_*/` — training logs (if B was run)
4. CLAUDE.md "Eval results" table has new rows for Exp A and (if relevant) Exp B
5. CLAUDE.md "Next steps" reflects the outcome:
   - If a winner: declare BC voice cloning solved on M3 Pro
   - If not: document that further gains need a 24 GB GPU + full-parameter training

---

## Pitfalls and tips

1. **Always use rtk** for shell commands (project convention).
2. **Always use uv** for package management; never raw pip into `/tmp/index-tts/.venv/`.
3. **Run names must be distinct** — never reuse a run name. Use `_v1`, `_v2` suffix on iterations.
4. **TensorBoard**: `.venv/bin/tensorboard --logdir runs/ --port 6006`.
5. **MPS memory**: flush every 25 steps. KD-style sustained training without flushing will hit 21 GB and OOM.
6. **Don't modify `/tmp/index-tts/` source** — write wrappers in `scripts/`. The cloned repo should stay clean so it can be re-synced.
7. **Centroid clip selection** — quality > quantity. 20 clean clips ≥ 100 noisy ones. Trust `audit.csv` ECAPA scores or re-audit if missing.
8. **Listen test matters** — ECAPA is a proxy. After every experiment, listen to at least 3 long-eval clips before deciding. Voice quality dimensions ECAPA misses: breath, pauses, prosody continuity.

---

## Acceptance criteria

Run Experiment A first. Run Experiment B only if A finishes with ECAPA < 0.82.

Update [../CLAUDE.md](../CLAUDE.md) "Eval results" with the new rows after each experiment. Update "Next steps" with the final decision.
