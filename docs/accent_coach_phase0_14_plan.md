# Phase 0.14 Plan — Segment-Splice, Accent-Transfer Probe, GPT-LoRA

**Author:** Opus (planning). **Executor:** claude-sonnet. **Date:** 2026-05-27.

This plan is written to be **executed, not interpreted**. Every step has exact
files, exact commands, and explicit acceptance checks. Where a value must be
discovered at runtime (e.g. an API return shape), the step says "VERIFY" and
gives the one-liner to run. Do not redesign; if a step's acceptance check fails,
stop and report rather than improvising.

## Hard rules (inherited from CLAUDE.md — non-negotiable)

- **Venvs**: scoring/DSP/Whisper → `.venv/bin/python`. Anything importing
  `indextts.*` (generation + LoRA training) → `vendor/index-tts/.venv/bin/python`.
  Never `python3` bare, never `pip`. Dependency changes via `uv add` only.
- **RTK prefix** every shell command (`rtk git ...`, `rtk pytest ...`).
- **≤ 400 code lines per file.** Split into a lib module if a script grows past it.
- **Ruff clean**: `uv run ruff check scripts/ accent_coach/` zero errors before any commit.
- **Tests green**: `uv run pytest tests/ -q` (30 tests).
- **Smoke test** before AND after any change touching IndexTTS-2 paths:
  `vendor/index-tts/.venv/bin/python scripts/indextts_smoke_test.py`
  (asserts cas_01 & sher_03 WER ≤ 0.10, ECAPA ≥ 0.74).
- **Do not touch** (read-only, locked): `vendor/`, `tts_output/eval_indextts_v2/`,
  `tts_output/refs/production/`, `tts_output/refs/indextts_baseline/`.
  Write new refs to `tts_output/refs/accent_test/`, new outputs under a `--run-name`.
- Run `bash scripts/check_repo.sh` before declaring any workstream done.

## Execution order

Run **WS-B first** (cheapest, and it tells us whether IndexTTS-2 can follow a
reference accent *at all* — that result reframes how much WS-C can hope to gain).
Then **WS-A** (independent DSP, no model training). Then **WS-C** (the big one).

---

# WS-B — Accent-transfer capability probe

**Question:** Given a non-RP reference clip, does IndexTTS-2 place the cloned
voice near that accent, near General American, or near RP? This measures whether
accent lives in the reference (zero-shot transferable) or is baked into the model.

**Why it matters for WS-C:** If accented references transfer cleanly, the
American-leaning vowels may be partly a *reference-choice* problem, not purely
model-bound — cheaper to fix than LoRA. If output clusters near GenAm regardless
of reference, that is direct evidence the bias is in the GPT (supports WS-C).

### B0. Build GenAm norm tables (new file)

Create `accent_coach/reference/genam_norms.py` mirroring the structure of
`accent_coach/reference/rp_norms.py` (same IPA keys, `(F1, F2)` Hz tuples, one
`# source` comment per value). Populate **male and female** dicts from
**Hillenbrand et al. (1995), "Acoustic characteristics of American English
vowels," JASA 97(5), Table V** (men) and the women's column.

- Map Hillenbrand's ARPABET/keywords to the IPA keys already used in
  `rp_norms.py` (iː, ɪ, ɛ, æ, ɑː, ɒ, ɔː, ʊ, uː, ʌ, ɜː, ə, eɪ, aɪ, ɔɪ, əʊ, aʊ).
- GenAm has no /ɒ/ (LOT–THOUGHT merger toward /ɑ/) and uses /ɝ/ for NURSE — map
  /ɜː/→/ɝ/, /ɒ/→/ɑ/; comment these mappings.
- **VERIFY** each value against the paper before committing; mark any value you
  cannot source `# TODO(cite)`. Do not invent numbers.
- Expose `get_genam_norms(mean_f0: float)` returning male table if `mean_f0 < 165`
  else female, matching the `get_rp_norms` signature.

Acceptance: `uv run python -c "from accent_coach.reference.genam_norms import get_genam_norms; print(len(get_genam_norms(120)))"` prints ≥ 15.

### B1. Build the three reference clips

Use `scripts/build_podcast_ref.py` (args: `--url --start --duration --out --sr 22050`).
The given windows contain applause/music/multiple speakers; you must **listen to
the downloaded audio and pick a clean, single-speaker ~10–14 s sub-window**. Write to:

- Georgia (Ginny & Georgia; speaker Brianne Howey, **female**, Southern US):
  `--url https://www.youtube.com/watch?v=QICSofOItus` → `tts_output/refs/accent_test/georgia.wav`
- Scottish: `--url https://www.youtube.com/watch?v=mxbow_aKoAM` window inside 0:27–2:51,
  avoid applause/music → `tts_output/refs/accent_test/scottish.wav` (note speaker sex in the manifest)
- Irish (Saoirse Ronan, **female**): `--url https://www.youtube.com/watch?v=9xCr6IQtYqk`
  window inside 0:27–1:05 → `tts_output/refs/accent_test/irish.wav`

For each, record in a `tts_output/refs/accent_test/PROVENANCE.md`: url, exact
start/duration chosen, why (clean speech), speaker sex, estimated mean F0
(`.venv/bin/python -c` with parselmouth or librosa.pyin).

Acceptance: three mono 22050 Hz WAVs, 10–14 s, audibly single-speaker, no music.

### B2. Generate the diagnostic phrase set per reference

Reuse the 5-vowel-dense calibration set already in the repo:
`tts_output/accent_coach/cal_25.csv` (or `tts_output/cross_eval_50/eval_short.csv`
if cal_25 lacks the 5 target vowels — pick whichever has more /ʌ ʊ ɔː aʊ ɜː/ tokens;
check by eye). For each reference run:

```bash
vendor/index-tts/.venv/bin/python scripts/indextts_gen.py \
    --phrases-csv tts_output/accent_coach/cal_25.csv \
    --out-dir tts_output/accent_coach/phase0_14/accent_probe/<georgia|scottish|irish> \
    --spk-ref tts_output/refs/accent_test/<ref>.wav \
    --num-beams 5
```

(If `indextts_gen.py` has no `--spk-ref` flag, check its `--help`; the production
ref is otherwise the default. Pass whatever flag selects the speaker reference.)

### B3. Score: where did each accent land?

Run **in `.venv`** (parselmouth + scoring). Reuse the existing pipeline functions
from `accent_coach.pipeline.experiment` (`extract_formants`,
`build_centroids_from_formants`, `score_against`). Write a small driver
`scripts/accent_coach_phase0_14_accent_probe.py` (≤ 400 lines) that, per reference:

1. `extract_formants(manifest, formants.csv)` on the generated clips.
2. `build_centroids_from_formants(...)` → generated vowel centroids.
3. Also extract formants + centroids on the **reference clip itself** (the source
   exemplar) so we have its vowel space.
4. Compute three Bark-space distances (use the existing Bark distance in
   `accent_coach/diagnostics/bark_distance.py`) per vowel and averaged:
   - **dist-to-source**: generated vs the reference's own centroids (sex-agnostic;
     this is the primary metric — "did it follow the reference?").
   - **dist-to-RP**: generated vs `get_rp_norms(mean_f0)` (sex-matched).
   - **dist-to-GenAm**: generated vs `get_genam_norms(mean_f0)` (sex-matched).
5. Emit `tts_output/accent_coach/phase0_14/accent_probe/grid.csv` and a findings
   doc `docs/accent_coach_phase0_14_accent_probe_findings.md`.

**Interpretation (this is a probe, not pass/fail):**
- dist-to-source << dist-to-both-norms → **model follows the reference accent**
  (good zero-shot transfer; weakens the "model-bound" claim).
- output near GenAm regardless of reference → **American bias in the GPT**
  (strong support for WS-C).
- output near RP → model already RP-leaning for these refs.

Write the verdict explicitly in the findings doc. Listen to 2 clips per accent
and note perceptual impression (does it *sound* Georgian/Scottish/Irish?).

---

# WS-A — Segment-splice formant shift (fix Lever B's global vocoder damage)

**Problem (from Phase 0.13a findings §5.1):** `shift_vowels_to_centroid` runs the
**whole** clip through `pyworld.wav2world → synthesize`, so every frame — even
unshifted phonemes — eats a fixed −0.37 DNSMOS round-trip loss. Goal: keep the
original waveform everywhere except the edited vowels.

### A1. Add a splice path to the DSP

Edit `accent_coach/dsp/formant_shift.py`. Add a `splice: bool = False` parameter
to `shift_vowels_to_centroid` (keep the existing full-resynth path as default so
nothing breaks). When `splice=True`:

For each target segment (reuse existing F1/F2 measurement + the `_warp_sp_frame`
warp; do **not** change the warp math):
1. Take the original audio as the output buffer (copy of input samples).
2. Cut a window = segment ± `CONTEXT_MS` (use 30 ms) of context on each side.
3. Run `pyworld.wav2world` on **just that window**; warp only the core (non-context)
   frames toward the target centroid; `pyworld.synthesize` just that window.
4. RMS-match the synthesized window's gain to the original window (avoid level jumps).
5. **Equal-power crossfade** (use `XFADE_MS` = 20 ms, `sqrt`-shaped weights) the
   synthesized window back into the output buffer over the context regions; replace
   the core samples outright.
6. Frames outside any target segment remain the original samples untouched.

Constraints: file is 197 lines now — stay ≤ 400. Add a one-line comment on *why*
splice exists (the global round-trip loss), nothing more. Run
`uv run ruff check accent_coach/` and `uv run pytest tests/ -q`.

### A2. Validate splice vs full-resynth

Add a `--splice` flag to `scripts/accent_coach_phase0_13_lever_b.py` that threads
through to `shift_vowels_to_centroid`. Re-run **only** `cell_all5` at N=3 into a
**new** dir (do not overwrite Phase 0.13 cells):

```bash
.venv/bin/python scripts/accent_coach_phase0_13_lever_b.py \
    --cell cell_all5 --replicates 3 --splice
```

Score the same way the existing driver does (composite + WER/ECAPA/DNSMOS).
Listen to 3 clips. Append results to
`docs/accent_coach_phase0_13_lever_b_findings.md` under a new "§7 Splice mode" section.

**Stop criteria:**
- **GREEN** — DNSMOS recovers ≥ +0.20 vs full-resynth cell_all5 (2.211) AND
  composite ≥ 73.0 (full-resynth cell_all5) AND WER ≤ 0.10. → splice is the new
  default for any signal-domain shift; note it.
- **YELLOW** — DNSMOS recovers but composite drops 1–3 pts (splice boundaries cost
  some accuracy). → keep both modes, document the tradeoff.
- **RED** — audible splice clicks, or composite drops > 3 pts, or DNSMOS no better.
  → revert `splice` default to False, document failure, move on.

**Out of scope for WS-A:** the /aʊ/ diphthong collapse (−33 pts) is a
single-midpoint-target problem, not a vocoder problem; splice will not fix it.
Note it; do not attempt trajectory warping here.

---

# WS-C — LoRA fine-tuning of the GPT (close the model-bound RP gap)

**Thesis (see docs/accent_coach_phase0_13_findings.md §5 and the architecture
notes below):** accent = which discrete semantic token the GPT emits per vowel.
The GPT (`UnifiedVoice`) is the autoregressive stage that *chooses* tokens; s2mel
only *renders* them. So LoRA on the GPT is the primary lever. s2mel-LoRA is the
documented fallback if and only if GPT-LoRA proves the token codebook can't
represent RP targets (RED outcome below).

**Architecture facts (verified by reading the vendored code — do not re-derive):**
- Pipeline: GPT (`indextts/gpt/model_v2.py:UnifiedVoice`) → semantic codes →
  s2mel DiT (`s2mel.pth`) → BigVGAN. The GPT predicts discrete **mel codes**
  (vocab 8194 = 8192 codebook + start 8192 + stop 8193).
- The GPT's target codes come from the **semantic codec**: audio → w2v-BERT
  features → `get_emb` (hidden_states[17], normalized by `semantic_mean/std`,
  `infer_v2.py:218-226`) → `semantic_codec.quantize(emb)` (`infer_v2.py:446`).
- Inner transformer is **stock HF GPT2** (built in
  `model_v2.py:build_hf_gpt_transformer`, line 262 re-imports stock `GPT2Model`).
  LoRA targets live at `gpt.gpt.h[i].attn.c_attn`, `.attn.c_proj`, `.mlp.c_fc`,
  `.mlp.c_proj` (HF GPT2 `Conv1D` layers).
- `UnifiedVoice.forward` (line 589) returns **latents, not logits**
  (`return_latent=True` path, line 630). For a CE loss you must call
  `get_logits(..., return_latent=False)` so `self.mel_head` is applied — see C3.

### C1. Build the training dataset

Source audio (on disk, raw): `tts_output/modern_rp_corpus/{fry,lindsey,bbc}/clips/*.wav`
(fry 1262, lindsey 260, bbc 451). Transcripts: the cleaned-corpus `transcripts.json`
files. Quality filters already computed: `cleaned_corpus/{fry,lindsey}/kept_clips.json`
(ECAPA sim, duration).

Write `scripts/accent_coach_phase0_14_build_dataset.py` (≤ 400 lines, run in
**vendor venv**) that produces a JSONL manifest of training examples:

1. Keep only clips with `kept_clips` sim ≥ 0.50 AND `0.8 s ≤ dur ≤ 10 s` (bounds
   mel-code length under `max_mel_tokens=1815`).
2. **Transcript quality gate** — the existing transcripts contain ASR errors
   (e.g. "Optalmic", "boringy"). Re-transcribe each kept clip with the project
   Whisper (`scripts/lib/transcribe.py`) and **drop clips whose re-transcription
   disagrees with the stored transcript by WER > 0.25** (bad text → bad
   text→code supervision). Keep the re-transcribed text as the training transcript.
3. Split **by clip, 90/10 train/val**, fixed seed 0. Hold the val set out entirely.
4. Emit `tts_output/accent_coach/phase0_14/lora/dataset/{train,val}.jsonl`, each
   line `{"wav": <abspath>, "text": <str>, "speaker": "fry|lindsey|bbc"}`.

Acceptance: print train/val counts; expect train ≈ 1400–1700 after filtering.
Sanity-listen 3 random kept clips and confirm transcript matches.

### C2. Build the target-extraction module (invert inference)

Write `scripts/lib/lora_targets.py` (≤ 400 lines, vendor venv). Given a loaded
`IndexTTS2`, for one (wav, text) example produce the tensors the GPT forward needs:

- `text_inputs` = `tts.tokenizer.tokenize(text)` → `convert_tokens_to_ids` → long tensor.
- `mel_codes` (the supervision target) — invert inference:
  1. load wav, resample 16k; `tts.extract_features(...)` → input_features, mask;
  2. `emb = tts.get_emb(input_features, mask)`;
  3. `out = tts.semantic_codec.quantize(emb)`.
     **VERIFY THE RETURN SHAPE FIRST** — run a one-off:
     `print(type(out), [getattr(x,'shape',None) for x in out], [getattr(x,'dtype',None) for x in out])`
     The GPT target is the **integer index sequence** (dtype long, values in
     `[0, 8191]`, shape `(1, T)`). `infer_v2.py:446` unpacks `_, S_ref = quantize(...)`;
     confirm which element is integer codes vs quantized embedding and pick the
     integer one. If `quantize` returns codes shaped `(1, n_q, T)`, take quantizer 0.
- `speaker_ref` = a **different random clip from the same speaker** (not the clip
  itself — self-conditioning lets the GPT copy). Compute its `spk_cond_emb` and
  `emo_cond_emb` via `get_emb` exactly as `infer_v2.py:439-446,489-495` does.

Acceptance: for one fry clip, `mel_codes` is long, 1×T, max < 8192; round-trips
through `semantic_codec.quantizer.vq2emb` without error.

### C3. Training-forward + loss

In the training script (C4), do **not** call `UnifiedVoice.forward` as-is (returns
latents). Instead replicate its body but get logits. Concretely, in a
`gpt_ce_loss(gpt, batch)` helper:

1. Reproduce conditioning assembly from `UnifiedVoice.forward` (lines 604-628):
   speaker latent via `get_conditioning`, emo via `get_emo_conditioning`+`emovec_layer`+`emo_layer`,
   `conds` cat, `build_aligned_inputs_and_targets` for text and mel codes, embeddings.
2. Call `gpt.get_logits(conds, text_emb, gpt.text_head, mel_emb, gpt.mel_head,
   return_latent=False)` → `(text_logits, mel_logits)`.
3. **Loss = cross-entropy on mel codes** (the model's native objective):
   `F.cross_entropy(mel_logits, mel_targets, ignore_index=gpt.stop_mel_token)`
   where `mel_targets` is the second output of `build_aligned_inputs_and_targets`
   on `mel_codes`. This is the primary and (initially) only loss.
   - Optional auxiliary (default OFF): `+ 0.1 * CE(text_logits, text_targets)`.
     Leave a flag but start with mel-only.

This is the correct loss because the GPT *is* a next-token predictor over mel
codes; we are continuing its pretraining objective on RP data.

### C4. LoRA config + training loop (local M3, CPU/MPS)

Install PEFT into the inference venv: `cd vendor/index-tts && uv add peft && cd -`.
Write `scripts/accent_coach_phase0_14_lora_train.py` (≤ 400 lines, vendor venv).

**LoRA targets (layer-selection — the crucial decision):**
- Apply LoRA to the GPT2 backbone only:
  `target_modules=["c_attn", "c_proj", "c_fc"]` (matches attn QKV+out and MLP across
  all 24 layers by name suffix). PEFT supports HF `Conv1D`; if your PEFT version
  errors on Conv1D, pin a newer one (`uv add "peft>=0.11"`).
- `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type=None)`.
- **Freeze everything else** — especially `conditioning_encoder`,
  `perceiver_encoder`, `emo_*`. This preserves zero-shot speaker conditioning and
  guards against identity drift (the same drift risk noted in prior KD work; we
  rely on small rank + frozen conditioning + the ECAPA gate in C5 instead of a
  frozen teacher).
- Make `mel_head` optionally trainable behind a flag (default OFF for run 1).

**Loop:**
- Device: try `mps`; on the first unsupported-op error, fall back to `cpu`
  (catch and log). Use **float32** (MPS fp16 is slower per `infer_v2.py:68`).
- Batch size 1, **gradient accumulation 16**. Enable gradient checkpointing on the
  GPT2 backbone if memory pressured.
- Optimizer AdamW, **lr 1e-4**, constant LR with 50-step linear warmup.
- **Dry-run gate first**: train on **32 clips for 200 steps** and confirm train
  loss strictly decreases. If it does not, STOP and report (loss/wiring bug) — do
  not launch the full run.
- Full run: 1–3 epochs over the train split. Checkpoint LoRA adapter every 300
  steps to `tts_output/accent_coach/phase0_14/lora/ckpt/step_<N>/` via
  `peft_model.save_pretrained(...)`.
- Log train loss every 20 steps; eval loss (C5.1) every checkpoint.
- **Expect this to be slow on M3** (likely hours/epoch). That is acceptable per
  the chosen local-hardware constraint; just log throughput (steps/min) after the
  dry run so total time is predictable.

### C5. Validation (the gates the user asked for)

**C5.1 Eval loss** — every checkpoint, compute mean `gpt_ce_loss` over the held-out
val split (no grad). MUST trend **down then plateau**. If eval loss rises while
train loss falls → overfit: stop at the best checkpoint, halve LR or reduce epochs.

**C5.2 Generated-audio quality** — every checkpoint, load base IndexTTS2 + apply the
adapter (see C6), generate the fixed `cal_25.csv` set with the **BC production
reference** (`tts_output/refs/production/ref_interview.wav`), then score in `.venv`:
- **accent-coach composite vs modern_rp** (`score_against(..., "modern_rp", ...)`) —
  primary success metric; baseline to beat is the Phase 0.13 `cell_baseline_b` = **67.53**.
- **WER** (Whisper) — must stay ≤ 0.10.
- **ECAPA-to-BC** (`ref_narrator.wav`) — must stay ≥ ~0.74. This is the
  identity-preservation gate; a crash means the LoRA overrode the BC conditioning.
- **DNSMOS OVR** — naturalness; flag drops > 0.3.

**C5.3 Listening** — 5 clips per checkpoint; confirm it sounds like BC and the 5
target vowels (/ʌ ʊ ɔː aʊ ɜː/) sound more RP, not distorted.

**C5.4 Regression** — `scripts/indextts_smoke_test.py` must still pass with the
adapter **disabled** (base unchanged), and ideally with it enabled.

### C5.5 Stop criteria (GREEN / YELLOW / RED)

Evaluated on the best checkpoint by composite, gated on identity/intelligibility:

- **GREEN** — composite **+≥10** vs 67.53 (→ ≥ 77.5) AND WER ≤ 0.10 AND
  ECAPA ≥ 0.74 AND DNSMOS drop ≤ 0.3. → ship the adapter; write findings; wire C6
  into the default generation path behind a flag.
- **YELLOW** — composite **+5 to +10** with identity/WER preserved. → iterate once:
  try (a) `mel_head` trainable, (b) r=32, or (c) +1 epoch. If still YELLOW, ship as
  optional and document.
- **RED** — composite **< +5**, OR ECAPA < 0.70 (identity lost), OR WER > 0.15, OR
  eval loss won't decrease after the dry-run passed. → GPT-LoRA insufficient.
  Document, then the next experiment (NOT this phase) is the **s2mel DiT LoRA**
  ablation or a base-model swap. Do not silently expand scope here.

### C6. Inference integration

Add a `--lora-adapter <path>` flag to `scripts/indextts_gen.py`. When set, after
constructing `IndexTTS2`, wrap the GPT backbone:
`tts.gpt.gpt = PeftModel.from_pretrained(tts.gpt.gpt, <path>)` (or `load_adapter` /
`set_adapter`), keep everything else identical, regenerate. Default (flag unset) =
base model, byte-for-byte unchanged. Confirm the smoke test passes with the flag unset.

---

## Deliverables checklist

- [ ] WS-B: `genam_norms.py`, three accent refs + PROVENANCE, probe driver,
      `accent_coach_phase0_14_accent_probe_findings.md` with an explicit verdict.
- [ ] WS-A: splice path in `formant_shift.py`, `--splice` flag, §7 appended to
      lever_b findings with GREEN/YELLOW/RED verdict.
- [ ] WS-C: dataset builder + manifests, `lora_targets.py`, training script,
      best adapter checkpoint, `accent_coach_phase0_14_lora_findings.md` with the
      composite/WER/ECAPA/DNSMOS table and the stop-criteria verdict, `--lora-adapter`
      flag in `indextts_gen.py`.
- [ ] `bash scripts/check_repo.sh` green; smoke test green before/after.

## Things that will bite you (read before starting)

1. `semantic_codec.quantize` return shape — VERIFY before building the dataset (C2).
   Picking the quantized embedding instead of integer codes silently trains garbage.
2. `UnifiedVoice.forward` returns latents, not logits — use `get_logits(return_latent=False)` (C3).
3. Accent refs are female; the modern-RP norm table is **male**. For WS-B use
   sex-matched norms and lean on dist-to-source (sex-agnostic) as the primary read.
4. Speaker reference for training must be a *different* same-speaker clip, not the
   target clip itself (C2), or the GPT learns to copy rather than generalize.
5. Identity drift: if ECAPA-to-BC crashes during WS-C, the LoRA is overriding
   conditioning — reduce rank/LR or freeze more, don't push through.
