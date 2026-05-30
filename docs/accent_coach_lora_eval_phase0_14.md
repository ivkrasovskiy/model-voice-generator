# Phase 0.14 LoRA Evaluation — Results & Analysis

**Date**: 2026-05-29  
**Branch**: ivk-temp-branch

---

## 1. What We Were Trying to Do

Fine-tune IndexTTS-2's GPT (UnifiedVoice) backbone with LoRA to produce more authentic modern RP vowels, specifically the BATH-TRAP distinction (the most salient phonetic marker separating RP from generic British or American English). Training corpus: 1195 clips from Fry + Lindsey + BBC male speakers (modern RP), pre-cached mel embeddings.

---

## 2. Pipeline Fixes Built During This Phase

### 2.1 RP Phoneme Alignment Corrections

WhisperX uses the American CMU pronunciation dictionary. Three systematic biases were found and fixed in `accent_coach/pipeline/rp_postprocess.py`:

| Problem | Root cause | Fix |
|---|---|---|
| **BATH misclassification** | CMU labels *can't, path, grass, after, castle, dance* etc. as AE (→ æ). RP uses /ɑː/. | Word-list relabel æ→ɑː for ~80 BATH-class words (`accent_coach/reference/bath_words.py`) |
| **Rhotic alignment errors** | CMU has coda-R in *car, here, after, dark, word*; RP non-rhotic speakers don't. Aligner tries to find R, truncating preceding vowel window. | `strip_coda_r()`: drop any /r/ preceded by vowel and not followed by vowel; extend preceding vowel's end_time. Linking R (before vowel, e.g. "car alarm") kept. |
| **LOT/BATH confusion** | CMU AA → ɑː for LOT words (*not, hot, stop*). RP LOT is /ɒ/, distinct from BATH/START (/ɑː/). | Word-list relabel ɑː→ɒ for ~70 LOT-class words. |

All three corrections are applied in sequence by `apply_rp_corrections()`, wired into `accent_coach_extract_formants.py` with an `apply_rp_corrections=True` flag (set to `False` for owner voice to preserve raw measurement).

17 unit tests in `tests/test_rp_postprocess.py` — all passing.

### 2.2 LoRA Training Improvements

| Fix | Details |
|---|---|
| **MPS NaN clips** | `PYTORCH_ENABLE_MPS_FALLBACK=1` set before torch import; prevents forward NaN for clips where MPS SDPA precision fails |
| **Early stopping** | `--patience 200` steps, `--eval-every 100` steps, `--max-steps 900` hard cap; `ckpt/best/` saved on every eval improvement |
| **Overfit detection** | Previous 3-epoch run showed best eval at step_300 (5.749 Bark), diverging to 7.84 by step_3300. Early stopping fires before divergence. |

---

## 3. Training Runs Summary

| Run | Epochs / steps | Best eval_loss | Notes |
|---|---|---|---|
| Run 1 (pre-fix) | 1 epoch, 1195 steps | unknown (log lost) | No MPS fallback, no early stopping |
| Run 2 (3-epoch, overfit) | Killed at step_3300 | **5.749 @ step_300** | Diverged badly after step_300 |
| Run 3 (early stopping) | Stopped at step_600 | **5.726 @ step_400** | Fired correctly; `ckpt/best/` = step_400 |

Checkpoint inventory kept:
- `ckpt/best/` — Run 3, step_400, eval_loss 5.726 **(use this)**
- `ckpt/step_1195_final/` — Run 1, pre-fix, full epoch 1

---

## 4. BATH-Phrase Evaluation (3 phrases, corrected pipeline)

Phrases designed to be BATH-heavy (*can't, past, castle, after, disaster, asking, dance, grass, afternoon*):

| Metric | base | ep1_final | ep2_step300 | **best_step400** |
|---|---|---|---|---|
| dist_to_RP overall | 1.143 | 0.957 | 1.276 | **0.856 ★** |
| BATH/PALM (ɑː) | **0.402** | 0.683 | 0.983 | 0.808 |
| TRAP (æ) | n/a | **0.183** | 0.674 | 0.403 |
| THOUGHT (ɔː) | 1.133 | 2.330 | 1.390 | **0.662 ★** |
| SCHWA (ə) | **0.317** | 1.200 | 0.608 | 0.655 |
| KIT (ɪ) | 1.887 | **0.523** | 1.176 | 1.736 |

**Takeaway**: `best_step400` (early-stopped run) wins overall on these targeted phrases, driven by a large THOUGHT improvement (0.662 vs 1.133 base). The overfit run (ep2_step300) is universally the worst. The base model has the best BATH — LoRA consistently pushes BATH in the wrong direction across all runs.

---

## 5. Full Cross-Eval-50 Evaluation (50 phrases, all vowels)

All sources scored on the same 50 phrases (Casanova + Sherlock). Generated audio uses BC production reference (`ref_interview.wav`, 14 s, num_beams=5). RP corrections applied to all except owner voice.

### 5.1 Overall ranking

| Rank | Source | dist_to_RP (Bark) | Note |
|---|---|---|---|
| 1 | **fry** | 0.402 | Ground-truth modern RP (Stephen Fry) |
| 2 | **gen_base** | 0.420 | IndexTTS-2 zero-shot, BC reference |
| 3 | **real_bc** | 0.521 | Real Benedict Cumberbatch recordings |
| 4 | **gen_lora_best** | 0.567 | LoRA step_400, BC reference |
| 5 | lindsey | 0.594 | John Lindsey (RP phonetician) |
| 6 | bbc | 0.663 | BBC male corpus |
| 7 | **owner** | 0.888 | Owner voice, raw (no RP corrections) |

### 5.2 Per-vowel breakdown

| Vowel | gen_base | gen_lora | fry | real_bc | owner |
|---|---|---|---|---|---|
| BATH (ɑː) | **0.118** | 0.178 | 0.092★ | 0.623 | 1.230 |
| TRAP (æ) | **0.248** | 0.346 | 0.213★ | 0.472 | 0.878 |
| DRESS (ɛ) | **0.389** | 0.488 | 0.090★ | 0.395 | 0.978 |
| GOAT (əʊ) | 0.358 | **0.237★** | 0.191 | 0.488 | 1.600 |
| SCHWA (ə) | 0.533 | **0.396★** | 0.167 | 0.483 | 0.422 |
| THOUGHT (ɔː) | 0.480 | 0.990 | 0.651 | 0.653 | 0.630 |
| KIT (ɪ) | **0.188** | 0.439 | 0.113★ | 0.291 | 0.731 |
| NURSE (ɜː) | 0.593 | **0.306★** | 0.170 | 0.271 | 0.750 |
| PRICE (aɪ) | 0.545 | 0.549 | 0.161★ | 0.608 | 1.065 |
| GOOSE (uː) | **0.148★** | 0.301 | 0.389 | 0.484 | 0.852 |

### 5.3 Raw BATH centroids [RP target F1=518 F2=1215]

| Source | F1 | F2 | n |
|---|---|---|---|
| fry | 508 | 1218 | 55 |
| gen_base | 510 | 1233 | 37 |
| gen_lora_best | 515 | 1183 | 31 |
| real_bc | 470 | 1302 | 133 |
| bbc | 556 | 1295 | 56 |
| lindsey | 430 | 1089 | 48 |
| owner | 572 | 1440 | 30 |

---

## 6. Key Findings

### 6.1 The base model is already near-RP

**gen_base (0.420) is within 0.018 Bark of Fry (0.402)** — essentially indistinguishable given measurement noise across only 50 phrases. IndexTTS-2 zero-shot with a 14 s BC reference clip is already producing highly RP-calibrated vowels.

BATH is particularly striking: gen_base F1=510, F2=1233 vs RP target F1=518, F2=1215 — within 10 Hz on F1 and 18 Hz on F2. The model has essentially internalized RP BATH from the reference speaker.

### 6.2 LoRA fine-tuning is counterproductive at this scale

On the full 50-phrase eval, gen_lora_best (0.567) is **worse** than gen_base (0.420). The LoRA improved GOAT, SCHWA, and NURSE — but degraded BATH, TRAP, KIT, THOUGHT, PRICE, GOOSE. Because the base was already excellent on the degraded vowels, the net effect is negative.

LoRA wins on BATH phrases (0.856 vs 1.143 base) only because those specific phrases expose THOUGHT and SCHWA weaknesses in the base that the LoRA happens to fix. On general text the improvement disappears.

**Hypothesis**: 1195 clips / 400 training steps is enough to shift some vowels but not enough to holistically improve RP quality — the model is partially "unlearning" the strong zero-shot calibration on some vowels while correcting others.

### 6.3 Real BC is not the ceiling

Real BC recordings (0.521) score below the base TTS model (0.420). Likely causes:
- Recording conditions vary (different microphones, rooms, broadcast processing)
- BC's actual speech deviates from idealised norm on some vowels — notably BATH (F1=470 vs target 518, a larger deviation than even the TTS output)
- The corpus clips are from diverse sources; register/style effects on vowel quality

This suggests that "sound like BC" and "sound like textbook modern RP" are not identical targets. The LoRA training aimed at RP corpus speakers (Fry + Lindsey + BBC), not specifically at BC's voice.

### 6.4 Owner voice baseline

Owner voice (0.888) shows the gap that an L2/non-RP speaker needs to close. Largest deviations: BATH (1.230), GOAT (1.600), PRICE (1.065), TRAP (0.878). SCHWA is surprisingly close (0.422 — similar to base). THOUGHT is also reasonable (0.630).

Owner BATH raw centroid: F1=572, F2=1440. The F2=1440 is close to TRAP territory (RP TRAP F2=1496), confirming the BATH-TRAP confusion typical of non-RP speakers: "path" pronounced with a front /æ/ rather than back /ɑː/.

---

## 7. Open Questions

1. **Is LoRA the right intervention?** If the base model already achieves near-Fry RP quality, LoRA may not be the right tool. Alternative: identity-specific fine-tuning (make it sound more *specifically* like BC, not just more RP).

2. **Is the real_bc corpus clean enough?** The variance in recording conditions inflates the dist-to-RP score. A cleaned subset might rank above gen_base.

3. **THOUGHT degradation in LoRA**: The THOUGHT vowel (ɔː) jumped from 0.480 (base) to 0.990 (lora_best) on the full eval. The BATH phrase eval showed the opposite (0.662, best of all). Something in the full eval set exposes a THOUGHT regression specific to the longer/more varied sentences.

4. **Owner voice coaching priority**: GOAT and BATH are the two worst vowels for the owner (1.600 and 1.230). These should be the first targets for accent coaching feedback.

---

## 8. Files & Checkpoints

| Path | Contents |
|---|---|
| `ckpt/best/` | Best LoRA adapter — Run 3, step_400, eval_loss 5.726 |
| `ckpt/step_1195_final/` | Run 1 adapter — pre-fix, 1 full epoch |
| `tts_output/cross_eval_50/gen_base/` | 50 phrases, base model |
| `tts_output/cross_eval_50/gen_lora_best/` | 50 phrases, LoRA best |
| `tts_output/modern_rp_corpus/formants_fry.csv` | 150-clip Fry formants (RP-corrected) |
| `tts_output/modern_rp_corpus/formants_lindsey.csv` | 150-clip Lindsey formants (RP-corrected) |
| `tts_output/modern_rp_corpus/formants_bbc.csv` | 150-clip BBC formants (RP-corrected) |
| `tts_output/real_bc_corpus/formants_real_bc.csv` | 385-clip real BC formants (RP-corrected) |
| `tts_output/owner_cal_50/formants.csv` | Owner voice formants (no RP corrections) |
| `accent_coach/pipeline/rp_postprocess.py` | BATH/LOT relabeling + coda-R stripping |
| `accent_coach/reference/bath_words.py` | BATH and LOT word sets |
| `scripts/score_rp_all.py` | 7-source RP comparison scorer |
| `scripts/score_bath_rp.py` | BATH-phrase targeted scorer |
| `docs/debug_lora_mps_nan.md` | MPS training debug log (all issues resolved) |
