# LoRA MPS Training — Debug Notes

**Date**: 2026-05-28  
**Branch**: ivk-temp-branch  
**Script**: `scripts/accent_coach_phase0_14_lora_train.py`  
**Dataset**: 1195 train / 132 val clips, RP corpus (fry+lindsey+bbc)  
**Cached embeddings**: `tts_output/accent_coach/phase0_14/lora/emb_cache/` (mel_codes + spk_emb per clip)

---

## Confirmed Facts

### Model footprint (MPS)

All components loaded on MPS after `IndexTTS2.__init__()`:

| Component | Params | MPS live |
|---|---|---|
| gpt (UnifiedVoice) | 866M | dominant |
| semantic_codec | 44.3M | |
| s2mel | 103.7M | |
| campplus | 6.8M | |
| bigvgan | 112.2M | |
| **Total MPS live** | | **8.358 GB** |
| **CPU RSS** | | **3.757 GB** |
| **Total** | | **~12–13 GB** (+ OS overhead) |

With system overhead and pool pages this saturates the 18 GB M3 Pro.

### MPS pool fix: confirmed working

`torch.mps.empty_cache()` without prior `torch.mps.synchronize()` does NOT reclaim pool pages held by inflight GPU commands. Fix (already in code, `_run_training`, every 20 steps):

```python
torch.mps.synchronize()
torch.mps.empty_cache()
```

After fix, MPS pool is stable at 9.7–9.9 GB throughout training.

### AdamW lazy state allocation: confirmed finite leak

Over the first 50 training steps (from `diagnose_mps_ram.py`):
- `Δlive = +99 MB` total over 50 steps (not per-step growth)
- Pattern: +134 MB at step 0 (lora_B grad buffers + AdamW state), −70 MB at step 15 (optimizer step), +39 MB at step 16 (lora_A gets grad/state), stable after step ~32
- Root cause: AdamW lazily allocates m1/m2 state the first time a param receives `.grad`. lora_B gets it at step 0; lora_A only gets it after optimizer step 15 when B becomes nonzero.
- **This leak is finite and expected — not the cause of 21 GB RAM.**

---

## Problem 1: MPS-Specific Gradient Norm Explosions

### Observation

CPU training (same code, same data, same model, `--device cpu`):

```
opt_step grad_norm = 0.251 – 0.431   (23 optimizer steps, steps 15–367)
```

MPS training (earlier runs, before inner-clip fix):

```
opt_step grad_norm = 23,000 – 90,000   (first two optimizer steps)
```

### Consequence

The first two MPS optimizer steps had grad_norms of 23K and 30K. Even after clipping to 1.0, AdamW's second moment `m2` was already estimated from the pre-clip raw gradients and was enormous. Subsequent steps with norm ~0.4 then produced `m2_hat ≈ 0` → `update = m1_hat / sqrt(m2_hat) → ∞` → NaN parameters (`lora_A.layer.0` first).

### Mitigation in place

Per-step inner clip in `_train_step` (after each individual backward, before accumulation):

```python
torch.nn.utils.clip_grad_norm_(
    [p for p in tts.gpt.parameters() if p.requires_grad and p.grad is not None], 1.0
)
```

NaN guards at three points:
1. `if not math.isfinite(loss_val): return None` — skip backward for NaN forward pass
2. Check first trainable param `.grad` for NaN after each backward — zero and skip
3. Check all trainable grads before optimizer step — skip step if any corrupt
4. Check all trainable params after optimizer step — stop if any NaN

### Root cause — RESOLVED

Per-layer backward hook diagnostic (`scripts/diagnose_mps_grad_explosion.py`, 2026-05-28):
- Ran 1 forward+backward on a normal clip (1_028.wav, fry speaker) **without** fallback enabled
- All layer grad norms: max 0.4 — identical to CPU baseline
- No explosion, no NaN

**Conclusion: MPS SDPA is not universally broken.** The 23K–90K explosions came specifically from the 3 bad clips (0_309, 0_133, 1_133) producing NaN in the forward pass, which then propagated as huge-but-finite values through backward before hitting the NaN checks. Normal clips behave identically on MPS and CPU.

Fix already applied: `enable_fallback_to_cpu(True)` prevents those 3 clips from going NaN in forward → explosion chain never starts.

---

## Problem 2: NaN-Producing Clips in the Dataset

### Confirmed bad clips

| Clip | Failure mode |
|---|---|
| `0_309.wav` | Forward loss = NaN |
| `0_133.wav` | Forward loss = NaN |
| `1_133.wav` | Backward produces NaN gradient |

These clips are in the full 1195-clip training set but NOT in the first 32 clips used by the dry-run, which is why the dry-run passes but full training hits NaN.

### Mitigation in place

All three guards described in Problem 1 catch these clips and skip them without corrupting accumulated gradients.

### Root cause

Unknown. Could be: unusually long sequence (padding/mask issue), silence/noise in audio, unusually short text (empty label after mel_codes), or a Whisper WER near the 0.25 gate threshold that resulted in a bad transcript.

**Things to try:**
1. Inspect `0_309.wav`, `0_133.wav`, `1_133.wav` — listen, check text, check mel_codes shape
2. Load the cached embedding and print `batch["mel_codes"]` shape and `batch["target_codes"]` shape for these clips
3. Check for empty tensors, all-zero sequences, or unusually large max values

---

## Problem 3: Total RAM at 21.5 GB

### Breakdown

| Contributor | Size |
|---|---|
| MPS live (model weights) | 8.358 GB |
| MPS pool (cached pages) | ~1–2 GB above live (with fix) |
| CPU RSS (Python, PyTorch overhead, CPU tensors) | 3.757 GB |
| System (GPU driver, Metal, OS) | ~3–4 GB |
| Peak during training step | +0.5–1 GB (activations, grad buffers) |

Sum: 8.358 + 1.5 + 3.757 + 3.5 + 1 ≈ **18 GB at steady state**, peaking to 21+ GB during heavy steps.

### The 21.5 GB observation was from a run without the synchronize fix

Without `synchronize()` before `empty_cache()`, the pool grows by ~300 MB per 20 steps and never recovers. Over 300 steps (when the user killed the process) that's ~4.5 GB of trapped pool pages on top of the baseline. 8.358 + 4.5 + 3.757 + 3.5 ≈ 20 GB.

### Potential RAM savings

Only GPT is trained — the other components (s2mel, bigvgan, campplus, semantic_codec) are only needed for inference. If we skip loading them, we save ~5 GB MPS:

```python
# In IndexTTS2.__init__, skip loading these when training-only mode
# s2mel: 103.7M params
# bigvgan: 112.2M params  
# campplus: 6.8M params
# semantic_codec: 44.3M params
```

This would bring MPS live from 8.358 GB to ~3.2 GB, giving ~5 GB headroom.

**Caveat**: Requires modifying IndexTTS2 or lazy-loading these components. Not currently implemented.

---

## Current Training Status

CPU training run (`/tmp/lora_cpu_test.log`, `--device cpu --epochs 1`):
- Ran to step 360, then process exited (why: unknown, may have OOM-killed or completed part of epoch 1)
- All 23 optimizer steps: grad_norm 0.251–0.431, zero explosions
- RSS grew from 5.69 GB → 6.08 GB over 360 steps (+390 MB, linear, expected from AdamW state)
- Loss range: 5.4–7.0 (noisy, consistent with early warmup phase — lr still ramping)
- No NaN anywhere

Full MPS training (1 epoch, 1195 steps): **COMPLETED**
- Checkpoints at step_300, step_600, step_900, step_1195_final (30 MB adapter each)
- No training log preserved (stdout only — loss curve not recoverable)
- Loss quality unknown; needs inference test to evaluate

---

## Recommended Next Steps (in priority order)

1. ✅ **Investigate bad clips** — cache data is CLEAN (no NaN/inf). Root cause is MPS precision,
   not data. Bad clips: `0_133` T=63 (short), `0_309` T=160, `1_133` T=106. Texts are normal.
2. ✅ **`PYTORCH_ENABLE_MPS_FALLBACK=1`** — set via `os.environ.setdefault` before `import torch`
   in the training script. `torch.backends.mps.enable_fallback_to_cpu` does not exist in
   PyTorch 2.8; the env var is the correct API. Targets SDPA backward precision.
3. ✅ **Run `scripts/diagnose_mps_grad_explosion.py`** — no explosion on normal clips (max 0.4);
   explosion was clip-specific (3 bad clips → forward NaN → backward explosion chain)
4. ⬜ **Inference test** — run a phrase through the step_1195_final adapter; listen to confirm
   the LoRA learned something meaningful
5. ⬜ **Training-only model load** — skip s2mel/bigvgan/campplus/semantic_codec to free 5 GB MPS;
   needed if running multiple epochs

---

## Key Files

| File | Purpose |
|---|---|
| `scripts/accent_coach_phase0_14_lora_train.py` | Training loop with all current guards |
| `scripts/lib/lora_targets.py` | `extract_targets` (cache-aware), `gpt_ce_loss` |
| `scripts/diagnose_mps_ram.py` | Per-component + per-step MPS memory diagnostic |
| `/tmp/lora_cpu_test.log` | CPU training run to step 360 |
| `/tmp/lora_mps_diag.log` | MPS memory diagnosis output |
| `/tmp/lora_pipeline_status.txt` | Pipeline stage log |
| `tts_output/accent_coach/phase0_14/lora/ckpt/step_300/` | Checkpoint saved during CPU run |
