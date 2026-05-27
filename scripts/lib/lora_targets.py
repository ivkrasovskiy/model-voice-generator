"""WS-C C2 — Produce LoRA training targets from a (wav, text, speaker_ref_wav) triple.

Given a loaded IndexTTS2 instance, returns the tensors that the GPT CE-loss
forward (C3) needs:
  - text_tokens  (1, L) int32  — BPE token IDs from tts.tokenizer
  - mel_codes    (1, T) int64  — semantic codec integer indices; VERIFY:
      codes, _ = tts.semantic_codec.quantize(emb)   [int64, values in 0..8191]
      (first return = integer codes; second = quantized embeddings)
  - spk_cond_emb (1, T', 1024) float32  — get_emb of a DIFFERENT same-speaker clip
  - emo_cond_emb (1, T', 1024) float32  — same as spk_cond_emb (IndexTTS2 default)

Acceptance test (run in vendor venv):
  vendor/index-tts/.venv/bin/python -c "
  import sys; sys.path.insert(0,'vendor/index-tts'); sys.path.insert(0,'scripts')
  from indextts.infer_v2 import IndexTTS2
  from lib.lora_targets import extract_targets
  tts = IndexTTS2('vendor/index-tts/checkpoints/config.yaml',
                  'vendor/index-tts/checkpoints', device='cpu')
  t = extract_targets(tts, 'tts_output/modern_rp_corpus/fry/clips/1_000.wav',
                      'Not what we call British',
                      'tts_output/modern_rp_corpus/fry/clips/1_001.wav', 'cpu')
  print(t['mel_codes'].shape, t['mel_codes'].dtype, t['mel_codes'].max().item())
  assert t['mel_codes'].dtype.is_signed and t['mel_codes'].max() < 8192
  print('OK')
  "
"""
from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def _load_wav_16k(path: str | Path, device: str = "cpu") -> torch.Tensor:
    """Load mono WAV, resample to 16 kHz, return (1, N) float32 on device."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    return wav.to(device)


def _emb_from_wav(tts, wav_16k: torch.Tensor) -> torch.Tensor:
    """Run extract_features + get_emb, return (1, T, 1024) float32."""
    inputs = tts.extract_features(
        wav_16k, sampling_rate=16000, return_tensors="pt"
    )
    input_features = inputs["input_features"].to(wav_16k.device)
    attention_mask = inputs["attention_mask"].to(wav_16k.device)
    with torch.no_grad():
        emb = tts.get_emb(input_features, attention_mask)
    return emb


@torch.no_grad()
def extract_targets(
    tts,
    wav_path: str | Path,
    text: str,
    spk_ref_path: str | Path,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Return training tensors for one (wav, text) example.

    Parameters
    ----------
    tts          : loaded IndexTTS2 instance
    wav_path     : path to the target clip (whose semantic codes become the label)
    text         : transcript of wav_path (used for text_tokens)
    spk_ref_path : a DIFFERENT clip from the same speaker (not wav_path itself);
                   avoids self-conditioning which lets the GPT copy rather than
                   generalise to text
    device       : "cpu" or "mps"
    """
    # --- text tokens ---
    text_token_list = tts.tokenizer.tokenize(text)
    text_ids = tts.tokenizer.convert_tokens_to_ids(text_token_list)
    text_tokens = torch.tensor(text_ids, dtype=torch.int32, device=device).unsqueeze(0)

    # --- mel codes (semantic integer indices, the training label) ---
    wav_16k = _load_wav_16k(wav_path, device)
    emb = _emb_from_wav(tts, wav_16k)
    # VERIFIED: quantize returns (int_codes, quant_emb); int_codes is int64, shape (1, T)
    mel_codes, _ = tts.semantic_codec.quantize(emb)  # (1, T) int64, values 0..8191

    # --- speaker conditioning (from a different clip) ---
    ref_16k = _load_wav_16k(spk_ref_path, device)
    spk_cond_emb = _emb_from_wav(tts, ref_16k)   # (1, T', 1024)
    emo_cond_emb = spk_cond_emb                   # IndexTTS2 default: emo = spk ref

    return {
        "text_tokens":  text_tokens,   # (1, L) int32
        "mel_codes":    mel_codes,     # (1, T) int64, 0..8191
        "spk_cond_emb": spk_cond_emb, # (1, T', 1024) float32
        "emo_cond_emb": emo_cond_emb, # (1, T', 1024) float32
    }


def gpt_ce_loss(gpt, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute mel-code cross-entropy loss for one training example.

    Reproduces the UnifiedVoice.forward conditioning assembly but calls
    get_logits(return_latent=False) to get actual logits instead of latents.
    References model_v2.py lines 604-630.
    """
    import torch.nn.functional as F

    device = next(gpt.parameters()).device

    text_tokens  = batch["text_tokens"].to(device)   # (1, L) int32
    mel_codes    = batch["mel_codes"].to(device)     # (1, T) int64
    spk_emb      = batch["spk_cond_emb"].to(device)  # (1, T', 1024)
    emo_emb      = batch["emo_cond_emb"].to(device)  # (1, T', 1024)

    # Speaker conditioning: (1, T', 1024) → (1, 32, model_dim)
    spk_cond = gpt.get_conditioning(spk_emb.transpose(1, 2))

    # Emotion conditioning
    emo_vec_ori = gpt.get_emo_conditioning(emo_emb.transpose(1, 2))
    emo_vec_syn = gpt.emovec_layer(emo_vec_ori)
    emo_vec     = gpt.emo_layer(emo_vec_syn)

    # Speed embedding (always zeros = normal speed)
    use_speed        = torch.zeros(1, dtype=torch.long, device=device)
    duration_emb     = gpt.speed_emb(torch.zeros_like(use_speed))
    duration_emb_half = gpt.speed_emb(torch.ones_like(use_speed))

    conds = torch.cat(
        [spk_cond + emo_vec.unsqueeze(1),
         duration_emb_half.unsqueeze(1),
         duration_emb.unsqueeze(1)],
        dim=1,
    )  # (1, 34, model_dim)

    # Text tokens → aligned inputs + targets
    text_lengths = torch.tensor([text_tokens.shape[1]], dtype=torch.long, device=device)
    text_padded  = gpt.set_text_padding(text_tokens.clone().long(), text_lengths)
    text_padded  = F.pad(text_padded, (0, 1), value=gpt.stop_text_token)
    text_inp, _  = gpt.build_aligned_inputs_and_targets(
        text_padded, gpt.start_text_token, gpt.stop_text_token
    )
    text_emb = (gpt.text_embedding(text_inp)
                + gpt.text_pos_embedding(text_inp))

    # Mel codes → aligned inputs + targets
    mel_lengths  = torch.tensor([mel_codes.shape[1]], dtype=torch.long, device=device)
    mel_padded   = gpt.set_mel_padding(mel_codes.clone(), mel_lengths)
    mel_padded   = F.pad(mel_padded, (0, 1), value=gpt.stop_mel_token)
    mel_inp, mel_tgt = gpt.build_aligned_inputs_and_targets(
        mel_padded, gpt.start_mel_token, gpt.stop_mel_token
    )
    mel_emb = (gpt.mel_embedding(mel_inp)
               + gpt.mel_pos_embedding(mel_inp))

    # Logits: return_latent=False → actual softmax inputs (B, vocab, T)
    _, mel_logits = gpt.get_logits(
        conds, text_emb, gpt.text_head, mel_emb, gpt.mel_head,
        return_latent=False,
    )

    # CE loss on mel codes; ignore padding at stop token position
    loss = F.cross_entropy(mel_logits, mel_tgt, ignore_index=gpt.stop_mel_token)
    return loss
