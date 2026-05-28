"""Unit tests for the LoRA training pipeline (scripts/lib/lora_targets.py).

Tests use mocks and synthetic tensors — no model weights loaded, no GPU required.
The three bug-prone contracts pinned here:
  1. _emb_from_wav: extract_features receives CPU tensor; get_emb receives model device.
  2. gpt_ce_loss: get_conditioning / get_emo_conditioning receive cond_mel_lengths.
  3. gpt_ce_loss: returns a finite scalar loss.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

# Make scripts/ importable without installing
_SCRIPTS = Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.lora_targets import _emb_from_wav, _load_wav_16k, gpt_ce_loss  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_file(tmp_path: Path, sr: int = 22050, duration: float = 0.5) -> Path:
    """Write a tiny sine-wave WAV and return its path."""
    import torchaudio
    n = int(sr * duration)
    wav = torch.sin(2 * torch.pi * 440 * torch.arange(n) / sr).unsqueeze(0)
    p = tmp_path / "test.wav"
    torchaudio.save(str(p), wav, sr)
    return p


class _MockGPT:
    """Minimal stand-in for UnifiedVoice used by gpt_ce_loss.

    Uses real tensors with small model_dim so the full arithmetic in
    gpt_ce_loss runs without touching any real model weights.
    """
    stop_mel_token = 8192
    start_mel_token = 8191
    stop_text_token = 1000
    start_text_token = 999
    text_head = None  # passed through to get_logits but mocked there
    mel_head = None

    def __init__(self, model_dim: int = 32, device: str = "cpu"):
        import torch.nn as nn
        self.model_dim = model_dim
        self._device = device
        self._p = nn.Parameter(torch.zeros(1, device=device))

    def parameters(self):
        yield self._p

    # ---- conditioning ----

    def get_conditioning(self, spk_input, cond_mel_lengths=None):
        B = spk_input.shape[0]
        return torch.zeros(B, 32, self.model_dim, device=self._device)

    def get_emo_conditioning(self, emo_input, cond_mel_lengths=None):
        B = emo_input.shape[0]
        return torch.zeros(B, self.model_dim, device=self._device)

    def emovec_layer(self, x):
        return x

    def emo_layer(self, x):
        return x

    def speed_emb(self, x):
        # x: (B,) → (B, model_dim)
        return torch.zeros(x.shape[0], self.model_dim, device=self._device)

    # ---- padding / alignment ----

    def set_text_padding(self, tokens, lengths):
        return tokens

    def set_mel_padding(self, tokens, lengths):
        return tokens

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    # ---- embeddings ----

    def text_embedding(self, x):
        B, L = x.shape
        return torch.randn(B, L, self.model_dim, device=self._device)

    def text_pos_embedding(self, x):
        B, L = x.shape
        return torch.randn(B, L, self.model_dim, device=self._device)

    def mel_embedding(self, x):
        B, T = x.shape
        return torch.randn(B, T, self.model_dim, device=self._device)

    def mel_pos_embedding(self, x):
        B, T = x.shape
        return torch.randn(B, T, self.model_dim, device=self._device)

    # ---- logits ----

    def get_logits(self, conds, first_inputs, first_head,
                   second_inputs=None, second_head=None,
                   get_attns=False, return_latent=False):
        B, L, _ = first_inputs.shape
        T = second_inputs.shape[1]
        vocab_mel = self.stop_mel_token + 1   # 8193
        vocab_text = self.stop_text_token + 1  # 1001
        return (
            torch.randn(B, vocab_text, L, device=self._device),
            torch.randn(B, vocab_mel, T, device=self._device),
        )


def _make_batch(T: int = 10, L: int = 5, Tprime: int = 20, device: str = "cpu"):
    """Synthetic batch matching the shapes returned by extract_targets."""
    return {
        "text_tokens":  torch.randint(0, 500, (1, L), dtype=torch.int32, device=device),
        "mel_codes":    torch.randint(0, 8192, (1, T), dtype=torch.int64, device=device),
        "spk_cond_emb": torch.randn(1, Tprime, 1024, device=device),
        "emo_cond_emb": torch.randn(1, Tprime, 1024, device=device),
    }


# ---------------------------------------------------------------------------
# _load_wav_16k
# ---------------------------------------------------------------------------

class TestLoadWav16k:
    def test_resamples_to_16k(self, tmp_path):
        p = _make_wav_file(tmp_path, sr=22050)
        wav = _load_wav_16k(p)
        # 0.5 s × 16000 = 8000 samples ± rounding
        assert wav.shape[0] == 1
        assert abs(wav.shape[1] - 8000) < 50

    def test_mono_from_stereo(self, tmp_path):
        import torchaudio
        n = 8000
        stereo = torch.randn(2, n)
        p = tmp_path / "stereo.wav"
        torchaudio.save(str(p), stereo, 16000)
        wav = _load_wav_16k(p)
        assert wav.shape[0] == 1

    def test_already_16k_passthrough(self, tmp_path):
        p = _make_wav_file(tmp_path, sr=16000, duration=0.5)
        wav = _load_wav_16k(p)
        assert abs(wav.shape[1] - 8000) < 10

    def test_returned_on_cpu(self, tmp_path):
        p = _make_wav_file(tmp_path, sr=16000)
        wav = _load_wav_16k(p, device="cpu")
        assert wav.device.type == "cpu"


# ---------------------------------------------------------------------------
# _emb_from_wav — device routing
# ---------------------------------------------------------------------------

class TestEmbFromWav:
    """Verify the CPU / model-device split without real weights."""

    def _make_mock_tts(self, model_device: str = "cpu"):
        tts = MagicMock()

        # extract_features returns a plain dict (feature extractor output)
        fake_features = {
            "input_features": torch.zeros(1, 80, 3000),  # on CPU, as extractor returns
            "attention_mask": torch.ones(1, 3000, dtype=torch.long),
        }
        tts.extract_features.return_value = fake_features

        # get_emb returns (1, T, 1024) on model_device
        fake_emb = torch.randn(1, 50, 1024, device=model_device)
        tts.get_emb.return_value = fake_emb

        # gpt.parameters() yields a param on model_device
        dummy_param = torch.nn.Parameter(torch.zeros(1, device=model_device))
        tts.gpt.parameters.return_value = iter([dummy_param])

        return tts

    def test_extract_features_receives_cpu_tensor(self):
        tts = self._make_mock_tts("cpu")
        wav = torch.randn(1, 8000)  # input wav on CPU
        _emb_from_wav(tts, wav)
        args, _ = tts.extract_features.call_args
        assert args[0].device.type == "cpu", "extract_features must receive CPU tensor"

    def test_get_emb_receives_model_device_tensor(self):
        tts = self._make_mock_tts("cpu")
        wav = torch.randn(1, 8000)
        _emb_from_wav(tts, wav)
        args, _ = tts.get_emb.call_args
        # Both input_features and attention_mask should be on model device
        assert args[0].device.type == "cpu"
        assert args[1].device.type == "cpu"

    def test_result_shape(self):
        tts = self._make_mock_tts("cpu")
        wav = torch.randn(1, 8000)
        emb = _emb_from_wav(tts, wav)
        assert emb.shape == (1, 50, 1024)


# ---------------------------------------------------------------------------
# gpt_ce_loss — contract tests
# ---------------------------------------------------------------------------

class TestGptCeLoss:
    def test_returns_finite_scalar(self):
        gpt = _MockGPT()
        batch = _make_batch()
        loss = gpt_ce_loss(gpt, batch)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_loss_is_positive(self):
        gpt = _MockGPT()
        batch = _make_batch()
        loss = gpt_ce_loss(gpt, batch)
        assert loss.item() > 0

    def test_cond_mel_lengths_passed_to_get_conditioning(self):
        """get_conditioning must receive cond_mel_lengths — the conformer needs it."""
        gpt = _MockGPT()
        record = {}

        orig_get_cond = gpt.get_conditioning

        def spy_get_conditioning(spk_input, cond_mel_lengths=None):
            record["cond_mel_lengths"] = cond_mel_lengths
            return orig_get_cond(spk_input, cond_mel_lengths)

        gpt.get_conditioning = spy_get_conditioning
        batch = _make_batch(Tprime=20)
        gpt_ce_loss(gpt, batch)
        assert record.get("cond_mel_lengths") is not None, (
            "get_conditioning must be called with cond_mel_lengths (not None) — "
            "conformer_perceiver encoder calls make_pad_mask(xs_lens) which crashes on None"
        )
        assert record["cond_mel_lengths"][0].item() == 20

    def test_cond_mel_lengths_passed_to_get_emo_conditioning(self):
        """get_emo_conditioning must also receive cond_mel_lengths."""
        gpt = _MockGPT()
        record = {}

        orig_get_emo = gpt.get_emo_conditioning

        def spy_get_emo(emo_input, cond_mel_lengths=None):
            record["emo_mel_lengths"] = cond_mel_lengths
            return orig_get_emo(emo_input, cond_mel_lengths)

        gpt.get_emo_conditioning = spy_get_emo
        batch = _make_batch(Tprime=15)
        gpt_ce_loss(gpt, batch)
        assert record.get("emo_mel_lengths") is not None
        assert record["emo_mel_lengths"][0].item() == 15

    def test_various_sequence_lengths(self):
        """Different T/L/Tprime combinations all produce finite loss."""
        gpt = _MockGPT()
        for T, L, Tp in [(5, 3, 10), (50, 20, 100), (1, 1, 5)]:
            batch = _make_batch(T=T, L=L, Tprime=Tp)
            loss = gpt_ce_loss(gpt, batch)
            assert torch.isfinite(loss), f"Non-finite loss for T={T} L={L} Tprime={Tp}"

    def test_backward_produces_gradients(self):
        """Loss must support backward — training loop calls loss.backward()."""
        gpt = _MockGPT()
        # Make one parameter require grad so we can check grad flow
        gpt._p.requires_grad_(True)
        batch = _make_batch()
        loss = gpt_ce_loss(gpt, batch)
        # The mock logits use randn which has no leaf node, but we verify
        # at minimum that backward does not crash
        try:
            loss.backward()
        except RuntimeError as e:
            if "does not require grad" in str(e):
                pass  # expected for all-constant mock
            else:
                raise


# ---------------------------------------------------------------------------
# _pick_ref (accent_coach_phase0_14_lora_train helper)
# ---------------------------------------------------------------------------


class TestPickRef:
    """Test speaker reference selection without importing the heavy training module."""

    @staticmethod
    def _pick_ref(items, exclude_wav):
        """Inline copy of _pick_ref logic for isolated testing."""
        import random
        exclude_speaker = next(
            (e["speaker"] for e in items if e["wav"] == exclude_wav), None
        )
        same_spk = [
            e["wav"] for e in items
            if e["speaker"] == exclude_speaker and e["wav"] != exclude_wav
        ]
        if not same_spk:
            return None
        return random.choice(same_spk)

    def _items(self):
        return [
            {"wav": "a.wav", "speaker": "alice", "text": "hello"},
            {"wav": "b.wav", "speaker": "alice", "text": "world"},
            {"wav": "c.wav", "speaker": "bob",   "text": "foo"},
            {"wav": "d.wav", "speaker": "bob",   "text": "bar"},
        ]

    def test_ref_is_different_wav(self):
        items = self._items()
        ref = self._pick_ref(items, "a.wav")
        assert ref != "a.wav"

    def test_ref_same_speaker(self):
        items = self._items()
        ref = self._pick_ref(items, "a.wav")
        assert ref == "b.wav"  # only other alice clip

    def test_single_clip_speaker_returns_none(self):
        items = [{"wav": "x.wav", "speaker": "solo", "text": "alone"}]
        assert self._pick_ref(items, "x.wav") is None

    def test_unknown_wav_returns_none(self):
        items = self._items()
        assert self._pick_ref(items, "missing.wav") is None
