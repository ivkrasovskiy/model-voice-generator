"""
Generate eval phrases with IndexTTS-2 using a SPEAKER-CENTROID reference.

Instead of conditioning on a single reference clip (the baseline), we average
the speaker-conditioning tensors (spk_cond_emb, style) across N high-quality
reference clips. The derived tensors that depend on a single time-grid
(S_ref, ref_mel, prompt_condition) are computed from one representative clip.

No vendor source is modified. We subclass IndexTTS2 and pre-seed its cache
fields before calling the unchanged .infer().

Usage:
    vendor/index-tts/.venv/bin/python scripts/indextts_centroid_gen.py \
        --phrases-csv tts_output/cross_eval_50/eval_short.csv \
        --out-dir tts_output/eval_indextts_centroid_short
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
PROJECT_ROOT = Path(__file__).parent.parent
INDEXTTS_ROOT = PROJECT_ROOT / "vendor" / "index-tts"
sys.path.insert(0, str(INDEXTTS_ROOT))

import soundfile as sf
import torch
import torchaudio
from indextts.infer_v2 import IndexTTS2

CFG_PATH = str(INDEXTTS_ROOT / "checkpoints/config.yaml")
MODEL_DIR = str(INDEXTTS_ROOT / "checkpoints")
DEFAULT_REFS_CSV = PROJECT_ROOT / "tts_output/centroid_refs.csv"
SENTINEL = "__centroid__"


def _emo_emb_from_audio(tts: IndexTTS2, audio_path: str) -> torch.Tensor:
    """Mirror infer_v2.py lines 489-495 to compute the emo_cond_emb for a clip."""
    emo_audio, _ = tts._load_and_cut_audio(audio_path, 15, sr=16000)
    emo_inputs = tts.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
    return tts.get_emb(
        emo_inputs["input_features"].to(tts.device),
        emo_inputs["attention_mask"].to(tts.device),
    )


class IndexTTS2Centroid(IndexTTS2):
    def _compute_ref_tensors(self, audio_path: str):
        """Mirror infer_v2.py lines 435-459 for a single clip."""
        audio, sr = self._load_and_cut_audio(audio_path, 15, verbose=False)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

        inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        spk_cond_emb = self.get_emb(input_features, attention_mask)

        _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
        ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(ref_mel.device),
            num_mel_bins=80, dither=0, sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self.campplus_model(feat.unsqueeze(0))

        prompt_condition = self.s2mel.models["length_regulator"](
            S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None,
        )[0]
        return {
            "spk_cond_emb": spk_cond_emb,
            "style": style,
            "S_ref": S_ref,
            "ref_mel": ref_mel,
            "prompt_condition": prompt_condition,
        }

    def _stack_and_mean_variable_dim(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """Stack tensors that differ along exactly one dim and mean across clips.

        Detects the variable axis automatically and truncates all clips to the
        minimum length along that axis before stacking.
        """
        shapes = [tuple(t.shape) for t in tensors]
        if len(set(shapes)) == 1:
            return torch.stack(tensors, dim=0).mean(dim=0)
        ref = shapes[0]
        var_dims = [
            i for i in range(len(ref))
            if any(s[i] != ref[i] for s in shapes[1:])
        ]
        if len(var_dims) != 1:
            raise RuntimeError(
                f"expected exactly one variable dim, got {var_dims} for shapes {shapes}"
            )
        d = var_dims[0]
        min_len = min(s[d] for s in shapes)
        slicer = [slice(None)] * len(ref)
        slicer[d] = slice(0, min_len)
        truncated = [t[tuple(slicer)] for t in tensors]
        return torch.stack(truncated, dim=0).mean(dim=0)

    def set_centroid_refs(self, ref_paths: list[str], representative_idx: int = 0):
        assert len(ref_paths) >= 2, "need >=2 refs for a meaningful centroid"
        print(f"  computing per-clip tensors for {len(ref_paths)} refs...")
        per_clip = []
        for i, p in enumerate(ref_paths):
            t = self._compute_ref_tensors(p)
            per_clip.append(t)
            if i == 0:
                print(
                    f"  shapes from clip 0: "
                    f"spk_cond_emb={tuple(t['spk_cond_emb'].shape)}, "
                    f"style={tuple(t['style'].shape)}, "
                    f"ref_mel={tuple(t['ref_mel'].shape)}"
                )

        spk_centroid = self._stack_and_mean_variable_dim(
            [t["spk_cond_emb"] for t in per_clip]
        )
        style_centroid = self._stack_and_mean_variable_dim(
            [t["style"] for t in per_clip]
        )

        rep = per_clip[representative_idx]
        rep_path = ref_paths[representative_idx]

        # Pre-seed the parent's spk cache. infer_v2.py line 428 checks
        # `cache_spk_cond is None or cache_spk_audio_prompt != spk_audio_prompt`.
        # Setting cache_spk_audio_prompt = SENTINEL and passing SENTINEL as
        # spk_audio_prompt on every .infer() call makes the cache hit, skipping
        # the per-call recompute.
        self.cache_spk_cond = spk_centroid
        self.cache_s2mel_style = style_centroid
        self.cache_s2mel_prompt = rep["prompt_condition"]
        self.cache_mel = rep["ref_mel"]
        self.cache_spk_audio_prompt = SENTINEL

        # Pre-seed the emo cache too — emo_audio_prompt defaults to
        # spk_audio_prompt (line 421-422). Compute emo_cond_emb from the
        # representative clip.
        self.cache_emo_cond = _emo_emb_from_audio(self, rep_path)
        self.cache_emo_audio_prompt = SENTINEL

        print(
            f"  centroid built: spk_cond_emb {tuple(spk_centroid.shape)}, "
            f"style {tuple(style_centroid.shape)}, "
            f"representative={Path(rep_path).name}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrases-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--refs-csv", default=str(DEFAULT_REFS_CSV),
                        help="CSV with column 'wav_path' listing centroid reference clips")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    refs_csv = Path(args.refs_csv)
    with open(refs_csv) as f:
        ref_rows = list(csv.DictReader(f))
    ref_paths = [str(PROJECT_ROOT / r["wav_path"]) for r in ref_rows]
    print(f"Centroid refs: {len(ref_paths)} clips from {refs_csv}")

    phrases_csv = Path(args.phrases_csv)
    with open(phrases_csv) as f:
        phrases = list(csv.DictReader(f))
    print(f"Phrases: {len(phrases)} from {phrases_csv}")

    print(f"Loading IndexTTS-2 (centroid subclass) on {args.device}...")
    tts = IndexTTS2Centroid(cfg_path=CFG_PATH, model_dir=MODEL_DIR, device=args.device)

    print("Building speaker centroid...")
    tts.set_centroid_refs(ref_paths, representative_idx=0)

    manifest = []
    for row in phrases:
        slug, prompt = row["slug"], row["prompt"]
        out_path = out_dir / f"indextts_centroid_{slug}.wav"
        print(f"\n--- {slug}: {prompt!r}")
        if out_path.exists():
            print("  already exists, skipping generation")
        else:
            tts.infer(
                spk_audio_prompt=SENTINEL,
                text=prompt,
                output_path=str(out_path),
                emo_audio_prompt=SENTINEL,
            )
        info = sf.info(str(out_path))
        print(f"  → {out_path.name} ({info.duration:.1f}s, sr={info.samplerate})")
        manifest.append({
            "label": "indextts_centroid",
            "step": 0,
            "slug": slug,
            "prompt": prompt,
            "wav_path": str(out_path),
            "sr": int(info.samplerate),
        })

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n✓ {len(manifest)} clips → {out_dir}")
    print(f"✓ manifest → {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
