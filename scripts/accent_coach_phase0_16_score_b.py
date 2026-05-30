"""Score Track B outputs: GA cloning + identity/accent disentanglement (Phase 0.16).

For each generated run, measures BOTH:
  - accent  : per-token distance to RP vs GenAm + rhoticity (coach_metrics)
  - identity: mean ECAPA cosine of the output to BC and to the Huberman donor

Answers:
  B1  clone_huberman          → does spk=GA produce GA accent? (accent should be
                                GenAm + rhotic; identity≈Huberman, not BC)
  B2i disentangle_BC+emoHub   → does the emo prompt carry accent? (if identity
                                stays BC but accent moves toward GA → disentangled)

  vendor/index-tts/.venv/bin/python is NOT needed — uses the scoring .venv.
  .venv/bin/python scripts/accent_coach_phase0_16_score_b.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from accent_coach_extract_formants import extract_formants  # noqa: E402

from accent_coach.diagnostics.coach_metrics import (  # noqa: E402
    load_tokens,
    rhoticity,
    score_source,
)
from scripts.lib.identity import cosine, embed_file, embed_wav, load_ecapa  # noqa: E402

P16 = PROJECT_ROOT / "tts_output/accent_coach/phase0_16"
BC_REF = PROJECT_ROOT / "tts_output/refs/production/ref_interview.wav"
HUB_REF = PROJECT_ROOT / "tts_output/refs/genam/huberman_ref.wav"
MEAN_F0 = 110.0

RUNS = [
    ("B1 clone_huberman", P16 / "clone_huberman"),
    ("B2i disentangle_BC+emoHub", P16 / "disentangle_bc_emoHuberman"),
]


def _identity_vs(manifest_dir: Path, ref_emb: np.ndarray, ecapa) -> float:
    import json
    manifest = json.loads((manifest_dir / "manifest.json").read_text())
    sims = []
    for row in manifest:
        wav, sr = sf.read(row["wav_path"])
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        sims.append(cosine(embed_wav(wav.astype(np.float32), sr, ecapa), ref_emb))
    return float(np.mean(sims)) if sims else float("nan")


def main() -> int:
    ecapa = load_ecapa()
    bc_emb = embed_file(BC_REF, ecapa)
    hub_emb = embed_file(HUB_REF, ecapa)

    print(f"{'run':30}{'RP dist':10}{'GenAm dist':12}{'rhotic':14}{'idBC':8}{'idHub':8}")
    print("-" * 90)
    for name, out_dir in RUNS:
        manifest = out_dir / "manifest.json"
        if not manifest.exists():
            print(f"{name:30} [SKIP] no manifest at {out_dir}")
            continue
        csv_out = out_dir / "formants_none.csv"
        if not csv_out.exists():
            extract_formants(manifest, csv_out, name.split()[0], target="none")
        toks = load_tokens(csv_out)
        rp = score_source(toks, "rp", MEAN_F0)["overall"]
        ga = score_source(toks, "genam", MEAN_F0)["overall"]
        rho = rhoticity(toks)
        rho_s = f"{rho['verdict']} {rho['f3_hz']:.0f}" if rho else "no_F3"
        id_bc = _identity_vs(out_dir, bc_emb, ecapa)
        id_hub = _identity_vs(out_dir, hub_emb, ecapa)
        rp_s = f"{rp['mean']:.3f}"
        ga_s = f"{ga['mean']:.3f}"
        print(f"{name:30}{rp_s:10}{ga_s:12}{rho_s:14}{id_bc:<8.3f}{id_hub:<8.3f}")

    print("\nAnchors: gen_base(BC,RP) scores RP~1.57/GenAm~1.75, non_rhotic, idBC high.")
    print("Read: B1 should be GenAm<RP + rhotic + idHub>idBC (clone worked).")
    print("      B2i: if idBC stays high AND GenAm<RP/rhotic → emo prompt carries accent")
    print("           (disentangled); if accent stays RP → single timbre ref is the limit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
