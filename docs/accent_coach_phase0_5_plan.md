# Accent Coach — Phase 0.5 Root-Cause Investigation Plan

> **Status**: ready for sonnet to execute. Pure measurement experiment — no scoring code changes. Owner reviews findings + listens to audio samples before any Phase 1 action.
>
> **Related**: [`accent_coach_phase0_findings.md`](accent_coach_phase0_findings.md), [`accent_coach_technical_spec.md`](accent_coach_technical_spec.md), [`accent_coach_plan.md`](accent_coach_plan.md).
>
> **How to read this**: sections 1-3 are context. Section 4 onward is the executable spec — one script per subsection with CLI args, output paths, and algorithm pseudocode. Execute in the order given in section 5.

## 1. Goal

Explain why owner and synth-BC score equally on the vowel composite (BC=75.5 vs owner=73.3). The cancellation comes from /æ ʌ ɛ/, where synth-BC scores *worse* than owner against Deterding 1997 RP. Attribute each phoneme's deviation to one of four root causes with numbers.

## 2. Hypotheses (with verdict thresholds)

| H | Statement | Verdict if … |
|---|---|---|
| H1 | Deterding 1997 RP is stale; modern SSBE shifted | median \|Modern RP − Deterding\| > 75 Hz over /æ ɛ ʌ/ |
| H2 | BC's natural speech deviates from RP on /æ ʌ ɛ/ | median \|Real BC − Modern RP\| > 75 Hz over /æ ɛ ʌ/ |
| H3 | IndexTTS distorts BC's vowel space | median \|Synth BC − Real BC\| > 75 Hz AND cross-ref variance < 50 Hz |
| H4 | Synth-BC's /æ ɛ/ collide with where Slavic L1 produces /ɛ/-substituted-/æ/ | D(synth-BC, owner) < D(synth-BC, real-BC) in F1/F2 Euclidean Hz |

## 3. Sources

### D1 — Real BC speech
- **Primary**: <https://youtu.be/cHmkAStZBkc> (same source as `tts_output/ref_interview.wav`). Pull full audio.
- **Secondary** (only if primary < 5 min net solo BC): <https://www.youtube.com/watch?v=UKfBtgDSCzw>. BC starts ~0:50. Two male voices present.

### D2 — Modern RP
- **BBC**: 1-2 segments from the BBC News channel **main** bulletins (NOT regional bulletins like BBC Look North). Sonnet picks YouTube URLs and surfaces them in `samples_to_verify/` before full processing.
- **Geoff Lindsey** (@DrGeoffLindsey on YouTube): 2-3 videos of him speaking in his own RP voice (NOT demonstrating other accents). **Owner pre-curates URLs**; if missing at run time, sonnet shortlists 5 candidates by title heuristics (prefer "RP", "British pronunciation", "Standard Southern British"; reject titles naming other accents) and **pauses**.

URLs go in a JSON config consumed by script 2 (template below).

### D3 — Synth BC (existing)
`tts_output/bc_cal_50/` — 50 IndexTTS-2 generations of calibration sentences.

### D4 — Owner (existing)
Owner's 50 calibration recordings.

---

## 4. Scripts to write

All scripts go in `scripts/`. Match the project's existing CLI/argparse + uv venv conventions. Use `.venv/` for everything except script 6's IndexTTS generation (uses `vendor/index-tts/.venv/`).

### 4.1 — `scripts/accent_coach_build_real_bc.py`

Builds D1. Pulls audio, diarizes, identifies BC via ECAPA-to-ref, drops overlap, writes clips.

**Inputs (CLI):**
- `--primary-url URL` (default: `https://youtu.be/cHmkAStZBkc`)
- `--secondary-url URL` (default: `https://www.youtube.com/watch?v=UKfBtgDSCzw`)
- `--secondary-start SECONDS` (default: `50`)
- `--ref-wav PATH` (default: `tts_output/ref_interview.wav`)
- `--out-dir PATH` (default: `tts_output/real_bc_corpus`)
- `--min-net-seconds INT` (default: `300`)

**Outputs:**
- `<out-dir>/raw/{primary,secondary}.wav` — 16 kHz mono, full audio
- `<out-dir>/clips/{0000.wav, 0001.wav, …}` — kept BC-only segments
- `<out-dir>/manifest.json` — list of `{clip_id, path, source_url, start_s, end_s, transcript}`
- `<out-dir>/samples_to_verify/{0..4}.wav` — 5 random kept clips for owner listen-check
- `<out-dir>/diarization_debug.csv` — `{speaker_id, ecapa_cos_to_ref, n_words, total_dur_s, kept_bool}` per source

**Algorithm (pseudocode):**

```python
def build_real_bc(primary_url, secondary_url, secondary_start, ref_wav, out_dir, min_net):
    ref_emb = ecapa_embedding(ref_wav)  # speechbrain/spkrec-ecapa-voxceleb

    all_segments = []
    for source_name, url, start_offset in [
        ("primary", primary_url, 0),
        ("secondary", secondary_url, secondary_start),
    ]:
        raw = f"{out_dir}/raw/{source_name}.wav"
        run(f"yt-dlp -x --audio-format wav -o {raw}.tmp {url}")
        run(f"ffmpeg -i {raw}.tmp -ac 1 -ar 16000 {raw}")
        if start_offset > 0:
            run(f"ffmpeg -ss {start_offset} -i {raw} {raw}.trimmed && mv {raw}.trimmed {raw}")

        # WhisperX with diarization
        words = whisperx_transcribe_and_diarize(
            raw, model="large-v2", batch_size=16, diarize=True,
        )  # list of {start, end, text, speaker}

        # ECAPA per speaker
        debug_rows = []
        for sid in sorted(set(w["speaker"] for w in words)):
            spk_audio = concat_word_spans(raw, [w for w in words if w["speaker"] == sid])
            sim = cosine(ecapa_from_array(spk_audio), ref_emb)
            debug_rows.append(dict(speaker_id=sid, ecapa_cos_to_ref=sim,
                                   n_words=sum(1 for w in words if w["speaker"]==sid),
                                   total_dur_s=sum(w["end"]-w["start"] for w in words if w["speaker"]==sid)))

        bc = max(debug_rows, key=lambda r: r["ecapa_cos_to_ref"])
        if bc["ecapa_cos_to_ref"] < 0.5:
            raise EscalateToOwner(f"No BC cluster in {source_name}: max cos={bc['ecapa_cos_to_ref']:.2f}")
        bc_sid = bc["speaker_id"]

        # Keep BC words, drop overlap with other speakers (≥ 50 ms)
        bc_words = [w for w in words if w["speaker"] == bc_sid]
        bc_words = [w for w in bc_words
                    if not has_overlap(w, [o for o in words if o["speaker"] != bc_sid], ms=50)]

        # Group into segments by 200 ms silence gaps
        segments = group_by_gap(bc_words, gap_ms=200)
        segments = [s for s in segments if s["end"] - s["start"] >= 1.0]

        all_segments.extend([(source_name, url, s) for s in segments])
        write_csv_append(debug_rows, f"{out_dir}/diarization_debug.csv", source_name)

        net = sum(s["end"] - s["start"] for _, _, s in all_segments)
        if source_name == "primary" and net >= min_net:
            break  # skip secondary

    net = sum(s["end"] - s["start"] for _, _, s in all_segments)
    if net < min_net:
        raise EscalateToOwner(f"Insufficient BC audio: {net:.1f}s < {min_net}s")

    # Write clips + manifest
    manifest = []
    for i, (source_name, url, seg) in enumerate(all_segments):
        clip_path = f"{out_dir}/clips/{i:04d}.wav"
        write_clip_from_raw(seg, clip_path)
        manifest.append(dict(clip_id=f"{i:04d}", path=clip_path, source_url=url,
                             start_s=seg["start"], end_s=seg["end"], transcript=seg["text"]))
    json_dump(manifest, f"{out_dir}/manifest.json")
    copy_random(manifest, n=5, dst=f"{out_dir}/samples_to_verify/")
```

**Dependencies (install via uv if missing):**
- `yt-dlp` CLI
- `ffmpeg` (system, already on macOS)
- `whisperx` — needs `HF_TOKEN` env var for pyannote/speaker-diarization-3.1
- `speechbrain` (already in project deps for ECAPA scoring)
- `librosa`, `soundfile`, `numpy` (already in project deps)

**Failure modes & escalations:**
- HF_TOKEN missing → raise with message "set HF_TOKEN to use pyannote/speaker-diarization-3.1"
- No cluster reaches cos ≥ 0.5 → raise EscalateToOwner; dump debug CSV + a sample per cluster
- Combined < min-net-seconds → raise EscalateToOwner

---

### 4.2 — `scripts/accent_coach_build_modern_rp.py`

Builds D2 from a JSON list of URLs with quality gates. No diarization needed (single-speaker content).

**Inputs (CLI):**
- `--urls-json PATH` — JSON list: `[{label, url, trim_head_s, trim_tail_s}, ...]`
- `--out-dir PATH` (default: `tts_output/modern_rp_corpus`)

**Outputs:**
- `<out-dir>/{label}/raw/{N}.wav` — 16 kHz mono, head/tail trimmed
- `<out-dir>/{label}/clips/{N}_{seg}.wav` — kept segments per URL
- `<out-dir>/manifest.json` — per clip: `{clip_id, label, url, source_idx, start_s, end_s, transcript}`
- `<out-dir>/samples_to_verify/{label}_{N}.wav` — first 5 s of each URL (owner verifies it's actually RP)
- `<out-dir>/quality_log.csv` — `{url, dur_s, silent_frac, min_pair_cos, decision}`

**URLs JSON template** (commit at `configs/accent_coach_phase0_5/modern_rp_urls.json`):

```json
[
    {"label": "bbc",     "url": "TODO", "trim_head_s": 15, "trim_tail_s": 15},
    {"label": "bbc",     "url": "TODO", "trim_head_s": 15, "trim_tail_s": 15},
    {"label": "lindsey", "url": "TODO", "trim_head_s": 15, "trim_tail_s": 15},
    {"label": "lindsey", "url": "TODO", "trim_head_s": 15, "trim_tail_s": 15}
]
```

If any URL is `"TODO"` at run time, sonnet shortlists 5 candidates per label by YouTube title heuristics and **pauses** for owner approval.

**Algorithm:**

```python
def build_modern_rp(urls_json_path, out_dir):
    config = json.load(open(urls_json_path))
    if any(e["url"] == "TODO" for e in config):
        shortlist_and_pause(missing_labels=[e["label"] for e in config if e["url"]=="TODO"])

    manifest, quality_log = [], []
    for i, entry in enumerate(config):
        label, url = entry["label"], entry["url"]
        raw = f"{out_dir}/{label}/raw/{i}.wav"
        run(f"yt-dlp -x --audio-format wav -o {raw}.tmp {url}")
        run(f"ffmpeg -i {raw}.tmp -ac 1 -ar 16000 {raw}")
        dur = audio_duration(raw)
        run(f"ffmpeg -ss {entry['trim_head_s']} -to {dur-entry['trim_tail_s']} -i {raw} {raw}.t && mv {raw}.t {raw}")
        dur = audio_duration(raw)

        # Quality gates
        if dur < 60:
            quality_log.append(dict(url=url, dur_s=dur, decision="reject_too_short")); continue
        silent_frac = silent_fraction(raw, vad="silero")
        if silent_frac > 0.20:
            quality_log.append(dict(url=url, dur_s=dur, silent_frac=silent_frac, decision="reject_too_silent")); continue
        embs = [ecapa_chunk(raw, start=s, dur=5) for s in random_starts(raw, n=5, dur=5)]
        min_pair = min(cosine(a,b) for a,b in combinations(embs, 2))
        if min_pair < 0.7:
            quality_log.append(dict(url=url, min_pair_cos=min_pair, decision="reject_multi_speaker")); continue
        quality_log.append(dict(url=url, dur_s=dur, silent_frac=silent_frac, min_pair_cos=min_pair, decision="keep"))

        # Transcribe (no diarization)
        words = whisperx_transcribe(raw, model="large-v2", batch_size=16, diarize=False)
        segments = group_by_gap(words, gap_ms=300)
        segments = [s for s in segments if s["end"] - s["start"] >= 1.0]

        for seg_idx, seg in enumerate(segments):
            clip = f"{out_dir}/{label}/clips/{i}_{seg_idx:03d}.wav"
            write_clip_from_raw(seg, clip)
            manifest.append(dict(clip_id=f"{label}_{i}_{seg_idx:03d}", path=clip,
                                 label=label, url=url, source_idx=i,
                                 start_s=seg["start"], end_s=seg["end"], transcript=seg["text"]))
        copy_first_seconds(raw, n=5, dst=f"{out_dir}/samples_to_verify/{label}_{i}.wav")

    json_dump(manifest, f"{out_dir}/manifest.json")
    write_csv(quality_log, f"{out_dir}/quality_log.csv")
```

---

### 4.3 — `scripts/accent_coach_extract_formants.py`

Thin wrapper over the existing alignment + formant pipeline. Runs over a manifest, emits per-token CSV.

**Inputs (CLI):**
- `--manifest PATH`
- `--out PATH`
- `--source-label STR` — label to tag every row (e.g., "real_bc", "modern_rp_bbc", "modern_rp_lindsey", "synth_bc", "owner")

**Outputs:**
- CSV with columns: `clip_id, source_label, phoneme, F1, F2, voiced_fraction, duration_s, transcript`

**Algorithm:**

```python
def extract_formants(manifest_path, out_csv, source_label):
    rows = []
    for entry in json.load(open(manifest_path)):
        # Use existing modules:
        word_timings = accent_coach.pipeline.alignment.run(entry["path"], entry["transcript"])
        formants = accent_coach.pipeline.formants.run(entry["path"], word_timings)
        for f in formants:
            rows.append(dict(clip_id=entry["clip_id"], source_label=source_label,
                             phoneme=f.phoneme, F1=f.F1, F2=f.F2,
                             voiced_fraction=f.voiced_fraction, duration_s=f.dur,
                             transcript=entry["transcript"]))
    write_csv(rows, out_csv)
```

**Hard constraint**: do not modify any code inside [`../accent_coach/pipeline/`](../accent_coach/pipeline/). Use Phase 0 parameters (voiced-fraction ≥ 35%, F1 ceiling 900 Hz, auto pitch ceiling) exactly as configured.

---

### 4.4 — `scripts/accent_coach_build_table.py`

Aggregate per-token formant CSVs into the 5-column comparison table.

**Inputs (CLI):**
- `--modern-rp-csv PATH`
- `--real-bc-csv PATH`
- `--synth-bc-csv PATH`
- `--owner-csv PATH`
- `--out PATH` (default: `docs/accent_coach_phase0_5_table.csv`)

**Outputs:**
- CSV: `phoneme, source, F1_mean, F2_mean, F1_std, F2_std, n_tokens`
- One row per `(phoneme, source)`. Phonemes: `/iː ɪ ʊ uː eɪ ɛ æ ʌ ɔː/`.
- Source values: `deterding_rp, modern_rp_bbc, modern_rp_lindsey, real_bc, synth_bc, owner`. The `modern_rp_*` rows let us also collapse to a combined `modern_rp` for the headline table.

**Algorithm:**

```python
PHONEMES = ["iː", "ɪ", "ʊ", "uː", "eɪ", "ɛ", "æ", "ʌ", "ɔː"]

def build_table(modern_rp_csv, real_bc_csv, synth_bc_csv, owner_csv, out_csv):
    rows = []

    # Deterding from accent_coach/reference/rp_norms.py — import directly
    from accent_coach.reference.rp_norms import DETERDING_1997  # or whatever the symbol is
    for ph, (f1, f2) in DETERDING_1997.items():
        if ph in PHONEMES:
            rows.append(dict(phoneme=ph, source="deterding_rp", F1_mean=f1, F2_mean=f2,
                             F1_std=None, F2_std=None, n_tokens=None))

    # Per-source aggregates
    for csv_path in [modern_rp_csv, real_bc_csv, synth_bc_csv, owner_csv]:
        df = pd.read_csv(csv_path)
        for source_label, sub in df.groupby("source_label"):
            for ph in PHONEMES:
                ph_sub = sub[sub.phoneme == ph]
                if len(ph_sub) == 0:
                    continue
                rows.append(dict(phoneme=ph, source=source_label,
                                 F1_mean=ph_sub.F1.mean(), F2_mean=ph_sub.F2.mean(),
                                 F1_std=ph_sub.F1.std(), F2_std=ph_sub.F2.std(),
                                 n_tokens=len(ph_sub)))

    # Also emit a combined "modern_rp" row (BBC + Lindsey pooled)
    # ... pool modern_rp_bbc + modern_rp_lindsey by re-reading modern_rp_csv ...

    write_csv(rows, out_csv)
```

---

### 4.5 — `scripts/accent_coach_vowel_space_plot.py`

F1/F2 scatter plot, all sources overlaid.

**Inputs:**
- `--table PATH` (default: `docs/accent_coach_phase0_5_table.csv`)
- `--out PATH` (default: `docs/img/accent_coach_phase0_5_vowel_space.png`)

**Outputs:** PNG at 150 DPI.

**Algorithm:**

```python
def plot(table_path, out_png):
    df = pd.read_csv(table_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"deterding_rp": "k", "modern_rp": "tab:blue", "real_bc": "tab:green",
              "synth_bc": "tab:orange", "owner": "tab:red"}
    for src in colors:
        sub = df[df.source == src]
        ax.scatter(sub.F2_mean, sub.F1_mean, c=colors[src], label=src, s=80, alpha=0.8)
        for _, r in sub.iterrows():
            ax.annotate(r.phoneme, (r.F2_mean, r.F1_mean), fontsize=11, xytext=(5,5), textcoords="offset points")
    ax.invert_xaxis()  # F2 high on left (standard convention)
    ax.invert_yaxis()  # F1 high at bottom
    ax.set_xlabel("F2 (Hz)"); ax.set_ylabel("F1 (Hz)")
    ax.set_title("Vowel space: Deterding RP vs Modern RP vs Real BC vs Synth BC vs Owner")
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=150)
```

---

### 4.6 — `scripts/accent_coach_refclip_robustness.py`

Generate 15 sentences with 5 different BC reference clips, extract formants, summarise variance.

**Inputs (CLI):**
- `--ref-clips-json PATH` — list of `{name, path}`
- `--sentences-csv PATH` — first 15 rows from `tts_output/cross_eval_50/eval_short.csv` (or the smoke-test set)
- `--out-dir PATH` (default: `tts_output/refclip_robustness`)
- `--skip-generation` flag (re-run formant extraction without regenerating audio)

**Outputs:**
- `<out-dir>/{ref_name}/clips/{N}.wav` (15 per ref)
- `<out-dir>/{ref_name}/manifest.json`
- `<out-dir>/{ref_name}/formants.csv`
- `<out-dir>/summary.csv` — per `(ref_name, phoneme)`: `F1_mean, F2_mean, n_tokens`

**Ref-clips JSON template** (sonnet writes this, picking 2 D1 clips + 2 audiobook clips):

```json
[
    {"name": "interview_main",   "path": "tts_output/ref_interview.wav"},
    {"name": "d1_segment_a",     "path": "tts_output/real_bc_corpus/clips/XXXX.wav"},
    {"name": "d1_segment_b",     "path": "tts_output/real_bc_corpus/clips/YYYY.wav"},
    {"name": "audiobook_a",      "path": "TBD — search tts_output/eval_indextts_v2/ and adjacent dirs"},
    {"name": "audiobook_b",      "path": "TBD — same"}
]
```

If no audiobook refs exist in `tts_output/`, use 4 D1 segments instead.

**Algorithm:**

```python
def refclip_robustness(refs_json, sentences_csv, out_dir, skip_generation):
    refs = json.load(open(refs_json))

    if not skip_generation:
        for ref in refs:
            # Reuse existing scripts/indextts_gen.py via subprocess + vendor venv
            subprocess.run([
                "vendor/index-tts/.venv/bin/python", "scripts/indextts_gen.py",
                "--phrases-csv", sentences_csv,
                "--ref-audio", ref["path"],
                "--out-dir", f"{out_dir}/{ref['name']}",
            ], check=True)

    for ref in refs:
        run([sys.executable, "scripts/accent_coach_extract_formants.py",
             "--manifest", f"{out_dir}/{ref['name']}/manifest.json",
             "--out", f"{out_dir}/{ref['name']}/formants.csv",
             "--source-label", f"synth_bc_ref_{ref['name']}"])

    # Build summary.csv: aggregate per (ref_name, phoneme)
    rows = []
    for ref in refs:
        df = pd.read_csv(f"{out_dir}/{ref['name']}/formants.csv")
        for ph, sub in df.groupby("phoneme"):
            rows.append(dict(ref_name=ref["name"], phoneme=ph,
                             F1_mean=sub.F1.mean(), F2_mean=sub.F2.mean(), n_tokens=len(sub)))
    write_csv(rows, f"{out_dir}/summary.csv")
```

**Compute budget**: 75 generations × ~60 s ≈ 75-90 min CPU. **PAUSE before running** to confirm with owner.

---

### 4.7 — `scripts/accent_coach_compute_verdicts.py`

Compute H1-H4 numerical verdicts from the table and ref-robustness summary.

**Inputs (CLI):**
- `--table PATH` (default: `docs/accent_coach_phase0_5_table.csv`)
- `--refclip-summary PATH` (default: `tts_output/refclip_robustness/summary.csv`)
- `--out-json PATH` (default: `docs/accent_coach_phase0_5_verdicts.json`)
- `--out-md PATH` (default: `docs/accent_coach_phase0_5_verdicts.md`) — pasteable into findings doc

**Outputs:**
- JSON: `{"H1": {"median_dF1": float, "median_dF2": float, "verdict": "stale"|"fresh"}, "H2": {...}, "H3": {"median_dF1": ..., "median_dF2": ..., "cross_ref_var_F1": ..., "cross_ref_var_F2": ..., "verdict": "architecture"|"ref_dependent"|"no_shift"}, "H4": {"per_phoneme": {"æ": {"d_synth_owner": ..., "d_synth_real": ..., "confirmed": bool}, "ɛ": {...}}}}`
- Markdown snippet with the same numbers formatted for the findings doc.

**Algorithm:**

```python
PROBLEM = ["æ", "ɛ", "ʌ"]

def verdict_H1(table):
    det = table[table.source == "deterding_rp"].set_index("phoneme")
    mod = table[table.source == "modern_rp"].set_index("phoneme")
    dF1 = np.median([abs(det.loc[p].F1_mean - mod.loc[p].F1_mean) for p in PROBLEM if p in det.index and p in mod.index])
    dF2 = np.median([abs(det.loc[p].F2_mean - mod.loc[p].F2_mean) for p in PROBLEM if p in det.index and p in mod.index])
    return dict(median_dF1=dF1, median_dF2=dF2, verdict="stale" if max(dF1, dF2) > 75 else "fresh")

def verdict_H2(table):
    # |real_bc - modern_rp| over PROBLEM
    ...

def verdict_H3(table, robustness):
    # |synth_bc - real_bc| > 75 Hz median
    real = table[table.source == "real_bc"].set_index("phoneme")
    synth = table[table.source == "synth_bc"].set_index("phoneme")
    dF1 = np.median([abs(real.loc[p].F1_mean - synth.loc[p].F1_mean) for p in PROBLEM])
    dF2 = np.median([abs(real.loc[p].F2_mean - synth.loc[p].F2_mean) for p in PROBLEM])

    # cross-ref variance from robustness summary
    var_F1 = robustness[robustness.phoneme.isin(PROBLEM)].groupby("phoneme").F1_mean.std().mean()
    var_F2 = robustness[robustness.phoneme.isin(PROBLEM)].groupby("phoneme").F2_mean.std().mean()

    shifts = max(dF1, dF2) > 75
    consistent = max(var_F1, var_F2) < 50
    verdict = ("architecture" if shifts and consistent
               else "ref_dependent" if shifts and not consistent
               else "no_shift")
    return dict(median_dF1=dF1, median_dF2=dF2, cross_ref_var_F1=var_F1, cross_ref_var_F2=var_F2, verdict=verdict)

def verdict_H4(table):
    out = {}
    for ph in ["æ", "ɛ"]:
        synth = table[(table.source=="synth_bc") & (table.phoneme==ph)].iloc[0]
        real  = table[(table.source=="real_bc")  & (table.phoneme==ph)].iloc[0]
        owner = table[(table.source=="owner")    & (table.phoneme==ph)].iloc[0]
        d_so = math.hypot(synth.F1_mean - owner.F1_mean, synth.F2_mean - owner.F2_mean)
        d_sr = math.hypot(synth.F1_mean - real.F1_mean,  synth.F2_mean - real.F2_mean)
        d_or = math.hypot(owner.F1_mean - real.F1_mean,  owner.F2_mean - real.F2_mean)
        out[ph] = dict(d_synth_owner=d_so, d_synth_real=d_sr, d_owner_real=d_or,
                       confirmed=(d_so < d_sr))
    return out
```

---

## 5. Execution order

```
# Step 1: D1
python scripts/accent_coach_build_real_bc.py
# CHECKPOINT 1: owner listens to tts_output/real_bc_corpus/samples_to_verify/*.wav

# Step 2: D2 (URLs in configs/accent_coach_phase0_5/modern_rp_urls.json)
python scripts/accent_coach_build_modern_rp.py \
    --urls-json configs/accent_coach_phase0_5/modern_rp_urls.json
# CHECKPOINT 2: owner listens to tts_output/modern_rp_corpus/samples_to_verify/*.wav

# Step 3: formants for D1, D2, D3, D4
python scripts/accent_coach_extract_formants.py \
    --manifest tts_output/real_bc_corpus/manifest.json \
    --out tts_output/real_bc_corpus/formants.csv \
    --source-label real_bc
python scripts/accent_coach_extract_formants.py \
    --manifest tts_output/modern_rp_corpus/manifest.json \
    --out tts_output/modern_rp_corpus/formants.csv \
    --source-label modern_rp  # sonnet emits both modern_rp_bbc / _lindsey at row level via the manifest's `label` field
python scripts/accent_coach_extract_formants.py \
    --manifest tts_output/bc_cal_50/manifest.json \
    --out tts_output/bc_cal_50/formants.csv \
    --source-label synth_bc
python scripts/accent_coach_extract_formants.py \
    --manifest <path to owner's calibration manifest> \
    --out tts_output/owner_cal_50/formants.csv \
    --source-label owner

# Step 4: table
python scripts/accent_coach_build_table.py \
    --modern-rp-csv tts_output/modern_rp_corpus/formants.csv \
    --real-bc-csv   tts_output/real_bc_corpus/formants.csv \
    --synth-bc-csv  tts_output/bc_cal_50/formants.csv \
    --owner-csv     tts_output/owner_cal_50/formants.csv

# Step 5: plot
python scripts/accent_coach_vowel_space_plot.py

# CHECKPOINT 3: PAUSE — confirm with owner before 75-gen batch
python scripts/accent_coach_refclip_robustness.py \
    --ref-clips-json configs/accent_coach_phase0_5/ref_clips.json \
    --sentences-csv  tts_output/cross_eval_50/eval_short.csv

# Step 6: verdicts
python scripts/accent_coach_compute_verdicts.py

# Step 7: sonnet hand-writes docs/accent_coach_phase0_5_findings.md
# - Paste the markdown verdict snippet (from compute_verdicts --out-md)
# - Embed docs/img/accent_coach_phase0_5_vowel_space.png
# - Add a listening checklist with paths
# - Add sonnet's own interpretation paragraph (not just numbers)
```

## 6. Pause / ask-owner checkpoints

1. D1 net < 5 min after pulling both primary and secondary URLs → ask owner.
2. After D1 `samples_to_verify/` written → owner listens, green-lights or rejects.
3. D2 URLs JSON missing or contains `"TODO"` → sonnet shortlists 5 candidates per missing label by title heuristics, pauses.
4. After D2 `samples_to_verify/` written → owner listens, green-lights.
5. Before step 6 (75-gen IndexTTS batch, ~90 min CPU) → "OK to spend ~90 min?" ping.
6. Any HF/model download failure → don't retry blindly, surface error.

## 7. Hard NOTs (out of scope)

- Do NOT modify [`../accent_coach/reference/rp_norms.py`](../accent_coach/reference/rp_norms.py) or any scoring code.
- Do NOT modify [`../accent_coach/pipeline/`](../accent_coach/pipeline/) (alignment, formants, etc.) — wrap, don't edit.
- Do NOT touch rhythm/stress code paths.
- Do NOT touch IndexTTS pins or `tts_output/eval_indextts_v2/`.
- Total compute budget: ≤ 8 h wall.

## 8. Deliverables checklist

- [ ] `scripts/accent_coach_build_real_bc.py` + run
- [ ] `scripts/accent_coach_build_modern_rp.py` + run
- [ ] `scripts/accent_coach_extract_formants.py` + 4 runs
- [ ] `scripts/accent_coach_build_table.py` + run
- [ ] `scripts/accent_coach_vowel_space_plot.py` + run
- [ ] `scripts/accent_coach_refclip_robustness.py` + run
- [ ] `scripts/accent_coach_compute_verdicts.py` + run
- [ ] `configs/accent_coach_phase0_5/modern_rp_urls.json` (populated)
- [ ] `configs/accent_coach_phase0_5/ref_clips.json` (populated)
- [ ] `tts_output/real_bc_corpus/` (raw + clips + manifest + samples_to_verify + diarization_debug.csv + formants.csv)
- [ ] `tts_output/modern_rp_corpus/` (same structure)
- [ ] `tts_output/refclip_robustness/` (per-ref subdirs + summary.csv)
- [ ] `docs/accent_coach_phase0_5_table.csv`
- [ ] `docs/img/accent_coach_phase0_5_vowel_space.png`
- [ ] `docs/accent_coach_phase0_5_verdicts.json` + `.md`
- [ ] `docs/accent_coach_phase0_5_findings.md` with TL;DR, table, plot, raw numbers, listening checklist, sonnet's interpretation paragraph
