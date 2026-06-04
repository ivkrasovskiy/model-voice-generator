# Accent Coach — Technical Specification for Claude Code

## Purpose

Add an accent assessment feature to an existing voice cloning repository. The user records a fixed set of calibration sentences; the system extracts phonetic features, compares against the same sentences synthesized in the user-selected target voice, and produces per-skill scores with diagnostic drill-downs.

Everything must run **fully locally on macOS**. No cloud calls, no external APIs, no telemetry. The repo already uses `uv`, `ruff`, and linters — follow those conventions strictly.

---

## Architecture Overview

```
accent_coach/
├── backend/                    # Python (FastAPI), local-only server
│   ├── api/                    # FastAPI route handlers
│   ├── pipeline/               # Audio analysis pipeline
│   │   ├── alignment.py        # Forced alignment (WhisperX)
│   │   ├── formants.py         # F1/F2 extraction (parselmouth)
│   │   ├── prosody.py          # Pitch, energy, rhythm (librosa)
│   │   ├── aspiration.py       # VOT extraction for /p/ /t/ /k/
│   │   └── features.py         # Feature aggregation
│   ├── comparison/             # User-vs-target comparison logic
│   │   ├── vowels.py           # Vowel space comparison
│   │   ├── rhythm.py           # Rhythm pattern comparison
│   │   ├── stops.py            # VOT comparison
│   │   ├── intonation.py       # Pitch contour comparison
│   │   └── aggregator.py       # Skill score aggregation
│   ├── diagnostics/            # Articulatory mapping (F1/F2 → advice)
│   │   ├── ipa_chart.py        # Canonical IPA reference data
│   │   └── advice.py           # Generate human-readable feedback
│   ├── calibration/            # Calibration sentence definitions
│   │   └── sentences.py        # The 25-sentence calibration set
│   ├── models/                 # Pydantic data models
│   ├── storage/                # Local SQLite for sessions/history
│   └── voice_clone/            # Existing repo integration point
├── frontend/                   # React + Vite
│   ├── src/
│   │   ├── components/
│   │   │   ├── RadarChart.tsx       # Wind-rose chart (D3 + React)
│   │   │   ├── VowelChart.tsx       # F1/F2 scatter
│   │   │   ├── SkillCard.tsx
│   │   │   ├── RecordingStudio.tsx
│   │   │   ├── MouthDiagram.tsx
│   │   │   └── ...
│   │   ├── pages/
│   │   │   ├── Home.tsx             # Voice library
│   │   │   ├── Calibration.tsx
│   │   │   ├── Dashboard.tsx        # Radar + skill cards
│   │   │   └── DrillDown/[skill].tsx
│   │   ├── lib/                     # API client, audio recording
│   │   └── styles/
│   └── package.json
├── pyproject.toml              # uv-managed
└── README.md
```

---

## Tech Stack

### Backend (Python)

- **Runtime:** Python 3.11+, managed by `uv`
- **Web framework:** FastAPI + Uvicorn
- **Audio analysis:**
  - `praat-parselmouth` — formant extraction (F1, F2)
  - `librosa` — pitch (f0), energy, rhythm, MFCC
  - `whisperx` — forced alignment (already familiar to user via existing Whisper use)
  - `torchaudio` — fallback aligner (MMS-based) if WhisperX fails
  - `numpy`, `scipy` — signal processing
- **Data:** `pydantic` for models, `sqlite3` (stdlib) for local session storage
- **Existing integration:** the cloned voice TTS module from the existing repo provides target audio generation

### Frontend

- **Framework:** React 18 + Vite + TypeScript
- **Routing:** React Router
- **State:** Zustand (lightweight, no Redux overhead)
- **Charts:** D3.js for custom charts (radar, vowel plot), Recharts as fallback for simpler views
- **Audio recording:** Web Audio API + MediaRecorder
- **Styling:** Tailwind CSS with custom design tokens matching the design brief

### Communication

- REST + WebSocket. Recording uploads as multipart; analysis status streamed via WebSocket.

### Linting / formatting

- Match existing repo: `ruff` for Python, `eslint` + `prettier` for frontend.

---

## Data Models (Pydantic)

```python
class PhonemeInstance(BaseModel):
    """One occurrence of a phoneme in an utterance."""
    phoneme: str              # IPA symbol, e.g., "æ"
    arpabet: str              # ARPABET, e.g., "AE"
    start_time: float         # seconds
    end_time: float
    sentence_id: int
    word: str
    is_stressed: bool

class VowelFeatures(BaseModel):
    phoneme: PhonemeInstance
    f1: float                 # Hz, sampled at vowel midpoint
    f2: float                 # Hz
    f3: float | None = None
    duration_ms: float
    pitch_mean: float

class StopFeatures(BaseModel):
    phoneme: PhonemeInstance
    vot_ms: float             # voice onset time
    burst_energy: float

class SentenceAnalysis(BaseModel):
    sentence_id: int
    sentence_type: Literal["statement", "yes_no_question", "wh_question",
                            "complex", "list", "exclamation"]
    duration_s: float
    syllable_durations: list[float]
    pitch_contour: list[float]    # downsampled to ~50 points
    stress_pattern: list[bool]
    vowels: list[VowelFeatures]
    stops: list[StopFeatures]

class ComparisonResult(BaseModel):
    skill_scores: dict[str, float]   # 0-100, keys: vowels, consonants,
                                      # aspiration, rhythm, stress, intonation
    composite_score: float
    vowel_diagnostics: list[VowelDiagnostic]
    rhythm_breakdown: RhythmBreakdown
    aspiration_breakdown: AspirationBreakdown
    intonation_breakdown: IntonationBreakdown

class VowelDiagnostic(BaseModel):
    phoneme: str
    target_f1: float
    target_f2: float
    user_f1: float
    user_f2: float
    deviation_magnitude: float       # normalized euclidean in formant space
    articulatory_advice: str         # e.g., "Tongue too low; raise toward palate"
```

---

## Pipeline Detail

### Step 1: Forced Alignment

Use **WhisperX** to align user audio to known transcript at phoneme level.

```python
# pipeline/alignment.py
def align_audio(audio_path: Path, transcript: str) -> list[PhonemeInstance]:
    """
    Returns phoneme-level alignment with timestamps.
    Uses WhisperX with English wav2vec2 align model.
    """
```

Fallback: if WhisperX fails or model isn't loaded, use `torchaudio.pipelines.MMS_FA` (Meta's forced aligner, works fully offline).

### Step 2: Vowel Formant Extraction

For each vowel phoneme instance, sample F1/F2 at the vowel midpoint using parselmouth's Burg method.

```python
# pipeline/formants.py
def extract_vowel_features(
    audio: np.ndarray, sr: int, phonemes: list[PhonemeInstance]
) -> list[VowelFeatures]:
    sound = parselmouth.Sound(audio, sampling_frequency=sr)
    formants = sound.to_formant_burg(
        max_number_of_formants=5,
        maximum_formant=5500,  # 5500 for adult male, 5500-6500 for female
    )
    # auto-detect speaker pitch range to pick the right max_formant
    ...
```

**Important:** maximum_formant must be tuned per speaker (5000 for adult male, 5500 for adult female, higher for child). Estimate from f0 range.

### Step 3: VOT Extraction (Aspiration)

For each /p/ /t/ /k/ in word-initial stressed positions:

```python
# pipeline/aspiration.py
def extract_vot(audio: np.ndarray, sr: int, stop: PhonemeInstance) -> float:
    """
    Voice Onset Time = time between burst release and voicing onset.
    Algorithm:
      1. Find burst: energy spike in 0-50ms range of phoneme.
      2. Find voicing onset: first zero-crossing periodicity in next 100ms.
      3. Return delta in ms.
    """
```

For English: VOT for stressed /p t k/ should be 60-100ms. Below 30ms suggests under-aspiration (common L1 transfer error from Romance, Slavic, Asian languages).

### Step 4: Prosody (Pitch, Rhythm, Stress)

```python
# pipeline/prosody.py
def extract_prosody(audio: np.ndarray, sr: int) -> ProsodyFeatures:
    f0 = librosa.yin(audio, fmin=70, fmax=400, sr=sr)
    energy = librosa.feature.rms(y=audio).flatten()
    # Syllable boundaries from alignment
    # Stress detection: combine duration + energy + pitch peak per syllable
    ...
```

### Step 5: Target Voice Analysis

Generate the same calibration sentences via the existing voice cloning module, then run the **identical** pipeline on the synthesized audio. This guarantees apples-to-apples comparison.

Cache the target voice analysis — it only needs to be computed once per (voice, calibration_set) pair, then stored as JSON.

---

## Comparison & Scoring

### Vowel Score

For each English vowel category that appears in the calibration set:
1. Take the mean F1, F2 of all user instances of that vowel.
2. Take the mean F1, F2 of all target instances of that vowel.
3. Compute normalized Euclidean distance in (F1, F2) space (normalize by speaker's overall vowel space size — Lobanov z-score normalization).
4. Per-vowel score: 100 * exp(-distance / scale).
5. Overall vowel score: weighted average across all vowels in calibration.

**Articulatory mapping for diagnostics:**
- Higher F1 in user vs target → tongue lower / jaw more open → advice: "raise tongue"
- Higher F2 in user vs target → tongue further forward → advice: "retract tongue"
- And combinations thereof.

### Consonant Score

- Fricatives: spectral center of gravity comparison.
- Approximants: formant transitions.
- For now, weight fricatives most heavily; consonants are harder than vowels and we want to be useful, not exhaustive.

### Aspiration Score

- For each stressed /p t k/ token: compute VOT.
- Score = how well user's VOT falls within the target's VOT distribution per phoneme.
- Critical for English; under-aspiration is a strong "non-native" marker.

### Rhythm Score

Three-signal composite (comparison mode): 40% nPVI match + 40% syllable pattern correlation
+ 20% function-word inflation score. Absolute mode (no target): nPVI only.

- **nPVI** — normalised pairwise variability index (Grabe & Low 2002). In comparison mode,
  scores against the target clip's own nPVI. In absolute mode, scores against corpus-derived
  RP reference (MIN=40, MAX=62, centre=51 — supersedes lab read-aloud norms from 2002).
- **Pattern correlation** — Pearson r of mean-normalised syllable duration vectors. Returns
  neutral 50 when syllable counts diverge > 30% (misaligned positions).
- **Function word inflation** — for every function word ("the", "a", "to"…), ratio
  user/target > 1.7 = inflated. Uses WhisperX word-level timestamps directly.

Syllable extraction: `extract_syllable_durations_from_words()` uses word-level timestamps
from WhisperX for inter-word contrast + per-word acoustic nucleus detection for within-word
stress contrast. See Phase 0.17 in `docs/accent_coach_history.md` for bug history.

### Stress Score

- For each multi-syllable word: which syllable does the user emphasize vs target?
- Boolean stress placement match per word, averaged.

### Intonation Score

- Dynamic Time Warping (DTW) distance between user pitch contour and target pitch contour, per sentence.
- Normalize by speaker's overall pitch range.
- Especially diagnostic for questions (English yes/no questions have rising contour).

### Composite

Weighted average of the 6 skill scores. Weights configurable; default: vowels 25%, consonants 15%, aspiration 15%, rhythm 15%, stress 15%, intonation 15%.

---

## Calibration Sentences

The 25 sentences are designed for **maximum phonetic coverage with minimum redundancy**. Each targets specific aspects.

```python
# calibration/sentences.py
CALIBRATION_SENTENCES = [
    # Vowels
    Sentence(1, "She sees the green leaf.", type="statement",
             targets=["FLEECE_vowel"]),
    Sentence(2, "Sit and fix the little ship.", type="statement",
             targets=["KIT_vowel"]),
    Sentence(3, "Ten men sent letters.", type="statement",
             targets=["DRESS_vowel"]),
    Sentence(4, "The cat sat on a black mat.", type="statement",
             targets=["TRAP_vowel"]),
    Sentence(5, "Father parked the car far away.", type="statement",
             targets=["PALM_BATH_vowel"]),
    Sentence(6, "The fox got lost in the fog.", type="statement",
             targets=["LOT_vowel"]),
    Sentence(7, "I bought a tall cup of coffee.", type="statement",
             targets=["THOUGHT_vowel"]),
    Sentence(8, "Look at the good cook's book.", type="statement",
             targets=["FOOT_vowel"]),
    Sentence(9, "Two blue shoes are too loose.", type="statement",
             targets=["GOOSE_vowel"]),
    Sentence(10, "The sun comes up above the cloud.", type="statement",
             targets=["STRUT_vowel", "MOUTH_diphthong"]),
    Sentence(11, "The bird heard a third word.", type="statement",
             targets=["NURSE_vowel"]),
    Sentence(12, "A banana, a sofa, and a problem.", type="list",
             targets=["schwa_reduction"]),

    # Diphthongs
    Sentence(13, "I'd like five wise riders.", type="statement",
             targets=["PRICE_diphthong"]),
    Sentence(14, "They made a great mistake today.", type="statement",
             targets=["FACE_diphthong"]),
    Sentence(15, "The boy enjoyed the noisy toys.", type="statement",
             targets=["CHOICE_diphthong"]),
    Sentence(16, "Go home and don't be slow.", type="statement",
             targets=["GOAT_diphthong"]),

    # Consonants — TH
    Sentence(17, "Both of them thought it through.", type="statement",
             targets=["TH_voiceless", "TH_voiced"]),

    # Aspiration
    Sentence(18, "Pat packed pink papers properly.", type="statement",
             targets=["P_aspiration"]),
    Sentence(19, "Tom told ten tall tales.", type="statement",
             targets=["T_aspiration"]),
    Sentence(20, "Karen kept calling Kevin's cat.", type="statement",
             targets=["K_aspiration"]),

    # Rhythm / sentence types
    Sentence(21, "Did you see the red car yesterday?",
             type="yes_no_question",
             targets=["rising_intonation"]),
    Sentence(22, "Where did you put the keys?",
             type="wh_question",
             targets=["falling_intonation_wh"]),
    Sentence(23, "Although she was tired, she finished the work.",
             type="complex",
             targets=["clause_boundary_intonation"]),
    Sentence(24, "Photograph, photographer, photography.",
             type="list",
             targets=["stress_shift"]),
    Sentence(25, "I'm going to the store to buy some bread.",
             type="statement",
             targets=["weak_forms", "linking"]),
]
```

---

## API Endpoints

```
POST   /api/sessions                  # Create new session, returns session_id
GET    /api/voices                    # List available target voices
POST   /api/voices/select             # Select voice for session
GET    /api/calibration/sentences     # Get the 25 sentences
POST   /api/recordings/{sentence_id}  # Upload user audio for sentence
                                       # (multipart, .wav, 16kHz mono)
POST   /api/analyze                   # Trigger full analysis
                                       # Returns 202, work happens async
WS     /api/analyze/progress          # Stream progress updates
GET    /api/results/{session_id}      # Get ComparisonResult
GET    /api/results/{session_id}/drill/{skill}
                                       # Get detailed breakdown per skill
GET    /api/history                   # List past sessions for trend view
```

---

## Frontend Component Spec

### RadarChart Component

```tsx
interface RadarChartProps {
  userScores: { axis: string; value: number }[];
  targetScores: { axis: string; value: number }[]; // always all 100s
  onAxisClick: (axis: string) => void;
}
```

- D3 for the rendering, React for the wrapping.
- Animate polygon draw on mount.
- Click handler on axis label or polygon vertex drills down to that skill.
- Responsive sizing.

### VowelChart Component

```tsx
interface VowelChartProps {
  pairs: {
    phoneme: string;
    target: { f1: number; f2: number };
    user: { f1: number; f2: number };
    advice: string;
  }[];
}
```

- Trapezoidal background reflecting IPA vowel space (drawn as SVG).
- Gold dots for target, blue for user.
- Arrows from gold → blue per vowel.
- Hover: shows phoneme name, F1/F2 values, articulatory advice.

### MouthDiagram Component

- Static SVG cross-section.
- Accepts a `highlight` prop indicating which articulatory zone to emphasize ("alveolar", "palatal", "velar", etc.).

---

## Storage

Local SQLite at `~/Library/Application Support/AccentCoach/sessions.db`.

Schema:
```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    target_voice_id TEXT,
    composite_score REAL,
    result_json TEXT       -- full ComparisonResult as JSON
);

CREATE TABLE recordings (
    session_id TEXT,
    sentence_id INTEGER,
    audio_path TEXT,        -- local file path
    PRIMARY KEY (session_id, sentence_id)
);
```

Audio files stored at `~/Library/Application Support/AccentCoach/audio/{session_id}/{sentence_id}.wav`.

---

## Integration with Existing Voice Cloning Repo

Add as a new subpackage. The existing TTS module is consumed by a single import:

```python
from voice_clone import synthesize  # existing function

# In calibration/target_audio.py
def precompute_target_audio(voice_id: str) -> dict[int, Path]:
    """Generate all 25 calibration sentences in target voice, cache to disk."""
    audio_paths = {}
    for sentence in CALIBRATION_SENTENCES:
        out_path = TARGET_CACHE_DIR / voice_id / f"{sentence.id}.wav"
        if not out_path.exists():
            audio = synthesize(text=sentence.text, voice_id=voice_id)
            sf.write(out_path, audio, 22050)
        audio_paths[sentence.id] = out_path
    return audio_paths
```

Don't fork the existing TTS code. Consume it through its public interface.

---

## Development & Quality

- All new Python code must pass `ruff check` and `ruff format`.
- Type hints mandatory; mypy strict mode for new modules.
- Frontend: `eslint` + `prettier` configurations matching repo standards.
- Tests: pytest for backend pipeline (with sample audio fixtures), Vitest for frontend.
- Test coverage target: 70% for pipeline modules; the comparison logic in particular needs unit tests with synthetic phoneme inputs.

Sample fixture audio for tests: 3 short utterances at varying pronunciation quality, hand-annotated with expected F1/F2 ranges.

---

## Performance Targets

On M2/M3 Mac:
- Forced alignment for 25-sentence batch: under 30 seconds.
- Full feature extraction: under 60 seconds.
- Comparison + scoring: under 5 seconds.
- Total time from "Submit" to results dashboard: under 2 minutes.

Show progress via WebSocket with per-step granularity ("Aligning sentence 12/25...", "Extracting formants...", "Scoring...").

---

## Out of Scope (Phase 1)

- Real-time analysis while user speaks (post-recording only for now).
- Multi-language support (English only).
- Custom calibration sentence sets (fixed 25 for now).
- Cloud sync (local-only).
- Mobile native (web only — though usable on iPad/iPhone Safari).

---

## Acceptance Criteria

1. User can select a target voice and run a full calibration session.
2. Wind-rose chart renders with 6 axes showing user vs target.
3. Each axis is drillable into a detailed view.
4. Vowel chart shows F1/F2 plot with target/user comparison and articulatory advice per vowel.
5. Rhythm breakdown distinguishes questions vs statements vs complex sentences.
6. Aspiration view shows VOT for each stressed /p t k/.
7. All processing happens locally; no network calls leave the machine (verifiable by running with airplane mode after model download).
8. Session history persists and can be reviewed.
9. All Python code passes `ruff check` with no warnings.
10. README documents setup, usage, and how the feature integrates with the existing voice cloning module.
