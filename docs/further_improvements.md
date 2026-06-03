# Accent Coach — Analysis Modules Spec
## Rhythm · Intonation · Consonants
### Input to Claude Code — implement these three modules

---

## Context & Assumptions

The existing codebase already produces:
- `alignment: list[PhonemeInstance]` — phoneme-level timestamps from WhisperX, using ARPABET labels
- `audio: np.ndarray` — raw audio at `sr=16000`
- `sentence_meta: SentenceMeta` — sentence type (statement / yes_no_question / wh_question / complex / list)
- The **target voice** generates the same text, so both `user_audio` and `target_audio` go through the same pipeline

All three modules follow this pattern:
```
extract_features(audio, sr, alignment) → FeatureSet
compare(user_features, target_features)  → SkillScore
```

Each `SkillScore` returns:
- `score: float` — 0–100
- `diagnostics: list[str]` — human-readable, specific (e.g. "Function words 'to', 'the', 'and' spoken too long — compress them to ~30ms")

Dependencies: `numpy`, `scipy`, `librosa`, `praat-parselmouth`. No new dependencies beyond what vowel module already uses.

---

---

# MODULE 1: RHYTHM

## What it measures

English rhythm is not about equal time between beats.
It is about **extreme duration contrast**: stressed syllables are long, unstressed syllables are aggressively compressed.

"I went to the store" → `i`(30ms) `WENT`(180ms) `tə`(25ms) `ðə`(20ms) `STORE`(200ms)

A non-native speaker gives equal weight to all syllables:
`I`(90ms) `WENT`(150ms) `TO`(85ms) `THE`(80ms) `STORE`(170ms)

Both might have the same total duration. But the *ratio* is wrong.

## Three signals to compute

### Signal 1: nPVI — global rhythm type

Normalized Pairwise Variability Index. Standard phonetics metric.
High nPVI = stress-timed (English ~55–65).
Low nPVI = syllable-timed (Spanish/French/Italian ~35–45).

```python
# pipeline/rhythm.py

import numpy as np
from dataclasses import dataclass, field


FUNCTION_WORDS = {
    "the", "a", "an", "to", "of", "in", "on", "at", "by", "for",
    "and", "but", "or", "so", "yet", "nor", "he", "she", "it",
    "we", "they", "him", "her", "us", "them", "his", "its", "our", "their",
    "was", "were", "is", "are", "am", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "will", "would",
    "could", "should", "might", "may", "can", "shall", "must",
    "that", "which", "who", "whom", "this", "there", "here",
}


def npvi(durations: list[float]) -> float:
    """
    Normalized Pairwise Variability Index.
    Input: list of syllable durations in seconds.
    English target: ~55–65. Below ~45 suggests syllable-timed L1 transfer.
    """
    n = len(durations)
    if n < 2:
        return 0.0
    d = np.array(durations)
    pairs = zip(d[:-1], d[1:])
    return float(
        (100 / (n - 1))
        * sum(abs(a - b) / ((a + b) / 2) for a, b in pairs if (a + b) > 0)
    )
```

### Signal 2: Per-syllable duration comparison vs target

Compare syllable-by-syllable, normalized by each speaker's own mean duration.
This removes speech rate differences — only the *pattern* matters.

```python
def normalize_durations(durations: list[float]) -> np.ndarray:
    """Divide by speaker's mean — removes overall speech rate difference."""
    d = np.array(durations)
    mean = d.mean()
    if mean == 0:
        return d
    return d / mean


def syllable_pattern_score(
    user_durations: list[float],
    target_durations: list[float],
) -> float:
    """
    Compare normalized duration patterns. Returns 0–100.
    Uses Pearson correlation of normalized duration vectors.
    High correlation = matching rhythm pattern.
    """
    if len(user_durations) != len(target_durations):
        # Align lengths if alignment produced different counts
        min_len = min(len(user_durations), len(target_durations))
        user_durations = user_durations[:min_len]
        target_durations = target_durations[:min_len]

    u = normalize_durations(user_durations)
    t = normalize_durations(target_durations)

    if np.std(u) == 0 or np.std(t) == 0:
        return 50.0

    correlation = np.corrcoef(u, t)[0, 1]
    # Correlation -1..1 → score 0..100
    return float(np.clip((correlation + 1) / 2 * 100, 0, 100))
```

### Signal 3: Function word inflation detection

The most common L2 rhythm error. Detectable directly from alignment.

```python
@dataclass
class FunctionWordAnalysis:
    word: str
    user_duration_ms: float
    target_duration_ms: float
    ratio: float           # user / target; > 2.0 is a clear error
    is_inflated: bool      # True if ratio > 1.7


def analyze_function_words(
    alignment: list,        # list of WordInstance with .word, .start, .end
    target_alignment: list,
) -> list[FunctionWordAnalysis]:
    """
    For each function word in the sentence, compare user vs target duration.
    Target typically: 20–50ms. Non-native: 70–120ms.
    """
    results = []
    target_map = {w.word.lower(): w for w in target_alignment}

    for word_instance in alignment:
        word = word_instance.word.lower().strip(".,!?")
        if word not in FUNCTION_WORDS:
            continue
        if word not in target_map:
            continue

        user_dur = (word_instance.end - word_instance.start) * 1000
        target_dur = (target_map[word].end - target_map[word].start) * 1000
        ratio = user_dur / target_dur if target_dur > 0 else 1.0

        results.append(FunctionWordAnalysis(
            word=word,
            user_duration_ms=user_dur,
            target_duration_ms=target_dur,
            ratio=ratio,
            is_inflated=ratio > 1.7,
        ))

    return results
```

## Aggregation and scoring

```python
@dataclass
class RhythmScore:
    score: float                              # 0–100
    npvi_user: float
    npvi_target: float
    pattern_correlation: float               # 0–1
    inflated_function_words: list[str]
    sentence_type_scores: dict[str, float]   # "statement", "question", etc.
    diagnostics: list[str]


def score_rhythm(
    user_syllable_durations: list[float],
    target_syllable_durations: list[float],
    user_word_alignment: list,
    target_word_alignment: list,
    sentence_type: str,
) -> RhythmScore:
    npvi_u = npvi(user_syllable_durations)
    npvi_t = npvi(target_syllable_durations)
    pattern_score = syllable_pattern_score(user_syllable_durations, target_syllable_durations)
    fw_analysis = analyze_function_words(user_word_alignment, target_word_alignment)

    inflated = [fw.word for fw in fw_analysis if fw.is_inflated]

    # Composite: 40% nPVI match, 40% pattern correlation, 20% function word reduction
    npvi_score = max(0, 100 - abs(npvi_u - npvi_t) * 2)
    fw_score = 100 * (1 - len(inflated) / max(len(fw_analysis), 1))
    composite = 0.4 * npvi_score + 0.4 * pattern_score + 0.2 * fw_score

    diagnostics = []
    if npvi_u < 45 and npvi_t > 55:
        diagnostics.append(
            f"Your rhythm is too even (nPVI={npvi_u:.0f} vs target {npvi_t:.0f}). "
            "Stressed syllables should be much longer than unstressed ones."
        )
    if inflated:
        words_str = ", ".join(f"'{w}'" for w in inflated[:4])
        diagnostics.append(
            f"Function words {words_str} are too long. "
            "These should be short and compressed — ~25–40ms in natural speech."
        )
    if pattern_score < 60:
        diagnostics.append(
            "The overall timing pattern differs from the target. "
            "Listen to which syllables the target rushes through vs. dwells on."
        )

    return RhythmScore(
        score=float(np.clip(composite, 0, 100)),
        npvi_user=npvi_u,
        npvi_target=npvi_t,
        pattern_correlation=pattern_score / 100,
        inflated_function_words=inflated,
        sentence_type_scores={sentence_type: float(np.clip(composite, 0, 100))},
        diagnostics=diagnostics,
    )
```

---

---

# MODULE 2: INTONATION

## What it measures

The pitch melody of the sentence. The most diagnostic signals:

| Feature | English rule | Common L2 error |
|---|---|---|
| Statement boundary | Final pitch falls | Flat or rising ending |
| Yes/No question boundary | Final pitch rises | Falls like a statement |
| Wh-question boundary | Final pitch falls (not rises) | Over-rising ("Where are you GOING↗") |
| Nuclear stress | One word gets strong pitch accent | Too many words equally stressed |

## Step 1: Extract and normalize pitch

```python
# pipeline/intonation.py

import librosa
import numpy as np
from scipy.ndimage import median_filter
from dataclasses import dataclass


def extract_f0(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract fundamental frequency (pitch) contour.
    Returns array of f0 values in Hz. 0 = unvoiced frame.
    Frame shift: 512 samples (~32ms at 16kHz).
    """
    f0 = librosa.yin(
        audio,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz — covers all speakers
        fmax=librosa.note_to_hz("C6"),   # ~1047 Hz
        sr=sr,
        frame_length=2048,
        hop_length=512,
    )
    # Smooth — remove single-frame pitch errors
    f0 = median_filter(f0, size=3)
    return f0


def f0_to_semitones(f0: np.ndarray) -> np.ndarray:
    """
    Normalize to semitones relative to speaker's own median pitch.
    Removes speaker-to-speaker pitch range differences.
    Voiced frames only.
    """
    voiced = f0[f0 > 0]
    if len(voiced) == 0:
        return np.zeros_like(f0)

    median_hz = np.median(voiced)
    semitones = np.where(
        f0 > 0,
        12 * np.log2(np.where(f0 > 0, f0 / median_hz, 1)),
        np.nan,
    )
    return semitones
```

## Step 2: Boundary tone — the most important single signal

```python
def boundary_slope(
    f0_semitones: np.ndarray,
    sr: int,
    hop_length: int = 512,
    tail_ms: int = 250,
) -> float:
    """
    Measure pitch slope in final `tail_ms` of utterance.
    Positive slope = rising (question-like).
    Negative slope = falling (statement-like).
    Near zero = flat.
    Returns slope in semitones/second.
    """
    n_tail_frames = int(tail_ms / 1000 * sr / hop_length)
    tail = f0_semitones[-n_tail_frames:]

    # Only voiced frames
    valid_mask = ~np.isnan(tail)
    if valid_mask.sum() < 3:
        return 0.0

    x = np.where(valid_mask)[0].astype(float)
    y = tail[valid_mask]
    slope, _ = np.polyfit(x, y, 1)

    # Convert from semitones/frame to semitones/second
    frames_per_second = sr / hop_length
    return float(slope * frames_per_second)


# Interpretation thresholds:
# slope > +1.5 st/s  → clearly rising
# slope < -1.5 st/s  → clearly falling
# -1.5 to +1.5       → flat / ambiguous
SLOPE_RISING_THRESHOLD = 1.5
SLOPE_FALLING_THRESHOLD = -1.5
```

## Step 3: DTW comparison — full contour shape

```python
def dtw_pitch_distance(
    user_semitones: np.ndarray,
    target_semitones: np.ndarray,
) -> float:
    """
    Dynamic Time Warping distance between pitch contours.
    Handles timing differences between user and target.
    Returns normalized distance (lower = more similar).
    """
    from librosa.sequence import dtw

    # Use only voiced frames (fill NaN with neighbor interpolation)
    def fill_nan(arr: np.ndarray) -> np.ndarray:
        mask = np.isnan(arr)
        arr = arr.copy()
        arr[mask] = np.interp(
            np.where(mask)[0],
            np.where(~mask)[0],
            arr[~mask],
        )
        return arr

    u = fill_nan(user_semitones).reshape(1, -1)
    t = fill_nan(target_semitones).reshape(1, -1)

    D, _ = dtw(u, t)
    # Normalize by path length
    normalized_dist = D[-1, -1] / (D.shape[0] + D.shape[1])
    return float(normalized_dist)


def dtw_score(distance: float) -> float:
    """Convert DTW distance to 0–100 score. Tuned empirically."""
    # distance ~0 → score 100, distance ~10 → score 0
    return float(np.clip(100 * np.exp(-distance / 5), 0, 100))
```

## Aggregation and scoring

```python
@dataclass
class IntonationScore:
    score: float
    boundary_slope_user: float          # semitones/second
    boundary_slope_target: float
    boundary_correct: bool              # does direction match?
    dtw_distance: float
    sentence_type: str
    diagnostics: list[str]


EXPECTED_BOUNDARY = {
    "statement":       "falling",
    "wh_question":     "falling",
    "yes_no_question": "rising",
    "complex":         "falling",
    "list":            "rising",     # list items rise, final item falls
    "exclamation":     "falling",
}


def classify_slope(slope: float) -> str:
    if slope > SLOPE_RISING_THRESHOLD:
        return "rising"
    if slope < SLOPE_FALLING_THRESHOLD:
        return "falling"
    return "flat"


def score_intonation(
    user_audio: np.ndarray,
    target_audio: np.ndarray,
    sr: int,
    sentence_type: str,
    hop_length: int = 512,
) -> IntonationScore:
    user_f0 = extract_f0(user_audio, sr)
    target_f0 = extract_f0(target_audio, sr)

    user_st = f0_to_semitones(user_f0)
    target_st = f0_to_semitones(target_f0)

    user_slope = boundary_slope(user_st, sr, hop_length)
    target_slope = boundary_slope(target_st, sr, hop_length)

    user_direction = classify_slope(user_slope)
    target_direction = classify_slope(target_slope)
    expected = EXPECTED_BOUNDARY.get(sentence_type, "falling")

    boundary_correct = user_direction == expected

    distance = dtw_pitch_distance(user_st, target_st)
    shape_score = dtw_score(distance)

    # Boundary direction is heavily weighted — it's the most salient error
    boundary_score = 100.0 if boundary_correct else 20.0
    composite = 0.5 * boundary_score + 0.5 * shape_score

    diagnostics = []
    if not boundary_correct:
        if sentence_type == "yes_no_question" and user_direction != "rising":
            diagnostics.append(
                "Yes/no questions should end with a rising pitch. "
                "Your voice falls at the end — it sounds like a statement."
            )
        elif sentence_type in ("statement", "wh_question") and user_direction == "rising":
            diagnostics.append(
                f"{'Statements' if sentence_type == 'statement' else 'Wh-questions'} "
                "should end with a falling pitch. "
                "Your voice rises at the end — this sounds uncertain."
            )
        elif user_direction == "flat":
            diagnostics.append(
                "Your pitch stays flat at the end of the sentence. "
                "Try making a clear fall/rise to match the sentence type."
            )

    if shape_score < 60:
        diagnostics.append(
            "The overall pitch melody differs from the target beyond the ending. "
            "Listen carefully to where the target's pitch peaks within the sentence."
        )

    return IntonationScore(
        score=float(np.clip(composite, 0, 100)),
        boundary_slope_user=user_slope,
        boundary_slope_target=target_slope,
        boundary_correct=boundary_correct,
        dtw_distance=distance,
        sentence_type=sentence_type,
        diagnostics=diagnostics,
    )
```

---

---

# MODULE 3: CONSONANTS

Consonants are not one problem. Each class requires a different algorithm.
Split by class — do not try to use a single metric across all consonants.

## ARPABET labels used by WhisperX

```python
# For reference when filtering alignment by phoneme class

FRICATIVES   = {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}
STOPS        = {"P", "B", "T", "D", "K", "G"}
VOICELESS_STOPS = {"P", "T", "K"}
AFFRICATES   = {"CH", "JH"}
NASALS       = {"M", "N", "NG"}
LIQUIDS      = {"L", "R"}
GLIDES       = {"W", "Y"}

# Fricatives split by voiced/voiceless pair — useful for error diagnosis
FRICATIVE_PAIRS = {
    "F":  {"voiced_pair": "V",  "place": "labiodental"},
    "V":  {"voiced_pair": "F",  "place": "labiodental"},
    "TH": {"voiced_pair": "DH", "place": "dental"},
    "DH": {"voiced_pair": "TH", "place": "dental"},
    "S":  {"voiced_pair": "Z",  "place": "alveolar"},
    "Z":  {"voiced_pair": "S",  "place": "alveolar"},
    "SH": {"voiced_pair": "ZH", "place": "postalveolar"},
    "ZH": {"voiced_pair": "SH", "place": "postalveolar"},
}
```

---

## 3A: Fricatives — Spectral Center of Gravity

**Why this works:** Each fricative has a characteristic spectral shape determined by vocal tract geometry. The CoG (average frequency weighted by energy) is stable and speaker-independent enough to be diagnostic.

**Reference CoG values (Hz) — empirically validated ranges:**

```python
# pipeline/consonants/fricatives.py

import numpy as np
from dataclasses import dataclass


# Center of gravity reference ranges in Hz
# Tuple: (low_end, center, high_end)
FRICATIVE_COG_REFERENCE = {
    "S":  (6000, 7500, 9000),   # High, hissing
    "Z":  (5500, 7000, 8500),   # Like S but slightly lower
    "SH": (3000, 4500, 6000),   # Lower, hushing
    "ZH": (2800, 4000, 5500),   # Like SH but slightly lower
    "F":  (4000, 6000, 8000),   # Broad, weaker than S
    "V":  (3500, 5500, 7500),
    "TH": (2000, 3500, 6000),   # Diffuse, low energy, dental
    "DH": (1500, 3000, 5500),
    "HH": (None, None, None),   # Aspiration — skip CoG for HH
}


def spectral_cog(audio_segment: np.ndarray, sr: int) -> float:
    """
    Spectral Center of Gravity for a fricative segment.
    Returns frequency in Hz.
    """
    # Apply Hanning window
    window = np.hanning(len(audio_segment))
    windowed = audio_segment * window

    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sr)

    # Focus on 1kHz–12kHz where fricative energy lives
    mask = (freqs >= 1000) & (freqs <= 12000)
    spectrum = spectrum[mask]
    freqs = freqs[mask]

    if spectrum.sum() == 0:
        return 0.0

    cog = float(np.sum(freqs * spectrum) / np.sum(spectrum))
    return cog


@dataclass
class FricativeAnalysis:
    phoneme: str           # ARPABET
    user_cog: float        # Hz
    target_cog: float      # Hz
    expected_cog: float    # from reference table
    cog_error: float       # abs(user_cog - target_cog)
    score: float           # 0–100
    diagnosis: str


def analyze_fricative(
    phoneme: str,
    user_audio_segment: np.ndarray,
    target_audio_segment: np.ndarray,
    sr: int,
) -> FricativeAnalysis | None:
    """
    Compare user vs target fricative quality.
    The target voice IS the reference — we measure deviation from it.
    Also cross-check against expected CoG range for sanity.
    """
    if phoneme not in FRICATIVE_COG_REFERENCE:
        return None
    low, center, high = FRICATIVE_COG_REFERENCE[phoneme]
    if center is None:
        return None

    user_cog = spectral_cog(user_audio_segment, sr)
    target_cog = spectral_cog(target_audio_segment, sr)
    error = abs(user_cog - target_cog)

    # Score: exponential decay. Error < 500Hz → ~95, Error 2000Hz → ~50
    score = float(np.clip(100 * np.exp(-error / 2000), 0, 100))

    # Generate specific diagnosis
    diagnosis = _fricative_diagnosis(phoneme, user_cog, target_cog, center)

    return FricativeAnalysis(
        phoneme=phoneme,
        user_cog=user_cog,
        target_cog=target_cog,
        expected_cog=float(center),
        cog_error=error,
        score=score,
        diagnosis=diagnosis,
    )


def _fricative_diagnosis(
    phoneme: str, user_cog: float, target_cog: float, expected: float
) -> str:
    error = user_cog - target_cog  # positive = user too high, negative = user too low

    if phoneme in ("TH", "DH"):
        if user_cog > 5500:
            return (
                f"/{phoneme}/ sounds like /{'S' if phoneme == 'TH' else 'Z'}/. "
                "Place the tongue tip between the teeth, not behind them. "
                "The sound should be weak and diffuse, not sharp."
            )
        if user_cog > 4000:
            return (
                f"/{phoneme}/ is close but the tongue may not be dental enough. "
                "Push the tip gently against the upper teeth."
            )

    if phoneme in ("S", "Z") and user_cog < 5000:
        return (
            f"/{phoneme}/ sounds too 'hushed' (like /{'SH' if phoneme == 'S' else 'ZH'}/). "
            "Move the tongue tip closer to the alveolar ridge."
        )

    if phoneme in ("SH", "ZH") and user_cog > 6500:
        return (
            f"/{phoneme}/ sounds too sharp (like /{'S' if phoneme == 'SH' else 'Z'}/). "
            "Move tongue slightly back and round the lips slightly."
        )

    if abs(error) < 500:
        return f"/{phoneme}/ sounds correct."

    direction = "too high-pitched/sharp" if error > 0 else "too low-pitched/soft"
    return f"/{phoneme}/ spectral quality is {direction} compared to target."
```

---

## 3B: Stops — Voice Onset Time (VOT)

**Why this works:** English voiceless stops (/p/, /t/, /k/) in stressed word-initial position have aspirated release: a burst of air before voicing begins. VOT = time between burst and voicing onset. English: 60–100ms. Most other languages: 10–30ms. Under-aspiration is one of the strongest non-native markers.

```python
# pipeline/consonants/stops.py

import numpy as np
import librosa
from dataclasses import dataclass


# English voiceless stop VOT reference (word-initial, stressed) in ms
VOT_REFERENCE_MS = {
    "P": (50, 75, 100),    # (min, center, max)
    "T": (55, 80, 110),
    "K": (60, 85, 125),
}

# Voiced stops should have near-zero or negative VOT (prevoicing)
VOT_VOICED_REFERENCE_MS = {
    "B": (-20, 5, 20),
    "D": (-20, 5, 20),
    "G": (-20, 5, 20),
}


@dataclass
class VOTAnalysis:
    phoneme: str
    user_vot_ms: float
    target_vot_ms: float
    expected_range: tuple[float, float, float]
    is_word_initial: bool
    is_stressed: bool
    score: float
    diagnosis: str


def extract_vot(
    audio: np.ndarray,
    sr: int,
    phoneme_start_s: float,
    phoneme_end_s: float,
) -> float:
    """
    Estimate Voice Onset Time for a stop consonant.
    Returns VOT in milliseconds.

    Algorithm:
    1. Extract window from phoneme_start to phoneme_start + 150ms
    2. Find burst: first frame of significant broadband energy
    3. Find voicing onset: first frame with strong periodicity (autocorrelation peak)
    4. VOT = voicing_onset - burst_onset

    This is a simplified but functional implementation.
    For higher accuracy, see AutoVOT (Keshet et al. 2014) if needed later.
    """
    WINDOW_S = 0.15  # look 150ms ahead for voicing onset
    HOP_LENGTH = 256  # ~16ms at 16kHz
    FRAME_LENGTH = 1024

    start_sample = int(phoneme_start_s * sr)
    end_sample = min(int((phoneme_start_s + WINDOW_S) * sr), len(audio))
    segment = audio[start_sample:end_sample]

    if len(segment) < FRAME_LENGTH:
        return 0.0

    # Step 1: Frame-wise RMS energy
    frames = librosa.util.frame(
        segment, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    energy = np.sqrt(np.mean(frames**2, axis=0))

    if energy.max() == 0:
        return 0.0

    # Step 2: Find burst — first frame where energy crosses 15% of max
    burst_threshold = 0.15 * energy.max()
    burst_frames = np.where(energy > burst_threshold)[0]
    if len(burst_frames) == 0:
        return 0.0
    burst_frame = int(burst_frames[0])

    # Step 3: Find voicing onset — periodicity via normalized autocorrelation
    voicing_frame = len(energy) - 1  # default: no voicing found
    min_lag = int(sr / 400)           # 400 Hz max f0
    max_lag = int(sr / 70)            # 70 Hz min f0

    for i in range(burst_frame, len(energy)):
        frame_start = i * HOP_LENGTH
        frame_end = frame_start + FRAME_LENGTH
        if frame_end > len(segment):
            break
        frame_audio = segment[frame_start:frame_end]

        # Normalized autocorrelation
        autocorr = np.correlate(frame_audio, frame_audio, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        if autocorr[0] == 0:
            continue
        autocorr /= autocorr[0]

        peak_in_range = autocorr[min_lag:max_lag]
        if len(peak_in_range) > 0 and peak_in_range.max() > 0.35:
            voicing_frame = i
            break

    vot_frames = voicing_frame - burst_frame
    vot_ms = float(vot_frames * HOP_LENGTH / sr * 1000)
    return max(0.0, vot_ms)


def analyze_stop(
    phoneme: str,
    user_audio: np.ndarray,
    target_audio: np.ndarray,
    sr: int,
    phoneme_start_s: float,
    phoneme_end_s: float,
    is_word_initial: bool,
    is_stressed: bool,
) -> VOTAnalysis | None:
    """
    Only analyze voiceless stops in word-initial stressed positions.
    That is where English aspiration is most contrastive and diagnostic.
    """
    if phoneme not in VOT_REFERENCE_MS:
        return None
    if not (is_word_initial and is_stressed):
        # Aspiration contrast is weak in other positions — skip
        return None

    user_vot = extract_vot(user_audio, sr, phoneme_start_s, phoneme_end_s)
    target_vot = extract_vot(target_audio, sr, phoneme_start_s, phoneme_end_s)
    ref_min, ref_center, ref_max = VOT_REFERENCE_MS[phoneme]

    # Score based on how close to target AND whether in expected English range
    # Primary: match target; secondary: match expected English VOT
    target_error = abs(user_vot - target_vot)
    range_error = max(0, ref_min - user_vot) + max(0, user_vot - ref_max)

    score = float(np.clip(
        100 * np.exp(-target_error / 30) * np.exp(-range_error / 40),
        0, 100
    ))

    diagnosis = _stop_diagnosis(phoneme, user_vot, target_vot, ref_min, ref_max)

    return VOTAnalysis(
        phoneme=phoneme,
        user_vot_ms=user_vot,
        target_vot_ms=target_vot,
        expected_range=(ref_min, ref_center, ref_max),
        is_word_initial=is_word_initial,
        is_stressed=is_stressed,
        score=score,
        diagnosis=diagnosis,
    )


def _stop_diagnosis(
    phoneme: str,
    user_vot: float,
    target_vot: float,
    ref_min: float,
    ref_max: float,
) -> str:
    if user_vot < ref_min - 15:
        return (
            f"/{phoneme.lower()}/ has too little aspiration (VOT={user_vot:.0f}ms, "
            f"English target: {ref_min:.0f}–{ref_max:.0f}ms). "
            "Add a stronger puff of breath after releasing the consonant."
        )
    if user_vot > ref_max + 20:
        return (
            f"/{phoneme.lower()}/ is over-aspirated (VOT={user_vot:.0f}ms). "
            "Slightly reduce the breath burst — you don't need to push so hard."
        )
    return f"/{phoneme.lower()}/ aspiration is good (VOT={user_vot:.0f}ms)."
```

---

## 3C: Rhotic /r/ — F3 Depression

**Why this works:** English /r/ (retroflex/bunched) dramatically lowers the third formant (F3) to ~1800–2200 Hz. This is the single most reliable acoustic signature of English rhoticity. Other languages' /r/ sounds (tapped, trilled, uvular) do not do this — F3 stays at 2400–2800 Hz.

```python
# pipeline/consonants/liquids.py

import parselmouth
import numpy as np
from dataclasses import dataclass


# F3 reference values for English /r/ (Hz)
R_F3_RHOTIC_MAX = 2200      # below this = good rhotic /r/
R_F3_ERROR_MIN = 2500       # above this = clearly non-rhotic
R_F3_WARNING_ZONE = (2200, 2500)  # borderline

# F2 reference for dark /l/ (syllable-final) vs clear /l/ (syllable-initial)
# Dark /l/: F2 typically drops to ~800–1200 Hz (velarized)
# Clear /l/: F2 ~1400–1800 Hz
L_DARK_F2_MAX = 1300
L_CLEAR_F2_MIN = 1400


@dataclass
class RhoticAnalysis:
    user_f3: float
    target_f3: float
    is_rhotic: bool       # True if user's F3 < R_F3_RHOTIC_MAX
    score: float
    diagnosis: str


@dataclass
class LateralAnalysis:
    phoneme: str
    position: str          # "initial" or "final"
    user_f2: float
    target_f2: float
    dark_l_correct: bool   # Only relevant for final position
    score: float
    diagnosis: str


def extract_formants_at_midpoint(
    audio: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    max_formant: float = 5500,
) -> tuple[float, float, float]:
    """Returns (F1, F2, F3) in Hz at phoneme midpoint."""
    midpoint = (start_s + end_s) / 2
    sound = parselmouth.Sound(audio, sampling_frequency=sr)
    formants = sound.to_formant_burg(
        time_step=0.005,
        max_number_of_formants=5,
        maximum_formant=max_formant,
        window_length=0.025,
    )
    f1 = formants.get_value_at_time(1, midpoint) or 0.0
    f2 = formants.get_value_at_time(2, midpoint) or 0.0
    f3 = formants.get_value_at_time(3, midpoint) or 0.0
    return f1, f2, f3


def analyze_r(
    user_audio: np.ndarray,
    target_audio: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    max_formant: float = 5500,
) -> RhoticAnalysis:
    """
    Compare F3 during /r/.
    Low F3 (< 2200 Hz) = good English rhotic.
    High F3 (> 2500 Hz) = non-rhotic substitution.
    """
    _, _, user_f3 = extract_formants_at_midpoint(user_audio, sr, start_s, end_s, max_formant)
    _, _, target_f3 = extract_formants_at_midpoint(target_audio, sr, start_s, end_s, max_formant)

    is_rhotic = user_f3 < R_F3_RHOTIC_MAX if user_f3 > 0 else False
    f3_error = abs(user_f3 - target_f3)

    # Score based on F3 deviation from target
    score = float(np.clip(100 * np.exp(-f3_error / 300), 0, 100))

    if user_f3 == 0:
        diagnosis = "/r/ segment too short to measure — ensure you hold the sound briefly."
    elif user_f3 > R_F3_ERROR_MIN:
        diagnosis = (
            f"/r/ sounds non-rhotic (F3={user_f3:.0f}Hz, target={target_f3:.0f}Hz). "
            "Curl or bunch the tongue tip further back, "
            "or squeeze the tongue sides against upper molars."
        )
    elif user_f3 > R_F3_WARNING_ZONE[0]:
        diagnosis = (
            f"/r/ is borderline (F3={user_f3:.0f}Hz). "
            "Try curling the tongue tip slightly more."
        )
    else:
        diagnosis = f"/r/ is rhotic and correct (F3={user_f3:.0f}Hz)."

    return RhoticAnalysis(
        user_f3=user_f3,
        target_f3=target_f3,
        is_rhotic=is_rhotic,
        score=score,
        diagnosis=diagnosis,
    )


def analyze_l(
    user_audio: np.ndarray,
    target_audio: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    position: str,         # "initial" or "final"
    max_formant: float = 5500,
) -> LateralAnalysis:
    """
    Dark /l/ (syllable-final) requires F2 drop via velarization.
    Many L2 speakers use clear /l/ everywhere.
    Only flag errors in final position — that's where the error is distinctive.
    """
    _, user_f2, _ = extract_formants_at_midpoint(user_audio, sr, start_s, end_s, max_formant)
    _, target_f2, _ = extract_formants_at_midpoint(target_audio, sr, start_s, end_s, max_formant)

    f2_error = abs(user_f2 - target_f2)
    score = float(np.clip(100 * np.exp(-f2_error / 300), 0, 100))

    dark_l_correct = True
    diagnosis = f"/l/ sounds correct (F2={user_f2:.0f}Hz)."

    if position == "final":
        dark_l_correct = user_f2 < L_DARK_F2_MAX
        if not dark_l_correct and user_f2 > L_CLEAR_F2_MIN:
            diagnosis = (
                f"Syllable-final /l/ is too 'clear' (F2={user_f2:.0f}Hz, target={target_f2:.0f}Hz). "
                "English /l/ at the end of syllables is 'dark' — "
                "raise the back of the tongue toward the soft palate while keeping the tip up."
            )

    return LateralAnalysis(
        phoneme="L",
        position=position,
        user_f2=user_f2,
        target_f2=target_f2,
        dark_l_correct=dark_l_correct,
        score=score,
        diagnosis=diagnosis,
    )
```

---

## 3D: Consonant Module Aggregator

```python
# pipeline/consonants/aggregator.py

from dataclasses import dataclass, field
from .fricatives import FricativeAnalysis, analyze_fricative
from .stops import VOTAnalysis, analyze_stop
from .liquids import RhoticAnalysis, LateralAnalysis, analyze_r, analyze_l
import numpy as np


@dataclass
class ConsonantScore:
    score: float                                      # 0–100 composite
    fricative_score: float
    stop_aspiration_score: float
    rhotic_score: float
    lateral_score: float
    fricative_analyses: list[FricativeAnalysis] = field(default_factory=list)
    stop_analyses: list[VOTAnalysis] = field(default_factory=list)
    rhotic_analyses: list[RhoticAnalysis] = field(default_factory=list)
    lateral_analyses: list[LateralAnalysis] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


def score_consonants(
    fricative_analyses: list[FricativeAnalysis],
    stop_analyses: list[VOTAnalysis],
    rhotic_analyses: list[RhoticAnalysis],
    lateral_analyses: list[LateralAnalysis],
) -> ConsonantScore:
    def mean_score(items) -> float:
        scores = [x.score for x in items if x is not None]
        return float(np.mean(scores)) if scores else 100.0

    fricative_score = mean_score(fricative_analyses)
    stop_score = mean_score(stop_analyses)
    rhotic_score = mean_score(rhotic_analyses)
    lateral_score = mean_score(lateral_analyses)

    # Weights: fricatives and stops most diagnostic
    composite = (
        0.35 * fricative_score
        + 0.35 * stop_score
        + 0.20 * rhotic_score
        + 0.10 * lateral_score
    )

    # Collect worst diagnostics (top 3 issues)
    all_diagnostics = (
        [(a.score, a.diagnosis) for a in fricative_analyses if a.score < 75]
        + [(a.score, a.diagnosis) for a in stop_analyses if a.score < 75]
        + [(a.score, a.diagnosis) for a in rhotic_analyses if a.score < 75]
        + [(a.score, a.diagnosis) for a in lateral_analyses if a.score < 75]
    )
    all_diagnostics.sort(key=lambda x: x[0])
    diagnostics = [d for _, d in all_diagnostics[:4]]

    return ConsonantScore(
        score=float(np.clip(composite, 0, 100)),
        fricative_score=fricative_score,
        stop_aspiration_score=stop_score,
        rhotic_score=rhotic_score,
        lateral_score=lateral_score,
        fricative_analyses=fricative_analyses,
        stop_analyses=stop_analyses,
        rhotic_analyses=rhotic_analyses,
        lateral_analyses=lateral_analyses,
        diagnostics=diagnostics,
    )
```

---

---

# HOW TO CALL THE FULL PIPELINE

```python
# pipeline/analyze.py
# This ties all three modules together. Called after recording is complete.

async def analyze_session(
    user_recordings: dict[int, np.ndarray],    # sentence_id → audio
    target_recordings: dict[int, np.ndarray],  # sentence_id → audio (cloned)
    alignments: dict[int, Alignment],          # sentence_id → WhisperX output
    sentence_meta: dict[int, SentenceMeta],
    sr: int = 16000,
) -> ComparisonResult:

    rhythm_scores = []
    intonation_scores = []
    consonant_score_inputs = []

    for sid in user_recordings:
        user_audio = user_recordings[sid]
        target_audio = target_recordings[sid]
        alignment = alignments[sid]
        meta = sentence_meta[sid]

        # Rhythm
        user_syl_durations = get_syllable_durations(alignment.user)
        target_syl_durations = get_syllable_durations(alignment.target)
        rhythm = score_rhythm(
            user_syl_durations, target_syl_durations,
            alignment.user.words, alignment.target.words,
            meta.sentence_type,
        )
        rhythm_scores.append(rhythm)

        # Intonation
        intonation = score_intonation(user_audio, target_audio, sr, meta.sentence_type)
        intonation_scores.append(intonation)

        # Consonants — iterate phonemes
        fricative_results, stop_results, r_results, l_results = [], [], [], []
        for ph in alignment.user.phonemes:
            arpabet = ph.phoneme.upper()
            u_seg = extract_segment(user_audio, ph.start, ph.end, sr)
            t_seg = extract_segment(target_audio, ph.start, ph.end, sr)

            if arpabet in FRICATIVES:
                result = analyze_fricative(arpabet, u_seg, t_seg, sr)
                if result:
                    fricative_results.append(result)
            elif arpabet in VOICELESS_STOPS:
                result = analyze_stop(
                    arpabet, user_audio, target_audio, sr,
                    ph.start, ph.end, ph.is_word_initial, ph.is_stressed
                )
                if result:
                    stop_results.append(result)
            elif arpabet == "R":
                result = analyze_r(user_audio, target_audio, sr, ph.start, ph.end)
                r_results.append(result)
            elif arpabet == "L":
                position = "final" if ph.is_syllable_final else "initial"
                result = analyze_l(user_audio, target_audio, sr, ph.start, ph.end, position)
                l_results.append(result)

        consonant_score_inputs.append(
            score_consonants(fricative_results, stop_results, r_results, l_results)
        )

    # Aggregate across all sentences
    return aggregate_to_comparison_result(rhythm_scores, intonation_scores, consonant_score_inputs)
```

---

# WHAT EACH MODULE OUTPUTS TO THE FRONTEND

```
Rhythm:
  → overall score 0–100
  → nPVI user vs target
  → list of inflated function words (to highlight in UI)
  → per-sentence-type breakdown (statement / question / complex)
  → diagnostic strings

Intonation:
  → overall score 0–100
  → boundary direction: "rising" | "falling" | "flat" (user vs expected)
  → DTW shape distance
  → per-sentence-type breakdown
  → diagnostic strings

Consonants:
  → overall score 0–100
    → sub-score: fricatives
    → sub-score: aspiration (stops)
    → sub-score: rhotics (/r/)
    → sub-score: laterals (/l/)
  → per-phoneme breakdown list (for the drill-down table)
  → diagnostic strings (top 3–4 issues)
```

All scores 0–100. All diagnostic strings are human-readable, specific, and actionable.
```