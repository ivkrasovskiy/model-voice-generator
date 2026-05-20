## Phase 0.5 Verdict Summary

### H1 — Is Deterding 1997 stale?
Median |Deterding − Modern RP|: F1=163.0 Hz, F2=194.9 Hz
**Verdict: STALE**
*(threshold 75.0 Hz)*

### H2 — Does real BC deviate from modern RP on /æ ɛ ʌ/?
Median |Real BC − Modern RP|: F1=19.6 Hz, F2=88.5 Hz
**Verdict: BC_DEVIATES**

### H3 — Does IndexTTS distort BC's vowel space?
Median |Synth BC − Real BC|: F1=49.3 Hz, F2=25.8 Hz
Cross-ref variance: F1=None Hz, F2=None Hz
**Verdict: NO_SHIFT**
*(shift threshold 75.0 Hz, variance threshold 50 Hz)*

### H4 — Do synth-BC /æ ɛ/ collide with owner?

| Phoneme | d(synth, owner) | d(synth, real) | d(owner, real) | Confirmed |
|---------|----------------|----------------|----------------|-----------|
| /æ/ | 308.4 Hz | 106.1 Hz | 303.9 Hz | no |
| /ɛ/ | 239.3 Hz | 54.0 Hz | 278.1 Hz | no |
