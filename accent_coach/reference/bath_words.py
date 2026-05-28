"""BATH-class and LOT-class word sets for British RP phoneme relabeling.

BATH_WORDS: words where CMU uses AE (→ æ) but RP uses ɑː.
  Source: Wells (1982) BATH lexical set + common conversational words.

LOT_WORDS: words where CMU uses AA (→ ɑː) but RP uses ɒ.
  CMU uses AO for THOUGHT (dog, fog) — those already map to ɔː correctly.
  Only AA-coded LOT words need relabeling here.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# BATH / PALM / START class — CMU AE1 → RP /ɑː/
# Wells (1982) §2.2 BATH set + derived forms
# ---------------------------------------------------------------------------
BATH_WORDS: frozenset[str] = frozenset({
    # Core BATH set (Wells 1982)
    "bath", "baths", "path", "paths", "half", "halves", "calf", "calves",
    "staff", "staffs", "laugh", "laughs", "laughed", "laughing", "laughter",
    "craft", "crafts", "draft", "drafts", "shaft", "shafts", "raft", "rafts",
    "fast", "faster", "fastest", "past", "pasture", "mast", "masts", "last",
    "blast", "blasted", "vast", "castle", "castles", "class", "classes",
    "glass", "glasses", "grass", "mass", "lass", "pass", "passes", "brass",
    "ask", "asks", "asked", "asking", "task", "tasks", "bask", "casket",
    "basket", "baskets", "rascal", "mask", "flask", "grasp", "grasped",
    "clasp", "clasped", "rasp", "hasp",
    "dance", "danced", "dancer", "dancing", "lance", "glance", "glanced",
    "prance", "chance", "chances", "enhance", "advance", "advances",
    "trance", "stance", "france", "entrance", "romance",
    "plant", "plants", "planted", "planting", "slant", "chant", "grant",
    "grants", "rant", "pant", "pants", "ant", "ants", "advantage",
    "after", "afternoon", "afternoons", "aftermath",
    "answer", "answers", "answered", "answering",
    "branch", "branches", "ranch", "blanch",
    "command", "commands", "commander", "demand", "demands",
    "contrast", "contrastive",
    "disaster", "disasters", "disastrous",
    "example", "examples",
    "fasten", "fastened", "fastening",
    "father", "fathers", "fatherly",
    "forecast", "forecasts",
    "ghastly", "ghastlier",
    "master", "masters", "mastered", "mastering", "masterly",
    "nasty", "nastier", "nastily",
    "pasta", "plaster", "plastered", "plastering",
    "rafter", "rafters",
    "rather", "sample", "samples", "sampled",
    "can't",  # contracted form — add explicitly
    "shan't", "aren't",  # further contractions
})

# ---------------------------------------------------------------------------
# LOT class — CMU AA (→ ɑː in IPA mapping) but RP /ɒ/
# Only AA-coded words need relabeling; AO-coded (dog, fog, boss) → ɔː already correct.
# ---------------------------------------------------------------------------
LOT_WORDS: frozenset[str] = frozenset({
    "lot", "lots", "not", "hot", "got", "shot", "spot", "stop", "stops",
    "stopped", "stopping", "top", "tops", "pop", "drop", "drops", "dropped",
    "shop", "shops", "clock", "clocks", "block", "blocks", "blocked",
    "lock", "locks", "locked", "sock", "socks", "rock", "rocks", "rocked",
    "knock", "knocked", "stock", "stocks", "dock", "docks", "mock",
    "pot", "pots", "dot", "dots", "cot", "rot", "slot", "slots",
    "box", "boxes", "fox", "foxes", "ox", "oxen",
    "odd", "odds", "nod", "nods", "god", "gods",
    "job", "jobs", "mob", "mobs", "rob", "robs", "knob", "knobs",
    "college", "colleges", "bottle", "bottles", "model", "models",
    "problem", "problems", "possible", "possibly", "politics", "political",
    "popular", "popularity", "proper", "properly", "property", "properties",
    "process", "processed", "product", "products", "profit", "profits",
    "project", "projects", "promise", "promises", "promised",
    "obvious", "obviously", "knowledge", "honest", "honestly",
    "hospital", "hospitals", "october", "copy", "copies",
    "chocolate", "chocolates", "tropical", "topic", "topics",
})
