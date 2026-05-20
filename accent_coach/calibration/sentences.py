from __future__ import annotations

from pydantic import BaseModel


class Sentence(BaseModel):
    id: int
    text: str
    sentence_type: str
    targets: list[str]


CALIBRATION_SENTENCES: list[Sentence] = [
    # --- Pure monophthongs (11) ---
    Sentence(
        id=1,
        text="Please leave the keys on the table.",
        sentence_type="statement",
        targets=["iː"],
    ),
    Sentence(
        id=2,
        text="The ship hit a big cliff in the mist.",
        sentence_type="statement",
        targets=["ɪ"],
    ),
    Sentence(
        id=3,
        text="He left the red pen on the desk.",
        sentence_type="statement",
        targets=["ɛ"],
    ),
    Sentence(
        id=4,
        text="The black cat sat on the flat mat.",
        sentence_type="statement",
        targets=["æ"],
    ),
    Sentence(
        id=5,
        text="The father parked the car in the barn.",
        sentence_type="statement",
        targets=["ɑː"],
    ),
    Sentence(
        id=6,
        text="The dog got lost on the rocky plot.",
        sentence_type="statement",
        targets=["ɒ"],
    ),
    Sentence(
        id=7,
        text="The thought of walking to the court was daunting.",
        sentence_type="statement",
        targets=["ɔː"],
    ),
    Sentence(
        id=8,
        text="The cook put the book on the wooden hook.",
        sentence_type="statement",
        targets=["ʊ"],
    ),
    Sentence(
        id=9,
        text="The moon moved through the smooth blue pool.",
        sentence_type="statement",
        targets=["uː"],
    ),
    Sentence(
        id=10,
        text="The young monk strung up the drum in the sun.",
        sentence_type="statement",
        targets=["ʌ"],
    ),
    Sentence(
        id=11,
        text="The nurse first heard the birds stir at dawn.",
        sentence_type="statement",
        targets=["ɜː"],
    ),
    # --- Schwa / weak forms (3) ---
    Sentence(
        id=12,
        text="A man and a woman arrived at a hotel.",
        sentence_type="statement",
        targets=["ə"],
    ),
    Sentence(
        id=13,
        text="The police would have arrested him if he had been there.",
        sentence_type="statement",
        targets=["ə"],
    ),
    Sentence(
        id=14,
        text="You can do it if you want to.",
        sentence_type="statement",
        targets=["ə"],
    ),
    # --- Diphthongs (5) ---
    Sentence(
        id=15,
        text="They came to play the same game on a rainy day.",
        sentence_type="statement",
        targets=["eɪ"],
    ),
    Sentence(
        id=16,
        text="I tried to find a bright light at night.",
        sentence_type="statement",
        targets=["aɪ"],
    ),
    Sentence(
        id=17,
        text="The boy enjoyed the noise from the toy.",
        sentence_type="statement",
        targets=["ɔɪ"],
    ),
    Sentence(
        id=18,
        text="The old road goes close to home.",
        sentence_type="statement",
        targets=["əʊ"],
    ),
    Sentence(
        id=19,
        text="The mouse found a house outside town.",
        sentence_type="statement",
        targets=["aʊ"],
    ),
    # --- TH consonants (2) ---
    Sentence(
        id=20,
        text="The weather there is better than I thought.",
        sentence_type="statement",
        targets=["ð"],
    ),
    Sentence(
        id=21,
        text="Think of three things that seem worth the effort.",
        sentence_type="statement",
        targets=["θ"],
    ),
    # --- Aspiration stops /p/ (2) ---
    Sentence(
        id=22,
        text="Peter put the parcel on the post.",
        sentence_type="statement",
        targets=["pʰ"],
    ),
    Sentence(
        id=23,
        text="Paul picked up the pen and passed it back.",
        sentence_type="statement",
        targets=["pʰ"],
    ),
    # --- Aspiration stops /t/ (2) ---
    Sentence(
        id=24,
        text="Tom took the train to the terminal.",
        sentence_type="statement",
        targets=["tʰ"],
    ),
    Sentence(
        id=25,
        text="Tell me the time when the tide turns.",
        sentence_type="statement",
        targets=["tʰ"],
    ),
    # --- Aspiration stops /k/ (2) ---
    Sentence(
        id=26,
        text="Kate kept the keys in the kitchen cabinet.",
        sentence_type="statement",
        targets=["kʰ"],
    ),
    Sentence(
        id=27,
        text="Come and carry the coat to the car.",
        sentence_type="statement",
        targets=["kʰ"],
    ),
    # --- Fricatives (4) ---
    Sentence(
        id=28,
        text="The sun sets slowly in the west.",
        sentence_type="statement",
        targets=["s"],
    ),
    Sentence(
        id=29,
        text="Zebras zigzag when they are surprised.",
        sentence_type="statement",
        targets=["z"],
    ),
    Sentence(
        id=30,
        text="She showed the chef the fresh fish dish.",
        sentence_type="statement",
        targets=["ʃ"],
    ),
    Sentence(
        id=31,
        text="The beige garage belongs to a prestigious regime.",
        sentence_type="statement",
        targets=["ʒ"],
    ),
    # --- Approximants / liquids (3) ---
    Sentence(
        id=32,
        text="The red river runs rapidly around the rocks.",
        sentence_type="statement",
        targets=["r"],
    ),
    Sentence(
        id=33,
        text="The lull of the full bell filled the hall.",
        sentence_type="statement",
        targets=["l"],
    ),
    Sentence(
        id=34,
        text="We walked west while the wind blew warm.",
        sentence_type="statement",
        targets=["w"],
    ),
    # --- Intonation: yes/no questions (2) ---
    Sentence(
        id=35,
        text="Are you coming to the meeting tomorrow?",
        sentence_type="yes_no_question",
        targets=["intonation_rise"],
    ),
    Sentence(
        id=36,
        text="Has she already finished the report?",
        sentence_type="yes_no_question",
        targets=["intonation_rise"],
    ),
    # --- Intonation: wh-questions (2) ---
    Sentence(
        id=37,
        text="Where did you put the documents?",
        sentence_type="wh_question",
        targets=["intonation_fall"],
    ),
    Sentence(
        id=38,
        text="What time does the last train leave?",
        sentence_type="wh_question",
        targets=["intonation_fall"],
    ),
    # --- Intonation: declarative (1) ---
    Sentence(
        id=39,
        text="The conference begins at nine and ends at five.",
        sentence_type="statement",
        targets=["intonation_plateau_fall"],
    ),
    # --- Intonation: complex (1) ---
    Sentence(
        id=40,
        text="If you want to improve your accent, practise every single day.",
        sentence_type="complex",
        targets=["intonation_complex"],
    ),
    # --- Rhythm / linking / weak forms (6) ---
    Sentence(
        id=41,
        text="An apple a day keeps the doctor away.",
        sentence_type="statement",
        targets=["rhythm", "linking"],
    ),
    Sentence(
        id=42,
        text="I would have told you if I had known earlier.",
        sentence_type="statement",
        targets=["rhythm", "weak_forms"],
    ),
    Sentence(
        id=43,
        text="He ought to have been more careful about it.",
        sentence_type="statement",
        targets=["rhythm", "weak_forms"],
    ),
    Sentence(
        id=44,
        text="She turned it off and went back to sleep.",
        sentence_type="statement",
        targets=["rhythm", "linking"],
    ),
    Sentence(
        id=45,
        text="Could you tell me how to get to the nearest station?",
        sentence_type="yes_no_question",
        targets=["rhythm", "weak_forms"],
    ),
    Sentence(
        id=46,
        text="I am not sure whether that is actually the right answer.",
        sentence_type="statement",
        targets=["rhythm", "linking"],
    ),
    # --- Stress-shift triplets (4) ---
    Sentence(
        id=47,
        text="Take a photograph of the photographer at the photography exhibition.",
        sentence_type="statement",
        targets=["stress_shift"],
    ),
    Sentence(
        id=48,
        text="The economy depends on economic policy and economic data.",
        sentence_type="statement",
        targets=["stress_shift"],
    ),
    Sentence(
        id=49,
        text="A democrat supports democracy and democratic values.",
        sentence_type="statement",
        targets=["stress_shift"],
    ),
    Sentence(
        id=50,
        text="The original origin of the originality is unclear.",
        sentence_type="statement",
        targets=["stress_shift"],
    ),
]


def get_by_id(sentence_id: int) -> Sentence:
    for s in CALIBRATION_SENTENCES:
        if s.id == sentence_id:
            return s
    raise KeyError(f"No calibration sentence with id={sentence_id}")
