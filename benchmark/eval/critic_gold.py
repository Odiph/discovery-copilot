"""The labelled question set the critic is judged against.

`benchmark/eval/analyse_eval.py` ends section 3 by saying what it cannot do:

    THE CRITIC IS NOT UNDER TEST HERE. It bands the coach's OWN candidates,
    never the interviewer's questions [...] Judging the bandings needs a
    harness that forces known-bad candidates in.

This module is the "known-bad candidates". `judge_critic.py` is the harness.

**Where the labels come from, and why they are trustworthy.** The five BAD
cases are lifted verbatim from `claudictive_script.py`, where they sit under
a comment that names them, in order, as "the five failure shapes the critic
names". They were written to be bad, one per rule, before any critic output
existed to fit them to -- authorial intent recorded in advance is the
strongest provenance available here, and it is much stronger than me reading
the rubric and inventing sentences that match it. The GOOD cases are real
interviewer turns from the same script, chosen from the stretches the script
itself treats as competent interviewing.

**The three tiers are not decoration.** They exist because the failure modes
they catch are different, and one of them is invisible without the others:

- `BAD`: a violation banded `clean` is the dangerous direction -- the coach
  hands the interviewer a leading question with no objection.
- `GOOD`: a clean question banded `clear_violation` is DROPPED by
  `rank_candidates` and the interviewer never learns it existed. The measured
  run banded 26% `clear_violation` with nothing to compare it against, so
  over-flagging has been invisible in every number produced so far. Negative
  controls are the only thing that makes it visible.
- `CONTESTED`: questions that violate the letter of the rubric while being
  good interviewing in context ("If I gave you a magic wand..."). These are
  REPORTED AND NEVER SCORED. Asserting an answer for them would quietly
  promote my reading of the rubric to ground truth, and then measure the
  critic's agreement with me rather than its correctness. What the harness
  prints for them is evidence for a human deciding whether the rubric itself
  needs changing.

**What this can never prove.** It measures the critic on questions it did not
write, judged out of the conversational flow they occurred in, with whatever
hypotheses the caller supplies. A critic that scores well here can still band
the coach's own in-flight candidates badly. This narrows the standing
UNVERIFIED assumption A7 in `app/coach/critic.py`; it does not close it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# The five rule flags, spelled once. Every case below is checked against this
# tuple, so a flag added to `Judgment` and forgotten here fails loudly in
# `judge_critic.py` rather than being silently unmeasured.
ALL_FLAGS = (
    "is_leading",
    "is_hypothetical_or_future_tense",
    "is_pitch",
    "is_opinion_seeking",
    "is_confirmation_only",
    "is_generic",
    "is_off_topic",
)

TIER_BAD = "bad"
TIER_GOOD = "good"
TIER_CONTESTED = "contested"

# `PROJECT.md:17-19`: Hebrew/English code-switching "is the defining technical
# constraint of the product, not a localisation detail". Until 2026-08-14 this
# file, the judge, the synthetic interview and the interviewee agent were
# entirely English -- so every quality number this project had ever produced was
# blind to the harder half of its own use case, and rubric tuning driven by
# those numbers would optimise the easy language while a Hebrew regression moved
# nothing.
#
# LANG_MIXED is not a third language, it is the actual register: Hebrew grammar
# carrying English technical nouns ("עשית deploy", "ה-build נכשל"). Israeli
# engineers do not switch languages at sentence boundaries, which is precisely
# what makes it hard for a model and why a clean Hebrew set alone would not
# measure the real case.
LANG_EN = "en"
LANG_HE = "he"
LANG_MIXED = "mixed"
ALL_LANGUAGES = (LANG_EN, LANG_HE, LANG_MIXED)

# **USER DECISION, 2026-08-14: interviews and surveys start 100% English.**
# So the non-English cases are RETAINED and NOT RUN by default. Retained rather
# than deleted because the decision is "for now" -- deleting them would mean
# re-deriving the translation pairing, and re-earning the finding below, on the
# day the product goes bilingual. Not run because a measurement of a register
# the product is not currently used in would either be ignored or, worse,
# treated as a blocking failure.
#
# **What was measured before that decision, kept so it is not re-learned the
# hard way.** Against `google/gemini-2.5-flash`, English produced no dangerous
# outcome in any run, while Hebrew banded `he-good-real-problem` -- a CONTROL
# question, the direct translation of "Has that ever caused a real problem?" --
# as a `clear_violation`, which drops it before the interviewer sees it. The
# English twin was never dropped. Over-flagging in Hebrew is the dangerous
# direction: the coach goes quiet in the language the interview is being
# conducted in while the surface still reports itself healthy. Small sample and
# nondeterministic, so it is a flagged risk rather than a settled number.
#
# `PROJECT.md:17-19` still calls Hebrew/English code-switching "the defining
# technical constraint of the product". That claim and this decision cannot both
# be current; whichever way it is resolved should be resolved in that file.
DEFAULT_LANGUAGES = (LANG_EN,)


@dataclass(frozen=True)
class GoldCase:
    """One labelled question.

    `must_flag` is deliberately a SUBSET assertion, not an exact-match one. A
    leading question is often also confirmation-seeking, and demanding the
    critic produce exactly one flag would fail it for being more right than
    the label. The one exact-match assertion is on GOOD cases, where
    `must_flag` is empty and `no_other_flags` is True: there, every flag is
    constrained to False, which is the whole point of a negative control.

    `target_hypothesis_id` is set only where the case is meant to exercise
    `is_confirmation_only`, which is judged against a hypothesis's own written
    falsifier and is meaningless without one.

    `translation_of` is what makes the multilingual half an EXPERIMENT rather
    than a second opinion. A Hebrew case is not a new question with a new
    judgement attached -- it is one of the English cases above, carried across
    with the SAME labels, so language is the only variable that moved. Invented
    Hebrew questions would have needed their own provenance argument and would
    have confounded "the critic is worse in Hebrew" with "these are different
    questions". A translation that changed a label would be a different question
    wearing the same id, so the labels are asserted equal in the tests.
    """

    case_id: str
    tier: str
    text: str
    move: str
    must_flag: "frozenset[str]" = field(default_factory=frozenset)
    no_other_flags: bool = False
    target_hypothesis_id: Optional[str] = None
    note: str = ""
    language: str = LANG_EN
    translation_of: Optional[str] = None
    # What the interviewee had just said when this question was asked
    # (2026-08-15). `is_generic` asks whether a question engages with that
    # utterance, so a case with no trigger cannot exercise it -- the harness
    # judges those with an explicitly empty trigger and the prompt sets the
    # flag false rather than guessing. Defaulted to "" so the existing cases,
    # whose provenance is the script's own question list rather than a
    # question-and-answer pair, stay exactly as they were labelled.
    trigger: str = ""

    def __post_init__(self) -> None:
        unknown = set(self.must_flag) - set(ALL_FLAGS)
        if unknown:
            raise ValueError(f"{self.case_id}: unknown flag(s) {sorted(unknown)}")
        if self.tier not in (TIER_BAD, TIER_GOOD, TIER_CONTESTED):
            raise ValueError(f"{self.case_id}: unknown tier {self.tier!r}")
        if self.tier == TIER_GOOD and self.must_flag:
            raise ValueError(f"{self.case_id}: a GOOD case must assert no flags")
        if self.language not in ALL_LANGUAGES:
            raise ValueError(f"{self.case_id}: unknown language {self.language!r}")
        if self.language != LANG_EN and not self.translation_of:
            raise ValueError(
                f"{self.case_id}: a non-English case must name the English case it "
                "translates -- an untraceable one has no provenance for its labels"
            )


# --- the plan the whole set is judged against (2026-08-19) -----------------------
#
# `is_off_topic` is the one flag that cannot be judged from a question and an
# utterance alone: "off topic" is meaningless without a topic, and until this
# flag every case here was judged with no plan supplied at all.
#
# **DERIVED FROM THE SCRIPT, NOT INVENTED.** Each line below summarises a
# stretch of `claudictive_script.py`'s own interviewer turns, which is the same
# provenance the BAD cases have and is checkable by reading them. The first
# version of this constant was written from memory of what the product was
# about and was WRONG on the subject matter -- it described code review and
# repositories when the interview is about gaming and an in-game overlay. Two
# GOOD cases were then banded `clear_violation` for being off a plan they were
# never off, and one flipped band between identical runs. That is the standing
# rule about fixtures encoding what you already expect, caught by the harness
# rather than by review, and it is the reason this comment is this long.
#
# **The circularity, stated rather than hidden.** The GOOD cases are themselves
# interviewer turns from this script, so a survey derived from the script
# covers them BY CONSTRUCTION. That means the good cases can never be evidence
# that this flag distinguishes relevant from irrelevant well -- they are only a
# negative control against the flag firing on the interview's own subject
# matter. The positive evidence is `bad-off-topic` alone, which is authored to
# fall outside every line below.
#
# ONE shared survey, never per-case: a per-case plan could be written to make
# its own label come out right, which is exactly the failure the tiers exist to
# prevent.
#
# **Written as TOPICS, never as questions**, and that is not a style choice.
# The first script-derived version phrased each line as the interviewer's own
# question, which made several lines near-verbatim copies of the GOOD cases --
# they are turns from the same interview. `is_generic`'s rubric penalises "the
# question restates the survey item verbatim", so the fixture then taught the
# critic to condemn a good case for matching a line this file had copied from
# it. `test_no_survey_line_restates_a_case` pins the separation.
GOLD_SURVEY = (
    "How a play session actually starts, from the machine going on to the "
    "game running.",
    "Needing something from outside the game mid-session, and what happens "
    "to it.",
    "Alt-tabbing out mid-session, and what coming back is like.",
    "How a new tool gets installed on the gaming machine.",
    "What makes someone remove an overlay again.",
    "Problems hit mid-game that could not be solved there and then.",
    "How an AI coding assistant is used, and what happens to its answers.",
    "What is already paid for on this machine.",
)


# --- the five planted failures, verbatim from claudictive_script.py --------------
# Each carries the rule it was written to break. Extra flags are allowed; the
# named one is required.
_BAD = (
    GoldCase(
        case_id="bad-leading",
        tier=TIER_BAD,
        text="So the Overwolf install was the frustrating part, right?",
        move="deepen",
        must_flag=frozenset({"is_leading"}),
        note="Supplies the emotion and asks for assent. The interviewee's scripted "
             "reply -- 'I wouldn't have called it frustrating until you said it' -- is "
             "the damage being done, in the script, on purpose.",
    ),
    GoldCase(
        case_id="bad-hypothetical",
        tier=TIER_BAD,
        text="Would you use it every day if it could also read your whole codebase?",
        move="advance",
        must_flag=frozenset({"is_hypothetical_or_future_tense"}),
        note="Asks about a future self using a product that does not exist.",
    ),
    GoldCase(
        case_id="bad-pitch",
        tier=TIER_BAD,
        text=("The whole idea is that you never have to leave the game again, which is "
              "what makes it different from anything else out there."),
        move="advance",
        must_flag=frozenset({"is_pitch"}),
        note="Not a question at all -- a positioning statement wearing a turn.",
    ),
    GoldCase(
        case_id="bad-opinion",
        tier=TIER_BAD,
        text="Don't you think most developers would want something like this?",
        move="advance",
        must_flag=frozenset({"is_opinion_seeking"}),
        note="Asks the interviewee to speculate about a population they are not.",
    ),
    GoldCase(
        case_id="bad-confirmation",
        tier=TIER_BAD,
        text="So you'd say it's useful, then?",
        move="test_hypothesis",
        must_flag=frozenset({"is_confirmation_only"}),
        target_hypothesis_id="h4",
        note="Aimed at h4 (read-not-write) and worded so no answer could disconfirm "
             "it. h4's falsifier asks for people wanting to EDIT code in the overlay; "
             "'useful' cannot reach that either way.",
    ),
    GoldCase(
        case_id="bad-generic",
        tier=TIER_BAD,
        text="Can you tell me more about that?",
        move="deepen",
        must_flag=frozenset({"is_generic"}),
        trigger=(
            "I had the overlay open on my second monitor and I alt-tabbed out anyway, "
            "because the search box lost focus every time a match started. So I ended "
            "up just keeping a browser tab open on the other screen instead."
        ),
        note=(
            "PROVENANCE, stated plainly because it differs from every case above. "
            "The five original bad cases were lifted verbatim from the planted "
            "failures in the synthetic interview script, which is where their labels "
            "come from. This one was NOT: the script's failures were authored before "
            "`is_generic` existed, and the script itself is not tracked in the "
            "repository, so planting a question in it would produce a case whose "
            "provenance nobody else could check -- weaker, not stronger. It is "
            "therefore an authored case, and the honest claim for it is narrower: it "
            "is a clear instance by construction rather than a question a real "
            "interviewer was observed asking. The trigger names three specific things "
            "(the lost focus, the alt-tab, the browser tab on the second screen) and "
            "the question engages with none of them; it would read identically after "
            "any answer in any interview, which is the whole definition of the flag. "
            "Treat a failure here as weaker evidence than a failure on the five above."
        ),
    ),
    GoldCase(
        case_id="bad-off-topic",
        tier=TIER_BAD,
        text=(
            "You said the drive is about fifty minutes -- what does that look like on "
            "a bad traffic day, door to door?"
        ),
        move="deepen",
        must_flag=frozenset({"is_off_topic"}),
        trigger=(
            "Honestly the worst part of my week is the commute. It is about fifty "
            "minutes each way and I have been doing it for three years now, and the "
            "podcast backlog is the only thing keeping me sane in the car."
        ),
        note=(
            "PROVENANCE: AUTHORED, with the same narrower claim as `bad-generic` above "
            "and for the same reason: the script's planted failures predate this flag "
            "and inventing a sixth planted failure in an untracked file would produce "
            "a label nobody else could check. "
            "It is constructed to isolate the flag from every neighbouring one. The "
            "question is NOT generic -- it names the fifty minutes and the door-to-door "
            "trip, so it engages with this utterance and could not be asked of any "
            "other interviewee. It is not leading, not hypothetical (it asks about a "
            "recurring past reality), not a pitch, not opinion-seeking, and targets no "
            "hypothesis. By the letter of every other rule it is a GOOD question, and "
            "that is exactly the point: it is a well-formed, engaged, past-tense "
            "question about the interviewee's commute, and no answer to it -- fast day "
            "or slow day -- could change anything on GOLD_SURVEY above. If the critic "
            "cannot flag this one, the flag is not doing the job it was added for, "
            "because a well-formed question is the only case where it matters."
        ),
    ),
)

# --- negative controls: competent interviewer turns from the same script ---------
# Every flag must be False and the band must not be clear_violation. These are
# what make over-flagging visible.
_GOOD = (
    GoldCase(
        case_id="good-last-time",
        tier=TIER_GOOD,
        text=("Tell me about the last time you needed something from outside the game "
              "while you were playing."),
        move="deepen",
        no_other_flags=True,
        note="The canonical Mom Test move: a specific past episode.",
    ),
    GoldCase(
        case_id="good-walkthrough",
        tier=TIER_GOOD,
        text=("Walk me through installing a new tool on your gaming machine the last "
              "time you did it."),
        move="fill_gap",
        no_other_flags=True,
    ),
    GoldCase(
        case_id="good-what-next",
        tier=TIER_GOOD,
        text=("Think of a problem you hit mid-game that you could not solve there and "
              "then. What did you actually do next?"),
        move="deepen",
        no_other_flags=True,
        note="Past behaviour, not preference. 'Actually' is doing real work.",
    ),
    GoldCase(
        case_id="good-distinction",
        tier=TIER_GOOD,
        text="What is different about coming back, between those two?",
        move="deepen",
        no_other_flags=True,
    ),
    GoldCase(
        case_id="good-current-spend",
        tier=TIER_GOOD,
        text="What do you currently pay for that sits on this machine?",
        move="fill_gap",
        no_other_flags=True,
        note="The script's own repair of a bad pricing question -- present-tense "
             "behaviour instead of stated willingness to pay.",
    ),
    GoldCase(
        case_id="good-real-problem",
        tier=TIER_GOOD,
        text="Has that ever caused a real problem?",
        move="deepen",
        no_other_flags=True,
        note="Short and closed-ended, which is not the same as leading. Included "
             "deliberately: a critic that flags brevity fails here.",
    ),
    GoldCase(
        case_id="good-uninstalled-before",
        tier=TIER_GOOD,
        text="Have you uninstalled overlays for that before?",
        move="test_hypothesis",
        no_other_flags=True,
        target_hypothesis_id="h3",
        note="Carries a hypothesis target AND a falsifiable answer -- the case that "
             "separates 'targets a hypothesis' from 'confirmation-only'. A critic "
             "that flags every hypothesis-aimed question fails here and nowhere else.",
    ),
    GoldCase(
        case_id="good-magic-wand-followup",
        tier=TIER_GOOD,
        text="You said you keep a note on your phone and use it one time in five. What happens the other four times?",
        move="deepen",
        no_other_flags=True,
        note="Built from the interviewee's own numbers, which is what makes it "
             "specific rather than leading.",
    ),
)

# --- contested: the rubric's letter against good practice ------------------------
# REPORTED, NEVER SCORED. See the module docstring.
_CONTESTED = (
    GoldCase(
        case_id="contested-magic-wand",
        tier=TIER_CONTESTED,
        text="If I gave you a magic wand for that specific problem, what does it do?",
        move="advance",
        note="Literally hypothetical, and a well-regarded discovery move. If the "
             "critic bands this clear_violation the coach can never propose it -- "
             "that may be correct, and it is a decision about the RUBRIC, not a bug.",
    ),
    GoldCase(
        case_id="contested-would-you-type",
        tier=TIER_CONTESTED,
        text="Would you type code during a game?",
        move="test_hypothesis",
        target_hypothesis_id="h4",
        note="Future-tense by construction, but it is the most direct probe at h4's "
             "falsifier that exists. If the rubric forbids it, h4 may be untestable "
             "by interview -- worth knowing either way.",
    ),
    GoldCase(
        case_id="contested-in-front-of-you",
        tier=TIER_CONTESTED,
        text="If it were in front of you, in the game, what would you actually do with it?",
        move="advance",
        note="Hypothetical framing over a concrete, already-described situation.",
    ),
    GoldCase(
        case_id="contested-enough",
        tier=TIER_CONTESTED,
        text="Just seeing it would have been enough?",
        move="deepen",
        note="Reads as leading, but it is checking back an inference the interviewee "
             "just volunteered -- a legitimate summarise-and-verify move.",
    ),
)

# --- the same questions in Hebrew ------------------------------------------------
# Paired translations, labels carried across unchanged. Product names stay in
# Latin script because that is how they are written and said in Israeli tech
# speech -- "Overwolf" transliterated into Hebrew characters would be a question
# nobody would ask, which would confound the measurement with unnaturalness.
_HEBREW = (
    GoldCase(
        case_id="he-bad-leading",
        tier=TIER_BAD,
        text="אז ההתקנה של Overwolf הייתה החלק המתסכל, נכון?",
        move="deepen",
        must_flag=frozenset({"is_leading"}),
        language=LANG_HE,
        translation_of="bad-leading",
    ),
    GoldCase(
        case_id="he-bad-hypothetical",
        tier=TIER_BAD,
        text="היית משתמש בזה כל יום אם זה היה יכול גם לקרוא את כל הקוד שלך?",
        move="advance",
        must_flag=frozenset({"is_hypothetical_or_future_tense"}),
        language=LANG_HE,
        translation_of="bad-hypothetical",
    ),
    GoldCase(
        case_id="he-bad-pitch",
        tier=TIER_BAD,
        text=("כל הרעיון הוא שאתה כבר לא צריך לצאת מהמשחק, וזה מה שעושה את זה שונה "
              "מכל דבר אחר שיש בשוק."),
        move="advance",
        must_flag=frozenset({"is_pitch"}),
        language=LANG_HE,
        translation_of="bad-pitch",
    ),
    GoldCase(
        case_id="he-bad-opinion",
        tier=TIER_BAD,
        text="אתה לא חושב שרוב המפתחים היו רוצים משהו כזה?",
        move="advance",
        must_flag=frozenset({"is_opinion_seeking"}),
        language=LANG_HE,
        translation_of="bad-opinion",
    ),
    GoldCase(
        case_id="he-bad-confirmation",
        tier=TIER_BAD,
        text="אז היית אומר שזה שימושי, נכון?",
        move="test_hypothesis",
        must_flag=frozenset({"is_confirmation_only"}),
        target_hypothesis_id="h4",
        language=LANG_HE,
        translation_of="bad-confirmation",
    ),
    GoldCase(
        case_id="he-good-last-time",
        tier=TIER_GOOD,
        text="ספר לי על הפעם האחרונה שהיית צריך משהו מחוץ למשחק בזמן שאתה משחק.",
        move="deepen",
        no_other_flags=True,
        language=LANG_HE,
        translation_of="good-last-time",
    ),
    GoldCase(
        case_id="he-good-walkthrough",
        tier=TIER_GOOD,
        text="קח אותי צעד צעד דרך הפעם האחרונה שהתקנת כלי חדש על מכונת הגיימינג שלך.",
        move="fill_gap",
        no_other_flags=True,
        language=LANG_HE,
        translation_of="good-walkthrough",
    ),
    GoldCase(
        case_id="he-good-what-next",
        tier=TIER_GOOD,
        text=("תחשוב על בעיה שנתקלת בה באמצע משחק ולא הצלחת לפתור אותה שם ואז. "
              "מה עשית בפועל אחרי זה?"),
        move="deepen",
        no_other_flags=True,
        language=LANG_HE,
        translation_of="good-what-next",
    ),
    GoldCase(
        case_id="he-good-current-spend",
        tier=TIER_GOOD,
        text="על מה אתה משלם היום שיושב על המכונה הזאת?",
        move="fill_gap",
        no_other_flags=True,
        language=LANG_HE,
        translation_of="good-current-spend",
    ),
    GoldCase(
        case_id="he-good-real-problem",
        tier=TIER_GOOD,
        text="זה אי פעם גרם לך לבעיה אמיתית?",
        move="deepen",
        no_other_flags=True,
        language=LANG_HE,
        translation_of="good-real-problem",
    ),
)

# --- and code-switched, which is the register the product actually meets ---------
# Hebrew grammar carrying English technical nouns, switching mid-sentence rather
# than at a boundary. This is the shape PROJECT.md calls the defining constraint,
# and it is the one a model is most likely to mishandle -- the rubric it is
# judged against is written in English, so the model must apply an English rule
# to a sentence that is only partly English.
_MIXED = (
    GoldCase(
        case_id="mix-bad-leading",
        tier=TIER_BAD,
        text="אז ה-install של Overwolf היה the frustrating part, נכון?",
        move="deepen",
        must_flag=frozenset({"is_leading"}),
        language=LANG_MIXED,
        translation_of="bad-leading",
    ),
    GoldCase(
        case_id="mix-bad-hypothetical",
        tier=TIER_BAD,
        text="היית עושה integrate לזה ב-workflow שלך אם היה לזה API מלא?",
        move="advance",
        must_flag=frozenset({"is_hypothetical_or_future_tense"}),
        language=LANG_MIXED,
        translation_of="bad-hypothetical",
    ),
    GoldCase(
        case_id="mix-bad-opinion",
        tier=TIER_BAD,
        text="אתה לא חושב שרוב ה-developers היו רוצים feature כזה?",
        move="advance",
        must_flag=frozenset({"is_opinion_seeking"}),
        language=LANG_MIXED,
        translation_of="bad-opinion",
    ),
    GoldCase(
        case_id="mix-good-what-next",
        tier=TIER_GOOD,
        text="מה עשית בפועל אחרי שה-build נכשל?",
        move="deepen",
        no_other_flags=True,
        language=LANG_MIXED,
        translation_of="good-what-next",
    ),
    GoldCase(
        case_id="mix-good-last-time",
        tier=TIER_GOOD,
        text="ספר לי על הפעם האחרונה שעשית alt-tab באמצע משחק.",
        move="deepen",
        no_other_flags=True,
        language=LANG_MIXED,
        translation_of="good-last-time",
    ),
)

GOLD_CASES: "tuple[GoldCase, ...]" = _BAD + _GOOD + _CONTESTED + _HEBREW + _MIXED


def cases_by_language(language: str) -> "tuple[GoldCase, ...]":
    return tuple(case for case in GOLD_CASES if case.language == language)


def cases_by_tier(tier: str) -> "tuple[GoldCase, ...]":
    return tuple(case for case in GOLD_CASES if case.tier == tier)


def scored_cases() -> "tuple[GoldCase, ...]":
    """Everything with an assertable label. Contested cases are excluded here
    and reported separately, which is the whole reason the tier exists."""
    return tuple(case for case in GOLD_CASES if case.tier in (TIER_BAD, TIER_GOOD))
