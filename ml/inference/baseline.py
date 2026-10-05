"""Rule-based baseline rewriter (CPU-only, deterministic, no model download).

It makes conservative, meaning-preserving edits that address common mechanical
writing patterns: stock filler phrases, inflated vocabulary, stacked
transitions, register mismatches (contractions) and overlong sentences.
Quoted passages, URLs, e-mails and citations are protected from edits.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ml.preprocessing.normalize_text import split_paragraphs, tidy_spacing, words

Replacement = str | list[str] | Callable[[re.Match], str]

_SENTENCE_START = re.compile(r"(?:^|[.!?][\"')\]”’]?\s+|\n\s*)$")
_CAP_MARK = "\x01"


@dataclass
class Rule:
    pattern: re.Pattern
    replacement: Replacement
    category: str


def _r(pattern: str, replacement: Replacement, category: str, flags: int = re.IGNORECASE) -> Rule:
    return Rule(re.compile(pattern, flags), replacement, category)


def _verb_forms(base: str, forms: dict[str, str]) -> Callable[[re.Match], str]:
    def repl(m: re.Match) -> str:
        return forms.get(m.group(1).lower(), base)

    return repl


# --- Filler / stock phrases (applied at every intensity) ------------------------------
FILLER_RULES: list[Rule] = [
    _r(r"\bit is (?:important|worth|crucial|essential|interesting) to (?:note|mention|highlight|emphasi[sz]e|remember|recogni[sz]e) that\s+", "", "filler"),
    _r(r"\bit (?:should|must) be (?:noted|mentioned|emphasi[sz]ed) that\s+", "", "filler"),
    _r(r"\bit goes without saying that\s+", "", "filler"),
    _r(r"\bneedless to say,\s*", "", "filler"),
    _r(r"\bin today's (?:fast-paced|modern|digital|ever-changing|rapidly (?:evolving|changing)|interconnected) (?:business |technological |digital )?(?:world|landscape|era|age|society|environment),\s*", "today, ", "filler"),
    _r(r"\b(?:it is|it's) (?:essential|crucial|important) to (?:recogni[sz]e|understand) (?:that )?", "", "filler"),
    _r(r"\bthe (?:intricacies|complexities) of\b", "the details of", "inflated"),
    _r(r"\bin order to\b", "to", "wordiness"),
    _r(r"\bdue to the fact that\b", "because", "wordiness"),
    _r(r"\bowing to the fact that\b", "because", "wordiness"),
    _r(r"\bat this (?:point|moment) in time\b", "now", "wordiness"),
    _r(r"\bin the event that\b", "if", "wordiness"),
    _r(r"\bfor the purpose of\b", "for", "wordiness"),
    _r(r"\bhas the (?:ability|capacity) to\b", "can", "wordiness"),
    _r(r"\bhave the (?:ability|capacity) to\b", "can", "wordiness"),
    _r(r"\b(?:a )?(?:plethora|myriad|multitude) of\b", "many", "inflated"),
    _r(r"\ba (?:wide|vast|diverse|broad) (?:range|array|variety|spectrum) of\b", "many", "inflated"),
    _r(r"\ba significant number of\b", "many", "inflated"),
    _r(r"\bthe vast majority of\b", "most", "inflated"),
    _r(r"\bprior to\b", "before", "wordiness"),
    _r(r"\bsubsequent to\b", "after", "wordiness"),
    _r(r"\bin close proximity to\b", "near", "wordiness"),
    _r(r"\b(?:with|in) regards? to\b", "about", "wordiness"),
    _r(r"\beach and every\b", "every", "wordiness"),
    _r(r"\bfirst and foremost\b", "first", "wordiness"),
    _r(r"\bof (?:paramount|utmost) importance\b", "very important", "inflated"),
    _r(r"\butili[sz](e|es|ed|ing)\b", _verb_forms("use", {"e": "use", "es": "uses", "ed": "used", "ing": "using"}), "inflated"),
    _r(r"\butili[sz]ation\b", "use", "inflated"),
    _r(r"\bleverag(e|es|ed|ing)(?= (?:the|a|an|our|their|this|these|its|his|her|existing|data|technology|AI|tools?|many|multiple|various|several)\b)",
       _verb_forms("use", {"e": "use", "es": "uses", "ed": "used", "ing": "using"}), "inflated"),
    _r(r"\bdelv(e|es|ed|ing) (?:deeper |deeply )?into\b",
       _verb_forms("explore", {"e": "explore", "es": "explores", "ed": "explored", "ing": "exploring"}), "inflated"),
    _r(r"\bnavigat(e|es|ed|ing) the (?:complexities|intricacies) of\b",
       _verb_forms("deal with", {"e": "deal with", "es": "deals with", "ed": "dealt with", "ing": "dealing with"}), "inflated"),
    _r(r"\b(play|plays|played|playing) an? (?:crucial|pivotal|vital|key|significant|critical|essential|instrumental) role in\b",
       lambda m: f"{m.group(1)} a big part in", "inflated_informal"),
    _r(r"\bit is (?:evident|clear|obvious) that\s+", "", "filler"),
    _r(r"\ba testament to\b", "proof of", "inflated"),
    _r(r"\bseamless(ly)?\b", lambda m: "smooth" + (m.group(1) or ""), "inflated"),
    _r(r"\bembark(s|ed|ing)? on\b", lambda m: {"s": "starts", "ed": "started", "ing": "starting"}.get((m.group(1) or "").lower(), "start"), "inflated"),
    _r(r"\bfoster(s|ed|ing)?\b", lambda m: "encourage" + {"s": "s", "ed": "d", "ing": ""}.get((m.group(1) or "").lower(), "")
       if (m.group(1) or "").lower() != "ing" else "encouraging", "inflated"),
    _r(r"\bin the realm of\b", "in", "inflated"),
    _r(r"(?:\b(an?) )?\bever-evolving\b", lambda m: ("a " if m.group(1) else "") + "changing", "inflated"),
    _r(r"\bpivotal\b", "key", "inflated"),
    _r(r"\bcommenc(e|es|ed|ing)\b", _verb_forms("start", {"e": "start", "es": "starts", "ed": "started", "ing": "starting"}), "inflated"),
]

# --- Sentence-initial transitions (medium/strong) ------------------------------------
_TRANSITION_PATTERN = re.compile(
    r"(?:(?<=^)|(?<=[.!?]\s)|(?<=[.!?][\"')”’]\s)|(?<=\n))"
    r"(Furthermore|Moreover|Additionally|In addition|Consequently|Thus|Hence|"
    r"In conclusion|In summary|To sum up|Ultimately|Overall|Notably|Importantly|Indeed),\s+"
)
TRANSITIONS = {
    "add": {
        "natural": ["Also,", "What's more,", "On top of that,", ""],
        "casual": ["Plus,", "Also,", "On top of that,", ""],
        "professional": ["In addition,", "Beyond this,", "Also,"],
        "academic": ["In addition,", "Further,", "Beyond this,"],
        "simple": ["Also,", ""],
        "concise": [""],
        "detailed": ["In addition,", "Beyond that,", "On top of this,"],
    },
    "result": {
        "natural": ["So", "As a result,", "That means"],
        "casual": ["So", "Because of that,"],
        "professional": ["As a result,", "Therefore,"],
        "academic": ["Consequently,", "As a result,", "Therefore,"],
        "simple": ["So", "Because of this,"],
        "concise": ["So"],
        "detailed": ["As a result,", "Because of this,"],
    },
    "summary": {
        "natural": ["Overall,", "All in all,", "In short,"],
        "casual": ["So,", "All in all,", "Bottom line:"],
        "professional": ["Overall,", "In short,", "In summary,"],
        "academic": ["In summary,", "Overall,", "Taken together,"],
        "simple": ["In short,", "Overall,"],
        "concise": ["In short,", ""],
        "detailed": ["Overall,", "To summarize,", "Taken together,"],
    },
    "emphasis": {
        "natural": [""], "casual": [""], "professional": ["Notably,", ""], "academic": ["Notably,", "Importantly,"],
        "simple": [""], "concise": [""], "detailed": ["Notably,", "It is worth adding that"],
    },
}
_TRANSITION_GROUP = {
    "furthermore": "add", "moreover": "add", "additionally": "add", "in addition": "add",
    "consequently": "result", "thus": "result", "hence": "result",
    "in conclusion": "summary", "in summary": "summary", "to sum up": "summary", "ultimately": "summary",
    "overall": "summary", "notably": "emphasis", "importantly": "emphasis", "indeed": "emphasis",
}

# --- Contractions --------------------------------------------------------------------
_CONTRACTIONS = [
    ("do not", "don't"), ("does not", "doesn't"), ("did not", "didn't"), ("is not", "isn't"),
    ("are not", "aren't"), ("was not", "wasn't"), ("were not", "weren't"), ("cannot", "can't"),
    ("can not", "can't"), ("will not", "won't"), ("would not", "wouldn't"), ("should not", "shouldn't"),
    ("could not", "couldn't"), ("have not", "haven't"), ("has not", "hasn't"), ("had not", "hadn't"),
    ("it is", "it's"), ("that is", "that's"), ("there is", "there's"), ("what is", "what's"),
    ("they are", "they're"), ("we are", "we're"), ("you are", "you're"), ("I am", "I'm"),
    ("let us", "let's"), ("we have", "we've"), ("they have", "they've"), ("you will", "you'll"),
    ("we will", "we'll"), ("they will", "they'll"), ("it will", "it'll"),
]
# Only contract when another word follows ("as it is." must stay uncontracted).
_CONTRACT_RULES = [
    Rule(re.compile(rf"\b{re.escape(full)}\b(?= [A-Za-z])", re.IGNORECASE), short, "contraction")
    for full, short in _CONTRACTIONS
]
_EXPAND_RULES = [
    Rule(re.compile(rf"\b{re.escape(short)}\b", re.IGNORECASE), full, "formal_register")
    for full, short in _CONTRACTIONS
    if full not in {"can not", "let us"}
] + [Rule(re.compile(r"\bcan['’]t\b", re.I), "cannot", "formal_register"),
     Rule(re.compile(r"\bwon’t\b", re.I), "will not", "formal_register"),
     Rule(re.compile(r"\b(\w+)n’t\b", re.I), lambda m: f"{m.group(1)} not", "formal_register")]

# --- Plain-language vocabulary (simple / casual / strong natural) ---------------------
SIMPLE_RULES: list[Rule] = [
    _r(r"\bapproximately\b", "about", "plain_words"),
    _r(r"\bdemonstrat(e|es|ed|ing)\b", _verb_forms("show", {"e": "show", "es": "shows", "ed": "showed", "ing": "showing"}), "plain_words"),
    _r(r"\bnumerous\b", "many", "plain_words"),
    _r(r"\bindividuals\b", "people", "plain_words"),
    _r(r"\bpurchas(e|es|ed|ing)\b", _verb_forms("buy", {"e": "buy", "es": "buys", "ed": "bought", "ing": "buying"}), "plain_words"),
    _r(r"\bassist(s|ed|ing)?\b(?! (?:professor|director|manager))",
       lambda m: "help" + {"s": "s", "ed": "ed", "ing": "ing"}.get((m.group(1) or "").lower(), ""), "plain_words"),
    _r(r"\bobtain(s|ed|ing)?\b", lambda m: {"s": "gets", "ed": "got", "ing": "getting"}.get((m.group(1) or "").lower(), "get"), "plain_words"),
    _r(r"\bsufficient\b", "enough", "plain_words"),
    _r(r"\bcomprehend(s|ed|ing)?\b", lambda m: "understand" + {"s": "s", "ed": "", "ing": "ing"}.get((m.group(1) or "").lower(), "")
       if (m.group(1) or "").lower() != "ed" else "understood", "plain_words"),
    _r(r"\bfacilitat(e|es|ed|ing)\b", _verb_forms("help", {"e": "help", "es": "helps", "ed": "helped", "ing": "helping"}), "plain_words"),
    _r(r"\bregarding\b", "about", "plain_words"),
    _r(r"\bsubsequently\b", "later", "plain_words"),
    _r(r"\bendeavou?r(s)? to\b", lambda m: "tries to" if m.group(1) else "try to", "plain_words"),
    _r(r"\bterminat(e|es|ed|ing)\b", _verb_forms("end", {"e": "end", "es": "ends", "ed": "ended", "ing": "ending"}), "plain_words"),
    _r(r"\bin addition to\b", "besides", "plain_words"),
]

# --- Concise: remove empty intensifiers and hedges ------------------------------------
CONCISE_RULES: list[Rule] = [
    _r(r"\b(?:very|really|truly|basically|actually|essentially|literally|quite|rather|somewhat) (?=[a-z])", "", "concise"),
    _r(r"\bin fact,\s*", "", "concise"),
    _r(r"\b(?:as a matter of fact|at the end of the day|all things considered),\s*", "", "concise"),
    _r(r"\bthe fact that\b", "that", "concise"),
    _r(r"\bis able to\b", "can", "concise"),
    _r(r"\bare able to\b", "can", "concise"),
    _r(r"\bin a timely manner\b", "promptly", "concise"),
    _r(r"\bon a (?:daily|regular) basis\b", lambda m: "daily" if "daily" in m.group(0).lower() else "regularly", "concise"),
]

# --- Formal register (professional / academic) ----------------------------------------
FORMAL_RULES: list[Rule] = [
    _r(r"(?:^|(?<=[.!?]\s))So,?\s+(?!(?:far|much|many|that|long|too|to)\b)(?=[a-z])", "therefore, ", "formal_register", flags=re.M),
    _r(r"(?:^|(?<=[.!?]\s))But\s+(?=[a-z])", "however, ", "formal_register", flags=re.M),
    _r(r"(?:^|(?<=[.!?]\s))Also,?\s+(?=[a-z])", "in addition, ", "formal_register", flags=re.M),
    _r(r"\bpretty much\b", "largely", "formal_register"),
    _r(r"\b(?:kind|sort) of (?=[a-z]+(?:ed|ing|ly|ive|ous|al|ful|able|ible)\b)", "somewhat ", "formal_register"),
    _r(r"\b(?:lots of|a lot of)\b", "much", "formal_register"),
    _r(r"\bbasically,?\s*", "", "formal_register"),
]

_FORMAL_ROLE_RULE = _r(
    r"\b(play|plays|played|playing) an? (?:crucial|pivotal|vital|key|significant|critical|essential|instrumental) role in\b",
    lambda m: f"{m.group(1)} an important role in", "inflated",
)

_LOTS_COUNTABLE = re.compile(r"\bmuch ([a-z]+s)\b")


class BaselineRewriter:
    """Deterministic rule-based rewriter used as a CPU fallback and a quality baseline."""

    name = "baseline-rules-v1"

    def rewrite(self, text: str, style: str = "natural", intensity: str = "medium") -> tuple[str, list[str]]:
        """Return (rewritten_text, applied_change_categories)."""
        style = style.lower()
        intensity = intensity.lower()
        seed = int(hashlib.sha256(f"{style}|{intensity}|{text}".encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        protected, text = _protect(text)
        applied: set[str] = set()

        level = {"light": 0, "medium": 1, "strong": 2}.get(intensity, 1)
        formal = style in {"professional", "academic"}
        # Formal styles get a register-appropriate variant of informal replacements.
        rules: list[Rule] = [
            _FORMAL_ROLE_RULE if (formal and r.category == "inflated_informal") else r for r in FILLER_RULES
        ]
        if formal and level >= 1:
            rules += _EXPAND_RULES + FORMAL_RULES
        if style == "casual" or (style in {"natural", "simple"} and level >= 1):
            rules += _CONTRACT_RULES
        if style in {"simple", "casual"} and level >= 1 or (style == "natural" and level == 2):
            rules += SIMPLE_RULES
        if style == "concise":
            rules += CONCISE_RULES + SIMPLE_RULES[:4]

        for rule in rules:
            text, changed = _apply(rule, text)
            if changed:
                applied.add(rule.category)

        if level >= 1:
            text, changed = _rewrite_transitions(text, style, rng, drop_every=2 if level == 2 else 3)
            if changed:
                applied.add("transitions")
        if level >= 1 and style != "detailed":
            limit = 26 if style in {"simple", "concise"} else (30 if level == 2 else 36)
            text, changed = _split_long_sentences(text, limit)
            if changed:
                applied.add("sentence_split")

        # "much people" -> "many people" after the formal "a lot of" rule.
        text = _LOTS_COUNTABLE.sub(lambda m: f"many {m.group(1)}", text)
        text = _finalize(text)
        text = _restore(text, protected)
        return text, sorted(applied)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_PROTECT = re.compile(
    r"\b(?:https?://|www\.)\S+"  # URLs
    r"|\b[\w.+-]+@[\w-]+\.[\w.-]+\b"  # e-mails
    r"|\[\d+(?:[,–-]\s*\d+)*\]"  # numeric citations
    r"|\([A-Z][^()]{0,80}?\d{4}[a-z]?\)"  # author-year citations
    r"|\"[^\"\n]{1,300}\"|“[^”\n]{1,300}”"  # direct quotations
    r"|`[^`\n]+`"  # inline code
)


def _protect(text: str) -> tuple[list[str], str]:
    spans: list[str] = []

    def repl(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"\x02{len(spans) - 1}\x03"

    return spans, _PROTECT.sub(repl, text)


def _restore(text: str, spans: list[str]) -> str:
    return re.sub(r"\x02(\d+)\x03", lambda m: spans[int(m.group(1))], text)


def _at_sentence_start(text: str, pos: int) -> bool:
    return bool(_SENTENCE_START.search(text[:pos]))


def _match_case(src: str, repl: str) -> str:
    if repl and src[:1].isupper() and not repl[:1].isupper():
        return repl[0].upper() + repl[1:]
    return repl


def _apply(rule: Rule, text: str) -> tuple[str, bool]:
    changed = False

    def repl(m: re.Match) -> str:
        nonlocal changed
        if isinstance(rule.replacement, str):
            new = rule.replacement
        elif callable(rule.replacement):
            new = rule.replacement(m)
        else:
            new = rule.replacement[0]
        new = _match_case(m.group(0), new)
        if _at_sentence_start(m.string, m.start()):
            new = _CAP_MARK if not new else new[0].upper() + new[1:]
        if new != m.group(0):
            changed = True
        return new

    return rule.pattern.sub(repl, text), changed


def _rewrite_transitions(text: str, style: str, rng: random.Random, drop_every: int) -> tuple[str, bool]:
    counter = {"n": 0}
    used: set[str] = set()
    changed = False

    def repl(m: re.Match) -> str:
        nonlocal changed
        group = _TRANSITION_GROUP[m.group(1).lower()]
        options = TRANSITIONS[group].get(style, TRANSITIONS[group]["natural"])
        counter["n"] += 1
        # Stacked transitions are a hallmark of mechanical prose: drop some entirely.
        if group in {"add", "emphasis"} and counter["n"] % drop_every == 0:
            choice = ""
        else:
            # Never reuse a transition already used in this text ("Also, ... Also, ...").
            fresh = [o for o in options if o and o not in used] or [""]
            choice = fresh[rng.randrange(len(fresh))]
            used.add(choice)
        changed = True
        return choice + " " if choice else _CAP_MARK

    return _TRANSITION_PATTERN.sub(repl, text), changed


_SPLIT_POINTS = [
    re.compile(r";\s+(?=[a-zA-Z])"),
    re.compile(r",\s+(?:which|and this)\s+(?:means|meant|shows|showed|suggests|suggested)\s+that\s+"),
    re.compile(r",\s+whereas\s+(?=[a-z])"),
    re.compile(r",\s+(?:and|but)\s+(?=(?:it|this|they|we|he|she|these|those|the|there|many|most|some)\b)"),
]
_SPLIT_LEADS = {1: "This means that ", 2: "By contrast, "}


def _split_long_sentences(text: str, limit: int) -> tuple[str, bool]:
    changed = False
    out_paragraphs = []
    for para in split_paragraphs(text) or [text]:
        parts = re.split(r"(?<=[.!?])\s+", para)
        new_parts = []
        for sent in parts:
            if len(words(sent)) <= limit:
                new_parts.append(sent)
                continue
            for idx, pattern in enumerate(_SPLIT_POINTS):
                m = pattern.search(sent)
                if not m:
                    continue
                left, right = sent[: m.start()].rstrip(), sent[m.end():].lstrip()
                if len(words(left)) < 7 or len(words(right)) < 6:
                    continue
                lead = _SPLIT_LEADS.get(idx, "")
                if idx == 3 and m.group(0).strip(" ,").split()[0].lower() == "but":
                    lead = "But "
                right = lead + right if lead else right[0].upper() + right[1:]
                new_parts.append(left.rstrip(",;:") + ". " + right)
                changed = True
                break
            else:
                new_parts.append(sent)
        out_paragraphs.append(" ".join(new_parts))
    return "\n\n".join(out_paragraphs), changed


def _finalize(text: str) -> str:
    # Capitalise the letter following removed sentence-initial phrases.
    text = re.sub(r"\x01\s*([\"'(\[“]?)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    text = text.replace(_CAP_MARK, "")
    # "an utilization" -> "an use" -> "a use"
    text = re.sub(r"\b([Aa])n (?=use\b)", r"\1 ", text)
    paragraphs = [tidy_spacing(p) for p in split_paragraphs(text)]
    return "\n\n".join(p for p in paragraphs if p)
