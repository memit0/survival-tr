"""Check that simplified steps stay faithful to the passage they cite.

Phase 1 version: enough to test Kumru. It can reject bad output, but passing does not prove an answer is right,
so a person still reviews the results. Phase 4 replaces the rough word and negation checks.
"""
import re

STEP = re.compile(r"^\d+\.\s*\[(S\d+)\]\s*(.+)$")
NUMBER_WORDS = {"sıfır": "0", "iki": "2", "üç": "3", "dört": "4", "beş": "5", "altı": "6", "yedi": "7", "sekiz": "8",
                "dokuz": "9", "on": "10", "yirmi": "20", "otuz": "30", "kırk": "40", "elli": "50", "yarım": "0.5"}
# ponytail: rough Turkish negation patterns; "-ma/-me" also ends some ordinary words, so Phase 4 must improve this.
NEGATION = re.compile(r"\b(asla|değil\w*|yok|olmaz|sakın|hiçbir|hiçbir zaman|kesinlikle)\b"
                      r"|\w{2,}(mayın|meyin|mayınız|meyiniz|mamalı\w*|memeli\w*|maz|mez|madan|meden)\b")
MIN_OVERLAP = 0.5  # share of a step's main words that must come from the cited sentence


def lower_tr(text):
    return text.replace("İ", "i").replace("I", "ı").lower()


def numbers(text):
    """Every number in the text, so "dört saat" and "4 saat" count as the same."""
    text = lower_tr(text).replace("½", "0.5")
    found = {n.replace(",", ".").lstrip("0") or "0" for n in re.findall(r"\d+(?:[.,]\d+)?", text)}
    return found | {NUMBER_WORDS[w] for w in re.findall(r"\w+", text) if w in NUMBER_WORDS}


def has_negation(text):
    return bool(NEGATION.search(lower_tr(text)))


def stems(text):
    """Main words, cut to 4 letters so Turkish suffixes don't hide a match."""
    return {w[:4] for w in re.findall(r"\w+", lower_tr(text)) if len(w) > 3}


def parse_steps(output):
    """[(sentence id, text), ...], or None if any non-empty line isn't a cited step."""
    steps = []
    for line in output.replace("*", "").strip().splitlines():  # bold marks are formatting, not meaning
        if line.strip():
            found = STEP.match(line.strip())
            if not found:
                return None
            steps.append(found.groups())
    return steps or None


def check(output, sentences, required=()):
    """Returns (passed, reasons). sentences: {"S1": text}. required: ids of warnings the steps must cite."""
    steps = parse_steps(output)
    if steps is None:
        return False, ["format: not all lines are cited steps"]
    reasons = []
    seen = set()
    for sid, text in steps:
        source = sentences.get(sid)
        if sid in seen:
            reasons.append(f"{sid}: cited twice")
        seen.add(sid)
        if re.search(r"[()_]", text):
            reasons.append(f"{sid}: filler marks")
        # "Herkesi, tek, bir, odada" keeps the source's words but isn't a usable sentence.
        if text.count(",") > source.count("," ) + 1 if source else False:
            reasons.append(f"{sid}: broken into word fragments")
        if source is None:
            reasons.append(f"{sid}: cited sentence doesn't exist")
            continue
        added = numbers(text) - numbers(source)
        if added:
            reasons.append(f"{sid}: number not in source: {sorted(added)}")
        missing = numbers(source) - numbers(text)
        if missing:
            reasons.append(f"{sid}: number dropped: {sorted(missing)}")
        if has_negation(source) != has_negation(text):
            reasons.append(f"{sid}: negation changed")
        words = stems(text)
        if words and len(words & stems(source)) / len(words) < MIN_OVERLAP:
            reasons.append(f"{sid}: wording too far from source")
    cited = {sid for sid, _ in steps}
    reasons += [f"{sid}: required warning missing" for sid in required if sid not in cited]
    return not reasons, reasons
