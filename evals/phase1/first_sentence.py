"""Re-score round 2 keeping only each rewrite's first sentence, the fairest case for Kumru. Runs no model.

Lists each distinct approved rewrite that differs from its source beyond punctuation and spacing, for human review.
Usage: python3 evals/phase1/first_sentence.py
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "prototype"))
from checker import check, lower_tr  # noqa: E402


def first_sentence(text):
    found = re.match(r".+?[.!?](?=\s|$)", text)
    return found.group(0) if found else text


def plain(text):
    return re.sub(r"[^\w]", "", lower_tr(text))


total = approved = same = 0
changed = {}
with open(os.path.join(ROOT, "evals", "results", "phase1-round2.jsonl"), encoding="utf-8") as f:
    for row in map(json.loads, f):
        for line in row["output"].splitlines():
            sid, text = re.match(r"\d+\. \[(S\d+)\] (.*)", line).groups()
            text = first_sentence(text)
            source = row["source"][sid]
            total += 1
            if check(f"1. [{sid}] {text}", {sid: source})[0]:
                approved += 1
                if plain(text) == plain(source):
                    same += 1
                else:
                    changed[(row["passage"], sid, text)] = source

print(f"{total} first sentences; {approved} approved by the checker; {same} of those only copy the source "
      f"(ignoring punctuation and spacing); {len(changed)} distinct approved rewrites actually differ:")
for (passage, sid, text), source in changed.items():
    print(f"\n{passage}/{sid}\n  source: {source}\n  output: {text}")
