"""Run every test question and diagnosis case through answer() with real local models.

Needs EmbeddingGemma on port 8098 and an LLM on port 8099.
Usage: python3 evals/phase2/answer_eval.py --name gemma-4-e2b [--limit N]
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "prototype"))
from answer import Request, answer  # noqa: E402
from dose import load_medicines  # noqa: E402
from llm import EMBED_MODELS, document_embedder  # noqa: E402
from pack import build, load_items  # noqa: E402
from search import normalize  # noqa: E402
from server import real_models  # noqa: E402


def load(name):
    with open(os.path.join(ROOT, "evals", "phase2", name), encoding="utf-8") as f:
        return json.load(f)


def report(rows):
    questions = [r for r in rows if r["set"] == "search"]
    cases = [r for r in rows if r["set"] == "diagnosis"]
    kinds = {}
    for row in rows:
        kinds[row["result"]["kind"]] = kinds.get(row["result"]["kind"], 0) + 1
    seconds = [r["seconds"] for r in rows]
    print(f"kinds: {kinds}; average {sum(seconds) / len(seconds):.1f}s, slowest {max(seconds):.1f}s")
    print(f"questions flagged by the checker: {sum(bool(r['faithfulness_flags']) for r in questions)}/{len(questions)}")
    no_match = [r for r in rows if not r["expected"]]
    print(f"no-resource tests marked low match: {sum(r['result']['low_match'] for r in no_match)}/{len(no_match)}")
    matching = [r for r in rows if r["expected"]]
    print(f"resource tests wrongly marked low match: {sum(r['result']['low_match'] for r in matching)}/{len(matching)}")
    answered = [r for r in cases if r["result"]["kind"] == "possible_diagnosis"]
    with_resource = [r for r in answered if r["expected"]]
    print(f"diagnoses produced: {len(answered)}/{len(cases)}; flagged by the checker: {sum(bool(r['faithfulness_flags']) for r in answered)}")
    print(f"right resource chosen: {sum(r['result']['data']['resource'] == r['expected'] for r in with_resource)}/{len(with_resource)}")
    print(f"accepted condition named: {sum(r['accepted_hit'] for r in with_resource)}/{len(with_resource)}")
    emergencies = [r for r in cases if r["severity"] == "emergency"]
    print(f"DANGEROUS MISSES (emergency rated minor, or no usable answer): {sum(r['dangerous_miss'] for r in emergencies)}/{len(emergencies)}: "
          f"{[r['id'] for r in emergencies if r['dangerous_miss']]}")
    print(f"emergency answers that never mention 112: {sum(r['no_112'] for r in emergencies)}/{len(emergencies)}: "
          f"{[r['id'] for r in emergencies if r['no_112']]}")
    minors = [r for r in answered if r["severity"] == "minor"]
    print(f"minor cases rated emergency: {sum(r['result']['data']['urgency'] == 'emergency' for r in minors)}/{len(minors)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    conn = sqlite3.connect(":memory:")
    build(conn, load_items(), document_embedder("embeddinggemma"), EMBED_MODELS["embeddinggemma"]["dimensions"])
    models, medicines = real_models("embeddinggemma"), load_medicines()
    tests = [{"set": "search", "mode": "ask", "id": q["id"], "text": q["text"], "expected": q["expected_card"], "severity": None, "accepted": []}
             for q in load("search_questions.json")["questions"]]
    tests += [{"set": "diagnosis", "mode": "diagnose", "id": c["id"], "text": c["text"], "expected": c["expected_resource"],
               "severity": c["severity"], "accepted": c["accepted"]} for c in load("diagnosis_cases.json")["cases"]]
    if args.limit:
        tests = tests[:args.limit] + tests[len(tests) // 2:len(tests) // 2 + args.limit]
    path = os.path.join(ROOT, "evals", "results", f"phase2-answers-{args.name}.jsonl")
    rows = []
    with open(path, "w", encoding="utf-8") as out:
        for test in tests:
            start = time.time()
            result = asdict(answer(Request(test["mode"], text=test["text"]), conn, models, medicines))
            row = {**test, "seconds": round(time.time() - start, 1), "result": result,
                   "faithfulness_flags": [w for w in result["warnings"] if w.startswith("Kaynakla") or "seçilen kaynaktan" in w]}
            diagnosed = result["kind"] == "possible_diagnosis"
            condition = normalize(f"{result['data']['condition']} {result['data']['resource']}") if diagnosed else ""
            row["accepted_hit"] = any(term in condition for term in test["accepted"])
            row["no_112"] = test["severity"] == "emergency" and "112" not in result["body"]
            row["dangerous_miss"] = test["severity"] == "emergency" and (not diagnosed or result["data"]["urgency"] == "minor")
            result["sources"] = [source["id"] for source in result["sources"]]
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{test['id']} {result['kind']:18} {row['seconds']:5.1f}s", flush=True)
    report(rows)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
