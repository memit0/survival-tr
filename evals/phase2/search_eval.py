"""Compare search methods on the test questions and diagnosis cases.

Needs a llama-server for the chosen embedding model on 127.0.0.1:8098, unless --model is omitted.
Usage: python3 evals/phase2/search_eval.py [--model embeddinggemma]
"""
import argparse
import json
import os
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "prototype"))
from llm import EMBED_MODELS, document_embedder, query_embedder  # noqa: E402
from pack import build, load_items  # noqa: E402
from search import keyword_search, merge_rankings, unit, vector_search  # noqa: E402

TOP = 3
TARGET = 0.90


def load_tests():
    folder = os.path.join(ROOT, "evals", "phase2")
    with open(os.path.join(folder, "search_questions.json"), encoding="utf-8") as f:
        tests = [{"set": "search", "id": q["id"], "text": q["text"], "expected": q["expected_card"]} for q in json.load(f)["questions"]]
    with open(os.path.join(folder, "diagnosis_cases.json"), encoding="utf-8") as f:
        tests += [{"set": "diagnosis", "id": c["id"], "text": c["text"], "expected": c["expected_resource"]} for c in json.load(f)["cases"]]
    return tests


def best_threshold(rows):
    """The cosine cut-off that best separates tests with a matching resource from those without."""
    candidates = sorted({row["top_cosine"] for row in rows})
    def correct(limit):
        return sum((row["top_cosine"] >= limit) == (row["expected"] is not None) for row in rows)
    best = max(candidates, key=correct)
    return best, correct(best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=EMBED_MODELS)
    args = parser.parse_args()
    conn = sqlite3.connect(":memory:")
    dimensions = EMBED_MODELS[args.model]["dimensions"] if args.model else None
    build(conn, load_items(), document_embedder(args.model) if args.model else None, dimensions)
    embed_query = query_embedder(args.model) if args.model else None

    rows = []
    for test in load_tests():
        rankings = {"keyword": keyword_search(conn, test["text"], 24)}
        row = dict(test)
        if embed_query:
            start = time.time()
            vector = unit(embed_query(test["text"]), dimensions)
            row["embed_seconds"] = round(time.time() - start, 3)
            rankings["vector"] = vector_search(conn, vector, 24)
            rankings["hybrid"] = merge_rankings(rankings["keyword"], rankings["vector"])
            row["top_cosine"] = round(rankings["vector"][0][1], 4)
        row["top3"] = {name: [item for item, _ in ranking[:TOP]] for name, ranking in rankings.items()}
        rows.append(row)

    name = args.model or "keyword"
    path = os.path.join(ROOT, "evals", "results", f"phase2-search-{name}.jsonl")
    with open(path, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"model: {name}")
    for method in rows[0]["top3"]:
        for test_set in ("search", "diagnosis"):
            scored = [r for r in rows if r["set"] == test_set and r["expected"]]
            hits = [r for r in scored if r["expected"] in r["top3"][method]]
            misses = [r["id"] for r in scored if r not in hits]
            mark = "PASS" if test_set == "search" and len(hits) / len(scored) >= TARGET else ("fail" if test_set == "search" else "    ")
            print(f"  {method:8} {test_set:10} top-{TOP}: {len(hits)}/{len(scored)} ({len(hits) / len(scored):.0%}) {mark} misses: {misses}")
    if embed_query:
        limit, correct = best_threshold(rows)
        seconds = [r["embed_seconds"] for r in rows]
        print(f"  low-match cut-off {limit:.3f}: separates {correct}/{len(rows)} (chosen on this same data, so optimistic)")
        print(f"  question embedding: average {sum(seconds) / len(seconds):.3f}s on this Mac")
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
