"""Article-level judgments remain stable when chunk boundaries change."""
from collections import defaultdict
from importlib.metadata import version, PackageNotFoundError
import json
import math
from pathlib import Path
import platform
import os
import time
import numpy as np
from .data import fingerprint, save_json


def validate_queries(rows, articles):
    seen, group_splits = set(), {}
    if not rows:
        raise ValueError("Evaluation requires questions")
    for row in rows:
        if not isinstance(row.get("qid"), str) or not row["qid"] or row["qid"] in seen:
            raise ValueError("Every question needs a unique non-empty qid")
        seen.add(row["qid"])
        if not isinstance(row.get("query"), str) or not row["query"].strip():
            raise ValueError(f"Missing query in {row['qid']}")
        if row.get("split") not in {"dev", "test", "train"}:
            raise ValueError("Question split must be dev, test or train")
        group = row.get("group", row["qid"])
        if group in group_splits and group_splits[group] != row["split"]:
            raise ValueError(f"Question family {group} leaks across splits")
        group_splits[group] = row["split"]
        labels = row.get("relevant", [])
        label_ids = set()
        for label in labels:
            aid, grade = label.get("article_id"), label.get("grade")
            if aid not in articles or aid in label_ids or grade not in (0, 1, 2):
                raise ValueError(f"Invalid/duplicate relevance label in {row['qid']}")
            label_ids.add(aid)
            for span in label.get("evidence", []):
                a, b = span.get("start"), span.get("end")
                if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= len(articles[aid]["text"]):
                    raise ValueError(f"Invalid evidence offset in {row['qid']}")
        if not isinstance(row.get("answerable"), bool):
            raise ValueError("Every question needs an explicit answerable boolean")
        if row["answerable"] != any(label["grade"] == 2 for label in labels):
            raise ValueError("answerable must agree with the presence of a grade-2 answer")
    return rows


def ndcg(ids, grades, k=10):
    gains = [2 ** grades.get(aid, 0) - 1 for aid in ids[:k]]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
    ideal = sorted((2 ** g - 1 for g in grades.values()), reverse=True)[:k]
    idcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def passage_answers(passage, labels):
    return any(label["article_id"] == passage["article_id"] and label["grade"] == 2 and
               any(passage["start"] <= span["start"] and passage["end"] >= span["end"]
                   for span in label.get("evidence", [])) for label in labels)


def score_query(row, result):
    grades = {r["article_id"]: r["grade"] for r in row.get("relevant", [])}
    correct = {aid for aid, grade in grades.items() if grade == 2}
    ids = [a["article_id"] for a in result["articles"]]
    first = next((i for i, aid in enumerate(ids[:10], 1) if aid in correct), None)
    evidence_labeled = any(r["grade"] == 2 and r.get("evidence") for r in row.get("relevant", []))
    metrics = {}
    if row["answerable"]:
        metrics = {"success_at_1": float(bool(ids and ids[0] in correct)),
                   "success_at_3": float(any(aid in correct for aid in ids[:3])),
                   "recall_at_3": len(set(ids[:3]) & correct) / len(correct),
                   "mrr_at_10": 1 / first if first else 0.,
                   "ndcg_at_10": ndcg(ids, grades),
                   "false_reject": float(not result["accepted"])}
        if evidence_labeled:
            metrics["passage_success_at_1"] = float(bool(result["articles"]) and passage_answers(result["articles"][0], row["relevant"]))
            metrics["candidate_success_at_20"] = float(any(passage_answers(p, row["relevant"]) for p in result["passages"][:20]))
    else:
        metrics["false_accept"] = float(result["accepted"])
    return {"qid": row["qid"], "query": row["query"], "group": row.get("group", row["qid"]),
            "category": row.get("category", "unspecified"), "answerable": row["answerable"],
            "metrics": metrics, "result": result}


def mean_metrics(rows):
    values = defaultdict(list)
    for row in rows:
        for name, value in row["metrics"].items():
            values[name].append(value)
    return {name: {"value": float(np.mean(v)), "n": len(v)} for name, v in sorted(values.items())}


def clustered_interval(rows, metric, samples=1000, seed=42):
    groups = defaultdict(list)
    for row in rows:
        if metric in row["metrics"]:
            groups[row["group"]].append(row["metrics"][metric])
    if not groups:
        return None
    values = list(groups.values())
    rng = np.random.default_rng(seed)
    draws = [np.mean([x for i in rng.integers(0, len(values), len(values)) for x in values[i]]) for _ in range(samples)]
    return [float(v) for v in np.quantile(draws, [.025, .975])]


def evaluate(searcher, queries, output, split="dev", method="dense", candidates=50, min_dense_score=None, warmup=2):
    validate_queries(queries, searcher.articles)  # Validate all splits before filtering.
    selected = [q for q in queries if q["split"] == split]
    if not selected:
        raise ValueError(f"No queries in split {split}")
    output = Path(output)
    if output.exists():
        raise ValueError("Report directory exists; use a fresh run name")
    for _ in range(warmup):
        searcher.search(selected[0]["query"], method, 10, candidates, min_dense_score)
    start = time.perf_counter()
    rows = [score_query(q, searcher.search(q["query"], method, 10, candidates, min_dense_score)) for q in selected]
    latencies = [r["result"]["latency_ms"] for r in rows]
    metrics = mean_metrics(rows)
    for name in metrics:
        metrics[name]["ci95_group_bootstrap"] = clustered_interval(rows, name)
    categories = {category: mean_metrics([r for r in rows if r["category"] == category])
                  for category in sorted({r["category"] for r in rows})}
    packages = {}
    for package in ("numpy", "snowballstemmer", "sentence-transformers", "torch", "onnxruntime"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    report = {"schema": 1, "split": split, "queries": len(rows), "query_fingerprint": fingerprint(selected),
              "article_fingerprint": fingerprint(list(searcher.articles.values())),
              "index": searcher.manifest, "method": method, "candidates": candidates,
              "min_dense_score": min_dense_score, "metrics": metrics, "categories": categories,
              "latency_ms": {"p50": float(np.percentile(latencies, 50)), "p95": float(np.percentile(latencies, 95))},
              "searcher_load_ms": searcher.load_ms, "warmup_queries": warmup,
              "evaluation_seconds": time.perf_counter() - start,
              "index_bytes": sum(p.stat().st_size for p in searcher.directory.rglob("*") if p.is_file()),
              "threads_requested": os.environ.get("TR_SEARCH_THREADS", "2"),
              "platform": platform.platform(), "python": platform.python_version(), "packages": packages,
              "label_policy": "grade 2 = satisfactory; unjudged assumed 0; evidence spans are alternative sufficient spans"}
    output.mkdir(parents=True)
    save_json(output / "summary.json", report)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    failures = [r for r in rows if r["metrics"].get("success_at_1", 1) == 0 or r["metrics"].get("false_accept", 0)]
    save_json(output / "failures.json", failures)
    lines = ["# Retrieval evaluation", "", f"Split: **{split}** · Method: **{method}** · Questions: **{len(rows)}**", "",
             "Demo scores are smoke tests, not real-world accuracy estimates.", "",
             "| Metric | Value | Questions | 95% group bootstrap interval |", "|---|---:|---:|---|"]
    for name, metric in metrics.items():
        lo, hi = metric["ci95_group_bootstrap"]
        lines.append(f"| {name} | {metric['value']:.3f} | {metric['n']} | [{lo:.3f}, {hi:.3f}] |")
    lines += ["", f"Warm P95: {report['latency_ms']['p95']:.2f} ms. See summary.json for configuration and predictions.jsonl for every result.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def compare(left_dir, right_dir):
    from .data import read_jsonl
    left, right = [json.loads((Path(d) / "summary.json").read_text()) for d in (left_dir, right_dir)]
    for field in ("query_fingerprint", "article_fingerprint", "split"):
        if left[field] != right[field]:
            raise ValueError(f"Reports differ in {field}; paired comparison is invalid")
    a, b = [read_jsonl(Path(d) / "predictions.jsonl") for d in (left_dir, right_dir)]
    by_id = {r["qid"]: r for r in b}
    differences = []
    for row in a:
        other = by_id[row["qid"]]
        differences.append({"qid": row["qid"], "group": row["group"], "metrics": {
            name: other["metrics"][name] - value for name, value in row["metrics"].items()}})
    return {"direction": "right minus left", "delta": mean_metrics(differences),
            "success_at_1_ci95": clustered_interval(differences, "success_at_1"),
            "improved_qids": [r["qid"] for r in differences if r["metrics"].get("success_at_1", 0) > 0],
            "regressed_qids": [r["qid"] for r in differences if r["metrics"].get("success_at_1", 0) < 0]}
