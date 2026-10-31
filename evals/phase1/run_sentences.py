"""Phase 1, round 2: Kumru rewrites one sentence at a time, and code adds each citation.

Round 1 (run.py) asked for a whole cited list; Kumru copied the prompt's example and added filler.
Needs a llama-server on 127.0.0.1:8099. Usage: python3 evals/phase1/run_sentences.py [--runs 3] [--limit N]
"""
import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "prototype"))
from checker import check, lower_tr  # noqa: E402

SERVER = "http://127.0.0.1:8099/v1/chat/completions"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SYSTEM = ("Acil durum metinlerini sadeleştiren bir yardımcısın. Anlamı değiştirme. Yeni bilgi ekleme. "
          "Sayıları ve olumsuz ifadeleri aynen koru. Yalnızca yeni cümleyi yaz.")


def rewrite(sentence, grammar, seed):
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Bu cümleyi daha kısa ve sade yaz:\n{sentence}"}],
            "temperature": 0.2, "repeat_penalty": 1.0, "max_tokens": 80, "seed": seed, "grammar": grammar}
    request = urllib.request.Request(SERVER, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with OPENER.open(request, timeout=120) as response:
        return json.load(response)["choices"][0]["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    with open(os.path.join(ROOT, "evals", "phase1", "passages.json"), encoding="utf-8") as f:
        passages = json.load(f)["passages"]
    passages = passages[:args.limit] if args.limit else passages
    with open(os.path.join(ROOT, "grammars", "sentence.gbnf"), encoding="utf-8") as f:
        grammar = f.read()
    path = os.path.join(ROOT, "evals", "results", "phase1-round2.jsonl")
    rows = []
    with open(path, "w", encoding="utf-8") as out:
        for passage in passages:
            for run in range(args.runs):
                start = time.time()
                texts = {sid: rewrite(text, grammar, 100 + run) for sid, text in passage["sentences"].items()}
                output = "\n".join(f"{n}. [{sid}] {text}" for n, (sid, text) in enumerate(texts.items(), 1))
                passed, reasons = check(output, passage["sentences"], passage["required"])
                failed = {reason.split(":")[0] for reason in reasons}
                copied = sum(lower_tr(t).strip(" .") == lower_tr(passage["sentences"][s]).strip(" .") for s, t in texts.items())
                rows.append({"passage": passage["id"], "run": run, "passed": passed, "reasons": reasons, "copied": copied,
                             "steps": len(texts), "steps_passed": len(texts) - len(failed & set(texts)),
                             "seconds": round(time.time() - start, 1), "source": passage["sentences"], "output": output})
                out.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
                out.flush()
                print(f"{passage['id']:16} {run} {'PASS' if passed else 'fail'} {rows[-1]['seconds']:5.1f}s", flush=True)
    steps = sum(r["steps"] for r in rows)
    print(f"passages: {sum(r['passed'] for r in rows)}/{len(rows)} passed; "
          f"steps: {sum(r['steps_passed'] for r in rows)}/{steps} passed; "
          f"{sum(r['copied'] for r in rows)}/{steps} exact copies; "
          f"average {sum(r['seconds'] for r in rows) / len(rows):.1f}s per passage")


if __name__ == "__main__":
    main()
