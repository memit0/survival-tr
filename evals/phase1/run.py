"""Phase 1: can Kumru simplify Turkish passages without changing their meaning?

Needs a llama-server running on 127.0.0.1:8099. Compares plain prompting with the citation grammar.
Usage: python3 evals/phase1/run.py [--runs 3] [--limit N]
"""
import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "prototype"))
from checker import check, lower_tr, parse_steps  # noqa: E402

SERVER = "http://127.0.0.1:8099/v1/chat/completions"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SYSTEM = ("Acil durum metinlerini sadeleştiren bir yardımcısın. Yalnızca verilen cümleleri kullan. Yeni bilgi ekleme. "
          "Sayıları ve uyarıları aynen koru. Olumsuz ifadeleri değiştirme.")
# The example's topic and wording appear in no test passage, so it can't hand the model an answer.
INSTRUCTION = ("Aşağıdaki cümleleri kısa ve anlaşılır adımlara dönüştür. Her cümle için bir adım yaz. Kalın yazı kullanma.\n\n"
               "Örnek cümleler:\n[S1] Çadırın fermuarını kapatın ve pencerelerini açmayın.\n"
               "[S2] Çantanıza en az 2 gün yetecek kuru gıda koyun.\n\n"
               "Örnek adımlar:\n1. [S1] Çadırın fermuarını kapatın, pencereleri açmayın.\n"
               "2. [S2] Çantanıza en az 2 gün yetecek kuru gıda koyun.\n\n"
               "Şimdi bu cümleleri dönüştür:\n{sentences}")


def ask(sentences, grammar, seed):
    listed = "\n".join(f"[{k}] {v}" for k, v in sentences.items())
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": INSTRUCTION.format(sentences=listed)}],
            "temperature": 0.25, "repeat_penalty": 1.1, "max_tokens": 300, "seed": seed}
    if grammar:
        body["grammar"] = grammar
    request = urllib.request.Request(SERVER, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.time()
    with OPENER.open(request, timeout=300) as response:
        data = json.load(response)
    return data["choices"][0]["message"]["content"], time.time() - start, data.get("timings", {}).get("predicted_per_second")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    with open(os.path.join(ROOT, "evals", "phase1", "passages.json"), encoding="utf-8") as f:
        passages = json.load(f)["passages"]
    passages = passages[:args.limit] if args.limit else passages
    with open(os.path.join(ROOT, "grammars", "writer.gbnf"), encoding="utf-8") as f:
        grammar = f.read()
    out_dir = os.path.join(ROOT, "evals", "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "phase1.jsonl")
    rows = []
    with open(path, "w", encoding="utf-8") as out:
        for passage in passages:
            for mode in ("plain", "grammar"):
                for run in range(args.runs):
                    text, seconds, speed = ask(passage["sentences"], grammar if mode == "grammar" else None, 100 + run)
                    passed, reasons = check(text, passage["sentences"], passage["required"])
                    steps = parse_steps(text) or []
                    copied = sum(lower_tr(t).strip(" .") == lower_tr(passage["sentences"].get(s, "")).strip(" .")
                                 for s, t in steps)  # exact copies are safe, but they simplify nothing
                    row = {"passage": passage["id"], "mode": mode, "run": run, "passed": passed, "reasons": reasons,
                           "seconds": round(seconds, 1), "tok_s": speed, "steps": len(steps), "copied": copied,
                           "output": text}
                    rows.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"{passage['id']:16} {mode:7} {run} {'PASS' if passed else 'fail'} {seconds:5.1f}s", flush=True)
    for mode in ("plain", "grammar"):
        some = [r for r in rows if r["mode"] == mode]
        print(f"{mode}: {sum(r['passed'] for r in some)}/{len(some)} passed the checker, "
              f"average {sum(r['seconds'] for r in some) / len(some):.1f}s, "
              f"{sum(r['copied'] for r in some)}/{sum(r['steps'] for r in some)} steps were exact copies")
    print("Results:", path)


if __name__ == "__main__":
    main()
