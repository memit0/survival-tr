"""Explicit commands for model preparation, indexing, searching, and evaluation."""
import argparse
import json
from pathlib import Path
import sys
from .data import prepare, read_jsonl, save_json
from .encoder import Encoder, bundle_fingerprint, download, load_tokenizer, quantize
from .evaluation import compare, evaluate
from .search import Searcher, build_index


def add_model(parser):
    parser.add_argument("--model", default="models/e5-small", help="Local model directory (never a Hub ID)")
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--onnx-file", help="Path relative to model directory, e.g. onnx/model_qint8_arm64.onnx")


def make_parser():
    p = argparse.ArgumentParser(description="Turkish offline retrieval; no generated answers")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download-model", help="ONLINE: download E5-small once")
    d.add_argument("--output", default="models/e5-small")
    d.add_argument("--revision", help="Pin a Hugging Face commit SHA")
    q = sub.add_parser("quantize", help="OFFLINE: export a separate INT8 ONNX bundle")
    q.add_argument("--source", default="models/e5-small")
    q.add_argument("--output", required=True)
    q.add_argument("--config", choices=["arm64", "avx2", "avx512", "avx512_vnni"], default="arm64")
    prep = sub.add_parser("prepare", help="Validate articles and split them while preserving evidence offsets")
    prep.add_argument("--articles", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--model", default="models/e5-small")
    prep.add_argument("--lexical-only", action="store_true", help="Use regex counts instead of model tokens; BM25 smoke tests only")
    prep.add_argument("--budget", type=int, default=240)
    prep.add_argument("--overlap", type=int, default=1, help="Overlap sentence/word units, always progressing")
    idx = sub.add_parser("index", help="Write a new immutable SQLite/vector index")
    idx.add_argument("--prepared", required=True)
    idx.add_argument("--output", required=True)
    idx.add_argument("--lexical-only", action="store_true")
    idx.add_argument("--stem", action="store_true")
    idx.add_argument("--ascii-fold", action="store_true")
    idx.add_argument("--examples", help="JSONL of source quotes with example questions that should find them")
    add_model(idx)
    for name in ("search", "evaluate", "offline-check"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--index", required=True)
        cmd.add_argument("--method", choices=["dense", "bm25", "hybrid"], default="dense")
        cmd.add_argument("--candidates", type=int, default=50)
        cmd.add_argument("--min-dense-score", type=float, help="Experimental calibrated cosine cutoff; no default")
        add_model(cmd)
        if name in ("search", "offline-check"):
            cmd.add_argument("query")
            cmd.add_argument("--top-k", type=int, default=3)
            cmd.add_argument("--json", action="store_true")
        else:
            cmd.add_argument("--queries", required=True)
            cmd.add_argument("--output", required=True)
            cmd.add_argument("--split", choices=["dev", "test", "train"], default="dev")
    article = sub.add_parser("article", help="Read a full article from the local index")
    article.add_argument("--index", required=True)
    article.add_argument("article_id")
    comp = sub.add_parser("compare", help="Paired comparison of two reports on identical questions/articles")
    comp.add_argument("left")
    comp.add_argument("right")
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        if args.command == "download-model":
            result = download(args.output, args.revision)
            save_json(Path(args.output) / "download-record.json", result)
            # The record itself changes the bundle. Indexing fingerprints the complete saved bundle.
        elif args.command == "quantize":
            result = quantize(args.source, args.output, args.config)
        elif args.command == "prepare":
            tokenizer = None if args.lexical_only else load_tokenizer(args.model)
            tid = "regex-words" if tokenizer is None else bundle_fingerprint(args.model)
            payload = prepare(read_jsonl(args.articles), args.budget, args.overlap, tokenizer, tid)
            if Path(args.output).exists():
                raise ValueError("Prepared output exists; choose a fresh experiment path")
            save_json(args.output, payload)
            result = {"articles": len(payload["articles"]), "passages": len(payload["chunks"]), "output": args.output}
        elif args.command == "index":
            encoder = None if args.lexical_only else Encoder(args.model, args.backend, args.onnx_file)
            examples = read_jsonl(args.examples) if args.examples else None
            result = build_index(json.loads(Path(args.prepared).read_text()), args.output, encoder, args.stem, args.ascii_fold, examples)
        elif args.command == "compare":
            result = compare(args.left, args.right)
        else:
            if args.command == "offline-check":
                # Must run in a FRESH process: block network before model/index loading.
                import socket
                def denied(*a, **kw):
                    raise RuntimeError("Network attempted during offline check")
                socket.socket.connect = denied
                socket.socket.connect_ex = denied
                socket.create_connection = denied
                socket.getaddrinfo = denied
            encoder = None if args.command == "article" or args.method == "bm25" else Encoder(args.model, args.backend, args.onnx_file)
            searcher = Searcher(args.index, encoder)
            if args.command == "article":
                if args.article_id not in searcher.articles:
                    raise ValueError("Unknown article ID")
                article = searcher.articles[args.article_id]
                print(article["title"] + "\n\n" + article["text"])
                return 0
            if args.command == "evaluate":
                result = evaluate(searcher, read_jsonl(args.queries), args.output,
                                  args.split, args.method, args.candidates, args.min_dense_score)
            else:
                result = searcher.search(args.query, args.method, args.top_k, args.candidates, args.min_dense_score)
                if args.command == "offline-check":
                    result["offline_check"] = "Python socket connections and DNS blocked before local model loading"
                if not args.json:
                    if not result["accepted"]:
                        print("Bu koleksiyonda yeterli bir yanıt bulunamadı.")
                    for i, hit in enumerate(result["excerpts"], 1):
                        print(f"\n{i}. {hit['title']} [{hit['article_id']}] score={hit['score']:.4f}\n{hit['text']}\n")
                    print(f"Search: {result['latency_ms']:.1f} ms")
                    return 0
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
