"""Exact dense retrieval and positive-IDF BM25; indexes contain no pickled objects."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
import numpy as np
from .data import fingerprint, save_json
from .text import analyze


class BM25:
    def __init__(self, docs, stem=False, ascii_fold=False, k1=1.5, b=0.75):
        self.stem, self.ascii_fold, self.k1, self.b = stem, ascii_fold, k1, b
        self.lengths = np.array([len(doc) for doc in docs], dtype=np.float32)
        self.avg = max(float(self.lengths.mean()), 1.0)
        self.postings = defaultdict(list)
        for index, doc in enumerate(docs):
            for token, freq in Counter(doc).items():
                self.postings[token].append((index, freq))

    def score(self, query):
        scores = np.zeros(len(self.lengths), dtype=np.float64)
        for term in set(analyze(query, self.stem, self.ascii_fold)):
            postings = self.postings.get(term, [])
            idf = math.log(1 + (len(scores) - len(postings) + .5) / (len(postings) + .5))
            for index, frequency in postings:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.avg)
                scores[index] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        return scores


def build_index(prepared, destination, encoder=None, stem=False, ascii_fold=False):
    expected = prepared.get("fingerprint")
    if expected != fingerprint({k: v for k, v in prepared.items() if k != "fingerprint"}):
        raise ValueError("Prepared corpus fingerprint mismatch; rerun prepare")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Index already exists; choose a new directory for each experiment")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".index-", dir=destination.parent))
    try:
        with sqlite3.connect(temporary / "corpus.sqlite") as db:
            db.execute("CREATE TABLE articles (article_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE chunks (position INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
            db.executemany("INSERT INTO articles VALUES (?, ?)", [
                (a["article_id"], json.dumps(a, ensure_ascii=False)) for a in prepared["articles"]])
            db.executemany("INSERT INTO chunks VALUES (?, ?)", [
                (i, json.dumps(c, ensure_ascii=False)) for i, c in enumerate(prepared["chunks"])])
        manifest = {"schema": 1, "corpus_fingerprint": expected, "chunking": prepared["config"],
                    "articles": len(prepared["articles"]), "passages": len(prepared["chunks"]),
                    "stem": stem, "ascii_fold": ascii_fold, "model": None}
        if encoder is not None:
            vectors = encoder.encode([c["embedding_text"] for c in prepared["chunks"]], "passage")
            if vectors.ndim != 2 or vectors.shape[0] != len(prepared["chunks"]) or not np.isfinite(vectors).all():
                raise ValueError("Invalid embedding matrix")
            norms = np.linalg.norm(vectors, axis=1)
            if not np.allclose(norms, 1, atol=.001):
                raise ValueError("Encoder must return normalized embeddings")
            np.save(temporary / "vectors.npy", vectors, allow_pickle=False)
            manifest["model"] = {"fingerprint": encoder.fingerprint, "backend": encoder.backend,
                                 "onnx_file": encoder.onnx_file, "dimensions": vectors.shape[1]}
        manifest["index_fingerprint"] = fingerprint(manifest)
        save_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def ordered(scores, positive_only=False):
    ids = np.arange(len(scores))
    if positive_only:
        ids = ids[scores > 0]
    return sorted(ids.tolist(), key=lambda i: (-float(scores[i]), i))


def fuse(dense_order, lexical_order, limit=50, constant=60):
    scores = defaultdict(float)
    for ranking in (dense_order[:limit], lexical_order[:limit]):
        for rank, index in enumerate(ranking, 1):
            scores[index] += 1 / (constant + rank)
    return sorted(scores, key=lambda i: (-scores[i], i)), scores


class Searcher:
    def __init__(self, directory, encoder=None):
        start = time.perf_counter()
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        with sqlite3.connect(f"file:{(self.directory / 'corpus.sqlite').resolve().as_posix()}?mode=ro", uri=True) as db:
            self.articles = {aid: json.loads(p) for aid, p in db.execute("SELECT article_id, payload FROM articles")}
            self.chunks = [json.loads(p) for _, p in db.execute("SELECT position, payload FROM chunks ORDER BY position")]
        self.encoder = encoder
        model = self.manifest["model"]
        self.vectors = None
        if encoder is not None:
            if model is None or model["fingerprint"] != encoder.fingerprint or model["backend"] != encoder.backend or model["onnx_file"] != encoder.onnx_file:
                raise ValueError("Model/backend does not match index; rebuild the index with this encoder")
            self.vectors = np.load(self.directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
            if self.vectors.shape != (len(self.chunks), model["dimensions"]):
                raise ValueError("Embedding matrix shape does not match index")
        self.bm25 = BM25([analyze(c["embedding_text"], self.manifest["stem"], self.manifest["ascii_fold"])
                          for c in self.chunks], self.manifest["stem"], self.manifest["ascii_fold"])
        self.load_ms = (time.perf_counter() - start) * 1000

    def search(self, query, method="dense", top_k=3, candidates=50, min_dense_score=None):
        if method not in {"dense", "bm25", "hybrid"}:
            raise ValueError("method must be dense, bm25 or hybrid")
        if top_k < 1 or candidates < 1:
            raise ValueError("top_k and candidates must be positive")
        if min_dense_score is not None and (method == "bm25" or not -1 <= min_dense_score <= 1):
            raise ValueError("min_dense_score requires dense/hybrid and a value in [-1,1]")
        start = time.perf_counter()
        if not analyze(query):
            return {"accepted": False, "articles": [], "passages": [], "latency_ms": 0.0}
        dense = None
        if method != "bm25":
            if self.encoder is None:
                raise ValueError("Dense/hybrid search requires a local model and a dense index")
            query_vector = self.encoder.encode([query], "query")[0]
            dense = self.vectors @ query_vector
        lexical = self.bm25.score(query) if method != "dense" else None
        if method == "dense":
            ranking, scores = ordered(dense), dense
        elif method == "bm25":
            ranking, scores = ordered(lexical, True), lexical
        else:
            ranking, scores = fuse(ordered(dense), ordered(lexical, True), candidates)
        passages = []
        articles, seen = [], set()
        for index in ranking:
            if min_dense_score is not None and float(dense[index]) < min_dense_score:
                continue
            chunk = self.chunks[index]
            result = {**chunk, "score": float(scores[index]),
                      "dense_score": None if dense is None else float(dense[index])}
            if len(passages) < max(candidates, 20):
                passages.append(result)
            if chunk["article_id"] not in seen:
                seen.add(chunk["article_id"])
                if len(articles) < top_k:
                    articles.append({**result, "source": self.articles[chunk["article_id"]].get("source", "")})
        return {"accepted": bool(articles), "articles": articles, "passages": passages,
                "latency_ms": (time.perf_counter() - start) * 1000}
