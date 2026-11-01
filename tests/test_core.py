import json
import socket
import subprocess
import sys
import numpy as np
import pytest
from turkish_search.data import prepare, fingerprint, validate_articles
from turkish_search.text import analyze
from turkish_search.search import BM25, Searcher, build_index, fuse
from turkish_search.cli import main
from turkish_search.evaluation import validate_queries, score_query, ndcg, evaluate, compare


ARTICLES = [
    {"article_id": "a", "title": "Kütüphane", "text": "# Üyelik\nÜyelik ücretsizdir. Kimlik kartıyla kayıt yapılır.\n\n# Saatler\nPazar günü kapalıdır."},
    {"article_id": "b", "title": "Kitap iadesi", "text": "Kitap iadesi giriş katında yapılır. Geciken kitaplar için görevliye danışın."},
]


class FakeEncoder:
    fingerprint, backend, onnx_file = "test-encoder", "torch", None
    def encode(self, texts, role):
        # Deterministic test fixture only; not a semantic model or accuracy benchmark.
        result = np.array([[1., .1] if "üye" in t.lower() or "Üye" in t else [.1, 1.] for t in texts], dtype=np.float32)
        return result / np.linalg.norm(result, axis=1, keepdims=True)


def test_turkish_analysis():
    assert analyze("İSTANBUL IĞDIR Ankara’nın") == ["istanbul", "ığdır", "ankara"]
    assert analyze("Şebeke çekmiyor", ascii_fold=True) == ["sebeke", "cekmiyor"]
    assert analyze("kitaplar", stem=True) == analyze("kitap", stem=True)
    assert analyze("değil") == ["değil"]  # Negation is not discarded.


def test_chunk_offsets_budget_and_tail():
    p = prepare(ARTICLES, budget=24, overlap=1)
    for c in p["chunks"]:
        a = next(a for a in ARTICLES if a["article_id"] == c["article_id"])
        assert c["text"] == a["text"][c["start"]:c["end"]]
        assert c["text"]
    assert any("Pazar" in c["text"] for c in p["chunks"])
    assert len({c["chunk_id"] for c in p["chunks"]}) == len(p["chunks"])


def test_long_sentence_splits_without_losing_words():
    text = " ".join(f"kelime{i}" for i in range(100))
    p = prepare([{"article_id": "a", "title": "Başlık", "text": text}], budget=20, overlap=0)
    assert " ".join(c["text"] for c in p["chunks"]) == text


def test_validation_and_tampering(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_articles(ARTICLES + [ARTICLES[0]])
    p = prepare(ARTICLES)
    p["chunks"][0]["text"] = "tampered"
    with pytest.raises(ValueError, match="fingerprint"):
        build_index(p, tmp_path / "index")


def test_bm25_positive_idf_and_no_match():
    b = BM25([["kitap"], ["kitap", "iade"]])
    assert b.score("kitap").min() > 0
    assert b.score("iade")[1] > b.score("iade")[0]
    assert not b.score("olmayankelime").any()


def test_rrf_and_dedup(tmp_path):
    order, scores = fuse([1, 2], [2, 1])
    assert scores[1] == scores[2]
    assert order == [1, 2]
    build_index(prepare(ARTICLES), tmp_path / "idx")
    s = Searcher(tmp_path / "idx")
    result = s.search("kitap", "bm25")
    assert result["articles"][0]["article_id"] == "b"
    assert len({r["article_id"] for r in result["articles"]}) == len(result["articles"])
    assert not s.search("zzzzzz", "bm25")["accepted"]
    assert not s.search("!!!", "bm25")["accepted"]
    with pytest.raises(ValueError, match="requires"):
        s.search("kitap", "dense")


def test_dense_exact_and_model_mismatch(tmp_path):
    e = FakeEncoder()
    build_index(prepare(ARTICLES), tmp_path / "idx", e)
    s = Searcher(tmp_path / "idx", e)
    assert s.search("Üyelik", "dense")["articles"][0]["article_id"] == "a"
    assert s.search("Üyelik", "hybrid")["accepted"]
    assert not s.search("Üyelik", "dense", min_dense_score=1)["accepted"]
    bad = FakeEncoder()
    bad.fingerprint = "different"
    with pytest.raises(ValueError, match="match"):
        Searcher(tmp_path / "idx", bad)


def test_second_excerpt_is_shown_from_the_same_article(tmp_path):
    text = " ".join(f"Kitap kuralı {i} burada yazılıdır." for i in range(30))
    build_index(prepare([{"article_id": "a", "title": "Kitap", "text": text}, *ARTICLES[1:]], budget=24, overlap=0), tmp_path / "idx")
    result = Searcher(tmp_path / "idx").search("kitap", "bm25", top_k=2)
    assert len({r["article_id"] for r in result["articles"]}) == len(result["articles"])
    counts = [r["article_id"] for r in result["excerpts"]]
    assert counts.count("a") == 2 and set(counts) == {"a", "b"}


def test_example_question_finds_its_passage_without_changing_shown_text(tmp_path):
    e = FakeEncoder()
    examples = [{"article_id": "b", "quote": "Geciken kitaplar", "questions": ["üye iade soruyor"]}]
    p = prepare(ARTICLES)
    build_index(p, tmp_path / "idx", e, examples=examples)
    s = Searcher(tmp_path / "idx", e)
    articles = s.search("üye", "dense")["articles"]
    hit = next(a for a in articles if a["article_id"] == "b")
    assert hit["matched_question"] == "üye iade soruyor"
    assert hit["dense_score"] == pytest.approx(articles[0]["dense_score"])  # Lifted level with the true "üye" passage.
    assert "üye" not in hit["text"].lower()  # The question finds text; it is never displayed as content.
    assert s.search("iade soruyor", "bm25")["articles"][0]["article_id"] == "b"
    with pytest.raises(ValueError, match="not found"):
        build_index(p, tmp_path / "bad", e, examples=[{"article_id": "b", "quote": "olmayan cümle", "questions": ["x"]}])
    assert not (tmp_path / "bad").exists()


def test_query_without_words_runs_through_the_cli(capsys, tmp_path):
    build_index(prepare(ARTICLES), tmp_path / "idx")
    assert main(["search", "--index", str(tmp_path / "idx"), "--method", "bm25", "!!!"]) == 0
    assert "bulunamadı" in capsys.readouterr().out


def questions():
    return [{"qid": "q1", "query": "Üyelik ücretsiz mi?", "group": "membership", "split": "dev", "answerable": True,
             "relevant": [{"article_id": "a", "grade": 2, "evidence": [{"start": 9, "end": 25}]}]},
            {"qid": "q2", "query": "zzzzzz", "split": "dev", "answerable": False, "relevant": []}]


def test_query_validation_leakage(tmp_path):
    q = questions()
    validate_queries(q, {a["article_id"]: a for a in ARTICLES})
    q.append({**q[0], "qid": "q3", "split": "test"})
    with pytest.raises(ValueError, match="leaks"):
        validate_queries(q, {a["article_id"]: a for a in ARTICLES})


def test_known_metrics_and_evidence():
    q = questions()[0]
    first = {"article_id": "a", "start": 0, "end": 50}
    r = score_query(q, {"accepted": True, "articles": [first], "passages": [first]})
    assert all(v == 1 for k, v in r["metrics"].items() if k != "false_reject")
    assert ndcg(["a"], {"a": 2}) == 1
    assert ndcg(["x", "a"], {"a": 2}) == pytest.approx(1 / np.log2(3))
    bad = score_query(q, {"accepted": True, "articles": [{**first, "end": 10}], "passages": []})
    assert bad["metrics"]["success_at_1"] == 1
    assert bad["metrics"]["passage_success_at_1"] == 0


def test_evaluate_compare_and_immutable_outputs(tmp_path):
    build_index(prepare(ARTICLES), tmp_path / "idx")
    s = Searcher(tmp_path / "idx")
    a = evaluate(s, questions(), tmp_path / "run-a", method="bm25", warmup=0)
    evaluate(s, questions(), tmp_path / "run-b", method="bm25", warmup=0)
    assert a["metrics"]["false_accept"]["value"] == 0
    assert a["metrics"]["success_at_1"]["value"] == 1
    assert compare(tmp_path / "run-a", tmp_path / "run-b")["delta"]["success_at_1"]["value"] == 0
    assert (tmp_path / "run-a" / "failures.json").exists()
    with pytest.raises(ValueError, match="exists"):
        evaluate(s, questions(), tmp_path / "run-a", method="bm25")
    with pytest.raises(ValueError, match="exists"):
        build_index(prepare(ARTICLES), tmp_path / "idx")


def test_offline_cli_fresh_process(tmp_path):
    build_index(prepare(ARTICLES), tmp_path / "idx")
    p = subprocess.run([sys.executable, "-m", "turkish_search", "offline-check", "--method", "bm25", "--index", str(tmp_path / "idx"), "kitap", "--json"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["accepted"]


def test_full_prefix_budget_with_tokenizer():
    class WordTokenizer:
        def encode(self, text, add_special_tokens=True):
            return list(range(len(text.split()) + (2 if add_special_tokens else 0)))
    tokenizer = WordTokenizer()
    text = " ".join(["kelime"] * 75)
    p = prepare([{"article_id": "x", "title": "İki kelime", "text": text}], budget=16, overlap=0, tokenizer=tokenizer)
    for c in p["chunks"]:
        assert len(tokenizer.encode("passage: " + c["embedding_text"])) <= 16
    assert " ".join(c["text"] for c in p["chunks"]) == text


def test_no_redundant_tail_chunk():
    p = prepare([{"article_id": "x", "title": "Başlık", "text": "Birinci cümle. İkinci cümle. Üçüncü cümle."}], budget=100, overlap=1)
    assert len(p["chunks"]) == 1


def test_model_fingerprint_ignores_download_temporaries(tmp_path):
    from turkish_search.encoder import bundle_fingerprint
    (tmp_path / "config.json").write_text('{}')
    original = bundle_fingerprint(tmp_path)
    (tmp_path / ".model.safetensors.partial").write_text('temporary')
    (tmp_path / "download-record.json").write_text('{}')
    assert bundle_fingerprint(tmp_path) == original
    (tmp_path / "config.json").write_text('{"changed": true}')
    assert bundle_fingerprint(tmp_path) != original
