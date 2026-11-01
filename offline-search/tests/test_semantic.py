"""Optional real-model integration. Run with TR_SEARCH_MODEL=/absolute/local/model pytest."""
import os
import socket
from pathlib import Path
import numpy as np
import pytest
from turkish_search.encoder import Encoder, load_tokenizer, bundle_fingerprint
from turkish_search.data import prepare
from turkish_search.search import Searcher, build_index


@pytest.mark.skipif(not os.environ.get("TR_SEARCH_MODEL"), reason="Set TR_SEARCH_MODEL to a downloaded E5-small bundle")
def test_real_e5_with_network_blocked(tmp_path, monkeypatch):
    def denied(*a, **kw):
        raise AssertionError("Network attempted during local integration test")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    path = Path(os.environ["TR_SEARCH_MODEL"])
    encoder = Encoder(path)
    articles = [
        {"article_id": "password", "title": "Parola", "text": "Parolanızı unuttuysanız giriş sayfasından parola sıfırlama bağlantısı isteyin."},
        {"article_id": "books", "title": "Kitaplar", "text": "Bir üye üç kitabı on dört günlüğüne ödünç alabilir."},
    ]
    prepared = prepare(articles, tokenizer=load_tokenizer(path), tokenizer_id=bundle_fingerprint(path))
    build_index(prepared, tmp_path / "index", encoder)
    s = Searcher(tmp_path / "index", encoder)
    result = s.search("Parolamı hatırlamıyorum nasıl yenilerim?")
    assert result["articles"][0]["article_id"] == "password"
    assert encoder.encode(["merhaba"], "query").shape == (1, 384)
    with pytest.raises(ValueError, match="limit"):
        encoder.encode(["kelime " * 1000], "query")
