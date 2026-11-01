"""Find the test resources closest to a question, using keywords, embeddings, or both."""
import math
import re
import struct

from checker import lower_tr

FOLD = str.maketrans("şğüöçıâîû", "sguociaiu")
RRF_K = 60


def normalize(text):
    """Turkish lowercase with Turkish letters mapped to plain ones, so typed text without them still matches."""
    return " ".join(re.findall(r"\w+", lower_tr(text).translate(FOLD)))


def trigrams(text):
    """Every three-letter chunk of each word. Whole-word FTS5 queries miss suffixes: "depremde" doesn't find "deprem"."""
    return list(dict.fromkeys(word[i:i + 3] for word in normalize(text).split() for i in range(len(word) - 2)))


def keyword_search(conn, text, limit=10):
    """[(item id, score)], best first."""
    grams = trigrams(text)
    if not grams:
        return []
    query = " OR ".join(f'"{gram}"' for gram in grams)
    rows = conn.execute("SELECT item_id, bm25(fts) FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?", (query, limit))
    return [(item_id, -score) for item_id, score in rows]


def unit(vector, dimensions=None):
    """Keep the first dimensions, then scale to length 1, so a dot product equals cosine similarity."""
    kept = vector[:dimensions] if dimensions else vector
    length = math.sqrt(sum(value * value for value in kept)) or 1.0
    return [value / length for value in kept]


def pack_vector(vector):
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob):
    return struct.unpack(f"<{len(blob) // 4}f", blob)


def vector_search(conn, query_vector, limit=10):
    """[(item id, cosine similarity)], best first.

    ponytail: compares every item, which is fine for a small pack. Add an index if packs reach many thousands of items.
    """
    scored = []
    for item_id, blob in conn.execute("SELECT item_id, vector FROM vectors"):
        scored.append((item_id, sum(a * b for a, b in zip(query_vector, unpack_vector(blob)))))
    return sorted(scored, key=lambda row: row[1], reverse=True)[:limit]


def merge_rankings(*rankings, k=RRF_K):
    """Reciprocal Rank Fusion: [(item id, score)], combining positions rather than incomparable raw scores."""
    scores = {}
    for ranking in rankings:
        for position, (item_id, _) in enumerate(ranking, 1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda row: row[1], reverse=True)
