"""Validate the test content and build the SQLite pack. Usage: python3 prototype/pack.py [--model embeddinggemma]"""
import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm import EMBED_MODELS, document_embedder  # noqa: E402
from search import normalize, pack_vector, unit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, ".local", "pack.db")
SOURCES = (("cards.json", "cards", "card"), ("diagnosis_resources.json", "resources", "resource"))


def load_items(content_dir=None):
    """All cards and resources. Raises ValueError for content that must never be built into a pack."""
    content_dir = content_dir or os.path.join(ROOT, "content")
    items, seen = [], set()
    pack_date = None
    for filename, key, collection in SOURCES:
        with open(os.path.join(content_dir, filename), encoding="utf-8") as f:
            data = json.load(f)
        if data.get("test_only") is not True:
            raise ValueError(f"{filename}: content must be marked test_only")
        pack_date = data["pack_date"]
        for item in data[key]:
            if item["id"] in seen:
                raise ValueError(f"duplicate id: {item['id']}")
            seen.add(item["id"])
            if not item["sentences"] or not set(item["required"]) <= set(item["sentences"]):
                raise ValueError(f"{item['id']}: required sentences must exist")
            items.append({**item, "collection": collection, "pack_date": pack_date})
    return items


def body_text(item):
    return " ".join(item["sentences"].values())


def build(conn, items, embed_document=None, dimensions=None):
    """Write the items, keyword index, and optional vectors into an empty database."""
    conn.executescript("""
        CREATE TABLE items (id TEXT PRIMARY KEY, data TEXT NOT NULL);
        CREATE VIRTUAL TABLE fts USING fts5(item_id UNINDEXED, body, tokenize='trigram');
        CREATE TABLE vectors (item_id TEXT PRIMARY KEY, vector BLOB NOT NULL);
    """)
    for item in items:
        text = body_text(item)
        conn.execute("INSERT INTO items VALUES (?, ?)", (item["id"], json.dumps(item, ensure_ascii=False)))
        conn.execute("INSERT INTO fts VALUES (?, ?)", (item["id"], normalize(f"{item['title']} {text}")))
        if embed_document:
            conn.execute("INSERT INTO vectors VALUES (?, ?)",
                         (item["id"], pack_vector(unit(embed_document(item["title"], text), dimensions))))
    conn.commit()


def get_item(conn, item_id):
    row = conn.execute("SELECT data FROM items WHERE id = ?", (item_id,)).fetchone()
    return json.loads(row[0]) if row else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="embeddinggemma", choices=EMBED_MODELS)
    parser.add_argument("--path", default=DEFAULT_PATH)
    args = parser.parse_args()
    temporary = args.path + ".tmp"
    if os.path.exists(temporary):
        os.remove(temporary)
    conn = sqlite3.connect(temporary)
    items = load_items()
    build(conn, items, document_embedder(args.model), EMBED_MODELS[args.model]["dimensions"])
    conn.close()
    os.replace(temporary, args.path)  # a failed build never leaves a half-written pack in place
    print(f"built {args.path}: {len(items)} items, model {args.model}")


if __name__ == "__main__":
    main()
