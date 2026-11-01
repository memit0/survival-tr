"""Validated article import and evidence-preserving, token-budgeted chunking."""
import hashlib
import json
import re
from pathlib import Path


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_jsonl(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
    return rows


def validate_articles(rows):
    seen = set()
    if not rows:
        raise ValueError("Corpus must contain at least one article")
    for row in rows:
        for field in ("article_id", "title", "text"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Article requires non-empty string {field}")
        if row["article_id"] in seen:
            raise ValueError(f"Duplicate article_id: {row['article_id']}")
        seen.add(row["article_id"])
    return rows


def embedding_text(title, heading, text):
    return f"{title}\n{heading}\n{text}" if heading else f"{title}\n{text}"


def sections(text):
    """Yield body offsets and current Markdown heading without rewriting source."""
    cursor, heading = 0, ""
    for match in re.finditer(r"^#{1,6}\s+(.+?)\s*$", text, re.MULTILINE):
        if text[cursor:match.start()].strip():
            yield cursor, match.start(), heading
        heading = match.group(1)
        cursor = match.end()
    if text[cursor:].strip():
        yield cursor, len(text), heading


def prepare(articles, budget=240, overlap=1, tokenizer=None, tokenizer_id="regex-words"):
    """Count the entire prefixed input including special tokens with a model tokenizer.

    Regex mode is only a lexical baseline. Overlong single words raise rather than
    silently truncate. Sentence/whitespace boundaries preserve original offsets.
    """
    validate_articles(articles)
    if budget < 16 or budget > 512 or overlap < 0:
        raise ValueError("budget must be 16..512 and overlap must be non-negative")
    count = (lambda s: len(tokenizer.encode(s, add_special_tokens=True))) if tokenizer else (
        lambda s: len(re.findall(r"\w+|[^\w\s]", s)))
    chunks = []
    for article in articles:
        text, title, aid = article["text"], article["title"], article["article_id"]
        for section_start, section_end, heading in sections(text):
            def fits(start, end):
                return count("passage: " + embedding_text(title, heading, text[start:end])) <= budget
            if count("passage: " + embedding_text(title, heading, "")) >= budget:
                raise ValueError(f"Title/heading exceeds token budget in {aid}")
            # Sentence boundaries include paragraph breaks; preserve punctuation.
            boundaries = [section_start]
            body = text[section_start:section_end]
            boundaries += [section_start + m.end() for m in re.finditer(r"[.!?]+(?:\s+|$)|\n\s*\n", body)]
            if boundaries[-1] != section_end:
                boundaries.append(section_end)
            units = []
            for start, end in zip(boundaries, boundaries[1:]):
                if not text[start:end].strip():
                    continue
                if fits(start, end):
                    units.append((start, end))
                    continue
                # Long sentences are split at word boundaries with exact token checks.
                words = list(re.finditer(r"\S+", text[start:end]))
                part_start = start + words[0].start()
                part_end = part_start
                for word in words:
                    wstart, wend = start + word.start(), start + word.end()
                    if not fits(wstart, wend):
                        raise ValueError(f"A word exceeds token budget in {aid}; clean input or increase budget")
                    if not fits(part_start, wend):
                        units.append((part_start, part_end))
                        part_start = wstart
                    part_end = wend
                units.append((part_start, part_end))
            i = 0
            while i < len(units):
                j = i + 1
                while j < len(units) and fits(units[i][0], units[j][1]):
                    j += 1
                start, end = units[i][0], units[j - 1][1]
                while start < end and text[start].isspace():
                    start += 1
                while end > start and text[end - 1].isspace():
                    end -= 1
                chunk_id = f"{aid}:{start}:{end}"
                chunks.append({"chunk_id": chunk_id, "article_id": aid,
                               "title": title, "heading": heading, "text": text[start:end],
                               "start": start, "end": end,
                               "embedding_text": embedding_text(title, heading, text[start:end])})
                if j == len(units):
                    break
                i = max(i + 1, j - overlap)
    if not chunks:
        raise ValueError("No passages produced")
    payload = {"articles": articles, "chunks": chunks,
               "config": {"budget": budget, "overlap": overlap, "tokenizer": tokenizer_id},
               "schema": 1}
    payload["fingerprint"] = fingerprint(payload)
    return payload


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
