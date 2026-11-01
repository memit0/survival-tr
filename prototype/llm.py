"""Small clients for the two local llama-server processes. Nothing here can reach beyond this computer."""
import json
import urllib.error
import urllib.request
from urllib.parse import urlsplit

EMBED_URL = "http://127.0.0.1:8098"
CHAT_URL = "http://127.0.0.1:8099"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

EMBED_MODELS = {
    "embeddinggemma": {"query": "task: search result | query: {text}", "document": "title: {title} | text: {text}", "dimensions": 256},
}


class ModelError(Exception):
    """The local model is missing, too slow, or returned something unusable."""


def _post(url, path, body, timeout):
    if urlsplit(url).hostname not in ("127.0.0.1", "localhost"):
        raise ModelError(f"refusing a non-local model address: {url}")
    request = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise ModelError(str(error)) from error


def embed(text, url=EMBED_URL, timeout=30):
    data = _post(url, "/v1/embeddings", {"input": text}, timeout)
    try:
        return data["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as error:
        raise ModelError("unexpected embedding response") from error


def chat(system, user, grammar=None, url=CHAT_URL, max_tokens=400, seed=7, timeout=180):
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.2, "max_tokens": max_tokens, "seed": seed, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}  # thinking used up the token budget before any answer
    if grammar:
        body["grammar"] = grammar
    data = _post(url, "/v1/chat/completions", body, timeout)
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ModelError("unexpected chat response") from error


def query_embedder(model, url=EMBED_URL):
    template = EMBED_MODELS[model]["query"]
    return lambda text: embed(template.format(text=text), url)


def document_embedder(model, url=EMBED_URL):
    template = EMBED_MODELS[model]["document"]
    return lambda title, text: embed(template.format(title=title, text=text), url)
