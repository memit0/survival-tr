"""Turkish analysis for BM25 only; never stem encoder inputs."""
import re
import unicodedata
from functools import lru_cache

_FOLD = str.maketrans("çğıöşü", "cgiosu")


@lru_cache(maxsize=1)
def _stemmer():
    import snowballstemmer
    return snowballstemmer.stemmer("turkish")


def analyze(text: str, stem: bool = False, ascii_fold: bool = False) -> list[str]:
    text = unicodedata.normalize("NFC", text).replace("I", "ı").replace("İ", "i").lower()
    # Treat suffixes after proper-name apostrophes separately from the name.
    text = re.sub(r"(?<=\w)['’][^\W\d_]+", "", text)
    words = re.findall(r"[^\W_]+", text, re.UNICODE)
    if stem:
        words = _stemmer().stemWords(words)
    # Fold after stemming: the Turkish stemmer expects Turkish spelling.
    if ascii_fold:
        words = [word.translate(_FOLD) for word in words]
    return words
