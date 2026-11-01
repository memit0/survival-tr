"""Run with: python3 -m unittest discover -s tests"""
import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prototype"))
from pack import build, get_item, load_items  # noqa: E402
from search import keyword_search, merge_rankings, normalize, trigrams, unit, vector_search  # noqa: E402

ITEMS = [
    {"id": "sarsinti", "title": "Deprem sırasında", "required": [], "sentences": {"S1": "Sarsıntı başlarsa yere çökün."}},
    {"id": "su", "title": "İçme suyu", "required": [], "sentences": {"S1": "Suyu en az 1 dakika kaynatın."}},
]


def fake_embed(title, text):
    return [1.0, 0.0] if "Deprem" in title else [0.0, 1.0]


def make_pack():
    conn = sqlite3.connect(":memory:")
    build(conn, ITEMS, fake_embed)
    return conn


class PackCase(unittest.TestCase):
    def pack(self):
        conn = make_pack()
        self.addCleanup(conn.close)
        return conn


class Normalize(PackCase):
    def test_turkish_letters_and_case(self):
        self.assertEqual(normalize("ALMIYOR"), "almiyor")
        self.assertEqual(normalize("İçme Suyu, Şimdi!"), "icme suyu simdi")
        self.assertEqual(normalize("yapmalıyım"), normalize("yapmaliyim"))

    def test_short_words_make_no_trigrams(self):
        self.assertEqual(trigrams("su ve"), [])
        self.assertEqual(trigrams("deprem")[:2], ["dep", "epr"])


class Keyword(PackCase):
    def test_suffix_still_matches(self):
        self.assertEqual(keyword_search(self.pack(), "depremde")[0][0], "sarsinti")

    def test_no_usable_text_returns_nothing(self):
        self.assertEqual(keyword_search(self.pack(), "?! a"), [])

    def test_search_syntax_is_treated_as_text(self):
        self.assertEqual(keyword_search(self.pack(), 'deprem" OR body:*')[0][0], "sarsinti")


class Vectors(PackCase):
    def test_unit_truncates_and_scales(self):
        self.assertEqual(unit([3.0, 4.0, 9.0], 2), [0.6, 0.8])

    def test_closest_vector_first(self):
        results = vector_search(self.pack(), [0.0, 1.0])
        self.assertEqual(results[0][0], "su")
        self.assertAlmostEqual(results[0][1], 1.0)


class Merge(PackCase):
    def test_item_ranked_well_in_both_lists_wins(self):
        merged = merge_rankings([("a", 9), ("b", 8)], [("c", 1), ("b", 0.5)])
        self.assertEqual(merged[0][0], "b")

    def test_score_uses_position(self):
        self.assertAlmostEqual(merge_rankings([("a", 0)], k=60)[0][1], 1 / 61)


class Content(PackCase):
    def test_real_content_loads(self):
        items = load_items()
        self.assertEqual(len(items), 24)
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        build(conn, items)
        self.assertEqual(get_item(conn, "inme")["collection"], "resource")
        self.assertIsNone(get_item(conn, "missing"))


if __name__ == "__main__":
    unittest.main()
