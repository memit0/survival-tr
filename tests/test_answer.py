"""Tests through the one entry point, using fake models. Run with: python3 -m unittest discover -s tests"""
import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prototype"))
from answer import Models, Request, answer  # noqa: E402
from dose import load_medicines  # noqa: E402
from llm import ModelError  # noqa: E402
from pack import build  # noqa: E402

ITEMS = [
    {"id": "su", "title": "İçme suyu", "severity": "caution", "required": ["S1"], "pack_date": "2026-10-31", "collection": "card",
     "sentences": {"S1": "Musluk suyunu içmeyin.", "S2": "Suyu en az 1 dakika kaynatın."}},
    {"id": "inme", "title": "İnme", "severity": "critical", "required": ["S1"], "pack_date": "2026-10-31", "collection": "resource",
     "sentences": {"S1": "Hemen 112'yi arayın."}},
]
MEDICINES = load_medicines()


def fake_embed_document(title, text):
    return [1.0, 0.0] if title == "İçme suyu" else [0.0, 1.0]


def make_pack():
    conn = sqlite3.connect(":memory:")
    build(conn, ITEMS, fake_embed_document)
    return conn


def models(reply, vector=(1.0, 0.0)):
    def chat(system, user, grammar):
        if isinstance(reply, Exception):
            raise reply
        return reply
    return Models(embed=lambda text: list(vector), chat=chat)


GOOD = "1. [S1] Musluk suyunu içmeyin."


def diagnosis(resource="inme", condition="Olası inme", urgency="ACİL", reason="[S1] Hemen 112'yi arayın.", step="112'yi arayın."):
    return f"KİMLİK: {resource}\nDURUM: {condition}\nACİLİYET: {urgency}\nNEDEN: {reason}\nADIM: {step}"


class PackCase(unittest.TestCase):
    def pack(self):
        conn = make_pack()
        self.addCleanup(conn.close)
        return conn


class Ask(PackCase):
    def ask(self, reply, text="musluk suyu", **kwargs):
        return answer(Request("ask", text=text), self.pack(), models(reply, **kwargs), MEDICINES)

    def test_answer_includes_source(self):
        result = self.ask(GOOD)
        self.assertEqual(result.kind, "answer")
        self.assertEqual(result.body, "Musluk suyunu içmeyin.")
        self.assertEqual(result.sources[0]["id"], "su")
        self.assertEqual(result.warnings, [])
        self.assertTrue(result.test_only)

    def test_critical_question_still_gets_llm_answer(self):
        reply = "1. [S1] Hemen 112'yi arayın."
        result = self.ask(reply, text="inme", vector=(0.0, 1.0))
        self.assertEqual(result.kind, "answer")

    def test_unfaithful_answer_is_shown_with_a_warning(self):
        result = self.ask("1. [S1] Musluk suyunu için.")
        self.assertEqual(result.kind, "answer")
        self.assertTrue(any("negation changed" in warning for warning in result.warnings))

    def test_weak_match_still_answers_but_is_labeled(self):
        result = self.ask(GOOD, vector=(0.1, 0.1))
        self.assertEqual(result.kind, "answer")
        self.assertTrue(result.low_match)

    def test_model_failures_fall_back_to_source_text(self):
        for reply in (ModelError("down"), "", "serbest metin", "1. [S99] Uydurma"):
            with self.subTest(reply=str(reply)):
                result = self.ask(reply)
                self.assertEqual(result.kind, "source_only")
                self.assertIn("Musluk suyunu içmeyin.", result.body)

    def test_embedding_failure_uses_keywords(self):
        def broken(text):
            raise ModelError("down")
        pack_models = Models(embed=broken, chat=lambda system, user, grammar: GOOD)
        result = answer(Request("ask", text="musluk suyu"), self.pack(), pack_models, MEDICINES)
        self.assertEqual(result.kind, "answer")
        self.assertTrue(result.low_match)
        self.assertTrue(result.warnings)

    def test_no_models_at_all_still_shows_keyword_match(self):
        result = answer(Request("ask", text="musluk suyu"), self.pack(), Models(), MEDICINES)
        self.assertEqual(result.kind, "source_only")

    def test_no_match_without_models_is_an_error_not_a_crash(self):
        self.assertEqual(answer(Request("ask", text="?!"), self.pack(), Models(), MEDICINES).kind, "error")

    def test_invalid_text_is_rejected(self):
        for text in (None, "", "   ", 5, "a" * 1001):
            with self.subTest(text=text):
                self.assertEqual(self.ask(GOOD, text=text).kind, "error")


class Diagnose(PackCase):
    def run_case(self, reply, vector=(0.0, 1.0)):
        return answer(Request("diagnose", text="yüzü düştü"), self.pack(), models(reply, vector=vector), MEDICINES)

    def test_returns_diagnosis_with_chosen_source_first(self):
        result = self.run_case(diagnosis())
        self.assertEqual(result.kind, "possible_diagnosis")
        self.assertEqual(result.sources[0]["id"], "inme")
        self.assertIn("Olası inme", result.body)

    def test_urgency_words_are_read_with_or_without_turkish_letters(self):
        self.assertEqual(self.run_case(diagnosis(urgency="KISA SÜREDE")).data["urgency"], "urgent")
        self.assertEqual(self.run_case(diagnosis(urgency="HAFIF")).data["urgency"], "minor")

    def test_invalid_diagnosis_falls_back(self):
        for reply in (diagnosis(resource="yok"), diagnosis(urgency="kesin"), diagnosis(condition=""),
                      diagnosis(reason="[S99] Uydurma"), diagnosis(reason="kaynaksız"), "serbest metin"):
            with self.subTest(reply=reply):
                self.assertEqual(self.run_case(reply).kind, "source_only")

    def test_reason_from_another_resource_warns(self):
        reply = diagnosis(reason="[S2] Musluk suyunu içmeyin.")
        self.assertTrue(any("seçilen kaynaktan" in warning for warning in self.run_case(reply).warnings))


class Dose(PackCase):
    def test_calculates_without_any_model(self):
        result = answer(Request("dose", medicine="sahteril", age="10", weight_kg="20"), self.pack(), Models(), MEDICINES)
        self.assertEqual(result.kind, "dose_result")
        self.assertEqual(result.data["mg"], "200.0")
        self.assertIn("uydurma", result.body)

    def test_refuses_bad_input(self):
        result = answer(Request("dose", medicine="aspirin", age="10", weight_kg="20"), self.pack(), Models(), MEDICINES)
        self.assertEqual(result.kind, "dose_refused")


class Mode(PackCase):
    def test_unknown_mode_is_an_error(self):
        self.assertEqual(answer(Request("delete"), self.pack(), Models(), MEDICINES).kind, "error")


if __name__ == "__main__":
    unittest.main()
