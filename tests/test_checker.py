"""Run with: python3 -m unittest discover -s tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prototype"))
from checker import check, has_negation, numbers  # noqa: E402

SOURCE = {"S1": "Jeneratörü kapı açık olsa bile asla garajda çalıştırmayın.",
          "S2": "Jeneratörü yalnızca dışarıda, evden en az 6 metre uzakta kullanın.",
          "S3": "Buzdolabı yiyecekleri yaklaşık dört saat güvende tutar."}


class Checker(unittest.TestCase):
    def test_faithful_steps_pass(self):
        output = "1. [S1] Kapı açık olsa bile jeneratörü asla garajda çalıştırmayın.\n" \
                 "2. [S2] Jeneratörü dışarıda, evden en az 6 metre uzakta kullanın.\n"
        self.assertEqual(check(output, SOURCE, ["S1"]), (True, []))

    def test_lost_negation_fails(self):
        passed, reasons = check("1. [S1] Kapı açıksa jeneratörü garajda çalıştırın.", SOURCE)
        self.assertFalse(passed)
        self.assertIn("S1: negation changed", reasons)

    def test_changed_number_fails(self):
        passed, reasons = check("1. [S2] Jeneratörü evden en az 3 metre uzakta kullanın.", SOURCE)
        self.assertFalse(passed)
        self.assertIn("S2: number not in source: ['3']", reasons)

    def test_dropped_number_fails(self):
        passed, reasons = check("1. [S2] Jeneratörü dışarıda, evden uzakta kullanın.", SOURCE)
        self.assertFalse(passed)
        self.assertIn("S2: number dropped: ['6']", reasons)

    def test_number_word_matches_digit(self):
        self.assertEqual(numbers("dört saat"), numbers("4 saat"))
        self.assertTrue(check("1. [S3] Buzdolabı yiyecekleri yaklaşık 4 saat güvende tutar.", SOURCE)[0])

    def test_made_up_citation_fails(self):
        self.assertIn("S9: cited sentence doesn't exist", check("1. [S9] Evden çıkın ve bekleyin.", SOURCE)[1])

    def test_missing_required_warning_fails(self):
        passed, reasons = check("1. [S2] Jeneratörü dışarıda, evden en az 6 metre uzakta kullanın.", SOURCE, ["S1"])
        self.assertFalse(passed)
        self.assertIn("S1: required warning missing", reasons)

    def test_uncited_text_fails(self):
        self.assertFalse(check("Jeneratörü dışarıda kullanın.", SOURCE)[0])

    def test_invented_wording_fails(self):
        passed, reasons = check("1. [S3] Telefon internet bağlantısı kurup uygulamayı güncelleyin.", SOURCE)
        self.assertFalse(passed)
        self.assertIn("S3: wording too far from source", reasons)

    def test_word_fragments_fail(self):
        passed, reasons = check("1. [S1] Jeneratörü, kapı, açık, olsa, bile, asla, garajda, çalıştırmayın.", SOURCE)
        self.assertFalse(passed)
        self.assertIn("S1: broken into word fragments", reasons)

    def test_repeated_step_fails(self):
        step = "Kapı açık olsa bile jeneratörü asla garajda çalıştırmayın."
        self.assertIn("S1: cited twice", check(f"1. [S1] {step}\n2. [S1] {step}", SOURCE)[1])

    def test_filler_marks_fail(self):
        self.assertIn("S1: filler marks", check("1. [S1] Jeneratörü (jeneratörü) asla garajda çalıştırmayın.", SOURCE)[1])

    def test_bold_marks_are_ignored(self):
        self.assertTrue(check("1. [S1] Kapı açık olsa **bile** jeneratörü **asla** garajda çalıştırmayın.", SOURCE)[0])

    def test_negation_patterns(self):
        for text in ("Tatmayın.", "Girmemelisiniz.", "Bu güvenli değil.", "Asla dokunma."):
            self.assertTrue(has_negation(text), text)
        self.assertFalse(has_negation("Kapıyı açın ve dışarı çıkın."))


if __name__ == "__main__":
    unittest.main()
