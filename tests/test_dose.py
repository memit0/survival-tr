"""Run with: python3 -m unittest discover -s tests. Expected values were calculated by hand."""
import os
import sys
import unittest
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prototype"))
from dose import DoseRefused, calculate, load_medicines  # noqa: E402

MEDICINES = load_medicines()


class Doses(unittest.TestCase):
    def test_per_kg(self):
        self.assertEqual(calculate(MEDICINES, "sahteril", 10, 20), ("Sahteril", Decimal("200.0")))

    def test_decimal_input_with_turkish_comma(self):
        self.assertEqual(calculate(MEDICINES, "denemol", "5", "12,5")[1], Decimal("31.3"))  # 31.25 rounds up

    def test_maximum_caps_the_dose(self):
        self.assertEqual(calculate(MEDICINES, "sahteril", 30, 90)[1], Decimal("500.0"))
        self.assertEqual(calculate(MEDICINES, "denemol", 30, 80)[1], Decimal("100.0"))

    def test_fixed_dose(self):
        self.assertEqual(calculate(MEDICINES, "ornekol", 12, 40), ("Örnekol", Decimal("100.0")))

    def test_limits_are_inclusive(self):
        self.assertEqual(calculate(MEDICINES, "sahteril", 1, 8)[1], Decimal("80.0"))

    def test_refusals(self):
        bad = [("yok", 10, 20), ("sahteril", 0, 20), ("sahteril", 10, 7), ("sahteril", 10, 201), ("sahteril", 121, 20),
               ("denemol", 1, 20), ("ornekol", 11, 50), ("ornekol", 20, 39), ("sahteril", None, 20), ("sahteril", 10, ""),
               ("sahteril", "abc", 20), ("sahteril", 10, "NaN"), ("sahteril", 10, "Infinity"), ("sahteril", True, 20),
               ("sahteril", 10, 20.5), ("sahteril", -3, 20)]
        for medicine, age, weight in bad:
            with self.subTest(medicine=medicine, age=age, weight=weight):
                with self.assertRaises(DoseRefused):
                    calculate(MEDICINES, medicine, age, weight)


if __name__ == "__main__":
    unittest.main()
