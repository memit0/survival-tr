"""Calculate test doses for invented medicines. The LLM is never involved. Never use this for a real person."""
import json
import os
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DoseRefused(Exception):
    """The input is missing, invalid, or outside the table's limits."""


def load_medicines(path=None):
    with open(path or os.path.join(ROOT, "content", "medicines.json"), encoding="utf-8") as f:
        return {medicine["id"]: medicine for medicine in json.load(f)["medicines"]}


def _number(value, label):
    if isinstance(value, (bool, float)) or value is None or value == "":  # floats can hide rounding errors
        raise DoseRefused(f"{label} gerekli ve sayı olmalı.")
    try:
        number = Decimal(str(value).strip().replace(",", "."))
    except InvalidOperation as error:
        raise DoseRefused(f"{label} sayı olmalı.") from error
    if not number.is_finite() or number <= 0:
        raise DoseRefused(f"{label} sıfırdan büyük olmalı.")
    return number


def calculate(medicines, medicine_id, age, weight_kg):
    """Returns (medicine name, mg as Decimal). Raises DoseRefused rather than guessing."""
    medicine = medicines.get(medicine_id)
    if medicine is None:
        raise DoseRefused("Bu ilaç test tablosunda yok.")
    age = _number(age, "Yaş")
    weight = _number(weight_kg, "Kilo")
    if not medicine["min_age_years"] <= age <= medicine["max_age_years"]:
        raise DoseRefused(f"Yaş {medicine['min_age_years']} ile {medicine['max_age_years']} arasında olmalı.")
    if not Decimal(medicine["min_weight_kg"]) <= weight <= Decimal(medicine["max_weight_kg"]):
        raise DoseRefused(f"Kilo {medicine['min_weight_kg']} ile {medicine['max_weight_kg']} kg arasında olmalı.")
    if medicine["kind"] == "fixed":
        mg = Decimal(medicine["fixed_mg"])
    else:
        mg = min(weight * Decimal(medicine["mg_per_kg"]), Decimal(medicine["max_mg"]))
    return medicine["name"], mg.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
