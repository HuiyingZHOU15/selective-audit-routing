from decimal import Decimal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finmr_structured import decimal_value, parse_question, split_sections


def test_decimal_value_normalizes_common_financial_formats():
    assert decimal_value("1,234") == Decimal("1234")
    assert decimal_value("(1,234)") == Decimal("-1234")


def test_parse_question_supports_duration_and_instant():
    duration = "Question1: What's the reported value of us-gaap:X in the Instance document for the period 2024-01-01 to 2024-09-30?"
    instant = "Question1: What's the reported value of us-gaap:Y in the Instance document for the period 2024-09-30?"
    assert parse_question(duration).start == "2024-01-01"
    assert parse_question(duration).end == "2024-09-30"
    assert parse_question(instant).start is None
    assert parse_question(instant).end == "2024-09-30"


def test_split_sections():
    sections = split_sections("prefix\n##Instance document\n<a/>\n##US GAAP Taxonomy\nID: x")
    assert sections["instance document"] == "<a/>"
    assert sections["us gaap taxonomy"] == "ID: x"
