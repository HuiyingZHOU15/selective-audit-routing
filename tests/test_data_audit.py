from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_data import aggregate_findings, normalize_text


def test_normalize_text_handles_curly_punctuation():
    assert normalize_text("Issuer’s  Revenue—Test") == "issuer s revenue test"


def test_aggregate_findings_collapses_standard_rows():
    common = {
        "Inspection Type": "Annual",
        "Registration ID": "1",
        "Firm Names": "Firm",
        "Inspection Year": "2025",
        "Country": "US",
        "Global Network": "Network",
        "Inspection Report Date": "1-Jan-26",
        "Inspection Report Section": "Part I.A",
        "Issuer Reference Key": "A",
        "Firm played a role but was not the lead auditor": "No",
        "Audits Affected by the Deficiencies Identified in Part I.A": "FS",
        "Classification of Audits with Part I.A Deficiencies": "Multiple",
        "Audit Area": "Revenue",
        "Firm-Identified Risk Assessment": "Fraud Risk",
        "Finding Count": "1",
        "Description in the Firm's Inspection Report": "Same description.",
    }
    rows = [
        {**common, "Auditing Standard": "AS 1105", "Paragraph of the Auditing Standard": "10"},
        {**common, "Auditing Standard": "AS 2301", "Paragraph of the Auditing Standard": "8"},
    ]
    result = aggregate_findings(pd.DataFrame(rows))
    assert len(result) == 1
    assert result.iloc[0]["source_row_count"] == 2
    assert result.iloc[0]["standard_refs"] == "AS 1105 | AS 2301"


def test_aggregate_findings_preserves_multiple_audit_areas():
    common = {
        "Inspection Type": "Annual",
        "Registration ID": "1",
        "Firm Names": "Firm",
        "Inspection Year": "2025",
        "Country": "US",
        "Global Network": "Network",
        "Inspection Report Date": "1-Jan-26",
        "Inspection Report Section": "Part I.A",
        "Issuer Reference Key": "A",
        "Firm played a role but was not the lead auditor": "No",
        "Audits Affected by the Deficiencies Identified in Part I.A": "FS",
        "Classification of Audits with Part I.A Deficiencies": "Multiple",
        "Firm-Identified Risk Assessment": "",
        "Finding Count": "1",
        "Description in the Firm's Inspection Report": "One finding with two areas.",
        "Auditing Standard": "AS 1105",
        "Paragraph of the Auditing Standard": "10",
    }
    rows = [
        {**common, "Audit Area": "Revenue"},
        {**common, "Audit Area": "Accounts Receivable"},
    ]
    result = aggregate_findings(pd.DataFrame(rows))
    assert len(result) == 1
    assert result.iloc[0]["audit_area_count"] == 2
    assert result.iloc[0]["audit_areas"] == "Accounts Receivable | Revenue"
