"""PIT guard for manual catalyst overrides.

Rows without source_filed_at used to apply on every as_of_date. They are
skipped. The production file is checked so a row this rule would drop is
visible in the test failure.
"""

from __future__ import annotations

import json
from pathlib import Path

from module_5_composite_v3 import apply_catalyst_overrides, catalyst_override_pit_reason

ROOT = Path(__file__).resolve().parent.parent
PRODUCTION = ROOT / "production_data" / "catalyst_overrides.json"
AS_OF = "2026-11-20"


def _summary(confidence: str = "LOW") -> dict:
    return {"integration": {"catalyst_confidence": confidence}, "scores": {}}


def test_missing_source_filed_at_is_skipped_and_not_applied():
    catalysts = {"CELC": _summary()}
    overrides = [{"ticker": "CELC", "confidence": 0.9, "event_type": "TRIAL_READOUT"}]
    applied, rejected = apply_catalyst_overrides(catalysts, overrides, AS_OF)
    assert applied == 0
    assert rejected == 1
    assert "catalyst_override_applied" not in catalysts["CELC"]["integration"]
    assert catalyst_override_pit_reason(overrides[0], AS_OF) == "missing source_filed_at"


def test_future_source_filed_at_is_skipped():
    catalysts = {"CELC": _summary()}
    overrides = [{"ticker": "CELC", "confidence": 0.9, "source_filed_at": "2026-12-01"}]
    applied, rejected = apply_catalyst_overrides(catalysts, overrides, AS_OF)
    assert applied == 0 and rejected == 1
    assert catalysts["CELC"]["integration"]["catalyst_confidence"] == "LOW"


def test_dated_override_still_applies():
    catalysts = {"CELC": _summary()}
    overrides = [
        {
            "ticker": "CELC",
            "confidence": 0.75,
            "source_filed_at": "2024-11-07",
            "event_type": "TRIAL_READOUT",
            "source": "filing",
        }
    ]
    applied, rejected = apply_catalyst_overrides(catalysts, overrides, AS_OF)
    assert applied == 1 and rejected == 0
    assert catalysts["CELC"]["integration"]["catalyst_confidence"] == "HIGH"
    assert catalysts["CELC"]["integration"]["catalyst_override_applied"] is True


def test_unparseable_source_filed_at_is_skipped():
    reason = catalyst_override_pit_reason({"source_filed_at": "not-a-date"}, AS_OF)
    assert reason is not None and reason.startswith("unparseable")


def test_production_catalyst_overrides_none_dropped():
    """Every row in the file the loader reads has a source_filed_at on or before 2026-11-20.

    #613 added COGT and IMCR, both dated 2026-11-07. CELC is dated 2024-11-07.
    A row this rule would drop fails the assertion and is named in the message.
    """
    payload = json.loads(PRODUCTION.read_text(encoding="utf-8"))
    dropped = []
    for row in payload["overrides"]:
        reason = catalyst_override_pit_reason(row, AS_OF)
        if reason:
            dropped.append(f"{row.get('ticker')}: {reason}")
    assert dropped == [], "catalyst overrides this rule would drop: " + "; ".join(dropped)
    assert [row["ticker"] for row in payload["overrides"]] == ["CELC", "COGT", "IMCR"]

    # The other candidate paths the loader checks are not present in the repo.
    assert not (ROOT / "data" / "catalyst_overrides.json").exists()
