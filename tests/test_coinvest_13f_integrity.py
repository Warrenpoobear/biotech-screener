"""Coinvest 13F integrity fixes (operator-authorized clock reset, 2026-11-01).

Covers three defects in the 13F -> coinvest_signals.json path:

1. Staleness: the PIT cache keeps a manager's latest filing forever, so a
   manager that stops filing (or moves to a new filer CIK, e.g. Broadfin)
   keeps contributing months-old positions.  Filings whose period_of_report
   is older than MAX_REPORT_AGE_DAYS before as_of are now excluded.
2. Firm grouping: separate filer CIKs of one firm (Foresite IV/V/VI, the old
   and new Broadfin CIKs) were counted as independent holders.  Members of a
   firm_group now count once, using only the group's latest report period.
3. Filing age: coinvest_signals.json stored days_since_latest_filing as of
   the build date, so every later run read a frozen age.  The builder now
   emits latest_filed_at and run_screen recomputes the age at load time.

Leakage: the data_available_timestamp of a 13F holding is its filed_at.
Every rule here only looks backward from as_of (filed_at <= as_of).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scripts.build_coinvest_features_from_13f import MAX_REPORT_AGE_DAYS, build_coinvest_features

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

UNIVERSE = {"AAAA", "BBBB", "CCCC"}

CIK_FS4 = "0001704132"  # Foresite Capital IV
CIK_FS5 = "0001792206"  # Foresite Capital V
CIK_BF_OLD = "0001601692"  # Broadfin (Kotler) - stopped filing after Q4 2025
CIK_BF_NEW = "0002110839"  # Broadfin Holdings LLC - files from Q1 2026
CIK_BAKER = "0001263508"


def _holding(ticker, value):
    return {
        "cusip": "000000000",
        "ticker": ticker,
        "issuer": f"{ticker} Inc",
        "shares": value * 10,
        "value_usd_thousands": value,
        "put_call": "",
    }


def _write_cache(tmp_path, as_of_date, managers):
    """Write a PIT cache where each manager has its own period / filing date.

    managers: list of (cik, name, holdings, period_of_report, filed_at)
    """
    date_dir = tmp_path / as_of_date
    (date_dir / "managers").mkdir(parents=True, exist_ok=True)
    entries = []
    for cik, name, holdings, por, filed in managers:
        entries.append(
            {
                "manager_cik": cik,
                "manager_name": name,
                "selected": True,
                "period_of_report": por,
                "filed_at": filed,
                "form_type": "13F-HR",
                "accession": "0000000000-00-000000",
                "holdings_count": len(holdings),
                "manager_json_path": f"managers/{cik}.json",
                "raw_xml_path": "",
                "rejection_reason": "",
            }
        )
        (date_dir / "managers" / f"{cik}.json").write_text(
            json.dumps(
                {
                    "manager_cik": cik,
                    "manager_name": name,
                    "as_of_date": as_of_date,
                    "period_of_report": por,
                    "filed_at": filed,
                    "form_type": "13F-HR",
                    "accession": "0000000000-00-000000",
                    "holdings": holdings,
                }
            )
        )
    index = {
        "schema_version": "sec_13f_pit_index.v1",
        "cache_type": "sec_13f_pit",
        "as_of_date": as_of_date,
        "created_at": "2026-10-30T12:00:00Z",
        "elite_only": True,
        "total_managers": len(managers),
        "managers_with_filing": len(managers),
        "coverage_pct": 100.0,
        "selection_policy": {"pit_rule": "test", "recent_filings_lookback_n": 8},
        "managers": sorted(entries, key=lambda m: m["manager_cik"]),
    }
    (date_dir / "index.json").write_text(json.dumps(index))
    return tmp_path


def _build(tmp_path, as_of, managers, groups=None, **kw):
    _write_cache(tmp_path, as_of, managers)
    tiers = {m[0]: 1 for m in managers}
    names = {m[0]: m[1] for m in managers}
    return build_coinvest_features(
        as_of_date=as_of,
        cache_root=tmp_path,
        universe_tickers=UNIVERSE,
        manager_tiers=tiers,
        manager_names=names,
        manager_groups=groups or {},
        no_prior=True,
        **kw,
    )


# ===================================================================
# 1. Staleness gate
# ===================================================================


class TestStalenessGate:
    def test_max_report_age_covers_one_missed_quarter(self):
        # Latest legitimate filing can be ~137d old just before a deadline;
        # one late/missed quarter adds ~92d. Anything older is abandoned.
        assert 229 <= MAX_REPORT_AGE_DAYS <= 240

    def test_abandoned_filer_excluded(self, tmp_path):
        out = _build(
            tmp_path,
            "2026-10-30",
            [
                (
                    CIK_BF_OLD,
                    "Broadfin",
                    [_holding("CCCC", 500)],
                    "2025-12-31",
                    "2026-02-17",
                ),
                (
                    CIK_BAKER,
                    "Baker Bros",
                    [_holding("AAAA", 500)],
                    "2026-06-30",
                    "2026-08-14",
                ),
            ],
        )
        assert "CCCC" not in out["tickers"]
        assert out["tickers"]["AAAA"]["coinvest_holders"] == ["Baker Bros"]
        excluded = out["provenance"]["stale_managers_excluded"]
        assert [m["cik"] for m in excluded] == [CIK_BF_OLD]

    def test_one_quarter_late_filer_kept(self, tmp_path):
        # Q1 period, 213 days before as_of: late but not abandoned.
        out = _build(
            tmp_path,
            "2026-10-30",
            [
                (
                    CIK_BAKER,
                    "Baker Bros",
                    [_holding("AAAA", 500)],
                    "2026-03-31",
                    "2026-05-15",
                )
            ],
        )
        assert "AAAA" in out["tickers"]
        assert out["provenance"]["stale_managers_excluded"] == []

    def test_gate_can_be_disabled(self, tmp_path):
        out = _build(
            tmp_path,
            "2026-10-30",
            [
                (
                    CIK_BF_OLD,
                    "Broadfin",
                    [_holding("CCCC", 500)],
                    "2025-12-31",
                    "2026-02-17",
                )
            ],
            max_report_age_days=None,
        )
        assert "CCCC" in out["tickers"]


# ===================================================================
# 2. Firm grouping
# ===================================================================


class TestFirmGrouping:
    def test_group_members_count_as_one_holder(self, tmp_path):
        out = _build(
            tmp_path,
            "2026-10-30",
            [
                (
                    CIK_FS4,
                    "Foresite Capital IV",
                    [_holding("AAAA", 300), _holding("BBBB", 700)],
                    "2026-06-30",
                    "2026-08-14",
                ),
                (
                    CIK_FS5,
                    "Foresite Capital V",
                    [_holding("AAAA", 200)],
                    "2026-06-30",
                    "2026-08-14",
                ),
            ],
            groups={CIK_FS4: "Foresite Capital", CIK_FS5: "Foresite Capital"},
        )
        a = out["tickers"]["AAAA"]
        assert a["coinvest_overlap_count"] == 1
        assert a["tier1_count"] == 1
        assert a["coinvest_holders"] == ["Foresite Capital"]
        # Position % uses combined holdings over combined book: 500 / 1200.
        assert a["max_tier1_position_pct"] == pytest.approx(500 / 1200 * 100, abs=0.01)

    def test_group_uses_latest_period_only(self, tmp_path):
        # Old Broadfin CIK (Q1) vs new CIK (Q2): only the Q2 book counts,
        # even though the Q1 filing is inside the staleness window.
        out = _build(
            tmp_path,
            "2026-10-30",
            [
                (
                    CIK_BF_OLD,
                    "Broadfin",
                    [_holding("CCCC", 500)],
                    "2026-03-31",
                    "2026-05-15",
                ),
                (
                    CIK_BF_NEW,
                    "Broadfin Holdings",
                    [_holding("AAAA", 500)],
                    "2026-06-30",
                    "2026-08-14",
                ),
            ],
            groups={CIK_BF_OLD: "Broadfin Capital", CIK_BF_NEW: "Broadfin Capital"},
        )
        assert "CCCC" not in out["tickers"]
        assert out["tickers"]["AAAA"]["coinvest_holders"] == ["Broadfin Capital"]

    def test_ungrouped_managers_unchanged(self, tmp_path):
        out = _build(
            tmp_path,
            "2026-10-30",
            [
                (
                    CIK_FS4,
                    "Foresite Capital IV",
                    [_holding("AAAA", 300)],
                    "2026-06-30",
                    "2026-08-14",
                ),
                (
                    CIK_BAKER,
                    "Baker Bros",
                    [_holding("AAAA", 300)],
                    "2026-06-30",
                    "2026-08-14",
                ),
            ],
        )
        assert out["tickers"]["AAAA"]["coinvest_overlap_count"] == 2


class TestRosterFilter:
    def test_cache_manager_off_roster_excluded(self, tmp_path):
        # PIT caches written before a roster change still list retired
        # managers; only CIKs in manager_tiers (the active roster) count.
        _write_cache(
            tmp_path,
            "2026-10-30",
            [
                (CIK_BAKER, "Baker Bros", [_holding("AAAA", 500)], "2026-06-30", "2026-08-14"),
                (CIK_FS4, "Foresite Capital IV", [_holding("BBBB", 500)], "2026-06-30", "2026-08-14"),
            ],
        )
        out = build_coinvest_features(
            as_of_date="2026-10-30",
            cache_root=tmp_path,
            universe_tickers=UNIVERSE,
            manager_tiers={CIK_BAKER: 1},
            manager_names={CIK_BAKER: "Baker Bros"},
            no_prior=True,
        )
        assert "BBBB" not in out["tickers"]
        assert [m["cik"] for m in out["provenance"]["managers_not_in_roster"]] == [CIK_FS4]


# ===================================================================
# 3. Filing age recomputed at load time
# ===================================================================


class TestFilingAgeAtLoad:
    def test_builder_emits_latest_filed_at(self, tmp_path):
        out = _build(
            tmp_path,
            "2026-08-21",
            [
                (
                    CIK_FS4,
                    "Foresite Capital IV",
                    [_holding("AAAA", 300)],
                    "2026-06-30",
                    "2026-08-11",
                ),
                (
                    CIK_BAKER,
                    "Baker Bros",
                    [_holding("AAAA", 300)],
                    "2026-06-30",
                    "2026-08-14",
                ),
            ],
        )
        assert out["tickers"]["AAAA"]["latest_filed_at"] == "2026-08-14"
        assert out["tickers"]["AAAA"]["days_since_latest_filing"] == 7

    def test_run_screen_recomputes_age(self):
        from run_screen import _refresh_coinvest_filing_age

        signals = {
            "AAAA": {
                "latest_filed_at": "2026-08-14",
                "days_since_latest_filing": 7,
                "coinvest_recency_state": "fresh",
            },
            "LEGACY": {
                "days_since_latest_filing": 7,
                "coinvest_recency_state": "fresh",
            },
        }
        _refresh_coinvest_filing_age(signals, "2026-10-30")
        assert signals["AAAA"]["days_since_latest_filing"] == 77
        assert signals["AAAA"]["coinvest_recency_state"] == "fresh"
        # Entries without latest_filed_at are left as built.
        assert signals["LEGACY"]["days_since_latest_filing"] == 7

    def test_run_screen_age_marks_stale(self):
        from run_screen import _refresh_coinvest_filing_age

        signals = {"AAAA": {"latest_filed_at": "2026-05-15", "days_since_latest_filing": 7}}
        _refresh_coinvest_filing_age(signals, "2026-10-30")
        assert signals["AAAA"]["days_since_latest_filing"] == 168
        assert signals["AAAA"]["coinvest_recency_state"] == "stale"

    def test_future_filing_never_yields_negative_age(self):
        from run_screen import _refresh_coinvest_filing_age

        signals = {"AAAA": {"latest_filed_at": "2026-11-14", "days_since_latest_filing": 0}}
        _refresh_coinvest_filing_age(signals, "2026-10-30")
        assert signals["AAAA"]["days_since_latest_filing"] == 0


# ===================================================================
# 4. Registry: firm groups declared (on active or retired entries)
# ===================================================================

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "production_data" / "manager_registry.json"


@pytest.fixture(scope="module")
def registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _all_entries(registry):
    out = {}
    for section in ("elite_core", "conditional", "retired"):
        for m in registry.get(section, []):
            out[m["cik"].zfill(10)] = m
    return out


class TestRegistryGroups:
    def test_broadfin_holdings_registered(self, registry):
        assert CIK_BF_NEW in _all_entries(registry)

    def test_broadfin_ciks_share_group(self, registry):
        e = _all_entries(registry)
        group = e[CIK_BF_OLD].get("firm_group")
        assert group and e[CIK_BF_NEW].get("firm_group") == group

    def test_foresite_funds_share_group(self, registry):
        e = _all_entries(registry)
        groups = {e[c].get("firm_group") for c in (CIK_FS4, CIK_FS5, "0002013341")}
        assert len(groups) == 1 and all(groups)


# ===================================================================
# 5. Coinvest roster (operator-specified, 2026-11-01)
# ===================================================================

ROSTER_2026_11 = {
    "0002018299",  # Boxer
    "0001727492",  # Caligan
    "0001802528",  # Fairmount
    "0001910456",  # Lynx1 (new)
    "0001831942",  # Commodore (new)
    "0001615982",  # Venrock Healthcare (new; filer Venrock Adviser LLC)
    "0001940272",  # ADAR1 (new)
    "0002005245",  # Siren
    "0001974915",  # Vestal Point
    "0001674712",  # Vivo
    "0001595855",  # Opaleye
    "0001907884",  # Kynam
    "0001892134",  # Frazier
    "0001792126",  # Logos
    "0001839435",  # Octagon (new)
    "0001839948",  # TCG Crossover
    "0001009258",  # Deerfield
    "0001703031",  # Bain Capital Life Sciences
    "0001631134",  # Sofinnova
    "0001773206",  # Integral
    "0001855655",  # Paradigm
    "0001773195",  # Affinity
    "0001856083",  # Deep Track
    "0001263508",  # Baker Bros
    "0001224962",  # Perceptive
    "0001633313",  # Avoro
    "0001569064",  # Suvretta (operator addition)
    "0001056807",  # BVF (operator addition)
    "0001389933",  # DAFNA (operator addition)
    "0001687078",  # MPM BioImpact (operator addition, new)
}


class TestRoster:
    def test_elite_core_is_operator_roster(self, registry):
        assert {m["cik"].zfill(10) for m in registry["elite_core"]} == ROSTER_2026_11

    def test_conditional_tier_unchanged(self, registry):
        assert len(registry["conditional"]) == 6

    def test_removed_managers_retired_not_deleted(self, registry):
        retired = {m["cik"].zfill(10): m for m in registry.get("retired", [])}
        assert len(retired) == 26
        assert {CIK_FS4, CIK_FS5, CIK_BF_OLD, CIK_BF_NEW} <= set(retired)
        assert all(m.get("retired_date") == "2026-11-01" for m in retired.values())
        assert not set(retired) & ROSTER_2026_11

    def test_retired_managers_excluded_from_signal(self):
        from elite_managers import get_all_managers

        active = {m["cik"].zfill(10) for m in get_all_managers()}
        assert len(active) == 30 + 6
        assert CIK_FS4 not in active and CIK_BF_OLD not in active
