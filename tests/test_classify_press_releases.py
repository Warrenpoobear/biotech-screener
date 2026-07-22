"""Tests for classify_press_releases.py and fetch date extraction."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.classify_press_releases import (
    _BIOTECH_RESCUE_INDICATORS,
    _BIOTECH_RESCUE_MIN_MATCHES,
    _classify_locally,
    _collision_counterfactual_would_rescue,
    _count_biotech_indicator_matches,
    _is_noise,
    _is_ticker_collision,
    _load_company_names,
    classify_releases,
)
from tools.fetch_company_press_releases import _extract_date_from_context

# ---------------------------------------------------------------------------
# Noise filter
# ---------------------------------------------------------------------------


class TestIsNoise:
    def test_law_firm_alert(self):
        assert _is_noise("INVESTOR ALERT: Pomerantz Law Firm Investigates Claims On Behalf of Investors")

    def test_class_action(self):
        assert _is_noise("Securities Class Action Filed Against XYZ Corp")

    def test_market_report(self):
        assert _is_noise("Lithium Metal Battery Materials Market Volume Worth 1,549,218 tons by 2035")

    def test_global_market_forecast(self):
        assert _is_noise("Global Market Research Report on Oncology Therapeutics 2025-2030")

    def test_joint_venture_analysis(self):
        assert _is_noise("Pharmaceuticals and Biotechnology Joint Venture Agreements Analysis Report 2025")

    def test_inducement_grant(self):
        assert _is_noise("4DMT Announces New Employment Inducement Grants")

    def test_hagens_berman(self):
        assert _is_noise("QURE ALERT: Hagens Berman Updates uniQure Investigation")

    def test_legitimate_clinical_not_noise(self):
        assert not _is_noise("Kodiak Sciences Announces Positive Topline Results in Phase 3 GLOW2 Trial")

    def test_legitimate_regulatory_not_noise(self):
        assert not _is_noise("FDA Approves New High Dose Regimen of SPINRAZA")

    def test_legitimate_corporate_not_noise(self):
        assert not _is_noise("Centessa Pharmaceuticals Reports Fourth Quarter Financial Results")

    def test_empty_headline(self):
        assert not _is_noise("")


# ---------------------------------------------------------------------------
# Clinical classification
# ---------------------------------------------------------------------------


class TestClassifyClinical:
    def test_phase_3_topline(self):
        r = _classify_locally("Kodiak Sciences Announces Positive Topline Results in Phase 3 GLOW2 Trial")
        assert r["event_category"] == "clinical"
        assert r["event_outcome_guess"] == "hit"
        assert r["thesis_change_flag"] is True

    def test_phase_2_data(self):
        r = _classify_locally("Maze Therapeutics Announces Positive Topline Data from Phase 2 HORIZON Trial")
        assert r["event_category"] == "clinical"

    def test_phase_1_data(self):
        r = _classify_locally("Oruka Therapeutics Announces Positive Interim Phase 1 Data for ORKA-002")
        assert r["event_category"] == "clinical"

    def test_did_not_meet_endpoint(self):
        r = _classify_locally("Theravance Biopharma Reports Phase 3 CYPRESS Study Did Not Meet Primary Endpoint")
        assert r["event_category"] == "clinical"
        assert r["event_outcome_guess"] == "miss"

    def test_pivotal_keyword(self):
        r = _classify_locally("Company Announces Pivotal Trial Results for Drug X")
        assert r["event_category"] == "clinical"

    def test_clinical_confidence(self):
        r = _classify_locally("Phase 3 data readout expected next week")
        assert r["confidence"] == 0.6


# ---------------------------------------------------------------------------
# Regulatory classification
# ---------------------------------------------------------------------------


class TestClassifyRegulatory:
    def test_fda_approval(self):
        r = _classify_locally("FDA Approves New High Dose Regimen of SPINRAZA")
        assert r["event_category"] == "regulatory"

    def test_nda_acceptance(self):
        r = _classify_locally("Celcuity Announces FDA Acceptance New Drug Application")
        assert r["event_category"] == "regulatory"

    def test_breakthrough_designation(self):
        r = _classify_locally("Cogent Biosciences Announces Breakthrough Therapy Designation")
        assert r["event_category"] == "regulatory"

    def test_complete_response_letter(self):
        r = _classify_locally("Company Receives Complete Response Letter from FDA")
        assert r["event_category"] == "regulatory"

    def test_pdufa_keyword(self):
        r = _classify_locally("PDUFA date set for May 24, 2026")
        assert r["event_category"] == "regulatory"


# ---------------------------------------------------------------------------
# M&A classification
# ---------------------------------------------------------------------------


class TestClassifyMNA:
    def test_acquisition(self):
        r = _classify_locally("Lilly to acquire Centessa Pharmaceuticals to advance treatments")
        assert r["event_category"] == "mna"
        assert r["mna_signal_flag"] is True
        assert r["exogenous_to_primary_catalyst"] is True
        assert r["severity"] == "critical"

    def test_merger(self):
        r = _classify_locally("Company X and Company Y Announce Definitive Merger Agreement")
        assert r["event_category"] == "mna"

    def test_definitive_agreement(self):
        r = _classify_locally("Company Enters Into Definitive Agreement to Be Acquired at $38 Per Share")
        assert r["event_category"] == "mna"

    def test_tender_offer(self):
        r = _classify_locally("Pfizer Commences Tender Offer for All Outstanding Shares")
        assert r["event_category"] == "mna"


# ---------------------------------------------------------------------------
# Financing classification
# ---------------------------------------------------------------------------


class TestClassifyFinancing:
    def test_public_offering(self):
        r = _classify_locally("Centessa Pharmaceuticals Announces Pricing of $250,000,000 Public Offering")
        assert r["event_category"] == "financing"
        assert r["financing_signal_flag"] is True
        assert r["price_direction_guess"] == "down"

    def test_private_placement(self):
        r = _classify_locally("Company Announces $50 Million Private Placement of Common Stock")
        assert r["event_category"] == "financing"

    def test_atm_offering(self):
        r = _classify_locally("Company Enters Into At-The-Market Offering Agreement")
        assert r["event_category"] == "financing"

    def test_registered_direct(self):
        r = _classify_locally("Company Announces Pricing of Registered Direct Offering")
        assert r["event_category"] == "financing"


# ---------------------------------------------------------------------------
# Safety classification
# ---------------------------------------------------------------------------


class TestClassifySafety:
    def test_clinical_hold(self):
        r = _classify_locally("FDA Places Partial Clinical Hold on Phase 2 Trial of Drug X")
        assert r["event_category"] == "safety"
        assert r["safety_signal_flag"] is True
        assert r["severity"] == "critical"

    def test_adverse_event(self):
        r = _classify_locally("Company Reports Serious Adverse Event in Phase 3 Trial")
        assert r["event_category"] == "safety"

    def test_voluntary_recall(self):
        r = _classify_locally("Company Announces Voluntary Recall of Product Due to Contamination")
        assert r["event_category"] == "safety"


# ---------------------------------------------------------------------------
# Default / other classification
# ---------------------------------------------------------------------------


class TestClassifyOther:
    def test_generic_corporate(self):
        r = _classify_locally("Company Provides Business Update for Q1 2026")
        assert r["event_category"] == "other"
        assert r["confidence"] == 0.3

    def test_informational_conference(self):
        r = _classify_locally("Company to Present at Upcoming Healthcare Conference in March")
        assert r["event_category"] == "other"
        assert r["informational_only"] is True
        assert r["confidence"] == 0.7

    def test_informational_financial_results(self):
        r = _classify_locally("Sionna Therapeutics Reports Fourth Quarter and Full Year 2025 Financial Results")
        assert r["event_category"] == "other"
        assert r["informational_only"] is True


# ---------------------------------------------------------------------------
# Priority order (clinical > regulatory > mna > financing > safety > other)
# ---------------------------------------------------------------------------


class TestClassificationPriority:
    def test_safety_beats_clinical(self):
        """Safety checked before clinical — 'clinical hold on Phase 2' is safety."""
        r = _classify_locally("FDA Places Clinical Hold on Phase 2 Trial")
        assert r["event_category"] == "safety"

    def test_clinical_beats_regulatory(self):
        """When headline matches both clinical and regulatory keywords, clinical wins
        because it's checked first (after safety)."""
        r = _classify_locally("FDA Accepts Phase 3 Topline Data Submission")
        assert r["event_category"] == "clinical"

    def test_clinical_guard_beats_informational(self):
        """Clinical guard takes priority over informational suppression: a Phase 3
        data presentation is a substantive catalyst even when announced 'at an
        investor conference' (Spec remediation 2026-06-02). So despite matching
        the 'conference'/'investor'/'present' informational keywords, the clinical
        guard keeps informational_only=False."""
        r = _classify_locally("Company to Present Phase 3 Data at Investor Conference")
        assert r["informational_only"] is False


# ---------------------------------------------------------------------------
# Date extraction from HTML context
# ---------------------------------------------------------------------------


class TestExtractDateFromContext:
    def test_time_datetime_tag(self):
        html = '<div><time datetime="2026-03-15T08:00:00Z">March 15, 2026</time> <a href="/news">Link</a></div>'
        assert _extract_date_from_context(html, html.index("href")) == "2026-03-15"

    def test_data_date_attribute(self):
        html = '<div data-publish-date="2026-04-01"><a href="/press">Press Release</a></div>'
        assert _extract_date_from_context(html, html.index("href")) == "2026-04-01"

    def test_iso_date_in_text(self):
        html = '<span>Published: 2026-03-20</span> <a href="/news">Headline</a>'
        assert _extract_date_from_context(html, html.index("href")) == "2026-03-20"

    def test_us_date_format(self):
        html = '<span>March 15, 2026</span> <a href="/news">Headline text here</a>'
        assert _extract_date_from_context(html, html.index("href")) == "2026-03-15"

    def test_us_date_no_comma(self):
        html = '<span>January 8 2026</span> <a href="/news">Headline</a>'
        assert _extract_date_from_context(html, html.index("href")) == "2026-01-08"

    def test_abbreviated_month(self):
        html = '<span>Feb 3, 2026</span> <a href="/news">Headline</a>'
        assert _extract_date_from_context(html, html.index("href")) == "2026-02-03"

    def test_no_date_returns_empty(self):
        html = '<div><a href="/news">Just a headline with no date anywhere</a></div>'
        assert _extract_date_from_context(html, html.index("href")) == ""

    def test_old_year_ignored(self):
        """Dates from old years should be ignored (not recent)."""
        html = '<span>2018-05-15</span> <a href="/news">Old news</a>'
        assert _extract_date_from_context(html, html.index("href")) == ""

    def test_priority_datetime_over_text(self):
        """datetime attribute should win over text dates."""
        html = '<time datetime="2026-03-20">March 25, 2026</time> <a href="/news">Link</a>'
        # datetime attr (2026-03-20) should beat text date (March 25)
        assert _extract_date_from_context(html, html.index("href")) == "2026-03-20"


# ---------------------------------------------------------------------------
# Ticker collision detection
# ---------------------------------------------------------------------------


class TestIsTickerCollision:
    NAMES = {
        "SION": ["sionna"],
        "ACAD": ["acadia"],
        "PHVS": ["pharvaris"],
        "KURA": ["kura", "oncology"],
        "FDMT": ["molecular"],  # "4D Molecular Therapeutics"
        "BIIB": ["biogen"],
        "BBIO": ["bridgebio"],
    }

    def test_company_name_match(self):
        assert not _is_ticker_collision("Sionna Therapeutics Reports Q4 Results", "SION", self.NAMES)

    def test_ticker_in_headline(self):
        assert not _is_ticker_collision("BIIB Reports Strong Earnings", "BIIB", self.NAMES)

    def test_true_collision_market_report(self):
        assert _is_ticker_collision(
            "Lithium Metal Battery Materials Market Volume Worth 1,549,218 tons by 2035",
            "SION",
            self.NAMES,
        )

    def test_true_collision_unrelated_company(self):
        assert _is_ticker_collision(
            "Eclipse Group showcases new electric vehicles at auto show 2026",
            "KURA",
            self.NAMES,
        )

    def test_brand_name_not_flagged(self):
        """4DMT is FDMT's brand — headline starts with brand, should not be flagged."""
        assert not _is_ticker_collision(
            "4DMT Completes Enrollment for 4FRONT-1 Phase 3 Clinical Trial",
            "FDMT",
            self.NAMES,
        )

    def test_no_company_registered_with_biotech_signal_not_flagged(self):
        """CH-2: unknown ticker in registry — headline contains biotech indicators,
        so the biotech-rescue fires and the item is not flagged as collision."""
        assert not _is_ticker_collision(
            "Unknown Co Announces Positive Phase 3 Clinical Trial Results for Drug X",
            "ZZZZ",
            self.NAMES,
        )

    def test_no_company_registered_no_biotech_signal_flagged(self):
        """CH-2: unknown ticker AND no biotech content — flag as collision.
        Previous behavior short-circuited to 'no collision' for any ticker
        missing from company_ir_sources.json, which let non-biotech headlines
        through. Now the ticker-match + biotech-rescue still run."""
        assert _is_ticker_collision(
            "Something about UNKNOWN ticker",
            "ZZZZ",
            self.NAMES,
        )

    def test_drug_name_headline(self):
        """BridgeBio's drug BBP-418 — company name not in headline but biotech content."""
        assert not _is_ticker_collision(
            "BBP-418 Demonstrates Consistent Efficacy and Favorable Safety Profile in Phase 3",
            "BBIO",
            self.NAMES,
        )

    def test_techbob_collision(self):
        assert _is_ticker_collision(
            "Techbob Academy Launches Tech Elite Incubation Program",
            "ACAD",
            self.NAMES,
        )


# ---------------------------------------------------------------------------
# CH-1: HTML-entity handling in noise + collision matching
# ---------------------------------------------------------------------------


class TestHTMLEntityDecoding:
    """Pattern matching must decode HTML entities like &amp;, &#174;, &#231;
    before comparing, so noise patterns containing `&` catch real-world headlines.
    Confirmed live bug (2026-04-18): C04 'LEVI &amp; KORSINSKY' slipped the noise
    filter because `&amp;` was not decoded.
    """

    def test_amp_entity_in_noise_match(self):
        hl = "GOSS: Management Optimistic About Topline -- LEVI &amp; KORSINSKY, LLP Investigates"
        assert _is_noise(hl)

    def test_amp_entity_collision_match(self):
        names = {"FOO": ["foobar"]}
        # `&amp;` stands in where the ticker collision rule needs to see the ampersand
        # version of a non-biotech bond headline.
        hl = "Caisse Fran&#231;aise de Financement Local: EMTN 2026-2 SOCIAL"
        assert _is_ticker_collision(hl, "FOO", names)

    def test_numeric_entity_decoded_in_name_match(self):
        """Numeric entities like &#174; (registered mark) should not prevent a
        name-word match on the decoded form."""
        names = {"XYZ": ["nerlynx"]}
        assert not _is_ticker_collision(
            "Knight Therapeutics Announces Health Canada Approval for NERLYNX&#174; (Neratinib)",
            "XYZ",
            names,
        )


# ---------------------------------------------------------------------------
# CH-3: short-prefix-token retention in company-name extraction
# ---------------------------------------------------------------------------


class TestLoadCompanyNamesShortPrefix:
    """_load_company_names must retain a short first token (e.g., 'SAB' from
    'SAB Biotherapeutics') even when longer tokens survive the stop-word filter.
    Previously, 'SAB' was dropped by the >3-char rule, leaving only
    ['biotherapeutics'], which did not match headlines like 'SAB BIO Announces...'
    and caused real biotech events to be false-flagged as collisions.
    """

    def _load_from_entries(self, entries: list) -> dict:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "production_data").mkdir()
            (root / "production_data" / "company_ir_sources.json").write_text(json.dumps({"sources": entries}))
            import tools.classify_press_releases as mod

            orig = mod.PROJECT_ROOT
            mod.PROJECT_ROOT = root
            try:
                return _load_company_names()
            finally:
                mod.PROJECT_ROOT = orig

    def test_short_prefix_retained(self):
        names = self._load_from_entries(
            [
                {"ticker": "SABS", "company": "SAB Biotherapeutics Inc"},
            ]
        )
        assert "sab" in names.get("SABS", [])

    def test_long_first_token_unchanged(self):
        names = self._load_from_entries(
            [
                {"ticker": "VRDN", "company": "Viridian Therapeutics Inc"},
            ]
        )
        assert names.get("VRDN") == ["viridian"]

    def test_all_stopword_name_fallback(self):
        """Company name made entirely of stop-words should still yield at least
        one discriminative token (first non-stop-word if any, or empty otherwise)."""
        names = self._load_from_entries(
            [
                {"ticker": "XYZ", "company": "The Group Company Inc"},
            ]
        )
        # All tokens are stop-words -> no entry created
        assert "XYZ" not in names

    def test_sabs_headline_no_longer_collides(self):
        """Acceptance regression for C05-style false positive."""
        names = self._load_from_entries(
            [
                {"ticker": "SABS", "company": "SAB Biotherapeutics Inc"},
            ]
        )
        assert not _is_ticker_collision(
            "SAB BIO Announces Pricing of $85 Million Public Offering of Common Stock",
            "SABS",
            names,
        )


# ---------------------------------------------------------------------------
# CH-5: new noise patterns added from fingpt_pilot seed audit
# ---------------------------------------------------------------------------


class TestNoisePatternsCH5:
    def test_halper_sadeh_law_firm(self):
        assert _is_noise("Halper Sadeh LLC is Investigating Whether BCO, THR, ACLX are Obtaining Fair Deals")

    def test_investment_opportunities_market_research(self):
        assert _is_noise("Exploring Investment Opportunities in the Expanding RNA Therapy Clinical Trials Market")

    def test_billion_valuation_by_year(self):
        assert _is_noise("RNA Therapy Market: A USD 4.28 Billion Valuation by 2031")

    def test_million_valuation_by_year(self):
        assert _is_noise("Niche Oncology Segment: A USD 250 Million Valuation by 2030")

    def test_ai_powered_device_clearance(self):
        assert _is_noise(
            "OneMedNet Real-World Data Supports FDA 510(k) Clearance of AI-Powered Scaida BrainCT-ICH Software"
        )

    def test_legitimate_approval_not_affected(self):
        """Ensure CH-5 additions don't suppress real FDA approvals."""
        assert not _is_noise("FDA Approves Amneal Denosumab Biosimilars Referencing Prolia and XGEVA")


# ---------------------------------------------------------------------------
# CH-4: biotech-rescue tightened to >=2 discriminative matches
# ---------------------------------------------------------------------------


class TestBiotechRescueIndicatorSet:
    """CH-4 drops generic tokens from the biotech-rescue match set.
    The four generic tokens 'approved', 'approval', 'drug', 'patient',
    'regulatory' must not appear in the discriminative set.
    """

    def test_generic_tokens_removed(self):
        for generic in ("approved", "approval", "drug", "patient", "regulatory"):
            assert generic not in _BIOTECH_RESCUE_INDICATORS, f"{generic!r} must not be discriminative"

    def test_representative_discriminative_tokens_retained(self):
        for keep in ("phase", "trial", "fda", "clinical", "efficacy", "endpoint", "pdufa"):
            assert keep in _BIOTECH_RESCUE_INDICATORS, f"{keep!r} must stay in discriminative set"

    def test_min_matches_is_two(self):
        assert _BIOTECH_RESCUE_MIN_MATCHES == 2


class TestCountBiotechIndicatorMatches:
    def test_two_matches_rescues(self):
        # "phase" + "endpoint" → 2 distinct matches
        hl = "acme announces positive phase 3 data meeting primary endpoint"
        assert _count_biotech_indicator_matches(hl) == 2

    def test_single_match_below_threshold(self):
        # only "fda" → 1 match
        assert _count_biotech_indicator_matches("fda 510(k) clearance for ai imaging") == 1

    def test_generic_drug_no_longer_counts(self):
        # "drug" alone should not count; "phase" does
        hl = "medicus pharma highlights orr in phase 2 skinject study and drug development plan"
        assert _count_biotech_indicator_matches(hl) == 1

    def test_zero_discriminative_in_non_biotech(self):
        assert _count_biotech_indicator_matches("viridian metals announces closing of first tranche of financing") == 0


class TestCollisionCH4:
    """Acceptance tests using the four specific collision misses from the
    CH-1/2/3/5 pass. After CH-4 tightening, all four must flag as collision
    EXCEPT A13 which requires a separate sector-mismatch rule (explicit scope
    carve-out; tracked as follow-up)."""

    def test_a06_medicus_pharma_flags(self):
        """A06 MRNA — 'Medicus Pharma ... Phase 2 ... Drug Development'.
        Only 'phase' is discriminative (drug dropped), 1 < 2 → flag."""
        assert _is_ticker_collision(
            "Medicus Pharma Business Update Call to Highlight 80% Overall Response Rate (ORR) in Phase 2 SkinJect Study and Agentic AI-enabled Drug Development Plan",
            "MRNA",
            {"MRNA": ["moderna"]},
        )

    def test_a10_onemednet_device_saas_flags(self):
        """A10 IRWD — only 'fda' match; 1 < 2 → flag."""
        assert _is_ticker_collision(
            "OneMedNet Real-World Data Supports FDA 510(k) Clearance of AI-Powered Scaida BrainCT-ICH Software",
            "IRWD",
            {"IRWD": ["ironwood"]},
        )

    def test_c07_dupixent_approved_flags(self):
        """C07 OCS — headline about SNY/REGN, OCS not in registry.
        'gene' fires as substring in 'regeneron' (1 match), but 1 < 2 → still flag."""
        assert _is_ticker_collision(
            "Press Release: Sanofi and Regeneron's Dupixent approved in Japan as the first targeted medicine to treat adults with bullous pemphigoid",
            "OCS",
            {},
        )

    def test_a13_viridian_metals_still_misses_out_of_scope(self):
        """A13 VRDN — 'Viridian Metals ...'. Viridian Therapeutics' name word
        'viridian' substring-matches the headline → check #1 returns 'not
        collision' before CH-4 rescue is even evaluated. Requires a separate
        sector-mismatch override (tracked as follow-up, out of CH-4 scope)."""
        names = {"VRDN": ["viridian"]}
        assert not _is_ticker_collision(
            "Viridian Metals Announces Closing of First Tranche of Financing and Short Form Vertical Amalgamation",
            "VRDN",
            names,
        )

    def test_two_discriminative_match_rescues(self):
        """Sanity: a headline with ≥2 discriminative terms for an unknown
        ticker still rescues correctly."""
        assert not _is_ticker_collision(
            "Unknown Co Reports Positive Phase 3 Trial Data With Primary Endpoint Met",
            "ZZZZ",
            {},
        )


class TestCH4CounterfactualShadow:
    """Shadow-logging hook: the counterfactual rule reports what the ≥1-match
    legacy behavior would say. Must not be wired into production."""

    def test_counterfactual_rescues_single_match(self):
        assert _collision_counterfactual_would_rescue("fda clearance for ai imaging")

    def test_counterfactual_rejects_zero_matches(self):
        assert not _collision_counterfactual_would_rescue("iron ore royalty acquisition update")


# ---------------------------------------------------------------------------
# P2: Soft-collision routing in classify_releases()
# ---------------------------------------------------------------------------


class TestSoftCollisionRouting:
    """P2 (2026-04-18): items collision-flagged because CH-4 tightened the
    rescue rule (match_count == 1) must stay visible in the escalation pool.
    Only hard collisions (match_count == 0, truly non-biotech) get converted
    to informational_only=True for silent drop.
    """

    @staticmethod
    def _with_empty_registry(monkeypatch, tmp_path):
        """Set PROJECT_ROOT to a tmp dir with an empty IR sources file so that
        _load_company_names() returns {} and all collision decisions come from
        the ticker-match + biotech-rescue logic."""
        import tools.classify_press_releases as mod

        (tmp_path / "production_data").mkdir()
        (tmp_path / "production_data" / "company_ir_sources.json").write_text('{"sources": []}')
        monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    def test_hard_collision_suppresses_to_informational(self, monkeypatch, tmp_path):
        """Zero biotech matches → hard collision → informational_only=True."""
        self._with_empty_registry(monkeypatch, tmp_path)
        rec = {
            "ticker": "ZZZZ",
            "company": "Unknown",
            "headline": "Fancamp Acquires Iron Ore Royalty and Provides Corporate Update",
            "source_url": "",
            "published_at_utc": "2026-04-01",
        }
        out = classify_releases([rec])
        assert len(out) == 1
        r = out[0]
        assert r["ticker_collision_flag"] is True
        assert r["collision_severity"] == "hard"
        assert r["informational_only"] is True
        assert "ticker_collision" in r["informational_reason"]

    def test_soft_collision_leaves_informational_false(self, monkeypatch, tmp_path):
        """One biotech match → soft collision → flag but keep informational_only=False
        so the item is still visible in the escalation pool."""
        self._with_empty_registry(monkeypatch, tmp_path)
        rec = {
            "ticker": "ZZZZ",
            "company": "Unknown",
            "headline": "Press Release: Sanofi and Regeneron's Dupixent approved in Japan as the first targeted medicine to treat adults with bullous pemphigoid",
            "source_url": "",
            "published_at_utc": "2026-04-01",
        }
        out = classify_releases([rec])
        assert len(out) == 1
        r = out[0]
        assert r["ticker_collision_flag"] is True
        assert r["collision_severity"] == "soft"
        # Critical: not silently dropped
        assert r["informational_only"] is False
        assert r["confidence"] <= 0.4

    def test_non_collision_has_severity_none(self, monkeypatch, tmp_path):
        """Real biotech event with ticker and company name present → no collision."""
        self._with_empty_registry(monkeypatch, tmp_path)
        rec = {
            "ticker": "ZZZZ",
            "company": "Sionna",
            "headline": "Sionna Announces Positive Phase 3 Trial Data With Primary Endpoint Met",
            "source_url": "",
            "published_at_utc": "2026-04-01",
        }
        out = classify_releases([rec])
        assert len(out) == 1
        r = out[0]
        assert r["ticker_collision_flag"] is False
        assert r["collision_severity"] == "none"

    def test_soft_collision_on_a17_tgtx_pattern(self, monkeypatch, tmp_path):
        """A17 TGTX regression: 'efficacy' single-match. Must be soft, not hard."""
        self._with_empty_registry(monkeypatch, tmp_path)
        rec = {
            "ticker": "ZZZZ",
            "company": "Unknown",
            "headline": "Long Term Data Published in JAMA Neurology Demonstrate Sustained Efficacy and Consistent Safety of BRIUMVI in Relapsing Multiple Sclerosis",
            "source_url": "",
            "published_at_utc": "2026-04-01",
        }
        out = classify_releases([rec])
        r = out[0]
        assert r["ticker_collision_flag"] is True
        assert r["collision_severity"] == "soft"
        assert r["informational_only"] is False

    def test_soft_collision_on_a12_pbyi_pattern(self, monkeypatch, tmp_path):
        """A12 PBYI regression: 'therapy' substring-match only. Must be soft."""
        self._with_empty_registry(monkeypatch, tmp_path)
        rec = {
            "ticker": "ZZZZ",
            "company": "Unknown",
            "headline": "Knight Therapeutics Announces Health Canada Approval for NERLYNX (Neratinib) to Treat HER2-Positive Metastatic Breast Cancer",
            "source_url": "",
            "published_at_utc": "2026-04-01",
        }
        out = classify_releases([rec])
        r = out[0]
        assert r["ticker_collision_flag"] is True
        assert r["collision_severity"] == "soft"
        assert r["informational_only"] is False
