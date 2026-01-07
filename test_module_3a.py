#!/usr/bin/env python3
"""
test_module_3a.py - Comprehensive Test Suite for Module 3A

Tests all components of the catalyst detection system.
"""

import unittest
from datetime import date, timedelta
from pathlib import Path
import json
import tempfile
import shutil

from ctgov_adapter import (
    CTGovStatus, CompletionType, CanonicalTrialRecord,
    process_trial_records_batch, AdapterConfig, AdapterError
)
from state_management import StateStore, StateSnapshot
from event_detector import (
    EventDetector, EventDetectorConfig, EventType,
    classify_status_change, classify_timeline_change,
    classify_date_confirmation, SimpleMarketCalendar
)
from catalyst_summary import CatalystAggregator, TickerCatalystSummary
from module_3_catalyst import compute_module_3_catalyst, Module3Config


# ============================================================================
# TEST DATA FIXTURES
# ============================================================================

def create_sample_trial_record(
    ticker: str = "TEST",
    nct_id: str = "NCT12345678",
    status: str = "Recruiting",
    last_update_posted: str = "2024-01-15"
) -> dict:
    """Create sample trial record matching user's format"""
    return {
        "ticker": ticker,
        "nct_id": nct_id,
        "phase": "PHASE2",
        "status": status,
        "randomized": True,
        "blinded": "DOUBLE",
        "primary_endpoint": "Overall Survival",
        "last_update_posted": last_update_posted,
        "source_date": None,
        "primary_completion_date": "2024-06-01",
        "primary_completion_type": "ESTIMATED",  # CT.gov uses ESTIMATED
        "completion_date": "2024-12-01",
        "completion_type": "ESTIMATED",  # CT.gov uses ESTIMATED
        "results_first_posted": None
    }


# ============================================================================
# ADAPTER TESTS
# ============================================================================

class TestCTGovAdapter(unittest.TestCase):
    """Test CT.gov adapter with user's data format"""
    
    def test_process_flattened_format(self):
        """Test adapter handles user's flattened format"""
        trial = create_sample_trial_record()
        config = AdapterConfig()
        
        records, stats = process_trial_records_batch(
            [trial],
            date(2024, 1, 15),
            config
        )
        
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].ticker, "TEST")
        self.assertEqual(records[0].nct_id, "NCT12345678")
        self.assertEqual(records[0].overall_status, CTGovStatus.RECRUITING)
    
    def test_pit_validation(self):
        """Test PIT validation rejects future data"""
        trial = create_sample_trial_record(last_update_posted="2024-01-20")
        config = AdapterConfig()
        
        with self.assertRaises(AdapterError):
            process_trial_records_batch(
                [trial],
                date(2024, 1, 15),  # as_of_date before last_update
                config
            )
    
    def test_missing_last_update_posted(self):
        """Test adapter fails on missing PIT anchor"""
        trial = create_sample_trial_record()
        trial['last_update_posted'] = None
        config = AdapterConfig()
        
        with self.assertRaises(AdapterError):
            process_trial_records_batch([trial], date(2024, 1, 15), config)


# ============================================================================
# STATE MANAGEMENT TESTS
# ============================================================================

class TestStateManagement(unittest.TestCase):
    """Test state store and snapshots"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.state_dir = Path(self.temp_dir)
        self.store = StateStore(self.state_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_snapshot(self):
        """Test snapshot persistence"""
        records = [
            CanonicalTrialRecord(
                ticker="TEST",
                nct_id="NCT001",
                overall_status=CTGovStatus.RECRUITING,
                last_update_posted=date(2024, 1, 15),
                primary_completion_date=date(2024, 6, 1),
                primary_completion_type=CompletionType.ANTICIPATED,
                completion_date=date(2024, 12, 1),
                completion_type=CompletionType.ANTICIPATED,
                results_first_posted=None
            )
        ]
        
        snapshot = StateSnapshot(
            snapshot_date=date(2024, 1, 15),
            records=records
        )
        
        self.store.save_snapshot(snapshot)
        loaded = self.store.load_snapshot(date(2024, 1, 15))
        
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.snapshot_date, date(2024, 1, 15))
        self.assertEqual(len(loaded.records), 1)
        self.assertEqual(loaded.records[0].nct_id, "NCT001")
    
    def test_get_most_recent_snapshot(self):
        """Test finding most recent snapshot"""
        dates = [date(2024, 1, 8), date(2024, 1, 15), date(2024, 1, 22)]
        
        for d in dates:
            snapshot = StateSnapshot(snapshot_date=d, records=[])
            self.store.save_snapshot(snapshot)
        
        most_recent = self.store.get_most_recent_snapshot()
        self.assertEqual(most_recent.snapshot_date, date(2024, 1, 22))


# ============================================================================
# EVENT CLASSIFICATION TESTS
# ============================================================================

class TestEventClassification(unittest.TestCase):
    """Test event classification logic"""
    
    def test_status_severe_negative(self):
        """Test severe negative status detection"""
        event_type, impact, direction = classify_status_change(
            CTGovStatus.RECRUITING,
            CTGovStatus.TERMINATED
        )
        
        self.assertEqual(event_type, EventType.CT_STATUS_SEVERE_NEG)
        self.assertEqual(impact, 3)
        self.assertEqual(direction, 'NEG')
    
    def test_status_downgrade(self):
        """Test status downgrade detection"""
        event_type, impact, direction = classify_status_change(
            CTGovStatus.RECRUITING,
            CTGovStatus.SUSPENDED
        )
        
        self.assertEqual(event_type, EventType.CT_STATUS_SEVERE_NEG)
        self.assertEqual(direction, 'NEG')
    
    def test_status_upgrade(self):
        """Test status upgrade detection"""
        event_type, impact, direction = classify_status_change(
            CTGovStatus.NOT_YET_RECRUITING,
            CTGovStatus.RECRUITING
        )
        
        self.assertEqual(event_type, EventType.CT_STATUS_UPGRADE)
        self.assertEqual(direction, 'POS')
    
    def test_timeline_pushout(self):
        """Test timeline pushout detection"""
        old_date = date(2024, 6, 1)
        new_date = date(2024, 9, 1)  # +90 days
        
        event_type, impact, direction = classify_timeline_change(
            old_date, new_date, noise_band_days=14
        )
        
        self.assertEqual(event_type, EventType.CT_TIMELINE_PUSHOUT)
        self.assertEqual(direction, 'NEG')
        self.assertGreater(impact, 0)
    
    def test_timeline_noise_ignored(self):
        """Test noise band ignores small changes"""
        old_date = date(2024, 6, 1)
        new_date = date(2024, 6, 8)  # +7 days (< 14)
        
        event_type, impact, direction = classify_timeline_change(
            old_date, new_date, noise_band_days=14
        )
        
        self.assertIsNone(event_type)
    
    def test_date_confirmation(self):
        """Test date confirmation detection"""
        event_type, impact, direction, actual_date = classify_date_confirmation(
            CompletionType.ESTIMATED,  # CT.gov uses ESTIMATED
            CompletionType.ACTUAL,
            date(2024, 1, 10),
            date(2024, 1, 15),
            recency_threshold_days=90
        )
        
        self.assertEqual(event_type, EventType.CT_DATE_CONFIRMED_ACTUAL)
        self.assertEqual(direction, 'POS')
        self.assertEqual(actual_date, date(2024, 1, 10))


# ============================================================================
# EVENT DETECTOR TESTS
# ============================================================================

class TestEventDetector(unittest.TestCase):
    """Test event detector"""
    
    def setUp(self):
        self.config = EventDetectorConfig()
        self.detector = EventDetector(self.config)
    
    def test_detect_status_change(self):
        """Test status change detection"""
        prior = CanonicalTrialRecord(
            ticker="TEST",
            nct_id="NCT001",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=date(2024, 1, 8),
            primary_completion_date=None,
            primary_completion_type=None,
            completion_date=None,
            completion_type=None,
            results_first_posted=None
        )
        
        current = CanonicalTrialRecord(
            ticker="TEST",
            nct_id="NCT001",
            overall_status=CTGovStatus.TERMINATED,
            last_update_posted=date(2024, 1, 15),
            primary_completion_date=None,
            primary_completion_type=None,
            completion_date=None,
            completion_type=None,
            results_first_posted=None
        )
        
        events = self.detector.detect_events(
            current, prior, date(2024, 1, 15)
        )
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.CT_STATUS_SEVERE_NEG)
    
    def test_no_events_on_initial_ingest(self):
        """Test no events detected when prior is None"""
        current = CanonicalTrialRecord(
            ticker="TEST",
            nct_id="NCT001",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=date(2024, 1, 15),
            primary_completion_date=None,
            primary_completion_type=None,
            completion_date=None,
            completion_type=None,
            results_first_posted=None
        )
        
        events = self.detector.detect_events(current, None, date(2024, 1, 15))
        
        self.assertEqual(len(events), 0)


# ============================================================================
# CATALYST AGGREGATOR TESTS
# ============================================================================

class TestCatalystAggregator(unittest.TestCase):
    """Test catalyst aggregation"""
    
    def setUp(self):
        self.calendar = SimpleMarketCalendar()
        self.aggregator = CatalystAggregator(self.calendar, decay_constant=30.0)
    
    def test_aggregate_events(self):
        """Test event aggregation"""
        from event_detector import CatalystEvent
        
        events = [
            CatalystEvent(
                nct_id="NCT001",
                event_type=EventType.CT_STATUS_SEVERE_NEG,
                direction='NEG',
                impact=3,
                confidence=0.95,
                disclosed_at=date(2024, 1, 15)
            )
        ]
        
        summary = self.aggregator.aggregate(
            "TEST",
            events,
            date(2024, 1, 15)
        )
        
        self.assertEqual(summary.ticker, "TEST")
        self.assertTrue(summary.severe_negative_flag)
        self.assertGreater(summary.catalyst_score_neg, 0)


# ============================================================================
# MARKET CALENDAR TESTS
# ============================================================================

class TestMarketCalendar(unittest.TestCase):
    """Test market calendar"""
    
    def setUp(self):
        self.calendar = SimpleMarketCalendar()
    
    def test_skip_weekend(self):
        """Test weekend skipping"""
        friday = date(2024, 1, 12)  # Friday
        monday = date(2024, 1, 15)  # Monday
        
        next_day = self.calendar.next_trading_day(friday)
        self.assertEqual(next_day, monday)
    
    def test_weekday_is_next_day(self):
        """Test weekday advances by 1"""
        monday = date(2024, 1, 15)
        tuesday = date(2024, 1, 16)
        
        next_day = self.calendar.next_trading_day(monday)
        self.assertEqual(next_day, tuesday)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestModule3Integration(unittest.TestCase):
    """Integration tests for full Module 3 pipeline"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        
        # Create trial_records.json
        trials = [
            create_sample_trial_record("TEST1", "NCT001", "Recruiting", "2024-01-15"),
            create_sample_trial_record("TEST2", "NCT002", "Recruiting", "2024-01-15")
        ]
        
        trial_records_path = self.data_dir / "trial_records.json"
        with open(trial_records_path, 'w') as f:
            json.dump(trials, f)
        
        self.trial_records_path = trial_records_path
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline_first_run(self):
        """Test complete pipeline on first run"""
        result = compute_module_3_catalyst(
            trial_records_path=self.trial_records_path,
            state_dir=self.data_dir / "ctgov_state",
            active_tickers={'TEST1', 'TEST2'},
            as_of_date=date(2024, 1, 15),
            output_dir=self.data_dir
        )
        
        # First run should create snapshot but detect 0 events
        self.assertIn('summaries', result)
        self.assertEqual(result['diagnostic_counts']['events_detected'], 0)
        
        # Verify state snapshot created
        state_file = self.data_dir / "ctgov_state" / "state_2024-01-15.jsonl"
        self.assertTrue(state_file.exists())


# ============================================================================
# RUN TESTS
# ============================================================================

def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCTGovAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestStateManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestEventClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestEventDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestCatalystAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestMarketCalendar))
    suite.addTests(loader.loadTestsFromTestCase(TestModule3Integration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
