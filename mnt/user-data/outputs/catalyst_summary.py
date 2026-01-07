#!/usr/bin/env python3
"""
catalyst_summary.py - Catalyst Event Aggregation

Aggregates events into ticker-level summaries with scoring.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional
import json
import hashlib
import logging

from event_detector import CatalystEvent, MarketCalendar, compute_event_score

logger = logging.getLogger(__name__)


# ============================================================================
# TICKER CATALYST SUMMARY
# ============================================================================

@dataclass
class TickerCatalystSummary:
    """Aggregated catalyst view for a single ticker"""
    ticker: str
    as_of_date: date
    catalyst_score_pos: float
    catalyst_score_neg: float
    catalyst_score_net: float
    nearest_positive_days: Optional[int]
    nearest_negative_days: Optional[int]
    severe_negative_flag: bool
    events: list[CatalystEvent]
    
    def to_dict(self, include_audit_hash: bool = True) -> dict:
        """Serialize to JSON-compatible dict"""
        base_dict = {
            'ticker': self.ticker,
            'as_of_date': self.as_of_date.isoformat(),
            'catalyst_score_pos': round(self.catalyst_score_pos, 4),
            'catalyst_score_neg': round(self.catalyst_score_neg, 4),
            'catalyst_score_net': round(self.catalyst_score_net, 4),
            'nearest_positive_days': self.nearest_positive_days,
            'nearest_negative_days': self.nearest_negative_days,
            'severe_negative_flag': self.severe_negative_flag,
            'events': [e.to_dict() for e in self.events]
        }
        
        if include_audit_hash:
            base_dict['audit_hash'] = self.compute_audit_hash()
        
        return base_dict
    
    def compute_audit_hash(self) -> str:
        """Compute audit hash from canonical JSON"""
        canonical_dict = self.to_dict(include_audit_hash=False)
        canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(',', ':'))
        hash_obj = hashlib.sha256(canonical_json.encode('utf-8'))
        return hash_obj.hexdigest()[:16]
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TickerCatalystSummary':
        """Deserialize from JSON"""
        from event_detector import EventType
        
        events = []
        for e in data['events']:
            event = CatalystEvent(
                source=e['source'],
                nct_id=e['nct_id'],
                event_type=EventType[e['event_type']],
                direction=e['direction'],
                impact=e['impact'],
                confidence=e['confidence'],
                disclosed_at=date.fromisoformat(e['disclosed_at']),
                fields_changed=e['fields_changed'],
                actual_date=date.fromisoformat(e['actual_date']) if e.get('actual_date') else None
            )
            events.append(event)
        
        return cls(
            ticker=data['ticker'],
            as_of_date=date.fromisoformat(data['as_of_date']),
            catalyst_score_pos=data['catalyst_score_pos'],
            catalyst_score_neg=data['catalyst_score_neg'],
            catalyst_score_net=data['catalyst_score_net'],
            nearest_positive_days=data['nearest_positive_days'],
            nearest_negative_days=data['nearest_negative_days'],
            severe_negative_flag=data['severe_negative_flag'],
            events=events
        )


# ============================================================================
# CATALYST AGGREGATOR
# ============================================================================

class CatalystAggregator:
    """Aggregates events into ticker-level summary"""
    
    def __init__(self, calendar: MarketCalendar, decay_constant: float = 30.0):
        self.calendar = calendar
        self.decay_constant = decay_constant
    
    def aggregate(
        self,
        ticker: str,
        events: list[CatalystEvent],
        as_of_date: date
    ) -> TickerCatalystSummary:
        """Aggregate events with directional scoring"""
        from event_detector import EventType
        
        pos_score = 0.0
        neg_score = 0.0
        severe_negative = False
        
        nearest_pos_days = None
        nearest_neg_days = None
        
        for event in events:
            score = compute_event_score(event, as_of_date, self.calendar, self.decay_constant)
            days_to_event = event.days_to_event(as_of_date, self.calendar)
            
            if event.direction == 'POS':
                pos_score += score
                if nearest_pos_days is None or days_to_event < nearest_pos_days:
                    nearest_pos_days = days_to_event
            
            elif event.direction == 'NEG':
                neg_score += score
                if nearest_neg_days is None or days_to_event < nearest_neg_days:
                    nearest_neg_days = days_to_event
                
                if event.event_type == EventType.CT_STATUS_SEVERE_NEG:
                    severe_negative = True
        
        net_score = pos_score - neg_score
        
        # Sort events for deterministic hashing
        sorted_events = sorted(
            events,
            key=lambda e: (e.nct_id, e.event_type.value, e.disclosed_at.isoformat())
        )
        
        return TickerCatalystSummary(
            ticker=ticker,
            as_of_date=as_of_date,
            catalyst_score_pos=pos_score,
            catalyst_score_neg=neg_score,
            catalyst_score_net=net_score,
            nearest_positive_days=nearest_pos_days,
            nearest_negative_days=nearest_neg_days,
            severe_negative_flag=severe_negative,
            events=sorted_events
        )


# ============================================================================
# OUTPUT WRITER
# ============================================================================

class CatalystOutputWriter:
    """Writes deterministic catalyst events output"""
    
    @staticmethod
    def write_catalyst_events(
        summaries: list[TickerCatalystSummary],
        as_of_date: date,
        output_path: str,
        prior_snapshot_date: Optional[date] = None,
        module_version: str = "3A.1.1"
    ):
        """Write catalyst_events_YYYY-MM-DD.json (deterministic)"""
        from pathlib import Path
        
        # Sort summaries by ticker for determinism
        summaries = sorted(summaries, key=lambda s: s.ticker)
        
        # Build output
        output = {
            'run_metadata': {
                'as_of_date': as_of_date.isoformat(),
                'prior_snapshot_date': prior_snapshot_date.isoformat() if prior_snapshot_date else None,
                'tickers_analyzed': len(summaries),
                'events_detected': sum(len(s.events) for s in summaries),
                'severe_negatives': sum(1 for s in summaries if s.severe_negative_flag),
                'module_version': module_version
            },
            'summaries': [s.to_dict() for s in summaries]
        }
        
        # Write with canonical JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, sort_keys=True)
        
        logger.info(f"Wrote catalyst events: {output_path}")
    
    @staticmethod
    def write_run_log(
        output_path: str,
        execution_time_seconds: float,
        config: dict,
        warnings: list = None,
        errors: list = None
    ):
        """Write run_log_YYYY-MM-DD.json (non-deterministic)"""
        from pathlib import Path
        from datetime import datetime
        import socket
        
        log = {
            'run_timestamp': datetime.utcnow().isoformat() + 'Z',
            'execution_time_seconds': round(execution_time_seconds, 2),
            'host': socket.gethostname(),
            'config': config,
            'warnings': warnings or [],
            'errors': errors or []
        }
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(log, f, indent=2, sort_keys=True)
        
        logger.info(f"Wrote run log: {output_path}")


if __name__ == "__main__":
    print("Catalyst Summary loaded successfully")
