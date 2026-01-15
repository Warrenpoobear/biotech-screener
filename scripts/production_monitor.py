#!/usr/bin/env python3
"""
Production Monitoring System

Tracks model performance in real-time and alerts on degradation.

Key Metrics:
1. Data coverage (% of universe with complete data)
2. Score distribution (detecting drift)
3. Prediction accuracy (realized vs predicted)
4. Anomaly detection (unusual patterns)

Alert Types:
- DATA_COVERAGE_DROP: Coverage fell below threshold
- SCORE_DRIFT: Mean score drifted from baseline
- IC_DEGRADATION: Predictive power declined
- RANK_INSTABILITY: Rankings changing too rapidly

Usage:
    monitor = ProductionMonitor()
    metrics = monitor.track_daily_metrics(date, scored_universe)
    alerts = monitor.check_for_anomalies(metrics)
    report = monitor.generate_weekly_report()

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import json
import math
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any


# Module metadata
__version__ = "1.0.0"


class ProductionMonitor:
    """
    Monitors live scoring performance and raises alerts.

    Tracks:
    1. Data coverage (what % of universe has complete data)
    2. Score distribution (detecting drift)
    3. Prediction accuracy (realized vs predicted)
    4. Anomaly detection (unusual patterns)
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        metrics_dir: str = "data/monitoring",
        baseline_file: Optional[str] = None
    ):
        """
        Initialize the production monitor.

        Args:
            metrics_dir: Directory to store daily metrics
            baseline_file: Path to baseline metrics file
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.baseline_metrics = self._load_baseline(baseline_file)
        self.alert_thresholds = {
            "coverage_drop": 0.15,      # 15% drop triggers alert
            "score_drift": 0.10,        # 10% drift in mean score
            "score_compression": 5.0,   # Std dev below this is suspicious
            "ic_degradation": 0.03,     # 3% drop in IC
            "rank_instability": 0.20,   # 20% rank correlation drop
            "high_confidence_min": 5,   # Minimum high-confidence picks
        }

        self.price_cache: Dict[str, Dict[str, float]] = {}

    def _load_baseline(self, filepath: Optional[str]) -> Dict:
        """
        Load baseline metrics from file or use defaults.
        """
        defaults = {
            "mean_score": 65.0,
            "score_std": 15.0,
            "data_coverage": 0.70,
            "typical_ic": 0.06,
            "typical_rank_correlation": 0.85,
            "typical_high_confidence": 15
        }

        if filepath and Path(filepath).exists():
            try:
                with open(filepath) as f:
                    loaded = json.load(f)
                    defaults.update(loaded)
            except Exception as e:
                print(f"Warning: Could not load baseline: {e}")

        return defaults

    def save_baseline(self, filepath: str) -> None:
        """Save current baseline metrics."""
        with open(filepath, 'w') as f:
            json.dump(self.baseline_metrics, f, indent=2)

    def track_daily_metrics(
        self,
        date: datetime,
        scored_universe: Dict[str, Dict]
    ) -> Dict:
        """
        Calculate and log daily metrics.

        Args:
            date: Scoring date
            scored_universe: Dict mapping ticker to scoring results

        Returns:
            Dict of daily metrics
        """
        if not scored_universe:
            return {
                "date": date.isoformat(),
                "error": "Empty universe"
            }

        scores = []
        coverage_complete = 0
        high_confidence = 0
        gated_count = 0

        for ticker, result in scored_universe.items():
            if isinstance(result, dict):
                score = result.get("final_score", result.get("composite_score"))
                if score is not None:
                    scores.append(float(score))

                coverage = result.get("data_coverage", 0)
                if coverage > 0.60 or result.get("has_complete_data", False):
                    coverage_complete += 1

                confidence = result.get("confidence", 0)
                if confidence > 0.75:
                    high_confidence += 1

                if result.get("gate_applied", False) or result.get("excluded", False):
                    gated_count += 1

        metrics = {
            "date": date.isoformat(),
            "universe_size": len(scored_universe),
            "scored_count": len(scores),
            "mean_score": sum(scores) / len(scores) if scores else 0,
            "score_std": self._std(scores) if len(scores) > 1 else 0,
            "score_min": min(scores) if scores else 0,
            "score_max": max(scores) if scores else 0,
            "data_coverage": coverage_complete / len(scored_universe) if scored_universe else 0,
            "num_high_confidence": high_confidence,
            "num_gated": gated_count,
            "timestamp": datetime.now().isoformat()
        }

        # Store in time series
        self._log_metrics(metrics)

        return metrics

    def _log_metrics(self, metrics: Dict) -> None:
        """Store metrics to file."""
        date_str = metrics["date"][:10]  # YYYY-MM-DD
        filepath = self.metrics_dir / f"metrics_{date_str}.json"

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

    def load_daily_metrics(self, date: datetime) -> Optional[Dict]:
        """Load metrics for a specific date."""
        date_str = date.strftime('%Y-%m-%d')
        filepath = self.metrics_dir / f"metrics_{date_str}.json"

        if filepath.exists():
            try:
                with open(filepath) as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def check_for_anomalies(
        self,
        current_metrics: Dict
    ) -> List[Dict]:
        """
        Compare current metrics to baseline and detect anomalies.

        Args:
            current_metrics: Today's metrics

        Returns:
            List of alert dicts
        """
        alerts = []

        # Coverage drop
        coverage = current_metrics.get("data_coverage", 0)
        baseline_coverage = self.baseline_metrics["data_coverage"]
        coverage_delta = baseline_coverage - coverage

        if coverage_delta > self.alert_thresholds["coverage_drop"]:
            alerts.append({
                "type": "DATA_COVERAGE_DROP",
                "severity": "HIGH",
                "message": f"Coverage dropped to {coverage:.1%} (baseline: {baseline_coverage:.1%})",
                "baseline": baseline_coverage,
                "current": coverage,
                "delta": coverage_delta,
                "timestamp": datetime.now().isoformat()
            })

        # Score distribution drift
        mean_score = current_metrics.get("mean_score", 0)
        baseline_mean = self.baseline_metrics["mean_score"]

        if baseline_mean > 0:
            score_drift = abs(mean_score - baseline_mean) / baseline_mean

            if score_drift > self.alert_thresholds["score_drift"]:
                alerts.append({
                    "type": "SCORE_DRIFT",
                    "severity": "MEDIUM",
                    "message": f"Mean score drifted {score_drift:.1%} from baseline",
                    "baseline": baseline_mean,
                    "current": mean_score,
                    "drift_pct": score_drift,
                    "timestamp": datetime.now().isoformat()
                })

        # Score compression (all scores too similar)
        score_std = current_metrics.get("score_std", 0)
        if score_std < self.alert_thresholds["score_compression"]:
            alerts.append({
                "type": "SCORE_COMPRESSION",
                "severity": "MEDIUM",
                "message": f"Score std dev is only {score_std:.2f} - check if signals are functioning",
                "current_std": score_std,
                "expected_std": self.baseline_metrics["score_std"],
                "timestamp": datetime.now().isoformat()
            })

        # Low high-confidence count
        high_conf = current_metrics.get("num_high_confidence", 0)
        if high_conf < self.alert_thresholds["high_confidence_min"]:
            alerts.append({
                "type": "LOW_CONFIDENCE",
                "severity": "LOW",
                "message": f"Only {high_conf} high-confidence picks (expected {self.baseline_metrics['typical_high_confidence']})",
                "current": high_conf,
                "expected": self.baseline_metrics["typical_high_confidence"],
                "timestamp": datetime.now().isoformat()
            })

        # Check for scoring failures
        universe_size = current_metrics.get("universe_size", 0)
        scored_count = current_metrics.get("scored_count", 0)
        if universe_size > 0 and scored_count / universe_size < 0.5:
            alerts.append({
                "type": "SCORING_FAILURE",
                "severity": "HIGH",
                "message": f"Only {scored_count}/{universe_size} tickers scored successfully",
                "success_rate": scored_count / universe_size,
                "timestamp": datetime.now().isoformat()
            })

        return alerts

    def calculate_realized_ic(
        self,
        scored_date: datetime,
        horizon_days: int = 30
    ) -> Dict:
        """
        Calculate IC after returns are known.

        Runs 30 days after scoring to see if predictions were accurate.

        Args:
            scored_date: Date when scoring was done
            horizon_days: Forward horizon in days

        Returns:
            Dict with IC, p-value, interpretation
        """
        # Load historical scores
        historical_metrics = self.load_daily_metrics(scored_date)
        if not historical_metrics:
            return {
                "ic": None,
                "reason": "No historical scores found"
            }

        # This would require storing the full scored universe
        # For now, return placeholder
        future_date = scored_date + timedelta(days=horizon_days)

        return {
            "scored_date": scored_date.isoformat(),
            "measured_date": future_date.isoformat(),
            "horizon_days": horizon_days,
            "ic": None,
            "status": "Requires full scored universe storage",
            "interpretation": "N/A"
        }

    def check_rank_stability(
        self,
        date1: datetime,
        date2: datetime
    ) -> Dict:
        """
        Check rank correlation between two scoring dates.

        High correlation = stable rankings (good)
        Low correlation = unstable rankings (potential issue)
        """
        metrics1 = self.load_daily_metrics(date1)
        metrics2 = self.load_daily_metrics(date2)

        if not metrics1 or not metrics2:
            return {
                "correlation": None,
                "reason": "Missing metrics for one or both dates"
            }

        # Would need full scored universe to calculate
        return {
            "date1": date1.isoformat(),
            "date2": date2.isoformat(),
            "correlation": None,
            "status": "Requires full scored universe storage"
        }

    def generate_weekly_report(self) -> Dict:
        """
        Generate aggregated weekly performance report.

        Returns:
            Dict with weekly summary and alerts
        """
        today = datetime.now()
        week_dates = [today - timedelta(days=i) for i in range(7)]

        weekly_metrics = []
        all_alerts = []

        for date in week_dates:
            daily_metrics = self.load_daily_metrics(date)
            if daily_metrics:
                weekly_metrics.append(daily_metrics)

                alerts = self.check_for_anomalies(daily_metrics)
                all_alerts.extend(alerts)

        if not weekly_metrics:
            return {
                "week_ending": today.isoformat(),
                "status": "NO_DATA",
                "message": "No metrics found for the past week"
            }

        # Aggregate statistics
        avg_coverage = sum(m["data_coverage"] for m in weekly_metrics) / len(weekly_metrics)
        avg_score = sum(m["mean_score"] for m in weekly_metrics) / len(weekly_metrics)
        total_scored = sum(m["scored_count"] for m in weekly_metrics)

        critical_alerts = [a for a in all_alerts if a["severity"] == "HIGH"]
        warning_alerts = [a for a in all_alerts if a["severity"] == "MEDIUM"]

        # Determine overall status
        if len(critical_alerts) > 0:
            status = "DEGRADED"
        elif len(warning_alerts) > 2:
            status = "WARNING"
        else:
            status = "HEALTHY"

        return {
            "week_ending": today.isoformat(),
            "days_analyzed": len(weekly_metrics),
            "total_scorings": total_scored,
            "average_coverage": avg_coverage,
            "average_score": avg_score,
            "total_alerts": len(all_alerts),
            "critical_alerts": len(critical_alerts),
            "warning_alerts": len(warning_alerts),
            "alerts": all_alerts,
            "status": status,
            "recommendations": self._generate_recommendations(all_alerts)
        }

    def _generate_recommendations(self, alerts: List[Dict]) -> List[str]:
        """Generate actionable recommendations from alerts."""
        recommendations = []

        alert_types = set(a["type"] for a in alerts)

        if "DATA_COVERAGE_DROP" in alert_types:
            recommendations.append(
                "URGENT: Investigate data pipeline - coverage has dropped significantly"
            )

        if "SCORE_COMPRESSION" in alert_types:
            recommendations.append(
                "Review signal calculations - scores are too clustered together"
            )

        if "SCORING_FAILURE" in alert_types:
            recommendations.append(
                "Check error logs - many tickers failing to score"
            )

        if "SCORE_DRIFT" in alert_types:
            recommendations.append(
                "Monitor score drift - may indicate market regime change or model issue"
            )

        if not recommendations:
            recommendations.append("System operating normally - no action required")

        return recommendations

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def generate_dashboard_data(self) -> Dict:
        """
        Generate data for monitoring dashboard.

        Returns 7-day time series of key metrics.
        """
        today = datetime.now()
        dates = [today - timedelta(days=i) for i in range(7)]

        time_series = {
            "dates": [],
            "coverage": [],
            "mean_score": [],
            "score_std": [],
            "high_confidence": [],
            "alerts_count": []
        }

        for date in reversed(dates):
            metrics = self.load_daily_metrics(date)

            time_series["dates"].append(date.strftime('%Y-%m-%d'))

            if metrics:
                time_series["coverage"].append(metrics.get("data_coverage", 0))
                time_series["mean_score"].append(metrics.get("mean_score", 0))
                time_series["score_std"].append(metrics.get("score_std", 0))
                time_series["high_confidence"].append(metrics.get("num_high_confidence", 0))

                alerts = self.check_for_anomalies(metrics)
                time_series["alerts_count"].append(len(alerts))
            else:
                time_series["coverage"].append(None)
                time_series["mean_score"].append(None)
                time_series["score_std"].append(None)
                time_series["high_confidence"].append(None)
                time_series["alerts_count"].append(None)

        return time_series

    def format_report(self, report: Dict) -> str:
        """Format weekly report as human-readable string."""
        lines = [
            "=" * 70,
            "PRODUCTION MONITORING WEEKLY REPORT",
            "=" * 70,
            f"Week Ending: {report['week_ending']}",
            f"Status: {report['status']}",
            "",
            "-" * 70,
            "SUMMARY METRICS",
            "-" * 70,
            f"Days Analyzed: {report.get('days_analyzed', 0)}",
            f"Total Scorings: {report.get('total_scorings', 0)}",
            f"Average Coverage: {report.get('average_coverage', 0):.1%}",
            f"Average Score: {report.get('average_score', 0):.1f}",
            "",
            "-" * 70,
            "ALERTS",
            "-" * 70,
            f"Critical: {report.get('critical_alerts', 0)}",
            f"Warning: {report.get('warning_alerts', 0)}",
            f"Total: {report.get('total_alerts', 0)}",
        ]

        alerts = report.get("alerts", [])
        if alerts:
            lines.append("")
            for alert in alerts[:10]:  # Show first 10
                lines.append(f"  [{alert['severity']}] {alert['type']}: {alert['message']}")

        lines.extend([
            "",
            "-" * 70,
            "RECOMMENDATIONS",
            "-" * 70,
        ])

        for rec in report.get("recommendations", []):
            lines.append(f"  • {rec}")

        lines.append("=" * 70)

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("PRODUCTION MONITOR - DEMO")
    print("=" * 70)

    monitor = ProductionMonitor()

    # Simulate scoring results
    sample_universe = {
        "NVAX": {"final_score": 84.3, "confidence": 0.85, "data_coverage": 0.9},
        "IMCR": {"final_score": 82.1, "confidence": 0.80, "data_coverage": 0.85},
        "KROS": {"final_score": 78.5, "confidence": 0.75, "data_coverage": 0.80},
        "ALKS": {"final_score": 75.0, "confidence": 0.70, "data_coverage": 0.75},
        "SIGA": {"final_score": 72.0, "confidence": 0.65, "data_coverage": 0.70},
    }

    print("\nTracking daily metrics...")
    metrics = monitor.track_daily_metrics(datetime.now(), sample_universe)
    print(f"  Mean Score: {metrics['mean_score']:.1f}")
    print(f"  Coverage: {metrics['data_coverage']:.1%}")
    print(f"  High Confidence: {metrics['num_high_confidence']}")

    print("\nChecking for anomalies...")
    alerts = monitor.check_for_anomalies(metrics)
    if alerts:
        for alert in alerts:
            print(f"  [{alert['severity']}] {alert['type']}: {alert['message']}")
    else:
        print("  No anomalies detected")

    print("\nGenerating weekly report...")
    report = monitor.generate_weekly_report()
    print(f"  Status: {report['status']}")
    print(f"  Days Analyzed: {report.get('days_analyzed', 0)}")
