"""
Alert Enricher
Enriches Wazuh alerts with ML classification results.
Outputs to log file, stdout, and optional webhook.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class AlertEnricher:
    """Enrich original Wazuh alerts with ML ensemble results."""

    def __init__(self, config: dict = None):
        """
        Initialize enricher.
        
        Config keys:
            log_file: Path to enriched alerts log file
            webhook_url: Optional webhook URL for alert forwarding
            min_level: Minimum intrusion confidence to trigger webhook (0-1)
            console_output: Whether to print to stdout (default: True)
        """
        config = config or {}
        self.log_file = config.get("log_file", "logs/enriched_alerts.jsonl")
        self.webhook_url = config.get("webhook_url", None)
        self.min_level = config.get("min_level", 0.7)
        self.console_output = config.get("console_output", True)

        # Ensure log directory exists
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

        # Stats
        self.stats = {
            "total_processed": 0,
            "intrusions_detected": 0,
            "normal_traffic": 0,
            "start_time": datetime.now().isoformat(),
        }

    def enrich(self, alerts: List[Dict], predictions: List[Dict]) -> List[Dict]:
        """
        Combine original alerts with ML predictions.
        
        Args:
            alerts: List of original Wazuh alert dicts
            predictions: List of prediction dicts from EnsembleEngine
            
        Returns:
            List of enriched alert dicts
        """
        enriched = []

        for alert, prediction in zip(alerts, predictions):
            enriched_alert = {
                "timestamp": datetime.now().isoformat(),
                "original_alert": {
                    "rule_id": alert.get("rule", {}).get("id", "unknown"),
                    "rule_level": alert.get("rule", {}).get("level", 0),
                    "rule_description": alert.get("rule", {}).get("description", ""),
                    "agent": alert.get("agent", {}),
                    "source_ip": alert.get("data", {}).get("srcip", ""),
                    "destination_ip": alert.get("data", {}).get("dstip", ""),
                },
                "ml_analysis": {
                    "ensemble_class": prediction["class_name"],
                    "ensemble_class_id": prediction["class_id"],
                    "confidence": round(prediction["confidence"], 4),
                    "is_intrusion": prediction["is_intrusion"],
                    "base_models": {
                        "random_forest": prediction["base_predictions"]["rf"]["class"],
                        "xgboost": prediction["base_predictions"]["xgb"]["class"],
                        "dnn": prediction["base_predictions"]["dnn"]["class"],
                    },
                    "model_agreement": self._check_agreement(prediction),
                },
                "severity": self._compute_severity(alert, prediction),
            }

            enriched.append(enriched_alert)
            self._output(enriched_alert)

            # Update stats
            self.stats["total_processed"] += 1
            if prediction["is_intrusion"]:
                self.stats["intrusions_detected"] += 1
            else:
                self.stats["normal_traffic"] += 1

        return enriched

    def _check_agreement(self, prediction: Dict) -> str:
        """Check if all base models agree on the classification."""
        classes = [
            prediction["base_predictions"]["rf"]["class"],
            prediction["base_predictions"]["xgb"]["class"],
            prediction["base_predictions"]["dnn"]["class"],
        ]
        if len(set(classes)) == 1:
            return "unanimous"
        elif len(set(classes)) == 2:
            return "majority"
        else:
            return "split"

    def _compute_severity(self, alert: Dict, prediction: Dict) -> str:
        """Compute combined severity from Wazuh rule level and ML confidence."""
        rule_level = alert.get("rule", {}).get("level", 0)
        ml_confidence = prediction["confidence"]
        is_intrusion = prediction["is_intrusion"]

        if not is_intrusion:
            return "info"
        elif rule_level >= 12 or ml_confidence >= 0.95:
            return "critical"
        elif rule_level >= 10 or ml_confidence >= 0.85:
            return "high"
        elif rule_level >= 7 or ml_confidence >= 0.75:
            return "medium"
        else:
            return "low"

    def _output(self, enriched_alert: Dict):
        """Output enriched alert to all configured destinations."""
        # Console output
        if self.console_output:
            self._print_alert(enriched_alert)

        # Log file (JSONL format)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(enriched_alert) + "\n")
        except Exception as e:
            logger.error(f"Log write error: {e}")

        # Webhook (if configured and severity warrants it)
        if (self.webhook_url and
                enriched_alert["ml_analysis"]["is_intrusion"] and
                enriched_alert["ml_analysis"]["confidence"] >= self.min_level):
            self._send_webhook(enriched_alert)

    def _print_alert(self, enriched_alert: Dict):
        """Pretty-print an enriched alert to console."""
        ml = enriched_alert["ml_analysis"]
        orig = enriched_alert["original_alert"]
        severity = enriched_alert["severity"]

        severity_icons = {
            "info": "ℹ️ ",
            "low": "🟡",
            "medium": "🟠",
            "high": "🔴",
            "critical": "🚨",
        }

        icon = severity_icons.get(severity, "❓")
        intrusion_icon = "[WARN]️  INTRUSION" if ml["is_intrusion"] else "✅ NORMAL"

        print(f"  {icon} [{severity.upper():8s}] {intrusion_icon} | "
              f"Class: {ml['ensemble_class']} ({ml['confidence']:.1%}) | "
              f"Rule: {orig['rule_id']} - {orig['rule_description'][:40]} | "
              f"Agreement: {ml['model_agreement']}")

    def _send_webhook(self, enriched_alert: Dict):
        """Send alert to webhook URL."""
        try:
            import requests
            response = requests.post(
                self.webhook_url,
                json=enriched_alert,
                timeout=5,
            )
            if response.status_code != 200:
                logger.warning(f"Webhook returned {response.status_code}")
        except Exception as e:
            logger.error(f"Webhook error: {e}")

    def get_stats(self) -> Dict:
        """Return processing statistics."""
        return {**self.stats, "last_updated": datetime.now().isoformat()}

    def print_stats(self):
        """Print summary statistics."""
        print(f"\n  {'-' * 40}")
        print(f"  Processing Statistics:")
        print(f"    Total processed:     {self.stats['total_processed']}")
        print(f"    Intrusions detected: {self.stats['intrusions_detected']}")
        print(f"    Normal traffic:      {self.stats['normal_traffic']}")
        detection_rate = (
            self.stats['intrusions_detected'] / max(self.stats['total_processed'], 1) * 100
        )
        print(f"    Detection rate:      {detection_rate:.1f}%")
        print(f"  {'-' * 40}")
