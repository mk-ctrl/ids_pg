"""
Inference Scheduler
Main entry point for the periodic pull->extract->infer->enrich detection loop.

Usage:
    python -m src.inference.scheduler                    # Live mode (requires Wazuh)
    python -m src.inference.scheduler --demo             # Demo with synthetic alerts
    python -m src.inference.scheduler --interval 30      # Custom interval
    python -m src.inference.scheduler --config configs/inference_config.yaml
"""

import os
import sys
import time
import signal
import logging
import argparse
from pathlib import Path
from datetime import datetime


try:
    import yaml
except ImportError:
    yaml = None

from src.inference.alert_puller import WazuhAlertPuller
from src.inference.feature_extractor import FeatureExtractor
from src.inference.ensemble_engine import EnsembleEngine
from src.inference.enricher import AlertEnricher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class IDSScheduler:
    """Main scheduler for the ML-enhanced IDS pipeline."""

    def __init__(self, config: dict, ui_callback=None):
        self.config = config
        self.interval = config.get("interval", 30)
        self.demo_mode = config.get("demo_mode", False)
        self.ui_callback = ui_callback
        self._running = False

        # Initialize components
        logger.info("Initializing IDS components...")

        self.alert_puller = WazuhAlertPuller(config.get("alert_puller", {}))
        self.feature_extractor = FeatureExtractor(
            config.get("scaler_path", None)
        )
        self.ensemble_engine = EnsembleEngine()
        self.enricher = AlertEnricher(config.get("enricher", {}))

    def start(self):
        """Start the detection loop (blocking)."""
        if not self.setup():
            return
            
        print(f"\n[OK] System ready. Monitoring every {self.interval}s...")
        print("  Press Ctrl+C to stop.\n")

        self.run_background()

    def setup(self):
        """Prepare system and load models."""
        print("=" * 60)
        print("  ML-Enhanced Intrusion Detection System")
        print("  " + "-" * 56)
        print(f"  Mode:     {'DEMO' if self.demo_mode else 'LIVE'}")
        print(f"  Interval: {self.interval}s")
        print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Load models
        if not self.ensemble_engine.load_models():
            print("\n[WARN] Could not load ML models. Train models first.")
            print("  See: python -m src.models.train_rf --help")
            return False

        # Setup graceful shutdown
        self._running = True
        try:
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)
        except ValueError:
            # Ignore if not called from main thread (e.g. running in UI thread)
            pass
            
        return True

    def run_background(self):
        """Run the detection loop in the background. Blocking call."""
        cycle_count = 0
        while self._running:
            cycle_count += 1
            try:
                self._run_cycle(cycle_count)
            except Exception as e:
                logger.error(f"Cycle {cycle_count} error: {e}")

            if self._running:
                time.sleep(self.interval)

        # Shutdown
        print("\n\n" + "=" * 60)
        print("  System shutting down...")
        self.enricher.print_stats()
        print("=" * 60)
        
    def stop(self):
        """Stop the background loop."""
        self._running = False

    def _run_cycle(self, cycle_num: int):
        """Execute one detection cycle."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n-- Cycle {cycle_num} [{timestamp}] --")

        # 1. Pull alerts
        if self.demo_mode:
            alerts = WazuhAlertPuller.create_demo_alerts(count=3)
        else:
            alerts = self.alert_puller.pull()

        if not alerts:
            print("  No new alerts.")
            return

        print(f"  Pulled {len(alerts)} alerts")

        # 2. Extract features
        features = self.feature_extractor.extract_batch(alerts)
        print(f"  Extracted features: {features.shape}")

        # 3. Run ensemble inference
        predictions = self.ensemble_engine.predict(features)

        # 4. Enrich and output
        enriched = self.enricher.enrich(alerts, predictions)
        
        # 5. UI Callback (if any)
        if self.ui_callback:
            self.ui_callback(enriched, self.enricher.get_stats())

        # Cycle summary
        intrusions = sum(1 for p in predictions if p["is_intrusion"])
        print(f"  Results: {intrusions} intrusions / {len(alerts)} alerts")

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        self._running = False


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file or return defaults."""
    if config_path and yaml:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f)

    # Default configuration
    return {
        "interval": 30,
        "demo_mode": False,
        "alert_puller": {
            "mode": "file",
            "alerts_file": "/var/ossec/logs/alerts/alerts.json",
            "lookback_seconds": 60,
        },
        "enricher": {
            "log_file": "logs/enriched_alerts.jsonl",
            "console_output": True,
            "min_level": 0.7,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="ML-Enhanced IDS Scheduler"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--interval", type=int, default=None,
                        help="Polling interval in seconds")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode with synthetic alerts")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.interval:
        config["interval"] = args.interval
    if args.demo:
        config["demo_mode"] = True

    scheduler = IDSScheduler(config)
    scheduler.start()


if __name__ == "__main__":
    main()
