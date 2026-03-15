"""
Feature Extractor for Live Wazuh Alerts
Transforms raw Wazuh JSON alerts into feature vectors compatible with the trained models.
"""

import logging
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Optional

from src.data.feature_config import (
    RAW_FEATURE_COLUMNS, WAZUH_FEATURE_MAP, DEFAULT_FEATURE_VALUE,
    SCALER_FILE, MODEL_DIR,
)

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract ML features from Wazuh alert JSON objects."""

    def __init__(self, scaler_path: str = None):
        """
        Initialize with a fitted StandardScaler.
        
        Args:
            scaler_path: Path to saved scaler. If None, uses default.
        """
        if scaler_path is None:
            scaler_path = str(Path(MODEL_DIR) / SCALER_FILE)

        try:
            self.scaler = joblib.load(scaler_path)
            self.n_features = self.scaler.n_features_in_
            logger.info(f"Scaler loaded: {self.n_features} features")
        except FileNotFoundError:
            logger.warning(f"Scaler not found at {scaler_path}. Using identity transform.")
            self.scaler = None
            self.n_features = len(RAW_FEATURE_COLUMNS)

    def extract_single(self, alert: Dict) -> np.ndarray:
        """
        Extract feature vector from a single Wazuh alert.
        
        Maps Wazuh JSON fields to the CIC-IDS2017 feature schema.
        Missing features are filled with the default value.
        """
        features = []

        for col in RAW_FEATURE_COLUMNS[:self.n_features]:
            value = DEFAULT_FEATURE_VALUE

            # Check direct mapping first
            for wazuh_key, feature_name in WAZUH_FEATURE_MAP.items():
                if feature_name == col:
                    value = self._get_nested(alert, wazuh_key, DEFAULT_FEATURE_VALUE)
                    break

            # Heuristic feature extraction from alert data
            if value == DEFAULT_FEATURE_VALUE:
                value = self._extract_heuristic(alert, col)

            # Ensure numeric
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = DEFAULT_FEATURE_VALUE

            features.append(value)

        return np.array(features, dtype=np.float64)

    def extract_batch(self, alerts: List[Dict]) -> np.ndarray:
        """Extract features from a batch of alerts."""
        if not alerts:
            return np.empty((0, self.n_features))

        feature_matrix = np.vstack([self.extract_single(a) for a in alerts])

        # Apply scaling
        if self.scaler is not None:
            feature_matrix = self.scaler.transform(feature_matrix)

        return feature_matrix

    def _get_nested(self, d: Dict, key_path: str, default=None):
        """Get a value from a nested dict using dot notation."""
        keys = key_path.split(".")
        current = d
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key, default)
            else:
                return default
        return current if current is not None else default

    def _extract_heuristic(self, alert: Dict, feature_name: str) -> float:
        """
        Heuristic feature extraction based on Wazuh alert structure.
        Derives CIC-IDS2017-like features from available alert fields.
        """
        data = alert.get("data", {})
        rule = alert.get("rule", {})

        # Port features
        if "Port" in feature_name:
            if "Destination" in feature_name or "dst" in feature_name.lower():
                return self._safe_float(data.get("dstport", 0))
            elif "Source" in feature_name or "src" in feature_name.lower():
                return self._safe_float(data.get("srcport", 0))

        # Rule level as a proxy for severity-related features
        if "Flag" in feature_name:
            level = rule.get("level", 0)
            if level >= 10:
                return 1.0
            return 0.0

        # Duration
        if "Duration" in feature_name:
            return self._safe_float(data.get("duration", 0))

        # Packet/byte counts - use rule level as proxy
        if any(kw in feature_name for kw in ["Packets", "Length", "Bytes", "Size"]):
            level = rule.get("level", 0)
            return float(level * 100)  # Scale up

        # IAT (Inter-Arrival Time) features
        if "IAT" in feature_name:
            return 0.0

        return DEFAULT_FEATURE_VALUE

    @staticmethod
    def _safe_float(value, default=0.0) -> float:
        """Safely convert to float."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
