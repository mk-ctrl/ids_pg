"""
Wazuh Alert Puller
Pulls alerts from Wazuh REST API or tails the alerts.json log file.

Can operate in two modes:
1. API mode: Queries Wazuh API for recent alerts
2. File mode: Reads from alerts.json log file
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

try:
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


class WazuhAlertPuller:
    """Pull alerts from Wazuh Manager."""

    def __init__(self, config: dict):
        """
        Initialize puller with config.
        
        Config keys:
            mode: "api" or "file"
            api_url: Wazuh API URL (e.g., https://localhost:55000)
            api_user: API username
            api_password: API password
            alerts_file: Path to alerts.json (for file mode)
            lookback_seconds: How far back to fetch alerts (default: 60)
        """
        self.mode = config.get("mode", "file")
        self.api_url = config.get("api_url", "https://localhost:55000")
        self.api_user = config.get("api_user", "wazuh-wui")
        self.api_password = config.get("api_password", "")
        self.alerts_file = config.get("alerts_file", "/var/ossec/logs/alerts/alerts.json")
        
        # Windows fallback for local testing
        import os
        if os.name == 'nt' and not os.path.exists(os.path.dirname(self.alerts_file)):
            self.alerts_file = "logs/wazuh_alerts_sim.json"
            
        self.lookback_seconds = config.get("lookback_seconds", 60)
        self._token = None
        self._token_time = None
        self._last_file_position = 0

    def pull(self) -> List[Dict]:
        """Pull recent alerts based on configured mode."""
        if self.mode == "api":
            return self._pull_from_api()
        else:
            return self._pull_from_file()

    def _authenticate(self) -> Optional[str]:
        """Authenticate with Wazuh API and get JWT token."""
        if not requests:
            logger.error("'requests' library required for API mode")
            return None

        # Reuse token if less than 15 minutes old
        if (self._token and self._token_time and
                (datetime.now() - self._token_time).seconds < 900):
            return self._token

        try:
            response = requests.post(
                f"{self.api_url}/security/user/authenticate",
                auth=(self.api_user, self.api_password),
                verify=False,
                timeout=10,
            )
            if response.status_code == 200:
                self._token = response.json()["data"]["token"]
                self._token_time = datetime.now()
                logger.info("Wazuh API authentication successful")
                return self._token
            else:
                logger.error(f"Auth failed: HTTP {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return None

    def _pull_from_api(self) -> List[Dict]:
        """Pull alerts via Wazuh REST API."""
        token = self._authenticate()
        if not token:
            return []

        headers = {"Authorization": f"Bearer {token}"}

        try:
            # Get recent alerts
            response = requests.get(
                f"{self.api_url}/alerts",
                headers=headers,
                params={
                    "limit": 100,
                    "sort": "-timestamp",
                    "q": f"timestamp>{int(time.time()) - self.lookback_seconds}",
                },
                verify=False,
                timeout=15,
            )

            if response.status_code == 200:
                data = response.json()
                alerts = data.get("data", {}).get("affected_items", [])
                logger.info(f"Pulled {len(alerts)} alerts from API")
                return alerts
            else:
                logger.error(f"API error: HTTP {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"API pull error: {e}")
            return []

    def _pull_from_file(self) -> List[Dict]:
        """Pull alerts by reading the alerts.json log file (tail mode)."""
        alerts_path = Path(self.alerts_file)

        if not alerts_path.exists():
            logger.warning(f"Alerts file not found: {alerts_path}")
            return []

        alerts = []
        try:
            with open(alerts_path, "r", encoding="utf-8") as f:
                # Seek to last known position
                f.seek(self._last_file_position)
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        alert = json.loads(line)
                        alerts.append(alert)
                    except json.JSONDecodeError:
                        continue

                # Update position
                self._last_file_position = f.tell()

            logger.info(f"Read {len(alerts)} new alerts from file")

        except Exception as e:
            logger.error(f"File read error: {e}")

        return alerts

    @staticmethod
    def create_demo_alerts(count: int = 5) -> List[Dict]:
        """Generate synthetic demo alerts for testing."""
        import random

        alert_templates = [
            {
                "rule": {"id": "100100", "level": 10, "description": "Suspicious PowerShell: Encoded command detected"},
                "agent": {"id": "001", "name": "win-endpoint-01", "ip": "192.168.1.100"},
                "data": {
                    "srcip": "192.168.1.100",
                    "dstip": "10.0.0.1",
                    "srcport": "49152",
                    "dstport": "443",
                    "win": {"eventdata": {"commandLine": "powershell -encodedCommand SGVsbG8="}},
                },
                "timestamp": datetime.now().isoformat(),
            },
            {
                "rule": {"id": "100111", "level": 10, "description": "RDP brute-force attack detected"},
                "agent": {"id": "001", "name": "win-endpoint-01", "ip": "192.168.1.100"},
                "data": {
                    "srcip": "10.0.0.50",
                    "dstip": "192.168.1.100",
                    "srcport": "12345",
                    "dstport": "3389",
                },
                "timestamp": datetime.now().isoformat(),
            },
            {
                "rule": {"id": "100120", "level": 12, "description": "Credential dumping: Suspicious LSASS access"},
                "agent": {"id": "002", "name": "win-server-01", "ip": "192.168.1.200"},
                "data": {
                    "srcip": "192.168.1.200",
                    "dstip": "192.168.1.200",
                    "srcport": "0",
                    "dstport": "0",
                },
                "timestamp": datetime.now().isoformat(),
            },
            {
                "rule": {"id": "554", "level": 3, "description": "Normal system event"},
                "agent": {"id": "001", "name": "win-endpoint-01", "ip": "192.168.1.100"},
                "data": {
                    "srcip": "192.168.1.100",
                    "dstip": "8.8.8.8",
                    "srcport": "50000",
                    "dstport": "53",
                },
                "timestamp": datetime.now().isoformat(),
            },
            {
                "rule": {"id": "100150", "level": 12, "description": "Ransomware precursor: Shadow copy deletion"},
                "agent": {"id": "001", "name": "win-endpoint-01", "ip": "192.168.1.100"},
                "data": {
                    "srcip": "192.168.1.100",
                    "dstip": "192.168.1.100",
                    "srcport": "0",
                    "dstport": "0",
                },
                "timestamp": datetime.now().isoformat(),
            },
        ]

        return [random.choice(alert_templates) for _ in range(count)]
