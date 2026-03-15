"""
IDS Intrusion Simulator
Injects synthetic Wazuh alerts into the local alerts.json file 
to trigger the ML-Enhanced IDS Dashboard in Live Mode.

Usage: 
    python simulate_intrusion.py ddos
    python simulate_intrusion.py normal
    python simulate_intrusion.py portscan
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Wazuh alerts path (default from config)
ALERTS_FILE = "/var/ossec/logs/alerts/alerts.json"

# In Windows, we'll create a local dummy file if /var/ossec doesn't exist
if os.name == 'nt' and not os.path.exists(os.path.dirname(ALERTS_FILE)):
    ALERTS_FILE = "logs/wazuh_alerts_sim.json"

def ensure_file_exists():
    Path(ALERTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, 'w') as f:
            pass

def inject_alert(alert_type):
    ensure_file_exists()
    
    timestamp = datetime.now().isoformat()
    
    if alert_type == "ddos":
        # High traffic, many packets, same dest port
        alert = {
            "timestamp": timestamp,
            "rule": {"id": "100200", "level": 12, "description": "High volume of traffic detected (Possible DDoS)"},
            "agent": {"id": "001", "name": "web-server-01", "ip": "10.0.0.5"},
            "data": {
                "srcip": "203.0.113.50", "dstip": "10.0.0.5",
                "srcport": "54321", "dstport": "80",
                "Total Length of Fwd Packets": 50000,
                "Flow Duration": 100000,
            }
        }
        print(f"🚨 Simulating DDoS Attack against 10.0.0.5:80...")
        
    elif alert_type == "portscan":
        # Many different ports, short duration
        alert = {
            "timestamp": timestamp,
            "rule": {"id": "100201", "level": 10, "description": "Multiple connections to different ports (Port Scan)"},
            "agent": {"id": "001", "name": "web-server-01", "ip": "10.0.0.5"},
            "data": {
                "srcip": "198.51.100.22", "dstip": "10.0.0.5",
                "srcport": "44444", "dstport": "8080",
                "Flow Duration": 50,
                "Fwd Packet Length Max": 0,
            }
        }
        print(f"🚨 Simulating Port Scan from 198.51.100.22...")
        
    else:
        # Normal web traffic
        alert = {
            "timestamp": timestamp,
            "rule": {"id": "554", "level": 3, "description": "Normal HTTP GET Request"},
            "agent": {"id": "001", "name": "web-server-01", "ip": "10.0.0.5"},
            "data": {
                "srcip": "192.168.1.100", "dstip": "10.0.0.5",
                "srcport": "50123", "dstport": "443",
                "Flow Duration": 1500,
                "Total Length of Fwd Packets": 150,
            }
        }
        print(f"✅ Simulating Normal Web Traffic...")

    # Write to file (append mode, just like Wazuh does)
    with open(ALERTS_FILE, "a") as f:
        f.write(json.dumps(alert) + "\n")
        
    print(f"  -> Wrote alert to {ALERTS_FILE}")
    print("  -> The UI Dashboard should pick this up within 30 seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject Wazuh ALerts")
    parser.add_argument("type", choices=["ddos", "portscan", "normal", "burst"], 
                        default="ddos", nargs="?")
    args = parser.parse_args()
    
    if args.type == "burst":
        print("💥 Initiating burst of 5 attacks...")
        for _ in range(3):
            inject_alert("ddos")
            time.sleep(0.5)
        for _ in range(2):
            inject_alert("portscan")
            time.sleep(0.5)
    else:
        inject_alert(args.type)
