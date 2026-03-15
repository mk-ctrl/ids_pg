# Setup Guide

Complete step-by-step guide to deploy the Hybrid Ensemble ML-Enhanced IDS.

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.10+ | ML pipeline |
| Docker Desktop | Latest | Wazuh deployment |
| Windows 10/11 | Pro/Enterprise | Target endpoint |
| 8 GB RAM | Minimum | Wazuh manager VM |

## 1. Clone & Install Dependencies

```bash
git clone <repo-url>
cd ramya-pg

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac

# Install Python packages
pip install -r requirements.txt
```

## 2. Deploy Wazuh (Docker)

```bash
cd docker
docker compose up -d
```

Wait 2-3 minutes, then access:
- **Wazuh Dashboard**: https://localhost:443
- **Default credentials**: admin / SecretPassword123!
- **API**: https://localhost:55000

Verify:
```bash
docker compose ps              # All 3 services should be "running"
curl -k https://localhost:55000 # Should return API info
```

## 3. Install Sysmon on Windows Target

1. Download [Sysmon](https://learn.microsoft.com/en-us/sysinternals/downloads/sysmon)
2. Install with our config:
   ```cmd
   sysmon64.exe -accepteula -i configs\sysmon-config.xml
   ```
3. Verify: `Get-WinEvent -LogName "Microsoft-Windows-Sysmon/Operational" -MaxEvents 5`

## 4. Install Wazuh Agent on Windows

1. Download: https://packages.wazuh.com/4.x/windows/wazuh-agent-4.8.0-1.msi
2. Install:
   ```cmd
   wazuh-agent-4.8.0-1.msi /q WAZUH_MANAGER="MANAGER_IP"
   ```
3. Copy agent config:
   ```cmd
   copy configs\ossec-agent.conf "C:\Program Files (x86)\ossec-agent\ossec.conf"
   ```
4. Edit `ossec.conf`: Replace `MANAGER_IP_HERE` with your Wazuh manager's IP
5. Start service:
   ```cmd
   net start WazuhSvc
   ```

## 5. Load Custom Rules

Copy rules to the Wazuh manager container:
```bash
docker cp configs/custom-rules.xml wazuh-manager:/var/ossec/etc/rules/custom-rules.xml
docker exec wazuh-manager /var/ossec/bin/wazuh-control restart
```

## 6. Download & Preprocess Dataset

```bash
# Download CIC-IDS2017 (~2.5 GB)
python -m src.data.download_dataset

# Preprocess (clean, normalize, split)
python -m src.data.preprocess

# Quick test with small sample
python -m src.data.preprocess --test-mode
```

## 7. Train ML Models

```bash
# Train each model individually
python -m src.models.train_rf
python -m src.models.train_xgb
python -m src.models.train_dnn --epochs 30

# Train stacking ensemble (requires all 3 base models)
python -m src.models.train_ensemble

# Quick smoke test
python -m src.models.train_rf --test-mode
python -m src.models.train_xgb --test-mode
python -m src.models.train_dnn --test-mode --epochs 2
python -m src.models.train_ensemble --test-mode
```

## 8. Run Inference Pipeline

```bash
# Demo mode (synthetic alerts)
python -m src.inference.scheduler --demo

# Live mode (requires Wazuh)
python -m src.inference.scheduler --config configs/inference_config.yaml

# Custom interval
python -m src.inference.scheduler --demo --interval 10
```

## 9. Run Evaluation

```bash
# Metrics demo
python -m src.evaluation.metrics --demo

# Full comparison (requires trained models + processed data)
python -m src.evaluation.compare
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Docker compose fails | Ensure Docker Desktop is running with sufficient RAM |
| Agent not connecting | Check firewall for ports 1514/1515, verify MANAGER_IP |
| ML import errors | Verify `pip install -r requirements.txt` completed |
| Dataset download fails | Check internet, retry or download manually from UNB |
| TensorFlow GPU errors | Install CPU version: `pip install tensorflow-cpu` |
