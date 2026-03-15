"""
Feature Configuration for CIC-IDS2017 Dataset
Centralized column definitions, label mappings, and constants.
"""

# -- CIC-IDS2017 Feature Columns ------------------------------------
# These are the raw column names from the CIC-IDS2017 dataset CSV files.
RAW_FEATURE_COLUMNS = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Total Length of Fwd Packets',
    'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
    'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
    'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
    'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
    'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

# Label column name in the dataset
LABEL_COLUMN = 'Label'

# -- Label Mapping --------------------------------------------------
# Map CIC-IDS2017 attack labels to our 5-class taxonomy
LABEL_MAPPING = {
    # Normal traffic
    'BENIGN': 'Normal',

    # DoS (Denial of Service) attacks
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'DDoS': 'DoS',
    'Heartbleed': 'DoS',

    # Probe / Reconnaissance attacks
    'PortScan': 'Probe',
    'FTP-Patator': 'Probe',
    'SSH-Patator': 'Probe',

    # R2L (Remote to Local) attacks
    'Web Attack \x96 Brute Force': 'R2L',
    'Web Attack \x96 XSS': 'R2L',
    'Web Attack \x96 Sql Injection': 'R2L',
    'Web Attack - Brute Force': 'R2L',
    'Web Attack - XSS': 'R2L',
    'Web Attack - Sql Injection': 'R2L',
    'Web Attack – Brute Force': 'R2L',
    'Web Attack – XSS': 'R2L',
    'Web Attack – Sql Injection': 'R2L',
    'Web Attack \xe2\x80\x93 Brute Force': 'R2L',
    'Web Attack \xe2\x80\x93 XSS': 'R2L',
    'Web Attack \xe2\x80\x93 Sql Injection': 'R2L',
    'Web Attack Â\x96 Brute Force': 'R2L',
    'Web Attack Â\x96 XSS': 'R2L',
    'Web Attack Â\x96 Sql Injection': 'R2L',
    'Infiltration': 'R2L',

    # U2R (User to Root) - privilege escalation
    'Bot': 'U2R',
}

# Numeric class IDs for ML models
CLASS_TO_ID = {
    'Normal': 0,
    'DoS': 1,
    'Probe': 2,
    'R2L': 3,
    'U2R': 4,
}

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

NUM_CLASSES = len(CLASS_TO_ID)

# -- Feature Selection ----------------------------------------------
# Top features by importance (from preliminary analysis)
# Used when --reduced flag is set for faster training
TOP_FEATURES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Total Length of Fwd Packets',
    'Total Length of Bwd Packets', 'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Fwd Packet Length Max', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Mean',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'Average Packet Size', 'Init_Win_bytes_forward',
    'Init_Win_bytes_backward', 'SYN Flag Count', 'ACK Flag Count',
    'PSH Flag Count', 'FIN Flag Count',
    'Subflow Fwd Bytes', 'Subflow Bwd Bytes',
    'Active Mean', 'Idle Mean',
]

# -- Model Configuration -------------------------------------------
MODEL_DIR = 'models'
DATA_RAW_DIR = 'data/raw'
DATA_PROCESSED_DIR = 'data/processed'

# Filenames for saved models
RF_MODEL_FILE = 'random_forest.joblib'
XGB_MODEL_FILE = 'xgboost.joblib'
DNN_MODEL_FILE = 'dnn_model.joblib'
META_MODEL_FILE = 'meta_classifier.joblib'
SCALER_FILE = 'scaler.joblib'

# -- Wazuh Alert Feature Mapping -----------------------------------
# Maps Wazuh JSON alert fields to CIC-IDS2017-style feature names
# Used by the inference pipeline to create compatible feature vectors
WAZUH_FEATURE_MAP = {
    'data.dstport': 'Destination Port',
    'data.srcport': 'Source Port',
    'data.bytes': 'Total Length of Fwd Packets',
    'data.duration': 'Flow Duration',
}

# Default feature values for missing Wazuh fields
DEFAULT_FEATURE_VALUE = 0.0

# -- Ensemble Configuration ----------------------------------------
ENSEMBLE_THRESHOLD = 0.7  # Confidence threshold for intrusion classification
POLLING_INTERVAL_SECONDS = 30  # Default polling interval
