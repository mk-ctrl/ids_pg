"""
Synthetic Dataset Generator for Pipeline Testing
Generates a realistic synthetic dataset matching CIC-IDS2017 feature schema.

Usage:
    python -m src.data.generate_synthetic
    python -m src.data.generate_synthetic --samples 50000
    python -m src.data.generate_synthetic --samples 5000 --output-dir data/raw
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.feature_config import RAW_FEATURE_COLUMNS, LABEL_COLUMN, CLASS_TO_ID


# Attack class distribution (roughly matching real CIC-IDS2017 proportions)
CLASS_DISTRIBUTION = {
    'BENIGN': 0.55,
    'DoS Hulk': 0.10,
    'DoS GoldenEye': 0.04,
    'DoS slowloris': 0.02,
    'DoS Slowhttptest': 0.02,
    'DDoS': 0.05,
    'PortScan': 0.08,
    'FTP-Patator': 0.03,
    'SSH-Patator': 0.02,
    'Web Attack - Brute Force': 0.03,
    'Web Attack - XSS': 0.02,
    'Web Attack - Sql Injection': 0.01,
    'Bot': 0.02,
    'Infiltration': 0.01,
}


def generate_feature_profiles():
    """
    Define feature distribution profiles for each attack class.
    Returns dict mapping class name to (mean_offset, std_multiplier) tuples.
    """
    profiles = {
        'BENIGN': {
            'Destination Port': (80, 200),
            'Flow Duration': (500000, 1000000),
            'Total Fwd Packets': (5, 3),
            'Total Backward Packets': (4, 3),
            'Total Length of Fwd Packets': (500, 300),
            'Total Length of Bwd Packets': (800, 500),
            'Flow Bytes/s': (10000, 5000),
            'Flow Packets/s': (50, 30),
            'Flow IAT Mean': (100000, 200000),
            'SYN Flag Count': (1, 0.5),
            'ACK Flag Count': (5, 3),
            'PSH Flag Count': (2, 2),
            'FIN Flag Count': (1, 0.5),
        },
        'DoS': {
            'Destination Port': (80, 50),
            'Flow Duration': (50000, 100000),
            'Total Fwd Packets': (100, 50),
            'Total Backward Packets': (2, 2),
            'Total Length of Fwd Packets': (5000, 3000),
            'Total Length of Bwd Packets': (100, 100),
            'Flow Bytes/s': (500000, 200000),
            'Flow Packets/s': (5000, 2000),
            'Flow IAT Mean': (1000, 5000),
            'SYN Flag Count': (50, 30),
            'ACK Flag Count': (10, 5),
            'PSH Flag Count': (0, 1),
            'FIN Flag Count': (0, 0.5),
        },
        'Probe': {
            'Destination Port': (30000, 15000),
            'Flow Duration': (10000, 50000),
            'Total Fwd Packets': (2, 1),
            'Total Backward Packets': (1, 1),
            'Total Length of Fwd Packets': (100, 50),
            'Total Length of Bwd Packets': (50, 30),
            'Flow Bytes/s': (5000, 3000),
            'Flow Packets/s': (200, 100),
            'Flow IAT Mean': (5000, 10000),
            'SYN Flag Count': (1, 0.5),
            'ACK Flag Count': (0, 0.5),
            'PSH Flag Count': (0, 0.5),
            'FIN Flag Count': (0, 0.3),
        },
        'R2L': {
            'Destination Port': (443, 100),
            'Flow Duration': (1000000, 2000000),
            'Total Fwd Packets': (20, 10),
            'Total Backward Packets': (15, 8),
            'Total Length of Fwd Packets': (2000, 1000),
            'Total Length of Bwd Packets': (3000, 2000),
            'Flow Bytes/s': (20000, 10000),
            'Flow Packets/s': (30, 20),
            'Flow IAT Mean': (50000, 100000),
            'SYN Flag Count': (1, 0.5),
            'ACK Flag Count': (10, 5),
            'PSH Flag Count': (5, 3),
            'FIN Flag Count': (1, 0.5),
        },
        'U2R': {
            'Destination Port': (8080, 2000),
            'Flow Duration': (2000000, 3000000),
            'Total Fwd Packets': (50, 20),
            'Total Backward Packets': (30, 15),
            'Total Length of Fwd Packets': (5000, 2000),
            'Total Length of Bwd Packets': (4000, 2000),
            'Flow Bytes/s': (30000, 15000),
            'Flow Packets/s': (40, 20),
            'Flow IAT Mean': (40000, 80000),
            'SYN Flag Count': (2, 1),
            'ACK Flag Count': (15, 8),
            'PSH Flag Count': (8, 4),
            'FIN Flag Count': (1, 0.5),
        },
    }
    return profiles


# Map attack labels to their class for profile lookup
LABEL_TO_CLASS = {
    'BENIGN': 'BENIGN',
    'DoS Hulk': 'DoS', 'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS', 'DoS Slowhttptest': 'DoS', 'DDoS': 'DoS',
    'PortScan': 'Probe', 'FTP-Patator': 'Probe', 'SSH-Patator': 'Probe',
    'Web Attack - Brute Force': 'R2L', 'Web Attack - XSS': 'R2L',
    'Web Attack - Sql Injection': 'R2L', 'Infiltration': 'R2L',
    'Bot': 'U2R',
}


def generate_synthetic_data(n_samples: int = 50000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic CIC-IDS2017-like dataset."""
    np.random.seed(seed)
    profiles = generate_feature_profiles()

    # Generate labels based on distribution
    labels = np.random.choice(
        list(CLASS_DISTRIBUTION.keys()),
        size=n_samples,
        p=list(CLASS_DISTRIBUTION.values()),
    )

    # Generate features
    data = {}
    for col in RAW_FEATURE_COLUMNS:
        values = np.zeros(n_samples)

        for i, label in enumerate(labels):
            attack_class = LABEL_TO_CLASS[label]
            profile = profiles[attack_class]

            if col in profile:
                mean, std = profile[col]
            else:
                # Default: small random values for unspecified features
                mean, std = 10, 20

            values[i] = max(0, np.random.normal(mean, std))

        data[col] = values

    data[LABEL_COLUMN] = labels

    df = pd.DataFrame(data)

    # Add some noise and correlations
    # Packet length features should correlate with total length
    for prefix in ['Fwd', 'Bwd']:
        total_col = f'Total Length of {prefix} Packets'
        if total_col in df.columns:
            for suffix in ['Max', 'Min', 'Mean', 'Std']:
                col = f'{prefix} Packet Length {suffix}'
                if col in df.columns:
                    df[col] = df[total_col] * np.random.uniform(0.1, 2.0, n_samples)

    # Ensure non-negative values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].clip(lower=0)

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic IDS dataset")
    parser.add_argument("--samples", type=int, default=50000,
                        help="Number of samples to generate (default: 50000)")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Synthetic CIC-IDS2017 Dataset Generator")
    print("=" * 60)
    print(f"  Samples:    {args.samples:,}")
    print(f"  Features:   {len(RAW_FEATURE_COLUMNS)}")
    print(f"  Classes:    {len(CLASS_DISTRIBUTION)}")
    print(f"  Output:     {output_dir.resolve()}")

    # Generate
    print("\n  Generating synthetic data...")
    df = generate_synthetic_data(n_samples=args.samples, seed=args.seed)

    # Show distribution
    print("\n  Label distribution:")
    for label, count in df[LABEL_COLUMN].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {label:<40s} {count:>6,} ({pct:.1f}%)")

    # Save as CSV (same format as real CIC-IDS2017)
    output_file = output_dir / "Synthetic-CIC-IDS2017.csv"
    df.to_csv(output_file, index=False)
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"\n  [OK] Saved: {output_file} ({size_mb:.1f} MB)")

    print("\n" + "=" * 60)
    print("[OK] Synthetic dataset ready!")
    print("  Next: python -m src.data.preprocess")
    print("=" * 60)


if __name__ == "__main__":
    main()
