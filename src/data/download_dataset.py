"""
CIC-IDS2017 Dataset Downloader
Downloads CSV files from the public mirror and verifies integrity.

Usage:
    python -m src.data.download_dataset
    python -m src.data.download_dataset --output-dir data/raw
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Run: pip install requests tqdm")
    sys.exit(1)


# -- CIC-IDS2017 Dataset Files -------------------------------------
# Public mirror hosted by University of New Brunswick
BASE_URL = "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/TrafficLabelling/"

DATASET_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar and resume support."""
    try:
        # Check if file partially exists for resume
        initial_size = dest_path.stat().st_size if dest_path.exists() else 0
        headers = {"Range": f"bytes={initial_size}-"} if initial_size > 0 else {}

        response = requests.get(url, stream=True, headers=headers, timeout=30)

        # If server doesn't support range, start fresh
        if response.status_code == 200:
            initial_size = 0
            mode = "wb"
        elif response.status_code == 206:
            mode = "ab"
        else:
            print(f"  [FAIL] HTTP {response.status_code} for {url}")
            return False

        total_size = int(response.headers.get("content-length", 0)) + initial_size
        filename = dest_path.name

        with open(dest_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=initial_size,
                unit="B",
                unit_scale=True,
                desc=f"  {filename[:45]}",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True

    except requests.exceptions.RequestException as e:
        print(f"  [FAIL] Download error: {e}")
        return False


def verify_file(filepath: Path) -> bool:
    """Basic verification: check file exists and has reasonable size."""
    if not filepath.exists():
        return False
    size_mb = filepath.stat().st_size / (1024 * 1024)
    if size_mb < 1:
        print(f"  [WARN] File too small ({size_mb:.1f} MB): {filepath.name}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download CIC-IDS2017 dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for downloaded files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already exist and pass verification",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CIC-IDS2017 Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Files to download: {len(DATASET_FILES)}")
    print()

    success_count = 0
    skip_count = 0

    for filename in DATASET_FILES:
        dest_path = output_dir / filename
        url = BASE_URL + filename.replace(" ", "%20")

        # Skip existing verified files
        if args.skip_existing and verify_file(dest_path):
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] Already exists ({size_mb:.1f} MB): {filename}")
            skip_count += 1
            continue

        print(f"\n  Downloading: {filename}")
        if download_file(url, dest_path):
            if verify_file(dest_path):
                size_mb = dest_path.stat().st_size / (1024 * 1024)
                print(f"  [OK] Downloaded ({size_mb:.1f} MB)")
                success_count += 1
            else:
                print(f"  [FAIL] Verification failed")
        else:
            print(f"  [FAIL] Download failed")

    print()
    print("=" * 60)
    print(f"Results: {success_count} downloaded, {skip_count} skipped, "
          f"{len(DATASET_FILES) - success_count - skip_count} failed")
    print("=" * 60)

    if success_count + skip_count == len(DATASET_FILES):
        print("\n[OK] All files ready. Run preprocessing:")
        print("  python -m src.data.preprocess")
    else:
        print("\n[WARN] Some files missing. Re-run to retry failed downloads.")
        print("  If downloads keep failing, manually download from:")
        print(f"  {BASE_URL}")


if __name__ == "__main__":
    main()
