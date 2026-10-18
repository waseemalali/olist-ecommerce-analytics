"""Prepare the raw Olist data for the pipeline.

The 9 Olist CSV files are stored in `data/raw/` as zip files so they fit
GitHub's file size limit. The geolocation table (~61 MB) is split into two
parts. This script:

  1. extracts every .zip in data/raw/
  2. joins the geolocation parts back into olist_geolocation_dataset.csv

Run it once before main.py:

    python prepare_data.py
"""
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
GEO_FILE = RAW_DIR / "olist_geolocation_dataset.csv"


def extract_zips():
    zips = sorted(RAW_DIR.glob("*.zip"))
    if not zips:
        raise SystemExit(f"No .zip files found in {RAW_DIR}")
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW_DIR)
        print(f"  extracted {z.name}")


def join_geolocation_parts():
    parts = sorted(RAW_DIR.glob("olist_geolocation_dataset_part*.csv"))
    if not parts:
        return
    with open(GEO_FILE, "w", encoding="utf-8", newline="") as out:
        for i, part in enumerate(parts):
            with open(part, encoding="utf-8", newline="") as f:
                header = f.readline()
                if i == 0:
                    out.write(header)
                out.write(f.read())
            part.unlink()
    print(f"  joined {len(parts)} parts -> {GEO_FILE.name}")


if __name__ == "__main__":
    print("Preparing Olist data in", RAW_DIR)
    extract_zips()
    join_geolocation_parts()
    csvs = sorted(p.name for p in RAW_DIR.glob("*.csv"))
    print(f"Done: {len(csvs)} CSV files ready. Now run: python main.py")
