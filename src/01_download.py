"""
01_download.py
--------------
Download crime data for all cities.

Auto-download via API:  NYC, Chicago, LA
Manual download needed: Karachi (Kaggle), London (data.police.uk)
"""

import os
import requests
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

NYC_API = "https://data.cityofnewyork.us/resource/qgea-i56i.json"
CHI_API = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
LA_API  = "https://data.lacity.org/resource/2nrs-mtv8.json"


def download_nyc():
    out_path = os.path.join(RAW_DIR, "nyc_raw.csv")
    if os.path.exists(out_path):
        print(f"[NYC] Already exists, skipping.")
        return
    print("[NYC] Downloading 2006-2024...")
    all_dfs = []
    for year in range(2006, 2025):
        print(f"  {year}...", end=" ")
        params = {
            "$limit": 600000,
            "$where": (f"cmplnt_fr_dt >= '{year}-01-01T00:00:00' AND "
                       f"cmplnt_fr_dt <= '{year}-12-31T23:59:59'"),
            "$select": ("cmplnt_fr_dt,cmplnt_fr_tm,"
                        "ofns_desc,law_cat_cd,"
                        "boro_nm,addr_pct_cd,"
                        "latitude,longitude"),
        }
        try:
            r = requests.get(NYC_API, params=params, timeout=120)
            r.raise_for_status()
            df_yr = pd.DataFrame(r.json())
            print(f"{len(df_yr):,}")
            all_dfs.append(df_yr)
        except Exception as e:
            print(f"Error: {e}")
    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
        df.to_csv(out_path, index=False)
        print(f"[NYC] Done: {len(df):,} -> {out_path}")


def download_chicago():
    out_path = os.path.join(RAW_DIR, "chicago_raw.csv")
    if os.path.exists(out_path):
        print(f"[Chicago] Already exists, skipping.")
        return
    print("[Chicago] Downloading 2001-2024...")
    all_dfs = []
    for year in range(2001, 2025):
        print(f"  {year}...", end=" ")
        params = {
            "$limit": 300000,
            "$where": f"year = {year}",
            "$select": ("date,"
                        "primary_type,description,"
                        "community_area,district,"
                        "latitude,longitude"),
        }
        try:
            r = requests.get(CHI_API, params=params, timeout=120)
            r.raise_for_status()
            df_yr = pd.DataFrame(r.json())
            print(f"{len(df_yr):,}")
            all_dfs.append(df_yr)
        except Exception as e:
            print(f"Error: {e}")
    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
        df.to_csv(out_path, index=False)
        print(f"[Chicago] Done: {len(df):,} -> {out_path}")


def download_la():
    out_path = os.path.join(RAW_DIR, "la_raw.csv")
    if os.path.exists(out_path):
        print(f"[LA] Already exists, skipping.")
        return
    print("[LA] Downloading 2020-2024...")
    all_dfs = []
    for year in range(2020, 2025):
        print(f"  {year}...", end=" ")
        params = {
            "$limit": 400000,
            "$where": (f"date_occ >= '{year}-01-01T00:00:00' AND "
                       f"date_occ <= '{year}-12-31T23:59:59'"),
            "$select": "date_occ,time_occ,crm_cd_desc,area_name,lat,lon",
        }
        try:
            r = requests.get(LA_API, params=params, timeout=120)
            r.raise_for_status()
            df_yr = pd.DataFrame(r.json())
            print(f"{len(df_yr):,}")
            all_dfs.append(df_yr)
        except Exception as e:
            print(f"Error: {e}")
    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
        df.to_csv(out_path, index=False)
        print(f"[LA] Done: {len(df):,} -> {out_path}")


def check_london():
    london_dir = os.path.join(RAW_DIR, "london")
    if os.path.exists(london_dir):
        files = [f for f in os.listdir(london_dir) if f.endswith('.csv')]
        if files:
            print(f"[London] Found {len(files)} CSV files in {london_dir}")
            return
    print("""
[London] Manual download required:
  1. Go to https://data.police.uk/data/
  2. Select: Metropolitan Police Service
  3. Date range: 2018-01 to 2023-12
  4. Download and extract ZIP
  5. Put all CSV files into data/raw/london/
""")


def check_karachi():
    kar_path = os.path.join(RAW_DIR, "karachi_raw.csv")
    if os.path.exists(kar_path):
        print(f"[Karachi] Found: {kar_path}")
    else:
        print("""
[Karachi] Manual download required:
  1. Go to https://www.kaggle.com/datasets/sarcasmos/karachi-urban-crime-analysis-with-demographic
  2. Download and extract ZIP
  3. Rename CSV to karachi_raw.csv
  4. Put in data/raw/
""")


if __name__ == "__main__":
    download_nyc()
    download_chicago()
    download_la()
    check_london()
    check_karachi()
    print("\nAll done!")