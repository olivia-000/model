"""
Washington DC crime data cleaner
Input:  dc_raw/Crime_Incidents_*.csv (2019-2024)
Output: data/processed/dc_clean.csv

Usage:
  python clean_dc.py \
    --input_dir ../../data/raw/dc_raw \
    --output ../../data/processed/dc_clean.csv
"""
import pandas as pd
import numpy as np
import glob
import os
import argparse

CRIME_MAP = {
    # violent
    'HOMICIDE':                   'violent',
    'ASSAULT W/DANGEROUS WEAPON': 'violent',
    'ROBBERY':                    'violent',
    'SEX ABUSE':                  'violent',
    # property
    'THEFT/OTHER':                'property',
    'THEFT F/AUTO':               'property',
    'MOTOR VEHICLE THEFT':        'property',
    'BURGLARY':                   'property',
    'ARSON':                      'property',
}

def clean(input_dir, output_path):
    files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    print(f'Found {len(files)} files:')
    for f in files:
        print(f'  {os.path.basename(f)}')

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False,
                             usecols=['REPORT_DAT','LATITUDE','LONGITUDE','OFFENSE'])
            dfs.append(df)
            print(f'  Loaded {os.path.basename(f)}: {len(df):,} rows')
        except Exception as e:
            print(f'  Error {os.path.basename(f)}: {e}')

    raw = pd.concat(dfs, ignore_index=True)
    print(f'\nRaw records: {len(raw):,}')

    # Drop missing coords
    raw = raw.dropna(subset=['LATITUDE','LONGITUDE','OFFENSE'])
    raw = raw[(raw['LATITUDE'] != 0) & (raw['LONGITUDE'] != 0)]

    # Map crime categories
    raw['crime_category'] = raw['OFFENSE'].map(CRIME_MAP).fillna('other')

    # Parse datetime
    raw['datetime'] = pd.to_datetime(raw['REPORT_DAT'], utc=True, errors='coerce')
    raw['datetime'] = raw['datetime'].dt.tz_localize(None)
    raw = raw.dropna(subset=['datetime'])

    out = pd.DataFrame({
        'city':           'DC',
        'datetime':       raw['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
        'latitude':       raw['LATITUDE'].astype(float),
        'longitude':      raw['LONGITUDE'].astype(float),
        'crime_category': raw['crime_category'],
    })

    out['datetime'] = pd.to_datetime(out['datetime'])
    out['hour']     = out['datetime'].dt.hour
    out['month']    = out['datetime'].dt.month
    out['weekday']  = out['datetime'].dt.weekday
    out['datetime'] = out['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    print(f'Clean records: {len(out):,}')
    print(f'Date range: {out["datetime"].min()} to {out["datetime"].max()}')
    print(f'\ncrime_category:\n{out["crime_category"].value_counts()}')

    out.to_csv(output_path, index=False)
    print(f'\nSaved to: {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output', default='dc_clean.csv')
    args = parser.parse_args()
    clean(args.input_dir, args.output)
