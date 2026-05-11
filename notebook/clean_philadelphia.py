"""
Philadelphia crime data cleaner
Input:  philly_raw/incidents_part1_part2_20XX.csv (2019-2024)
Output: data/processed/philadelphia_clean.csv

Usage:
  python clean_philadelphia.py \
    --input_dir ../../data/raw/philly_raw \
    --output ../../data/processed/philadelphia_clean.csv
"""
import pandas as pd
import numpy as np
import glob
import os
import argparse

CRIME_MAP = {
    # violent
    'Homicide - Criminal':              'violent',
    'Rape':                             'violent',
    'Aggravated Assault Firearm':       'violent',
    'Aggravated Assault No Firearm':    'violent',
    'Robbery Firearm':                  'violent',
    'Robbery No Firearm':               'violent',
    'Other Sex Offenses (Not Commercialized)': 'violent',
    # property
    'Thefts':                           'property',
    'Theft from Vehicle':               'property',
    'Motor Vehicle Theft':              'property',
    'Burglary Residential':             'property',
    'Burglary Non-Residential':         'property',
    'Vandalism/Criminal Mischief':      'property',
    'Fraud':                            'property',
    # other
    'All Other Offenses':               'other',
    'Other Assaults':                   'other',
    'Narcotic / Drug Law Violations':   'other',
    'Weapon Violations':                'other',
    'DRIVING UNDER THE INFLUENCE':      'other',
    'Disorderly Conduct':               'other',
    'Prostitution and Commercialized Vice': 'other',
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
                             usecols=['dispatch_date_time','lat','lng','text_general_code','hour'])
            dfs.append(df)
            print(f'  Loaded {os.path.basename(f)}: {len(df):,} rows')
        except Exception as e:
            print(f'  Error {os.path.basename(f)}: {e}')

    raw = pd.concat(dfs, ignore_index=True)
    print(f'\nRaw records: {len(raw):,}')

    # Drop missing coords
    raw = raw.dropna(subset=['lat','lng','text_general_code'])
    raw = raw[(raw['lat'] != 0) & (raw['lng'] != 0)]

    # Map crime categories
    raw['crime_category'] = raw['text_general_code'].map(CRIME_MAP).fillna('other')

    # Parse datetime
    raw['datetime'] = pd.to_datetime(raw['dispatch_date_time'], utc=True, errors='coerce')
    raw['datetime'] = raw['datetime'].dt.tz_localize(None)
    raw = raw.dropna(subset=['datetime'])

    out = pd.DataFrame({
        'city':           'Philadelphia',
        'datetime':       raw['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
        'latitude':       raw['lat'].astype(float),
        'longitude':      raw['lng'].astype(float),
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
    parser.add_argument('--output', default='philadelphia_clean.csv')
    args = parser.parse_args()
    clean(args.input_dir, args.output)
