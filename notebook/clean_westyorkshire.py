"""
West Yorkshire crime data cleaner
Input:  westyorkshire_raw/YYYY-MM/YYYY-MM-west-yorkshire-street.csv
Output: data/processed/west_yorkshire_clean.csv

Usage:
  python clean_westyorkshire.py \
    --input_dir ../data/raw/westyorkshire_raw \
    --output ../data/processed/west_yorkshire_clean.csv
"""
import pandas as pd
import numpy as np
import glob
import os
import argparse

CRIME_MAP = {
    'Violence and sexual offences': 'violent',
    'Robbery':                      'violent',
    'Possession of weapons':        'violent',
    'Burglary':                     'property',
    'Vehicle crime':                'property',
    'Theft from the person':        'property',
    'Other theft':                  'property',
    'Shoplifting':                  'property',
    'Criminal damage and arson':    'property',
    'Bicycle theft':                'property',
    'Anti-social behaviour':        'other',
    'Drugs':                        'other',
    'Public order':                 'other',
    'Other crime':                  'other',
    'Stalking and harassment':      'other',
    'Miscellaneous crimes against society': 'other',
}

def clean(input_dir, output_path):
    files = sorted(glob.glob(os.path.join(input_dir, '**', '*west-yorkshire-street.csv'), recursive=True))

    if not files:
        print(f'No west-yorkshire-street.csv found in {input_dir}')
        return

    print(f'Found {len(files)} files: {os.path.basename(files[0])} ~ {os.path.basename(files[-1])}')

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False,
                             usecols=['Month','Longitude','Latitude','Crime type'])
            dfs.append(df)
        except Exception as e:
            print(f'  Error {os.path.basename(f)}: {e}')

    raw = pd.concat(dfs, ignore_index=True)
    print(f'Raw records: {len(raw):,}')

    raw = raw.dropna(subset=['Latitude','Longitude','Crime type'])
    raw = raw[(raw['Latitude'] != 0) & (raw['Longitude'] != 0)]

    raw['crime_category'] = raw['Crime type'].map(CRIME_MAP).fillna('other')

    raw['datetime'] = pd.to_datetime(raw['Month'], format='%Y-%m')
    np.random.seed(42)
    n = len(raw)
    raw['datetime'] = (raw['datetime']
                       + pd.to_timedelta(np.random.randint(0, 28, n), unit='D')
                       + pd.to_timedelta(np.random.randint(0, 24, n), unit='h'))

    out = pd.DataFrame({
        'city':           'West Yorkshire',
        'datetime':       raw['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
        'latitude':       raw['Latitude'].astype(float),
        'longitude':      raw['Longitude'].astype(float),
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
    parser.add_argument('--output', default='west_yorkshire_clean.csv')
    args = parser.parse_args()
    clean(args.input_dir, args.output)
