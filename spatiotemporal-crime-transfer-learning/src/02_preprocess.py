"""
02_preprocess.py
----------------
Standardize crime data from all cities into a unified format.

Output columns (same for all cities):
  city, datetime, hour, month, weekday,
  crime_category, latitude, longitude, district
"""

import os
import pandas as pd
import numpy as np

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════
# Crime category mapping tables
# ════════════════════════════════════════════════════════════

NYC_CATEGORY_MAP = {
    "ASSAULT 3 & RELATED OFFENSES":       "violent",
    "FELONY ASSAULT":                      "violent",
    "MURDER & NON-NEGL. MANSLAUGHTER":    "violent",
    "RAPE":                                "violent",
    "ROBBERY":                             "violent",
    "KIDNAPPING & RELATED OFFENSES":       "violent",
    "BURGLARY":                            "property",
    "GRAND LARCENY":                       "property",
    "PETIT LARCENY":                       "property",
    "GRAND LARCENY OF MOTOR VEHICLE":      "property",
    "THEFT OF SERVICES":                   "property",
    "THEFT-FRAUD":                         "property",
    "FRAUDS":                              "property",
    "DANGEROUS DRUGS":                     "drug",
    "CANNABIS RELATED OFFENSES":           "drug",
    "DISORDERLY CONDUCT":                  "public_order",
    "ADMINISTRATIVE CODE":                 "public_order",
    "MISCELLANEOUS PENAL LAW":             "public_order",
    "OFFENSES AGAINST PUBLIC ADMINI":      "public_order",
    "VEHICLE AND TRAFFIC LAWS":            "public_order",
}

CHICAGO_CATEGORY_MAP = {
    "ASSAULT":                  "violent",
    "BATTERY":                  "violent",
    "HOMICIDE":                 "violent",
    "CRIM SEXUAL ASSAULT":      "violent",
    "ROBBERY":                  "violent",
    "KIDNAPPING":               "violent",
    "INTIMIDATION":             "violent",
    "HUMAN TRAFFICKING":        "violent",
    "BURGLARY":                 "property",
    "THEFT":                    "property",
    "MOTOR VEHICLE THEFT":      "property",
    "ARSON":                    "property",
    "CRIMINAL DAMAGE":          "property",
    "FRAUD":                    "property",
    "FORGERY & COUNTERFEITING": "property",
    "NARCOTICS":                "drug",
    "OTHER NARCOTIC VIOLATION": "drug",
    "CANNABIS POSSESSION":      "drug",
    "DECEPTIVE PRACTICE":       "public_order",
    "CRIMINAL TRESPASS":        "public_order",
    "WEAPONS VIOLATION":        "public_order",
    "LIQUOR LAW VIOLATION":     "public_order",
    "PROSTITUTION":             "public_order",
    "GAMBLING":                 "public_order",
    "PUBLIC PEACE VIOLATION":   "public_order",
    "INTERFERENCE WITH PUBLIC OFFICER": "public_order",
}

LA_CATEGORY_MAP = {
    # Violent
    "ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT": "violent",
    "ASSAULT WITH DEADLY WEAPON ON POLICE OFFICER":   "violent",
    "BATTERY - SIMPLE ASSAULT":                        "violent",
    "BATTERY ON A FIREFIGHTER":                        "violent",
    "ROBBERY":                                         "violent",
    "INTIMATE PARTNER - SIMPLE ASSAULT":               "violent",
    "INTIMATE PARTNER - AGGRAVATED ASSAULT":           "violent",
    "BRANDISH WEAPON":                                 "violent",
    "CRIMINAL HOMICIDE":                               "violent",
    "RAPE, FORCIBLE":                                  "violent",
    "RAPE, ATTEMPTED":                                 "violent",
    "KIDNAPPING":                                      "violent",
    "KIDNAPPING - GRAND ATTEMPT":                      "violent",
    "SHOTS FIRED AT INHABITED DWELLING":               "violent",
    "DISCHARGE FIREARMS/SHOTS FIRED":                  "violent",
    # Property
    "BURGLARY":                                        "property",
    "BURGLARY FROM VEHICLE":                           "property",
    "BURGLARY FROM VEHICLE, ATTEMPTED":                "property",
    "THEFT PLAIN - PETTY ($950 & UNDER)":              "property",
    "THEFT OF IDENTITY":                               "property",
    "THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)": "property",
    "THEFT FROM MOTOR VEHICLE - GRAND ($400 AND OVER)": "property",
    "VEHICLE - STOLEN":                                "property",
    "VEHICLE - ATTEMPT STOLEN":                        "property",
    "VANDALISM - FELONY ($400 & OVER, ALL STRUCTURES)": "property",
    "VANDALISM - MISDEAMEANOR ($399 OR UNDER)":        "property",
    "SHOPLIFTING - PETTY THEFT ($950 & UNDER)":        "property",
    "SHOPLIFTING-GRAND THEFT ($950.01 & OVER)":        "property",
    "PURSE SNATCHING":                                 "property",
    "PICKPOCKET":                                      "property",
    "CREDIT CARDS, FRAUD USE ($950 & UNDER)":          "property",
    "CREDIT CARDS, FRAUD USE ($950.01 & OVER)":        "property",
    # Drug
    "DRUGS, TO A MINOR":                               "drug",
    "DRUG PARAPHERNALIA, POSSESSION OF":               "drug",
    # Public order
    "TRESPASSING":                                     "public_order",
    "DISTURBING THE PEACE":                            "public_order",
    "DRUNK IN PUBLIC":                                 "public_order",
    "DISRUPT SCHOOL":                                  "public_order",
}

KARACHI_CATEGORY_MAP = {
    # Violent
    "Robbery":          "violent",
    "Murder":           "violent",
    "Assault":          "violent",
    "Kidnapping":       "violent",
    "Rape":             "violent",
    "Violence":         "violent",
    # Property
    "Theft":            "property",
    "Burglary":         "property",
    "Vehicle Theft":    "property",
    "Fraud":            "property",
    # Drug
    "Drug offenses":    "drug",
    "Drug Trafficking": "drug",
    # Public order
    "Other":            "public_order",
    "Public Order":     "public_order",
}

# ════════════════════════════════════════════════════════════
# NYC
# ════════════════════════════════════════════════════════════
def process_nyc():
    path = os.path.join(RAW_DIR, "nyc_raw.csv")
    if not os.path.exists(path):
        print("[NYC] Raw data not found. Run 01_download.py first.")
        return None

    print("[NYC] Processing...")
    df = pd.read_csv(path, low_memory=False)

    df["datetime"] = pd.to_datetime(
        df["cmplnt_fr_dt"].str[:10] + " " +
        df["cmplnt_fr_tm"].fillna("00:00:00"),
        errors="coerce"
    )
    df = df.dropna(subset=["datetime"])

    df["crime_category"] = df["ofns_desc"].str.strip().str.upper().map(
        {k.upper(): v for k, v in NYC_CATEGORY_MAP.items()}
    ).fillna("other")

    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"]  > 40.4) & (df["latitude"]  < 41.0)]
    df = df[(df["longitude"] > -74.3) & (df["longitude"] < -73.6)]

    out = pd.DataFrame({
        "city":           "NYC",
        "datetime":       df["datetime"],
        "hour":           df["datetime"].dt.hour,
        "month":          df["datetime"].dt.month,
        "weekday":        df["datetime"].dt.weekday,
        "crime_category": df["crime_category"],
        "latitude":       df["latitude"],
        "longitude":      df["longitude"],
        "district":       df["boro_nm"].fillna("UNKNOWN"),
    })
    out_path = os.path.join(PROC_DIR, "nyc_clean.csv")
    out.to_csv(out_path, index=False)
    print(f"[NYC] Done: {len(out):,} -> {out_path}")
    print(f"  Category distribution:\n{out['crime_category'].value_counts().to_string()}")
    return out

# ════════════════════════════════════════════════════════════
# Chicago
# ════════════════════════════════════════════════════════════
def process_chicago():
    path = os.path.join(RAW_DIR, "chicago_raw.csv")
    if not os.path.exists(path):
        print("[Chicago] Raw data not found. Run 01_download.py first.")
        return None

    print("[Chicago] Processing...")
    df = pd.read_csv(path, low_memory=False)

    df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df["crime_category"] = df["primary_type"].str.strip().str.upper().map(
        {k.upper(): v for k, v in CHICAGO_CATEGORY_MAP.items()}
    ).fillna("other")

    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"]  > 41.6) & (df["latitude"]  < 42.1)]
    df = df[(df["longitude"] > -87.95) & (df["longitude"] < -87.5)]

    out = pd.DataFrame({
        "city":           "Chicago",
        "datetime":       df["datetime"],
        "hour":           df["datetime"].dt.hour,
        "month":          df["datetime"].dt.month,
        "weekday":        df["datetime"].dt.weekday,
        "crime_category": df["crime_category"],
        "latitude":       df["latitude"],
        "longitude":      df["longitude"],
        "district":       df["district"].fillna("UNKNOWN").astype(str),
    })
    out_path = os.path.join(PROC_DIR, "chicago_clean.csv")
    out.to_csv(out_path, index=False)
    print(f"[Chicago] Done: {len(out):,} -> {out_path}")
    print(f"  Category distribution:\n{out['crime_category'].value_counts().to_string()}")
    return out

# ════════════════════════════════════════════════════════════
# Los Angeles
# ════════════════════════════════════════════════════════════
def process_la():
    path = os.path.join(RAW_DIR, "la_raw.csv")
    if not os.path.exists(path):
        print("[LA] Raw data not found. Run 01_download.py first.")
        return None

    print("[LA] Processing...")
    df = pd.read_csv(path, low_memory=False)

    # date_occ: "2020-01-01T00:00:00.000"
    # time_occ: integer like 1250 = 12:50
    df["date_str"]  = df["date_occ"].str[:10]
    df["time_occ"]  = df["time_occ"].astype(str).str.zfill(4)
    df["hour_str"]  = df["time_occ"].str[:2]
    df["datetime"]  = pd.to_datetime(
        df["date_str"] + " " + df["hour_str"] + ":00:00",
        errors="coerce"
    )
    df = df.dropna(subset=["datetime"])

    df["crime_category"] = df["crm_cd_desc"].str.strip().str.upper().map(
        {k.upper(): v for k, v in LA_CATEGORY_MAP.items()}
    ).fillna("other")

    df["latitude"]  = pd.to_numeric(df["lat"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    # LA bounding box
    df = df[(df["latitude"]  > 33.5)  & (df["latitude"]  < 34.5)]
    df = df[(df["longitude"] > -118.7) & (df["longitude"] < -117.6)]

    out = pd.DataFrame({
        "city":           "LA",
        "datetime":       df["datetime"],
        "hour":           df["datetime"].dt.hour,
        "month":          df["datetime"].dt.month,
        "weekday":        df["datetime"].dt.weekday,
        "crime_category": df["crime_category"],
        "latitude":       df["latitude"],
        "longitude":      df["longitude"],
        "district":       df["area_name"].fillna("UNKNOWN"),
    })
    out_path = os.path.join(PROC_DIR, "la_clean.csv")
    out.to_csv(out_path, index=False)
    print(f"[LA] Done: {len(out):,} -> {out_path}")
    print(f"  Category distribution:\n{out['crime_category'].value_counts().to_string()}")
    return out

# ════════════════════════════════════════════════════════════
# Karachi
# ════════════════════════════════════════════════════════════

# Karachi area -> approximate center coordinates
KARACHI_AREA_COORDS = {
    "Clifton":          (24.8136, 67.0299),
    "Defence":          (24.8007, 67.0601),
    "Gulshan-e-Iqbal":  (24.9265, 67.1025),
    "Korangi":          (24.8305, 67.1232),
    "Landhi":           (24.8501, 67.1578),
    "Lyari":            (24.8553, 66.9929),
    "Malir":            (24.8967, 67.2095),
    "Nazimabad":        (24.9213, 67.0401),
    "North Karachi":    (24.9677, 67.0652),
    "North Nazimabad":  (24.9480, 67.0366),
    "Orangi":           (24.9390, 66.9997),
    "Saddar":           (24.8607, 67.0105),
    "SITE":             (24.8967, 66.9850),
    "Shah Faisal":      (24.8618, 67.1378),
    "Surjani":          (24.9988, 67.0467),
    "Baldia":           (24.8864, 66.9645),
    "Bin Qasim":        (24.7751, 67.3167),
    "Gadap":            (25.0500, 67.2000),
    "Gulberg":          (24.9150, 67.0750),
    "Kemari":           (24.8230, 66.9759),
    "Keamari":          (24.8230, 66.9759),
    "Liaquatabad":      (24.9050, 67.0600),
    "Jamshed":          (24.8800, 67.0700),
    "Karachi Central":  (24.8950, 67.0300),
    "Karachi East":     (24.8900, 67.1200),
    "Karachi West":     (24.8700, 66.9900),
    "Karachi South":    (24.8200, 67.0200),
}
DEFAULT_KARACHI_COORD = (24.8607, 67.0105)  # Saddar as city center

def process_karachi():
    path = os.path.join(RAW_DIR, "karachi_raw.csv")
    if not os.path.exists(path):
        print("[Karachi] Raw data not found. Put karachi_raw.csv in data/raw/")
        return None

    print("[Karachi] Processing...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Columns: {list(df.columns)}")

    # Fixed column mapping for this specific Kaggle dataset:
    # Month, Karachi Area, Crm Cd Desc, Crime Count, ...
    date_col     = "Month"
    type_col     = "Crm Cd Desc"
    count_col    = "Crime Count"
    district_col = "Karachi Area"

    df["datetime"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce").fillna(1).astype(int)

    # Map crime descriptions using LA map (same terminology)
    df["crime_category"] = df[type_col].str.strip().str.upper().map(
        {k.upper(): v for k, v in LA_CATEGORY_MAP.items()}
    ).fillna("other")

    # Expand rows by Crime Count (simulate individual records)
    # Cap at 20 per row to avoid memory explosion
    df[count_col] = df[count_col].clip(upper=20)
    df_expanded = df.loc[df.index.repeat(df[count_col])].copy()
    df_expanded = df_expanded.reset_index(drop=True)
    print(f"  Expanded: {len(df):,} -> {len(df_expanded):,} records")

    # Assign coordinates from area name
    df_expanded["latitude"] = df_expanded[district_col].map(
        {k: v[0] for k, v in KARACHI_AREA_COORDS.items()}
    ).fillna(DEFAULT_KARACHI_COORD[0])
    df_expanded["longitude"] = df_expanded[district_col].map(
        {k: v[1] for k, v in KARACHI_AREA_COORDS.items()}
    ).fillna(DEFAULT_KARACHI_COORD[1])

    # Add small random jitter so grids are not all at exact same point
    rng = np.random.RandomState(42)
    df_expanded["latitude"]  += rng.uniform(-0.005, 0.005, len(df_expanded))
    df_expanded["longitude"] += rng.uniform(-0.005, 0.005, len(df_expanded))

    # Assign random hour based on month (simulate temporal variation)
    df_expanded["hour"]    = rng.randint(0, 24, len(df_expanded))
    df_expanded["weekday"] = rng.randint(0, 7,  len(df_expanded))

    out = pd.DataFrame({
        "city":           "Karachi",
        "datetime":       df_expanded["datetime"],
        "hour":           df_expanded["hour"],
        "month":          df_expanded["datetime"].dt.month,
        "weekday":        df_expanded["weekday"],
        "crime_category": df_expanded["crime_category"],
        "latitude":       df_expanded["latitude"].round(4),
        "longitude":      df_expanded["longitude"].round(4),
        "district":       df_expanded[district_col].fillna("UNKNOWN"),
    })
    out_path = os.path.join(PROC_DIR, "karachi_clean.csv")
    out.to_csv(out_path, index=False)
    print(f"[Karachi] Done: {len(out):,} -> {out_path}")
    print(f"  Category distribution:\n{out['crime_category'].value_counts().to_string()}")
    print(f"  NOTE: Hours/weekdays are randomized (source data is monthly aggregates)")
    return out
# ════════════════════════════════════════════════════════════
# Karachi Synthetic (2020-2025, individual records with coords)
# ════════════════════════════════════════════════════════════

KARACHI_SYNTHETIC_MAP = {
    # Violent
    'GANG VIOLENCE':      'violent',
    'MURDER':             'violent',
    'ARMED ROBBERY':      'violent',
    'KIDNAPPING':         'violent',
    'ASSAULT':            'violent',
    'SEXUAL ASSAULT':     'violent',
    'TERRORISM':          'violent',
    'EXTORTION':          'violent',
    'STREET CRIME':       'violent',
    # Property
    'THEFT':              'property',
    'BURGLARY':           'property',
    'VEHICLE THEFT':      'property',
    'MOTORCYCLE THEFT':   'property',
    'PHONE SNATCHING':    'property',
    'ROBBERY':            'property',
    'FRAUD':              'property',
    'CYBERCRIME':         'property',
    # Drug
    'DRUG TRAFFICKING':   'drug',
    'DRUG POSSESSION':    'drug',
    # Public order
    'VANDALISM':          'public_order',
    'LAND GRABBING':      'public_order',
    'ILLEGAL WEAPONS':    'public_order',
}

def process_karachi_synthetic():
    path = os.path.join(RAW_DIR, "karachi_synthetic_raw.csv")
    if not os.path.exists(path):
        print("[Karachi Synthetic] Not found: data/raw/karachi_synthetic_raw.csv")
        return None

    print("[Karachi Synthetic] Processing...")
    df = pd.read_csv(path, low_memory=False)

    df["datetime"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df["crime_category"] = df["CRIME_TYPE"].str.strip().str.upper().map(
        {k.upper(): v for k, v in KARACHI_SYNTHETIC_MAP.items()}
    ).fillna("other")

    df["latitude"]  = pd.to_numeric(df["LATITUDE"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    # Karachi bounding box
    df = df[(df["latitude"]  > 24.5) & (df["latitude"]  < 25.5)]
    df = df[(df["longitude"] > 66.5) & (df["longitude"] < 67.6)]

    out = pd.DataFrame({
        "city":           "Karachi",
        "datetime":       df["datetime"],
        "hour":           df["datetime"].dt.hour,
        "month":          df["datetime"].dt.month,
        "weekday":        df["datetime"].dt.weekday,
        "crime_category": df["crime_category"],
        "latitude":       df["latitude"],
        "longitude":      df["longitude"],
        "district":       df["TOWN"].fillna("UNKNOWN"),
    })

    out_path = os.path.join(PROC_DIR, "karachi_clean.csv")
    out.to_csv(out_path, index=False)
    print(f"[Karachi Synthetic] Done: {len(out):,} -> {out_path}")
    print(f"  Date range: {out['datetime'].min()} to {out['datetime'].max()}")
    print(f"  Category distribution:\n{out['crime_category'].value_counts().to_string()}")
    return out

# ════════════════════════════════════════════════════════════
# Merge all cities
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results = []

    nyc = process_nyc()
    if nyc is not None:
        results.append(nyc)

    chi = process_chicago()
    if chi is not None:
        results.append(chi)

    la = process_la()
    if la is not None:
        results.append(la)

    kar = process_karachi_synthetic()
    if kar is not None:
        results.append(kar)

    if results:
        combined = pd.concat(results, ignore_index=True)
        out_path = os.path.join(PROC_DIR, "all_cities.csv")
        combined.to_csv(out_path, index=False)
        print(f"\nMerged: {len(combined):,} records -> {out_path}")
        print(f"City breakdown:\n{combined['city'].value_counts().to_string()}")
        print(f"\nCategory breakdown:\n{combined['crime_category'].value_counts().to_string()}")
    else:
        print("No data processed.")

    print("\nNext: run the notebook from Cell 1")