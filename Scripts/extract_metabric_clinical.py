import os
import requests
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Results', 'metabric')
INTERMEDIATE_PATH = os.path.join(OUTPUT_DIR, 'intermediate')
FINAL_PATH = os.path.join(OUTPUT_DIR, 'final')

BASE_URL = "https://www.cbioportal.org/api/studies/brca_metabric/clinical-data"

PATIENT_FEATURES = [
    'AGE_AT_DIAGNOSIS',
    'SEX',
    'OS_MONTHS',
    'OS_STATUS',
    'CHEMOTHERAPY',
    'HORMONE_THERAPY',
    'RADIO_THERAPY',
    'COHORT',
    'LYMPH_NODES_EXAMINED_POSITIVE',
    'NPI',
    'CELLULARITY',
    'INFERRED_MENOPAUSAL_STATE',
    'HISTOLOGICAL_SUBTYPE',
]

SAMPLE_FEATURES = [
    'TUMOR_STAGE',
    'TUMOR_SIZE',
    'ER_STATUS',
    'PR_STATUS',
    'HER2_STATUS',
    'GRADE',
]

def fetch_clinical_data(data_type):
    url = f"{BASE_URL}?clinicalDataType={data_type}&projection=DETAILED"
    print(f"Fetching {data_type} clinical data from cBioPortal...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {data_type} data: {e}")
        return pd.DataFrame()

def process_patient_data(df_raw):
    if df_raw.empty:
        return pd.DataFrame()
    df = df_raw.pivot(index='patientId', columns='clinicalAttributeId', values='value').reset_index()
    available = [c for c in PATIENT_FEATURES if c in df.columns]
    missing = [c for c in PATIENT_FEATURES if c not in df.columns]
    if missing:
        print(f"Patient features not found in cBioPortal: {missing}")
    keep = ['patientId'] + available
    return df[keep].copy()

def process_sample_data(df_raw):
    if df_raw.empty:
        return pd.DataFrame()
    df = df_raw.pivot(
        index=['patientId', 'sampleId'],
        columns='clinicalAttributeId',
        values='value'
    ).reset_index()
    available = [c for c in SAMPLE_FEATURES if c in df.columns]
    missing = [c for c in SAMPLE_FEATURES if c not in df.columns]
    if missing:
        print(f"Sample features not found in cBioPortal: {missing}")
    keep = ['patientId', 'sampleId'] + available
    return df[keep].copy()

def clean_and_encode(df):
    # Survival outcome
    if 'OS_STATUS' in df.columns:
        df['event'] = (df['OS_STATUS'].str.upper().str.contains('DECEASED|1:DECEASED', na=False)).astype(int)
    if 'OS_MONTHS' in df.columns:
        df['survival_months'] = pd.to_numeric(df['OS_MONTHS'], errors='coerce')

    # Demographics
    if 'AGE_AT_DIAGNOSIS' in df.columns:
        df['age'] = pd.to_numeric(df['AGE_AT_DIAGNOSIS'], errors='coerce')
    if 'SEX' in df.columns:
        df['gender'] = df['SEX'].str.upper().str[0]

    # Receptor status: encode as binary; unknown/borderline set to NaN (MNAR candidates)
    for col in ['ER_STATUS', 'PR_STATUS', 'HER2_STATUS']:
        if col in df.columns:
            s = df[col].str.upper().str.strip()
            df[col + '_bin'] = np.where(s == 'POSITIVE', 1,
                               np.where(s == 'NEGATIVE', 0, np.nan))

    # Tumour features
    if 'TUMOR_SIZE' in df.columns:
        df['tumor_size'] = pd.to_numeric(df['TUMOR_SIZE'], errors='coerce')
    if 'LYMPH_NODES_EXAMINED_POSITIVE' in df.columns:
        df['lymph_nodes_positive'] = pd.to_numeric(df['LYMPH_NODES_EXAMINED_POSITIVE'], errors='coerce')
    if 'TUMOR_STAGE' in df.columns:
        df['tumour_stage'] = pd.to_numeric(df['TUMOR_STAGE'], errors='coerce')
    if 'NPI' in df.columns:
        df['npi'] = pd.to_numeric(df['NPI'], errors='coerce')
    if 'GRADE' in df.columns:
        df['grade'] = pd.to_numeric(df['GRADE'], errors='coerce')

    # Treatment flags
    for col, new in [('CHEMOTHERAPY', 'chemotherapy'), ('HORMONE_THERAPY', 'hormone_therapy'), ('RADIO_THERAPY', 'radio_therapy')]:
        if col in df.columns:
            s = df[col].str.upper().str.strip()
            df[new] = np.where(s == 'YES', 1, np.where(s == 'NO', 0, np.nan))

    # Categorical
    if 'INFERRED_MENOPAUSAL_STATE' in df.columns:
        df['menopausal_state'] = df['INFERRED_MENOPAUSAL_STATE'].str.upper().str.strip()
    if 'HISTOLOGICAL_SUBTYPE' in df.columns:
        df['histological_subtype'] = df['HISTOLOGICAL_SUBTYPE'].str.strip()
    if 'CELLULARITY' in df.columns:
        df['cellularity'] = df['CELLULARITY'].str.strip()
    if 'COHORT' in df.columns:
        df['cohort'] = pd.to_numeric(df['COHORT'], errors='coerce')

    raw_cols = [c for c in PATIENT_FEATURES + SAMPLE_FEATURES if c in df.columns]
    df = df.drop(columns=raw_cols)

    return df

def generate_missingness_report(df):
    # Exclude technical/administrative variables from being treated as model features
    # (cohort represents data collection site/hospital, not a clinical property)
    exclude = ['patientId', 'sampleId', 'cohort']
    feature_cols = [c for c in df.columns if c not in exclude]
    missing_count = df[feature_cols].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100.0
    
    report = pd.DataFrame({
        'feature': missing_count.index,
        'missing_count': missing_count.values,
        'missing_percentage': missing_pct.values
    }).sort_values('missing_percentage', ascending=False)
    report['high_missingness'] = (report['missing_percentage'] > 70.0).astype(int)
    report.to_csv(os.path.join(FINAL_PATH, 'missingness_profile_metabric.csv'), index=False)

    print(f"Total features: {len(feature_cols)}")
    print(f"Total patients: {len(df)}")
    for t in [10, 30, 50, 70]:
        print(f"Features above {t}% missingness: {(missing_pct > t).sum()}")
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-cache', type=str, help='Path to cached metabric cohort CSV')
    args = parser.parse_args()
    
    FINAL_OUT = os.path.join(FINAL_PATH, 'metabric_cohort.csv')

    if args.from_cache:
        print(f"Loading cached cohort from {args.from_cache}")
        df = pd.read_csv(args.from_cache)
    else:
        df_patient_raw = fetch_clinical_data('PATIENT')
        df_sample_raw  = fetch_clinical_data('SAMPLE')

        df_patient = process_patient_data(df_patient_raw)
        df_sample  = process_sample_data(df_sample_raw)

        df_patient.to_csv(os.path.join(INTERMEDIATE_PATH, 'metabric_patient_raw.csv'), index=False)
        df_sample.to_csv(os.path.join(INTERMEDIATE_PATH, 'metabric_sample_raw.csv'), index=False)

        df = pd.merge(df_patient, df_sample, on='patientId', how='inner')
        print(f"Merged cohort before cleaning: {df.shape}")

        df = clean_and_encode(df)

        # Exclude patients with missing survival outcome
        before = len(df)
        df = df[df['survival_months'].notna() & df['event'].notna()]
        print(f"Excluded {before - len(df)} patients with missing survival outcome")

    # V2 rule: survival_months <= 0 filter
    before_len = len(df)
    drop_mask = df['survival_months'] <= 0
    dropped = df[drop_mask]
    
    print(f"Dropping {len(dropped)} rows with survival_months <= 0.")
    if len(dropped) > 0:
        print(f"Dropped patientIds: {dropped['patientId'].tolist()}")
        
    df = df[~drop_mask].copy()
    
    # Assertions
    assert len(df) == len(df['patientId'].unique()), "Duplicate patientId found."
    assert (df['survival_months'] > 0).all(), "Found survival_months <= 0."
    assert set(df['event'].dropna().unique()).issubset({0, 1}), "Event not in {0, 1}."

    df.to_csv(FINAL_OUT, index=False)

    report = generate_missingness_report(df)

    print(f"Final dataset dimensions: {df.shape}")
    print(f"Event rate (OS): {df['event'].mean():.4f}")
    print(f"Median survival (OS, months): {df['survival_months'].median():.1f}")
    print("Extraction complete.")

if __name__ == "__main__":
    main()
