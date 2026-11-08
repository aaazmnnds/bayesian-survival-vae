import os
import argparse
import re
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')
def sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v) for v in obj]
    if hasattr(obj, 'tolist'):
        return [sanitize(v) for v in obj.tolist()]
    return obj


FEATURE_WINDOW_H = 24
LAB_PRE_WINDOW_H = 6
HORIZON_DAYS = 28
MISSING_THRESHOLD = 0.70
NEAR_CONSTANT_SHARE = 0.99
COMORBIDITY_MODE = "strict"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EICU_PATH = None
OUT_DIR = os.path.join(PROJECT_ROOT, 'Results', 'final')
FINAL_OUT = os.path.join(PROJECT_ROOT, 'Results', 'eicu', 'final', 'eicu_sepsis_cohort.csv')
LOG_OUT = os.path.join(OUT_DIR, 'eicu_extraction_log.json')
BOUNDS_FILE = os.path.join(OUT_DIR, 'bounds_approved.csv')

LAB_MAPPING = {
    'albumin': 'Albumin',
    'anion gap': 'AnionGap',
    'bicarbonate': 'Bicarbonate',
    'BUN': 'BUN',
    'calcium': 'Calcium',
    'chloride': 'Chloride',
    'creatinine': 'Creatinine',
    'glucose': 'Glucose',
    'sodium': 'Sodium',
    'potassium': 'Potassium',
    'lactate': 'Lactate',
    'Hgb': 'Hemoglobin',
    'Hct': 'Hematocrit',
    'platelets x 1000': 'Platelets',
    'WBC x 1000': 'WBC',
    '-lymphs': 'Lymphocytes',
    '-polys': 'Neutrophils',
    'PT - INR': 'INR',
    'PT': 'PT',
    'PTT': 'PTT',
    'ALT (SGPT)': 'ALT',
    'AST (SGOT)': 'AST',
    'total bilirubin': 'Bilirubin_Total',
    'pH': 'pH',
    'paO2': 'pO2',
    'paCO2': 'pCO2',
    'Base Excess': 'BaseExcess',
    'magnesium': 'Magnesium',
    'phosphate': 'Phosphate',
    'LDH': 'LDH',
    'troponin - T': 'Troponin_T',
    'CRP': 'CRP'
}

COMORBIDITIES = {
    'CCI_MI':       {'icd10': ['I21','I22','I252'], 'icd9': ['410','412']},
    'CCI_CHF':      {'icd10': ['I50','I42'],         'icd9': ['428','4254']},
    'CCI_PVD':      {'icd10': ['I71','I790'],        'icd9': ['4402','4408','4409']},
    'CCI_Stroke':   {'icd10': ['I60','I61','I62','I63','I64'], 'icd9': ['430','431','432','433','434','436']},
    'CCI_Dementia': {'icd10': ['F00','F01','F02','F03'], 'icd9': ['290']},
    'CCI_COPD':     {'icd10': ['J40','J41','J42','J43','J44'], 'icd9': ['491','492','496']},
    'CCI_Renal':    {'icd10': ['N18','N19'],          'icd9': ['585','586']},
    'CCI_Liver':    {'icd10': ['K70','K71','K72','K73','K74'], 'icd9': ['571']},
    'CCI_Cancer':   {'icd10': ['C0','C1','C2','C3','C4'], 'icd9': ['140','141','142','143','144','145','146','147','148','149','150']},
    'CCI_Diabetes': {'icd10': ['E10','E11','E12','E13','E14'], 'icd9': ['250']},
    'CCI_HIV':      {'icd10': ['B20','B21','B22','B23','B24'], 'icd9': ['042','043','044']},
}

def load_bounds():
    if not os.path.exists(BOUNDS_FILE):
        raise FileNotFoundError(f"Bounds file not found: {BOUNDS_FILE}")
    
    bounds_df = pd.read_csv(BOUNDS_FILE)
    bounds_dict = {}
    missing_bounds = []
    
    for _, row in bounds_df.iterrows():
        var = row['variable']
        lower = row['lower']
        upper = row['upper']
        
        if pd.isna(lower) or pd.isna(upper) or str(lower).strip() == "" or str(upper).strip() == "":
            missing_bounds.append(var)
        else:
            bounds_dict[var] = (float(lower), float(upper))
            
    if missing_bounds:
        raise ValueError(f"Variables missing lower or upper bounds in {BOUNDS_FILE}: {missing_bounds}")
        
    return bounds_dict

def apply_bounds(df, var_col, val_col, bounds_dict, log_counts):
    vars_present = df[var_col].unique()
    mask = pd.Series(False, index=df.index)
    
    for v in vars_present:
        if v in bounds_dict:
            lower, upper = bounds_dict[v]
            v_mask = df[var_col] == v
            out_of_bounds = v_mask & ((df[val_col] < lower) | (df[val_col] > upper))
            mask = mask | out_of_bounds
            log_counts[v] = log_counts.get(v, 0) + int(out_of_bounds.sum())
            
    df.loc[mask, val_col] = np.nan
    return df

def normalize_unit(var, unit):
    u = str(unit).strip()
    if var in ['Sodium', 'Potassium', 'Chloride', 'Bicarbonate', 'AnionGap', 'BaseExcess'] and u in ['mEq/L', 'mmol/L']:
        return 'mEq/L_or_mmol/L'
    if u in ['K/uL', 'K/mcL']:
        return 'K/uL_or_K/mcL'
    if u in ['IU/L', 'Units/L']:
        return 'IU/L_or_Units/L'
    return u

def check_units(df, var_col, unit_col, extract_log, log_key, log_missing_key):
    units_dict = {}
    missing_dict = {}
    errors = []
    vars_present = df[var_col].dropna().unique()
    for v in vars_present:
        v_df = df[df[var_col] == v]
        missing_count = int(v_df[unit_col].isna().sum())
        missing_dict[v] = missing_count
        
        v_df = v_df.dropna(subset=[unit_col])
        if v_df.empty:
            continue
        v_df['norm_unit'] = v_df[unit_col].apply(lambda x: normalize_unit(v, x))
        counts = v_df['norm_unit'].value_counts().to_dict()
        units_dict[v] = counts
        if len(counts) > 1:
            errors.append(f"{v}: {counts}")
    extract_log[log_key] = units_dict
    extract_log[log_missing_key] = missing_dict
    if errors:
        raise ValueError("Variables with multiple units detected:\n" + "\n".join(errors))

def generate_suspected_infection(patient_df, extract_log):
    cohort_ids = set(patient_df['patientunitstayid'])
    med_file = os.path.join(EICU_PATH, 'medication.csv.gz')
    antibiotics = [
        'vancomycin', 'piperacillin', 'meropenem', 'cefepime', 'metronidazole',
        'ciprofloxacin', 'ampicillin', 'gentamicin', 'azithromycin',
        'levofloxacin', 'clindamycin', 'linezolid', 'daptomycin',
        'ceftriaxone', 'ceftazidime'
    ]
    pattern = r'\b(?:' + '|'.join(antibiotics) + r')\b'
    
    meds_chunks = []
    meds_read = 0
    meds_matched = 0
    
    if os.path.exists(med_file):
        for chunk in pd.read_csv(med_file, chunksize=1000000, low_memory=False):
            chunk = chunk[chunk['patientunitstayid'].isin(cohort_ids)]
            meds_read += len(chunk)
            chunk['drugname'] = chunk['drugname'].fillna('').str.lower()
            chunk = chunk[chunk['drugname'].str.contains(pattern, regex=True)]
            meds_matched += len(chunk)
            if not chunk.empty:
                meds_chunks.append(chunk[['patientunitstayid', 'drugstartoffset']])
                
    extract_log['sepsis3_meds_read'] = meds_read
    extract_log['sepsis3_meds_matched'] = meds_matched
    print(f"Sepsis-3: med rows read={meds_read}, matched={meds_matched}")
                
    if not meds_chunks:
        return pd.DataFrame(columns=['patientunitstayid', 'suspected_infection_time'])
        
    abx = pd.concat(meds_chunks, ignore_index=True)
    if not abx.empty:
        diff_h = (abx['drugstartoffset'] - 1440) / 60.0
        extract_log.setdefault('temporal_audit', {})['medication'] = {
            'min_h': float(diff_h.min()), 'max_h': float(diff_h.max()), 'rows': len(abx)
        }
        # assert max_h will be done after D1 filter which drops > 1440

    abx['drugstartoffset'] = pd.to_numeric(abx['drugstartoffset'], errors='coerce')
    
    # D1: Tightened to [-1440, 1440]
    abx = abx[(abx['drugstartoffset'] >= -1440) & (abx['drugstartoffset'] <= 1440)]
    if not abx.empty:
        diff_h = (abx['drugstartoffset'] - 1440) / 60.0
        extract_log.setdefault('temporal_audit', {})['medication'] = {
            'min_h': float(diff_h.min()), 'max_h': float(diff_h.max()), 'rows': len(abx)
        }
        assert diff_h.max() <= 0, f"medication max_h > 0: {diff_h.max()}"

    
    stays_with_abx = abx['patientunitstayid'].nunique()
    extract_log['sepsis3_stays_with_abx'] = stays_with_abx
    print(f"Sepsis-3: stays with antibiotic in window={stays_with_abx}")
    
    first_abx = abx.groupby('patientunitstayid')['drugstartoffset'].min().reset_index()
    first_abx = first_abx.rename(columns={'drugstartoffset': 'suspected_infection_time'})
    return first_abx

def compute_sofa_scores(patient_df, extract_log):
    cohort_ids = set(patient_df['patientunitstayid'])
    lab_file = os.path.join(EICU_PATH, 'lab.csv.gz')
    
    po2_chunks, plt_chunks, bili_chunks, cr_chunks = [], [], [], []
    
    if os.path.exists(lab_file):
        for chunk in pd.read_csv(lab_file, chunksize=1000000, low_memory=False):
            chunk = chunk[(chunk['patientunitstayid'].isin(cohort_ids)) & (chunk['labresultoffset'] >= 0) & (chunk['labresultoffset'] <= 1440)]
            chunk['labname'] = chunk['labname'].fillna('').str.lower()
            
            po2 = chunk[chunk['labname'].isin(['paO2', 'pao2', 'PaO2'])]
            if not po2.empty: po2_chunks.append(po2[['patientunitstayid', 'labresult']])
            
            plt = chunk[chunk['labname'].str.contains('platelets', na=False)]
            if not plt.empty: plt_chunks.append(plt[['patientunitstayid', 'labresult']])
            
            bili = chunk[chunk['labname'].str.contains('bilirubin', na=False) & ~chunk['labname'].str.contains('direct|indirect', na=False)]
            if not bili.empty: bili_chunks.append(bili[['patientunitstayid', 'labresult']])
            
            cr = chunk[chunk['labname'].str.contains('creatinine', na=False)]
            if not cr.empty: cr_chunks.append(cr[['patientunitstayid', 'labresult']])

    po2_df = pd.concat(po2_chunks) if po2_chunks else pd.DataFrame(columns=['patientunitstayid', 'labresult'])
    po2_df['labresult'] = pd.to_numeric(po2_df['labresult'], errors='coerce')
    po2_min = po2_df.groupby('patientunitstayid')['labresult'].min().reset_index().rename(columns={'labresult': 'po2'})
    
    apache_file = os.path.join(EICU_PATH, 'apacheApsVar.csv.gz')
    apache_chunks = []
    for chunk in pd.read_csv(apache_file, chunksize=1000000, low_memory=False):
            chunk = chunk[chunk['patientunitstayid'].isin(cohort_ids)]
            if 'fio2' in chunk.columns or 'eyes' in chunk.columns:
                cols = ['patientunitstayid'] + [c for c in ['fio2', 'eyes', 'motor', 'verbal'] if c in chunk.columns]
                apache_chunks.append(chunk[cols])
                
    apache_df = pd.concat(apache_chunks) if apache_chunks else pd.DataFrame(columns=['patientunitstayid', 'fio2', 'eyes', 'motor', 'verbal'])
    
    if 'fio2' in apache_df.columns:
        apache_df['fio2'] = pd.to_numeric(apache_df['fio2'], errors='coerce')
        apache_df['fio2'] = np.where(apache_df['fio2'] > 1.0, apache_df['fio2'] / 100.0, apache_df['fio2'])
        fio2_max = apache_df.groupby('patientunitstayid')['fio2'].max().reset_index()
    else:
        fio2_max = pd.DataFrame(columns=['patientunitstayid', 'fio2'])
        
    resp = pd.merge(po2_min, fio2_max, on='patientunitstayid', how='outer')
    resp['fio2'] = resp['fio2'].fillna(0.21)
    resp['fio2'] = np.where(resp['fio2'] <= 0, 0.21, resp['fio2'])
    resp['pao2_fio2'] = resp['po2'] / resp['fio2']
    
    resp['sofa_resp'] = 0
    resp.loc[resp['pao2_fio2'] < 400, 'sofa_resp'] = 1
    resp.loc[resp['pao2_fio2'] < 300, 'sofa_resp'] = 2
    resp.loc[resp['pao2_fio2'] < 200, 'sofa_resp'] = 3
    resp.loc[resp['pao2_fio2'] < 100, 'sofa_resp'] = 4
    
    plt_df = pd.concat(plt_chunks) if plt_chunks else pd.DataFrame(columns=['patientunitstayid', 'labresult'])
    plt_df['labresult'] = pd.to_numeric(plt_df['labresult'], errors='coerce')
    plt_min = plt_df.groupby('patientunitstayid')['labresult'].min().reset_index()
    plt_min['sofa_coag'] = 0
    plt_min.loc[plt_min['labresult'] < 150, 'sofa_coag'] = 1
    plt_min.loc[plt_min['labresult'] < 100, 'sofa_coag'] = 2
    plt_min.loc[plt_min['labresult'] < 50, 'sofa_coag'] = 3
    plt_min.loc[plt_min['labresult'] < 20, 'sofa_coag'] = 4
    
    bili_df = pd.concat(bili_chunks) if bili_chunks else pd.DataFrame(columns=['patientunitstayid', 'labresult'])
    bili_df['labresult'] = pd.to_numeric(bili_df['labresult'], errors='coerce')
    bili_max = bili_df.groupby('patientunitstayid')['labresult'].max().reset_index()
    bili_max['sofa_liver'] = 0
    bili_max.loc[bili_max['labresult'] >= 1.2, 'sofa_liver'] = 1
    bili_max.loc[bili_max['labresult'] >= 2.0, 'sofa_liver'] = 2
    bili_max.loc[bili_max['labresult'] >= 6.0, 'sofa_liver'] = 3
    bili_max.loc[bili_max['labresult'] >= 12.0, 'sofa_liver'] = 4
    
    vital_per_file = os.path.join(EICU_PATH, 'vitalPeriodic.csv.gz')
    map_chunks = []
    for chunk in pd.read_csv(vital_per_file, chunksize=1000000, low_memory=False):
            chunk = chunk[(chunk['patientunitstayid'].isin(cohort_ids)) & (chunk['observationoffset'] >= 0) & (chunk['observationoffset'] <= 1440)]
            if 'systemicmean' in chunk.columns:
                c = chunk[['patientunitstayid', 'systemicmean']].dropna()
                if not c.empty:
                    map_chunks.append(c.rename(columns={'systemicmean': 'map'}))
                    
    vital_aper_file = os.path.join(EICU_PATH, 'vitalAperiodic.csv.gz')
    for chunk in pd.read_csv(vital_aper_file, chunksize=1000000, low_memory=False):
            chunk = chunk[(chunk['patientunitstayid'].isin(cohort_ids)) & (chunk['observationoffset'] >= 0) & (chunk['observationoffset'] <= 1440)]
            if 'noninvasivemean' in chunk.columns:
                c = chunk[['patientunitstayid', 'noninvasivemean']].dropna()
                if not c.empty:
                    map_chunks.append(c.rename(columns={'noninvasivemean': 'map'}))
                    
    map_df = pd.concat(map_chunks) if map_chunks else pd.DataFrame(columns=['patientunitstayid', 'map'])
    map_df['map'] = pd.to_numeric(map_df['map'], errors='coerce')
    map_min = map_df.groupby('patientunitstayid')['map'].min().reset_index()
    
    infusion_file = os.path.join(EICU_PATH, 'infusionDrug.csv.gz')
    vaso_chunks = []
    vaso_drugs = ['norepinephrine', 'epinephrine', 'dopamine', 'dobutamine']
    for chunk in pd.read_csv(infusion_file, chunksize=1000000, low_memory=False):
            chunk = chunk[(chunk['patientunitstayid'].isin(cohort_ids)) & (chunk['infusionoffset'] >= 0) & (chunk['infusionoffset'] <= 1440)]
            chunk['drugname'] = chunk['drugname'].fillna('').str.lower()
            pattern = r'(?:' + '|'.join(vaso_drugs) + r')'
            v = chunk[chunk['drugname'].str.contains(pattern, regex=True)]
            if not v.empty:
                v = v[['patientunitstayid', 'drugname', 'drugrate', 'infusionoffset']]
                v['drugrate'] = pd.to_numeric(v['drugrate'], errors='coerce')
                vaso_chunks.append(v)
                
    vaso_df = pd.concat(vaso_chunks) if vaso_chunks else pd.DataFrame(columns=['patientunitstayid', 'drugname', 'drugrate', 'infusionoffset'])
    
    if not vaso_df.empty:
        diff_h = (vaso_df['infusionoffset'] - 1440) / 60.0
        extract_log.setdefault('temporal_audit', {})['infusionDrug'] = {
            'min_h': float(diff_h.min()), 'max_h': float(diff_h.max()), 'rows': len(vaso_df)
        }
        assert diff_h.max() <= 0, f"infusionDrug max_h > 0: {diff_h.max()}"

    
    cv = pd.merge(map_min, patient_df[['patientunitstayid']], on='patientunitstayid', how='right')
    cv['sofa_cv'] = 0
    cv.loc[cv['map'] < 70, 'sofa_cv'] = 1
    
    if not vaso_df.empty:
        dopamine = vaso_df[vaso_df['drugname'].str.contains('dopamine', na=False)]
        epi_norepi = vaso_df[vaso_df['drugname'].str.contains('epinephrine|norepinephrine', na=False)]
        dobutamine = vaso_df[vaso_df['drugname'].str.contains('dobutamine', na=False)]
        
        dopa_max = dopamine.groupby('patientunitstayid')['drugrate'].max()
        epi_max = epi_norepi.groupby('patientunitstayid')['drugrate'].max()
        dobut_max = dobutamine.groupby('patientunitstayid')['drugrate'].max()
        
        d = cv['patientunitstayid'].map(dopa_max).fillna(0)
        e = cv['patientunitstayid'].map(epi_max).fillna(0)
        dob = cv['patientunitstayid'].map(dobut_max).fillna(0)
        
        cond4 = (d > 15) | (e > 0.1)
        cond3 = (d > 5) | (e > 0)
        cond2 = (d > 0) | (dob > 0)
        
        cv['sofa_cv'] = np.where(cond4, 4,
                          np.where(cond3, 3,
                            np.where(cond2, 2, cv['sofa_cv'])))
                
    if all(c in apache_df.columns for c in ['eyes', 'motor', 'verbal']):
        apache_df['gcs_computed'] = (
            pd.to_numeric(apache_df['eyes'], errors='coerce') +
            pd.to_numeric(apache_df['motor'], errors='coerce') +
            pd.to_numeric(apache_df['verbal'], errors='coerce')
        )
        gcs_min = apache_df.groupby('patientunitstayid')['gcs_computed'].min().reset_index()
        gcs_min = gcs_min.rename(columns={'gcs_computed': 'gcs'})
    else:
        gcs_min = pd.DataFrame(columns=['patientunitstayid', 'gcs'])
        
    gcs_min['sofa_cns'] = 0
    gcs_min.loc[gcs_min['gcs'] < 15, 'sofa_cns'] = 1
    gcs_min.loc[gcs_min['gcs'] < 13, 'sofa_cns'] = 2
    gcs_min.loc[gcs_min['gcs'] < 10, 'sofa_cns'] = 3
    gcs_min.loc[gcs_min['gcs'] < 6, 'sofa_cns'] = 4
    
    cr_df = pd.concat(cr_chunks) if cr_chunks else pd.DataFrame(columns=['patientunitstayid', 'labresult'])
    cr_df['labresult'] = pd.to_numeric(cr_df['labresult'], errors='coerce')
    cr_max = cr_df.groupby('patientunitstayid')['labresult'].max().reset_index()
    cr_max['sofa_renal'] = 0
    cr_max.loc[cr_max['labresult'] >= 1.2, 'sofa_renal'] = 1
    cr_max.loc[cr_max['labresult'] >= 2.0, 'sofa_renal'] = 2
    cr_max.loc[cr_max['labresult'] >= 3.5, 'sofa_renal'] = 3
    cr_max.loc[cr_max['labresult'] >= 5.0, 'sofa_renal'] = 4
    
    sofa_df = patient_df[['patientunitstayid']].copy()
    for d, c in [(resp, 'sofa_resp'), (plt_min, 'sofa_coag'), (bili_max, 'sofa_liver'), 
                 (cv, 'sofa_cv'), (gcs_min, 'sofa_cns'), (cr_max, 'sofa_renal')]:
        if c in d.columns:
            sofa_df = pd.merge(sofa_df, d[['patientunitstayid', c]], on='patientunitstayid', how='left')
        else:
            sofa_df[c] = 0
            
    sofa_df = sofa_df.fillna(0)
    sofa_df['sofa_total'] = sofa_df[['sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cv', 'sofa_cns', 'sofa_renal']].sum(axis=1)
    
    sofa_total = sofa_df['sofa_total']
    stays_with_sofa = sofa_df['patientunitstayid'].nunique()
    ge_2_count = int((sofa_total >= 2).sum())
    extract_log['sepsis3_stays_with_sofa'] = stays_with_sofa
    extract_log['sepsis3_sofa_distribution'] = {
        'min': float(sofa_total.min()) if not sofa_total.empty else None,
        'median': float(sofa_total.median()) if not sofa_total.empty else None,
        'max': float(sofa_total.max()) if not sofa_total.empty else None,
        'count_ge_2': ge_2_count
    }
    print(f"Sepsis-3: stays with SOFA={stays_with_sofa}, min={extract_log['sepsis3_sofa_distribution']['min']}, median={extract_log['sepsis3_sofa_distribution']['median']}, max={extract_log['sepsis3_sofa_distribution']['max']}, count_ge_2={ge_2_count}")
    
    return sofa_df

def extract_labs(base_cohort, bounds_dict, bounds_counts, extract_log):
    print("Extracting laboratory features...")
    cohort_ids = set(base_cohort['patientunitstayid'])
    print(f"  Cohort IDs: {len(cohort_ids)}, dtype: {base_cohort['patientunitstayid'].dtype}")
    print(f"  Examples: {list(cohort_ids)[:3]}")
    lab_file = os.path.join(EICU_PATH, 'lab.csv.gz')
    
    lab_chunks = []
    for chunk in pd.read_csv(lab_file, chunksize=1000000, low_memory=False):
            chunk = chunk[chunk['patientunitstayid'].isin(cohort_ids)]
            # B3: labs window [-6 h, 24 h] from ICU admission
            chunk = chunk[(chunk['labresultoffset'] >= -360) & (chunk['labresultoffset'] <= 1440)]

            chunk = chunk[chunk['labname'].isin(LAB_MAPPING.keys())]
            if not chunk.empty:
                chunk['labresult'] = pd.to_numeric(chunk['labresult'], errors='coerce')
                chunk['lab_mapped'] = chunk['labname'].map(LAB_MAPPING)
                lab_chunks.append(chunk[['patientunitstayid', 'lab_mapped', 'labresult', 'labmeasurenamesystem', 'labresultoffset']])
                
    if not lab_chunks:
        for val in LAB_MAPPING.values():
            base_cohort[val] = np.nan
        return base_cohort
        
    labs_df = pd.concat(lab_chunks).dropna(subset=['labresult'])
    if not labs_df.empty:
        diff_h = (labs_df['labresultoffset'] - 1440) / 60.0
        extract_log.setdefault('temporal_audit', {})['lab'] = {
            'min_h': float(diff_h.min()), 'max_h': float(diff_h.max()), 'rows': len(labs_df)
        }
        assert diff_h.max() <= 0, f"lab max_h > 0: {diff_h.max()}"
        assert diff_h.min() >= -30, f"lab min_h < -30: {diff_h.min()}"

    if not labs_df.empty:
        diff_h = (labs_df['labresultoffset'] - 1440) / 60.0
        extract_log.setdefault('temporal_audit', {})['lab'] = {
            'min_h': float(diff_h.min()), 'max_h': float(diff_h.max()), 'rows': len(labs_df)
        }
        assert diff_h.max() <= 0, f"lab max_h > 0: {diff_h.max()}"
        assert diff_h.min() >= -30, f"lab min_h < -30: {diff_h.min()}"

    
    crp_mask = labs_df['lab_mapped'] == 'CRP'
    labs_df.loc[crp_mask, 'labresult'] = labs_df.loc[crp_mask, 'labresult'] * 10.0
    extract_log['crp_converted'] = True
    
    check_units(labs_df, 'lab_mapped', 'labmeasurenamesystem', extract_log, 'lab_unit_counts', 'lab_missing_units')
    
    # Apply Bounds
    labs_df = apply_bounds(labs_df, 'lab_mapped', 'labresult', bounds_dict, bounds_counts)
    
    min_labs = ['Albumin', 'Bicarbonate', 'Calcium', 'Hemoglobin', 'Hematocrit', 'Platelets', 'Lymphocytes', 'pH', 'pO2', 'BaseExcess', 'Magnesium', 'Phosphate']
    max_labs = ['AnionGap', 'BUN', 'Creatinine', 'Glucose', 'Potassium', 'Lactate', 'WBC', 'Neutrophils', 'INR', 'PT', 'PTT', 'ALT', 'AST', 'Bilirubin_Total', 'pCO2', 'LDH', 'Troponin_T', 'CRP']
    both_labs = ['Sodium', 'Chloride']
    
    labs_min = labs_df[labs_df['lab_mapped'].isin(min_labs)].groupby(['patientunitstayid', 'lab_mapped'])['labresult'].min().unstack()
    labs_max = labs_df[labs_df['lab_mapped'].isin(max_labs)].groupby(['patientunitstayid', 'lab_mapped'])['labresult'].max().unstack()
    
    sc_df = labs_df[labs_df['lab_mapped'].isin(both_labs)]
    sc_min = sc_df.groupby(['patientunitstayid', 'lab_mapped'])['labresult'].min().unstack()
    if not sc_min.empty:
        sc_min.columns = [f"{col}_min" for col in sc_min.columns]
    sc_max = sc_df.groupby(['patientunitstayid', 'lab_mapped'])['labresult'].max().unstack()
    if not sc_max.empty:
        sc_max.columns = [f"{col}_max" for col in sc_max.columns]
        
    pivot = pd.concat([labs_min, labs_max, sc_min, sc_max], axis=1).reset_index()
    out = pd.merge(base_cohort, pivot, on='patientunitstayid', how='left')
    
    expected_labs = [v for v in list(dict.fromkeys(LAB_MAPPING.values())) if v not in ['Sodium', 'Chloride']] + ['Sodium_min', 'Sodium_max', 'Chloride_min', 'Chloride_max']
    for val in expected_labs:
        if val not in out.columns:
            out[val] = np.nan
            
    return out

def extract_vitals(cohort_labs, bounds_dict, bounds_counts, extract_log):
    print("Extracting vital signs...")
    cohort_ids = set(cohort_labs['patientunitstayid'])
    print(f"  Cohort IDs: {len(cohort_ids)}, dtype: {cohort_labs['patientunitstayid'].dtype}")
    print(f"  Examples: {list(cohort_ids)[:3]}")
    
    vital_per_file = os.path.join(EICU_PATH, 'vitalPeriodic.csv.gz')
    vital_aper_file = os.path.join(EICU_PATH, 'vitalAperiodic.csv.gz')
    nurse_chart_file = os.path.join(EICU_PATH, 'nurseCharting.csv.gz')
    
    vitals_data = []
    has_temp_per = set()
    per_min_h, per_max_h, per_rows = float('inf'), float('-inf'), 0
    for chunk in pd.read_csv(vital_per_file, chunksize=1000000, low_memory=False):
            chunk = chunk[chunk['patientunitstayid'].isin(cohort_ids)]
            chunk = chunk[(chunk['observationoffset'] >= 0) & (chunk['observationoffset'] <= FEATURE_WINDOW_H * 60)]
            if not chunk.empty:
                per_rows += len(chunk)
                diff_h = (chunk['observationoffset'] - 1440) / 60.0
                per_min_h = min(per_min_h, float(diff_h.min()))
                per_max_h = max(per_max_h, float(diff_h.max()))
            cols = [c for c in chunk.columns if c in [
                'patientunitstayid', 'heartrate', 'respiration', 'sao2',
                'temperature', 'systemicsystolic', 'systemicdiastolic', 'systemicmean'
            ]]
            if 'temperature' in chunk.columns:
                has_temp_per.update(chunk[chunk['temperature'].notna()]['patientunitstayid'].unique())
            if len(cols) > 1:
                sub = chunk[cols].copy()
                sub['vital_source'] = 'vitalPeriodic'
                vitals_data.append(sub)
                
    aper_min_h, aper_max_h, aper_rows = float('inf'), float('-inf'), 0
    for chunk in pd.read_csv(vital_aper_file, chunksize=1000000, low_memory=False):
            chunk = chunk[chunk['patientunitstayid'].isin(cohort_ids)]
            chunk = chunk[(chunk['observationoffset'] >= 0) & (chunk['observationoffset'] <= FEATURE_WINDOW_H * 60)]
            if not chunk.empty:
                aper_rows += len(chunk)
                diff_h = (chunk['observationoffset'] - 1440) / 60.0
                aper_min_h = min(aper_min_h, float(diff_h.min()))
                aper_max_h = max(aper_max_h, float(diff_h.max()))
            cols = [c for c in chunk.columns if c in ['patientunitstayid', 'noninvasivesystolic', 'noninvasivediastolic', 'noninvasivemean']]
            if len(cols) > 1:
                subset = chunk[cols].rename(columns={'noninvasivesystolic': 'systemicsystolic', 'noninvasivediastolic': 'systemicdiastolic', 'noninvasivemean': 'systemicmean'})
                subset['vital_source'] = 'vitalAperiodic'
                vitals_data.append(subset)
                
    has_temp_nurse = set()
    nurse_valname_counts = {}
    excluded_temps = 0
    converted_temps = 0
    
    nc_min_h, nc_max_h, nc_rows = float('inf'), float('-inf'), 0
    for chunk in pd.read_csv(nurse_chart_file, chunksize=1000000, low_memory=False):
            chunk = chunk[chunk['patientunitstayid'].isin(cohort_ids)]
            chunk = chunk[(chunk['nursingchartoffset'] >= 0) & (chunk['nursingchartoffset'] <= FEATURE_WINDOW_H * 60)]
            temp_mask = chunk['nursingchartcelltypevallabel'].str.contains('Temperature', na=False, case=False)
            temps = chunk[temp_mask].copy()
            if not temps.empty:
                nc_rows += len(temps)
                diff_h = (temps['nursingchartoffset'] - 1440) / 60.0
                nc_min_h = min(nc_min_h, float(diff_h.min()))
                nc_max_h = max(nc_max_h, float(diff_h.max()))
            if not temps.empty:
                valcounts = temps['nursingchartcelltypevalname'].value_counts(dropna=False).to_dict()
                for k, v in valcounts.items():
                    nurse_valname_counts[str(k)] = nurse_valname_counts.get(str(k), 0) + int(v)
                
                def parse_temp_unit(x):
                    s = str(x).lower()
                    if '(c)' in s: return 'C'
                    if '(f)' in s: return 'F'
                    return np.nan
                
                temps['unit_string'] = temps['nursingchartcelltypevalname'].apply(parse_temp_unit)
                
                valid_unit = temps['unit_string'].notna()
                excluded_temps += int((~valid_unit).sum())
                temps = temps[valid_unit].copy()
                
                if not temps.empty:
                    temps['temperature'] = pd.to_numeric(temps['nursingchartvalue'], errors='coerce')
                    temps = temps.dropna(subset=['temperature']).copy()
                    if not temps.empty:
                        f_mask = temps['unit_string'] == 'F'
                        converted_temps += int(f_mask.sum())
                        temps.loc[f_mask, 'temperature'] = (temps.loc[f_mask, 'temperature'] - 32) * 5 / 9
                        temps.loc[f_mask, 'unit_string'] = 'C'
                        
                        has_temp_nurse.update(temps['patientunitstayid'].unique())
                        temps = temps[['patientunitstayid', 'temperature', 'unit_string']].copy()
                        temps['vital_source'] = 'nurseCharting'
                        vitals_data.append(temps)
                        
    extract_log['nursecharting_temp_valname_counts'] = nurse_valname_counts
    extract_log['nursecharting_temp_excluded_no_unit'] = excluded_temps
    extract_log['nursecharting_temp_converted_f_to_c'] = converted_temps
                    
    extract_log['temp_patients_vitalPeriodic'] = len(has_temp_per)
    extract_log['temp_patients_nurseCharting'] = len(has_temp_nurse)
    extract_log['temp_patients_either'] = len(has_temp_per | has_temp_nurse)

    if per_rows > 0:
        extract_log.setdefault('temporal_audit', {})['vitalPeriodic'] = {
            'min_h': per_min_h, 'max_h': per_max_h, 'rows': per_rows
        }
        assert per_max_h <= 0, f"vitalPeriodic max_h > 0: {per_max_h}"
    if aper_rows > 0:
        extract_log.setdefault('temporal_audit', {})['vitalAperiodic'] = {
            'min_h': aper_min_h, 'max_h': aper_max_h, 'rows': aper_rows
        }
        assert aper_max_h <= 0, f"vitalAperiodic max_h > 0: {aper_max_h}"
    if nc_rows > 0:
        extract_log.setdefault('temporal_audit', {})['nurseCharting'] = {
            'min_h': nc_min_h, 'max_h': nc_max_h, 'rows': nc_rows
        }
        assert nc_max_h <= 0, f"nurseCharting max_h > 0: {nc_max_h}"

                
    apache_file = os.path.join(EICU_PATH, 'apacheApsVar.csv.gz')
    gcs_data = []
    intubated_data = []
    apache_minus1_counts = {}
    matched_apache_rows = 0
    for chunk in pd.read_csv(apache_file, chunksize=1000000, low_memory=False):
            chunk = chunk[chunk['patientunitstayid'].isin(cohort_ids)]
            matched_apache_rows += len(chunk)
            if not chunk.empty:
                for col in chunk.columns:
                    if chunk[col].dtype in ['int64', 'float64', 'int32']:
                        minus1_mask = chunk[col] == -1
                        if minus1_mask.any():
                            apache_minus1_counts[col] = apache_minus1_counts.get(col, 0) + int(minus1_mask.sum())
                            chunk.loc[minus1_mask, col] = np.nan
                
                # intubated (APACHE day 1), read AFTER the -1 -> NaN replacement
                if 'intubated' in chunk.columns:
                    keep = ['patientunitstayid', 'intubated'] + (['apacheapsvarid'] if 'apacheapsvarid' in chunk.columns else [])
                    intubated_data.append(chunk[keep].copy())
                            
                gcs_cols = [c for c in ['eyes', 'motor', 'verbal'] if c in chunk.columns]
                if len(gcs_cols) == 3:
                    has_all_3 = chunk[['eyes', 'motor', 'verbal']].notna().all(axis=1)
                    valid = chunk[has_all_3].copy()
                    if not valid.empty:
                        valid['GCS'] = pd.to_numeric(valid['eyes']) + pd.to_numeric(valid['motor']) + pd.to_numeric(valid['verbal'])
                        gcs_data.append(valid[['patientunitstayid', 'GCS']])
                        
    extract_log['apacheApsVar_minus1_counts'] = apache_minus1_counts
    
    if not intubated_data:
        raise ValueError(f"apacheApsVar has no 'intubated' column for the cohort. Cohort IDs: {len(cohort_ids)}, matched apacheApsVar rows: {matched_apache_rows}")
    df_intub = pd.concat(intubated_data, ignore_index=True)
    # apacheApsVar holds APACHE day-1 values (one row per stay expected); if a stay has more
    # than one row, keep the first by apacheapsvarid and log how many stays were affected.
    if 'apacheapsvarid' in df_intub.columns:
        df_intub = df_intub.sort_values(['patientunitstayid', 'apacheapsvarid'])
    extract_log['intubated_24h_stays_with_multiple_apache_rows'] = int(df_intub['patientunitstayid'].duplicated().sum())
    df_intub = df_intub.drop_duplicates(subset=['patientunitstayid'], keep='first')
    df_intub['intubated_24h'] = pd.to_numeric(df_intub['intubated'], errors='coerce')
    extract_log['intubated_24h_raw_values'] = {
        str(k): int(v) for k, v in df_intub['intubated_24h'].value_counts(dropna=False).to_dict().items()
    }
    df_intub = df_intub[['patientunitstayid', 'intubated_24h']]
                
    df_vitals = pd.concat(vitals_data) if vitals_data else pd.DataFrame()
    if 'unit_string' not in df_vitals.columns:
        df_vitals['unit_string'] = np.nan
        
    if not df_vitals.empty:
        rename_map = {
            'heartrate': 'HR',
            'respiration': 'RR',
            'sao2': 'O2Sat',
            'temperature': 'Temp',
            'systemicsystolic': 'SBP',
            'systemicdiastolic': 'DBP',
            'systemicmean': 'MAP'
        }
        df_vitals = df_vitals.rename(columns=rename_map)
        
        # Pre-apply BP bounds for consistency rule
        for v in ['SBP', 'DBP', 'MAP']:
            if v in bounds_dict and v in df_vitals.columns:
                lo, hi = bounds_dict[v]
                df_vitals.loc[(df_vitals[v] < lo) | (df_vitals[v] > hi), v] = float('nan')
                
        def apply_bp_rule(df, mask):
            cond_dbp_ge_sbp = (df['SBP'].notna()) & (df['DBP'].notna()) & (df['DBP'] >= df['SBP'])
            cond_map_gt_sbp = (df['MAP'].notna()) & (df['SBP'].notna()) & (df['MAP'] > df['SBP'])
            cond_map_lt_dbp = (df['MAP'].notna()) & (df['DBP'].notna()) & (df['MAP'] < df['DBP'])
            drop_mask = mask & (cond_dbp_ge_sbp | cond_map_gt_sbp | cond_map_lt_dbp)
            
            df.loc[drop_mask, ['SBP', 'DBP', 'MAP']] = float('nan')
            return drop_mask.sum()

        if 'SBP' in df_vitals.columns and 'DBP' in df_vitals.columns and 'MAP' in df_vitals.columns:
            dropped_per = apply_bp_rule(df_vitals, df_vitals['vital_source'] == 'vitalPeriodic')
            dropped_aper = apply_bp_rule(df_vitals, df_vitals['vital_source'] == 'vitalAperiodic')
            extract_log['bp_consistency_dropped_periodic'] = int(dropped_per)
            extract_log['bp_consistency_dropped_aperiodic'] = int(dropped_aper)
        
        # Melt to apply bounds generically (BP bounds will re-apply cleanly)
        val_vars = [c for c in df_vitals.columns if c not in ['patientunitstayid', 'vital_source', 'unit_string']]
        df_melt = df_vitals.melt(id_vars=['patientunitstayid', 'vital_source', 'unit_string'], value_vars=val_vars, var_name='vital_name', value_name='valuenum')
        df_melt = df_melt.dropna(subset=['valuenum'])
        
        def assign_vital_units(row):
            u = row['unit_string']
            if pd.notna(u):
                u_str = str(u).lower()
                if 'temperature' in u_str:
                    if '(f)' in u_str: return 'F'
                    if '(c)' in u_str: return 'C'
                return u
            
            if row['vital_source'] in ['vitalPeriodic', 'vitalAperiodic']:
                v = row['vital_name']
                if v in ['HR', 'RR']: return 'bpm'
                if v == 'O2Sat': return '%'
                if v == 'Temp': return 'C'
                if v in ['SBP', 'DBP', 'MAP']: return 'mmHg'
            return np.nan
            
        df_melt['derived_unit'] = df_melt.apply(assign_vital_units, axis=1)
        
        f_mask = (df_melt['vital_name'] == 'Temp') & (df_melt['derived_unit'] == 'F')
        df_melt.loc[f_mask, 'valuenum'] = (df_melt.loc[f_mask, 'valuenum'] - 32.0) * 5.0 / 9.0
        df_melt.loc[f_mask, 'derived_unit'] = 'C'
        
        heuristic_f_mask = (df_melt['vital_name'] == 'Temp') & (df_melt['vital_source'].isin(['vitalPeriodic', 'vitalAperiodic'])) & (df_melt['valuenum'] > 45)
        df_melt.loc[heuristic_f_mask, 'valuenum'] = (df_melt.loc[heuristic_f_mask, 'valuenum'] - 32.0) * 5.0 / 9.0
        
        src_dict = {}
        for (v, src), count in df_melt.groupby(['vital_name', 'vital_source']).size().items():
            if v not in src_dict: src_dict[v] = {}
            src_dict[v][src] = int(count)
        extract_log['vital_source_counts'] = src_dict
        
        check_units(df_melt, 'vital_name', 'derived_unit', extract_log, 'vital_unit_counts', 'vital_missing_units')
        
        df_melt = apply_bounds(df_melt, 'vital_name', 'valuenum', bounds_dict, bounds_counts)
        
        def p05(x): return x.quantile(0.05)
        def p95(x): return x.quantile(0.95)
        agg_df = df_melt.groupby(['patientunitstayid', 'vital_name'])['valuenum'].agg(['mean', p05, p95]).unstack()
        agg_df.columns = [f"{col[1]}_{col[0]}" for col in agg_df.columns]
        agg_df = agg_df.reset_index()
    else:
        agg_df = pd.DataFrame({'patientunitstayid': list(cohort_ids)})
        
    df_gcs = pd.concat(gcs_data) if gcs_data else pd.DataFrame()
    if not df_gcs.empty:
        df_gcs['GCS'] = pd.to_numeric(df_gcs['GCS'], errors='coerce')
        df_gcs['vital_name'] = 'GCS'
        df_gcs['vital_source'] = 'apacheApsVar'
        df_gcs['derived_unit'] = 'points'
        
        src_dict_gcs = {'GCS': {'apacheApsVar': len(df_gcs.dropna(subset=['GCS']))}}
        extract_log['gcs_source_counts'] = src_dict_gcs
        
        check_units(df_gcs, 'vital_name', 'derived_unit', extract_log, 'gcs_unit_counts', 'gcs_missing_units')
        df_gcs = apply_bounds(df_gcs, 'vital_name', 'GCS', bounds_dict, bounds_counts)
        
        gcs_agg = df_gcs.groupby('patientunitstayid')['GCS'].agg(['mean', 'min', 'max'])
        gcs_agg.columns = [f"GCS_{c}" for c in gcs_agg.columns]
        gcs_agg = gcs_agg.reset_index()
        agg_df = pd.merge(agg_df, gcs_agg, on='patientunitstayid', how='outer')
        
    out = pd.merge(cohort_labs, agg_df, on='patientunitstayid', how='left')
    
    vital_names = ['HR', 'RR', 'SBP', 'DBP', 'MAP', 'Temp', 'O2Sat', 'GCS']
    for v in vital_names:
        for stat in ['mean', 'min', 'max']:
            col = f"{v}_{stat}"
            if col not in out.columns:
                out[col] = np.nan
    
    # intubated_24h from apacheApsVar.intubated; stays without an APACHE row stay NaN
    out = pd.merge(out, df_intub, on='patientunitstayid', how='left')
    n_out = len(out)
    extract_log['intubated_24h_share_1'] = float((out['intubated_24h'] == 1).mean()) if n_out else float('nan')
    extract_log['intubated_24h_share_nan'] = float(out['intubated_24h'].isna().mean()) if n_out else float('nan')
    print(f"  intubated_24h: share 1 = {extract_log['intubated_24h_share_1']:.3f}, "
          f"share NaN = {extract_log['intubated_24h_share_nan']:.3f}")
                
    return out

def extract_comorbidities(cohort_vitals, extract_log):
    print("Extracting comorbidities...")
    cohort_ids = set(cohort_vitals['patientunitstayid'])
    print(f"  Cohort IDs: {len(cohort_ids)}, dtype: {cohort_vitals['patientunitstayid'].dtype}")
    print(f"  Examples: {list(cohort_ids)[:3]}")
    diag_file = os.path.join(EICU_PATH, 'diagnosis.csv.gz')
    past_hist_file = os.path.join(EICU_PATH, 'pastHistory.csv.gz')
    
    cm_data = {key: set() for key in COMORBIDITIES}
    
    diag_min_h, diag_max_h, diag_rows = float('inf'), float('-inf'), 0
    for chunk in pd.read_csv(diag_file, chunksize=1000000, low_memory=False):
        chunk = chunk[chunk['patientunitstayid'].isin(cohort_ids)]
        if COMORBIDITY_MODE == "strict":
            chunk = chunk[chunk['diagnosisoffset'] <= 1440]
            
        if not chunk.empty and 'diagnosisoffset' in chunk.columns:
            diag_rows += len(chunk)
            diff_h = (chunk['diagnosisoffset'] - 1440) / 60.0
            diag_min_h = min(diag_min_h, float(diff_h.min()))
            diag_max_h = max(diag_max_h, float(diff_h.max()))
            
        if 'icd9code' not in chunk.columns:
            continue
        chunk['icd9code'] = chunk['icd9code'].fillna('').astype(str)
        for key, codes in COMORBIDITIES.items():
            icd9_list = codes['icd9']
            pattern = '|'.join([f"^{c}" for c in icd9_list])
            mask = chunk['icd9code'].apply(lambda x: any(re.match(pattern, code.strip()) for code in x.split(',')))
            cm_data[key].update(chunk.loc[mask, 'patientunitstayid'])
            
    # Past history regex mapping
    ph_patterns = {
        'CCI_MI':       r'\b(myocardial infarction|mi)\b',
        'CCI_CHF':      r'\b(congestive heart failure|chf|heart failure)\b',
        'CCI_PVD':      r'\b(peripheral vascular disease|pvd)\b',
        'CCI_Stroke':   r'\b(stroke|cva|tia|transient ischemic attack)\b',
        'CCI_Dementia': r'\b(dementia|alzheimer)\b',
        'CCI_COPD':     r'\b(copd|chronic obstructive pulmonary disease)\b',
        'CCI_Renal':    r'\b(renal failure|esrd|ckd|chronic kidney disease)\b',
        'CCI_Liver':    r'\b(cirrhosis|hepatic failure|liver failure)\b',
        'CCI_Cancer':   r'\b(cancer|carcinoma|malignancy|tumor)\b',
        'CCI_Diabetes': r'\b(diabetes|dm)\b',
        'CCI_HIV':      r'\b(hiv|aids)\b'
    }
    
    ph_match_counts = {k: {} for k in ph_patterns}
                
    ph_min_h, ph_max_h, ph_rows = float('inf'), float('-inf'), 0
    ph_dropped_rows = 0
    ph_pts_before = set()
    ph_pts_after = set()
    if COMORBIDITY_MODE == "strict":
        for chunk in pd.read_csv(past_hist_file, chunksize=1000000, low_memory=False):
            chunk = chunk[chunk['patientunitstayid'].isin(cohort_ids)]
            
            orig_len = len(chunk)
            ph_pts_before.update(chunk['patientunitstayid'])
            chunk = chunk[chunk['pasthistoryoffset'] <= 1440]
            ph_pts_after.update(chunk['patientunitstayid'])
            ph_dropped_rows += (orig_len - len(chunk))
            
            if not chunk.empty:
                ph_rows += len(chunk)
                diff_h = (chunk['pasthistoryoffset'] - 1440) / 60.0
                ph_min_h = min(ph_min_h, float(diff_h.min()))
                ph_max_h = max(ph_max_h, float(diff_h.max()))
                
            chunk['pasthistorypath'] = chunk['pasthistorypath'].fillna('').str.lower()
            
            if not chunk.empty:
                for k, pat in ph_patterns.items():
                    if k == 'CCI_Cancer':
                        mask = chunk['pasthistorypath'].str.contains('/cancer/', regex=False) & \
                               ~chunk['pasthistorypath'].str.contains('/cancer therapy/', regex=False) & \
                               ~chunk['pasthistorypath'].str.contains('cancer-primary site/none', regex=False)
                    else:
                        mask = chunk['pasthistorypath'].str.contains(pat, regex=True)
                        
                    matches = chunk.loc[mask, 'pasthistorypath']
                    if not matches.empty:
                        cm_data[k].update(chunk.loc[mask, 'patientunitstayid'])
                        for m in matches.unique():
                            ph_match_counts[k][m] = ph_match_counts[k].get(m, 0) + (matches == m).sum()

    extract_log['pastHistory_matches'] = {k: v for k, v in ph_match_counts.items() if v}

    extract_log['pasthistory_filtered_rows'] = ph_dropped_rows
    extract_log['pasthistory_filtered_patients'] = len(ph_pts_before) - len(ph_pts_after)
    
    if diag_rows > 0:
        extract_log.setdefault('temporal_audit', {})['diagnosis'] = {
            'min_h': diag_min_h, 'max_h': diag_max_h, 'rows': diag_rows
        }
        assert diag_max_h <= 0, f"diagnosis max_h > 0: {diag_max_h}"
    if ph_rows > 0:
        extract_log.setdefault('temporal_audit', {})['pastHistory'] = {
            'min_h': ph_min_h, 'max_h': ph_max_h, 'rows': ph_rows
        }
        assert ph_max_h <= 0, f"pastHistory max_h > 0: {ph_max_h}"


    for key in COMORBIDITIES:
        cohort_vitals[key] = cohort_vitals['patientunitstayid'].isin(cm_data[key]).astype(int)
        
    return cohort_vitals

def main():
    parser = argparse.ArgumentParser(description="Extract eICU sepsis cohort.")
    parser.add_argument("--eicu-dir", type=str, required=True, help="Path to raw eICU folder")
    args = parser.parse_args()
    global EICU_PATH
    EICU_PATH = args.eicu_dir

    bounds_dict = load_bounds()
    flow_counts = {}
    temporal_audit = {}
    temporal_audit = {}
    bounds_counts = {}
    extract_log = {
        'constants': {
            'FEATURE_WINDOW_H': FEATURE_WINDOW_H,
            'LAB_PRE_WINDOW_H': LAB_PRE_WINDOW_H,
            'HORIZON_DAYS': HORIZON_DAYS,
            'MISSING_THRESHOLD': MISSING_THRESHOLD,
            'NEAR_CONSTANT_SHARE': NEAR_CONSTANT_SHARE,
            'COMORBIDITY_MODE': COMORBIDITY_MODE
        },
        'definitions': {
            'suspected_infection': 'antibiotics only, window [-1440, 1440] min',
            'SOFA_window': '[0, 1440] min',
            'SOFA_renal': 'without urine output',
            'SOFA_respiratory': 'without a ventilation indicator'
        }
    }
    
    try:
        # 1. Required files check
        required_files = [
            'patient.csv.gz', 'medication.csv.gz', 'lab.csv.gz', 'apacheApsVar.csv.gz',
            'vitalPeriodic.csv.gz', 'vitalAperiodic.csv.gz', 'infusionDrug.csv.gz',
            'diagnosis.csv.gz', 'nurseCharting.csv.gz', 'pastHistory.csv.gz'
        ]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(EICU_PATH, f))]
        if missing_files:
            raise FileNotFoundError(f"Missing required eICU files in {EICU_PATH}:\n" + "\n".join(missing_files))

        # Base cohort loading
        patient_file = os.path.join(EICU_PATH, 'patient.csv.gz')
        df_pat = pd.read_csv(patient_file)

        flow_counts['1_initial_icu_stays'] = len(df_pat)
        print(f"1_initial_icu_stays: {len(df_pat)}")
        if len(df_pat) == 0: raise ValueError("Flow step '1_initial_icu_stays' yielded 0 patients.")

        df_pat['age'] = df_pat['age'].astype(str).replace('> 89', '90')
        df_pat['age'] = pd.to_numeric(df_pat['age'], errors='coerce')
        df_pat = df_pat[df_pat['age'] >= 18]
        flow_counts['2_adults_>=18'] = len(df_pat)
        print(f"2_adults_>=18: {len(df_pat)}")
        if len(df_pat) == 0: raise ValueError("Flow step '2_adults_>=18' yielded 0 patients.")

        infection = generate_suspected_infection(df_pat, extract_log)
        sofa = compute_sofa_scores(df_pat, extract_log)
        
        print(f"Merge 1: df_pat dtype={df_pat['patientunitstayid'].dtype}, infection dtype={infection['patientunitstayid'].dtype}")
        df_pat = pd.merge(df_pat, infection, on='patientunitstayid', how='left')
        print(f"Merge 2: df_pat dtype={df_pat['patientunitstayid'].dtype}, sofa dtype={sofa['patientunitstayid'].dtype}")
        df_pat = pd.merge(df_pat, sofa[['patientunitstayid', 'sofa_total']], on='patientunitstayid', how='left')
        df_pat['sepsis3_flag'] = ((df_pat['suspected_infection_time'].notnull()) & (df_pat['sofa_total'] >= 2)).astype(int)

        sepsis3_matches_before_filter = int((df_pat['sepsis3_flag'] == 1).sum())
        extract_log['sepsis3_matches_before_filter'] = sepsis3_matches_before_filter
        print(f"Sepsis-3 matches before time_zero filter: {sepsis3_matches_before_filter}")

        # Sepsis-3 with onset by time zero
        # For eICU, time zero is 24h = 1440 minutes.
        # We require suspected_infection_time <= 1440 and sofa_total calculated from [0, 1440] is already fixed.
        df_pat = df_pat[(df_pat['sepsis3_flag'] == 1) & (df_pat['suspected_infection_time'] <= 1440)]
        flow_counts['3_sepsis3_by_time_zero'] = len(df_pat)
        print(f"3_sepsis3_by_time_zero: {len(df_pat)}")
        if len(df_pat) == 0: raise ValueError("Flow step '3_sepsis3_by_time_zero' yielded 0 patients.")

        # First qualifying stay
        df_pat = df_pat[df_pat['unitvisitnumber'] == 1]
        df_pat = df_pat.sort_values(['uniquepid', 'hospitaldischargeyear', 'patienthealthsystemstayid'])

        dup_counts = df_pat.duplicated(subset=['uniquepid'], keep=False).sum()
        extract_log['uniquepid_with_multiple_stays'] = int(dup_counts)

        df_pat = df_pat.drop_duplicates(subset=['uniquepid'], keep='first')
        flow_counts['4_first_icu_stay'] = len(df_pat)
        print(f"4_first_icu_stay: {len(df_pat)}")
        if len(df_pat) == 0: raise ValueError("Flow step '4_first_icu_stay' yielded 0 patients.")

        # Alive at ICU + 24h
        # death time = hospitaldischargeoffset when status = Expired
        df_pat['death_offset'] = np.where(df_pat['hospitaldischargestatus'] == 'Expired', df_pat['hospitaldischargeoffset'], np.nan)
        alive_at_tz = df_pat['death_offset'].isna() | (df_pat['death_offset'] > 1440)
        df_pat = df_pat[alive_at_tz]
        flow_counts['5_alive_at_time_zero'] = len(df_pat)
        print(f"5_alive_at_time_zero: {len(df_pat)}")
        if len(df_pat) == 0: raise ValueError("Flow step '5_alive_at_time_zero' yielded 0 patients.")

        # Still in ICU at 24h
        in_icu_at_tz = df_pat['unitdischargeoffset'] > 1440
        df_pat = df_pat[in_icu_at_tz]
        flow_counts['6_in_icu_at_time_zero'] = len(df_pat)
        print(f"6_in_icu_at_time_zero: {len(df_pat)}")
        if len(df_pat) == 0: raise ValueError("Flow step '6_in_icu_at_time_zero' yielded 0 patients.")
        
        missing_status = df_pat['hospitaldischargestatus'].isna()
        missing_offset = (df_pat['hospitaldischargestatus'] == 'Expired') & df_pat['hospitaldischargeoffset'].isna()
        
        extract_log['excluded_missing_hospitaldischargestatus'] = int(missing_status.sum())
        extract_log['excluded_expired_missing_offset'] = int(missing_offset.sum())
        
        df_pat = df_pat[~(missing_status | missing_offset)]
        flow_counts['7_known_outcome'] = len(df_pat)
        print(f"7_known_outcome: {len(df_pat)}")
        if len(df_pat) == 0: raise ValueError("Flow step '7_known_outcome' yielded 0 patients.")

        df_pat['gender'] = df_pat['gender'].replace({'Male': 'M', 'Female': 'F', 'Unknown': np.nan, 'Other': np.nan})

        cohort_l = extract_labs(df_pat, bounds_dict, bounds_counts, extract_log)
        cohort_v = extract_vitals(cohort_l, bounds_dict, bounds_counts, extract_log)
        cohort_c = extract_comorbidities(cohort_v, extract_log)

        # B5: Outcome
        # death in (1440, 1440 + HORIZON_DAYS * 1440]
        death_after_tz = cohort_c['death_offset'] > 1440
        death_within_horizon = cohort_c['death_offset'] <= 1440 + (HORIZON_DAYS * 1440)

        cohort_c['event'] = (death_after_tz & death_within_horizon).astype(int)

        cohort_c['time_days'] = HORIZON_DAYS
        cohort_c.loc[cohort_c['event'] == 1, 'time_days'] = (cohort_c.loc[cohort_c['event'] == 1, 'death_offset'] - 1440) / 1440.0

        # B6: Hospice
        hospice_mask = cohort_c['hospitaldischargelocation'].str.contains('Hospice', na=False, case=False)
        extract_log['hospice_discharges'] = int(hospice_mask.sum())

        # B8: Drop missing > 70% and log near-constant
        # 100% missingness guard BEFORE 70% rule
        cohort_c['SOFA'] = cohort_c['sofa_total']
        
        expected_labs = [v for v in list(dict.fromkeys(LAB_MAPPING.values())) if v not in ['Sodium', 'Chloride']] + ['Sodium_min', 'Sodium_max', 'Chloride_min', 'Chloride_max']
        allowlist_features = ['gender', 'age', 'admissionheight', 'admissionweight'] + \
                             expected_labs + \
                             [f"{v}_{agg}" for v in ['HR', 'RR', 'SBP', 'DBP', 'MAP', 'Temp', 'O2Sat'] for agg in ['mean', 'p05', 'p95']] + \
                             ['GCS_mean', 'GCS_min', 'GCS_max', 'intubated_24h'] + \
                             list(COMORBIDITIES.keys()) + ['SOFA']

        if 'ethnicity' in cohort_c.columns:
            cohort_c = cohort_c.drop(columns=['ethnicity'])
            extract_log['excluded_ethnicity_on_purpose'] = True
            
        # Apply height and weight bounds before missingness threshold
        if 'Height' in bounds_dict:
            h_lo, h_hi = bounds_dict['Height']
            out_h = (cohort_c['admissionheight'] < h_lo) | (cohort_c['admissionheight'] > h_hi)
            extract_log['height_out_of_bounds'] = int(out_h.sum())
            cohort_c.loc[out_h, 'admissionheight'] = float('nan')
            
        if 'Weight' in bounds_dict:
            w_lo, w_hi = bounds_dict['Weight']
            out_w = (cohort_c['admissionweight'] < w_lo) | (cohort_c['admissionweight'] > w_hi)
            extract_log['weight_out_of_bounds'] = int(out_w.sum())
            cohort_c.loc[out_w, 'admissionweight'] = float('nan')

        missing_pct = cohort_c[allowlist_features].isna().mean()

        full_missing = missing_pct[missing_pct == 1.0].index.tolist()
        if full_missing:
            raise ValueError(f"The following features have 100% missingness: {full_missing}. Extraction step failed.")

        to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
        extract_log['dropped_missing'] = {c: float(missing_pct[c]) for c in to_drop}

        feature_cols = [c for c in allowlist_features if c not in to_drop]

        near_constant = {}
        for c in feature_cols:
            val_counts = cohort_c[c].value_counts(normalize=True, dropna=True)
            if not val_counts.empty and val_counts.iloc[0] > NEAR_CONSTANT_SHARE:
                near_constant[c] = float(val_counts.iloc[0])

        extract_log['near_constant'] = near_constant

        # B9: Output columns
        final_cols = ['uniquepid', 'patientunitstayid', 'event', 'time_days'] + feature_cols

        present_not_allowlisted = [c for c in cohort_c.columns if c not in final_cols]
        print(f"Columns dropped because they are not in the allowlist: {present_not_allowlisted}")
        extract_log['dropped_not_in_allowlist'] = present_not_allowlisted

        final_cohort = cohort_c[final_cols].copy()

        # B11: Assertions
        assert len(final_cohort) == final_cohort['uniquepid'].nunique()
        assert (final_cohort['time_days'] > 0).all()
        assert (final_cohort['time_days'] <= HORIZON_DAYS).all()
        assert set(final_cohort['event'].unique()).issubset({0, 1})

        forbidden_words = ['time', 'offset', 'los', 'disch', 'death', 'dod', 'expire', 'intime', 'outtime', 'admit']
        for c in final_cohort.columns:
            if c in ['event', 'time_days']: continue
            for w in forbidden_words:
                assert w not in c.lower(), f"Forbidden word '{w}' in column '{c}'"

        extract_log['flow_counts'] = flow_counts
        extract_log['bounds_counts'] = bounds_counts
        extract_log['final_N'] = len(final_cohort)
        extract_log['event_rate'] = float(final_cohort['event'].mean())
        extract_log['median_time_days'] = float(final_cohort['time_days'].median())
        
        final_cohort.to_csv(FINAL_OUT, index=False)
        print(f"Extraction complete. Saved {FINAL_OUT}")

    except Exception as e:
        extract_log['error'] = str(e)
        raise
    finally:
        with open(LOG_OUT, 'w') as f:
            json.dump(sanitize(extract_log), f, indent=2)

if __name__ == "__main__":
    main()
