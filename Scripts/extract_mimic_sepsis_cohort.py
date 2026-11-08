import os
import argparse
import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

FEATURE_WINDOW_H = 24
LAB_PRE_WINDOW_H = 6
HORIZON_DAYS = 28
MISSING_THRESHOLD = 0.70
NEAR_CONSTANT_SHARE = 0.99
COMORBIDITY_MODE = "strict"
ARTERIAL_ONLY_BG = True
SPECIMEN_TYPE_ITEMID = 52033

GCS_ITEMIDS = {'eye': 220739, 'verbal': 223900, 'motor': 223901}
GCS_COMPONENT_RANGES = {'eye': (1, 4), 'verbal': (1, 5), 'motor': (1, 6)}
GCS_TOTAL_BOUNDS = (3, 15)
INTUBATED_VERBAL_VALUE = 'No Response-ETT'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'Results')
OUT_DIR = os.path.join(OUTPUT_PATH, 'final')
FINAL_OUT = os.path.join(OUT_DIR, 'mimic_sepsis_cohort.csv')
LOG_OUT = os.path.join(OUT_DIR, 'mimic_extraction_log.json')
MIMIC_PATH = None
MIN_AGE = 18
LAB_ITEMS = {
    'Albumin':          {'ids': [50862], 'agg': 'min'},
    'AnionGap':         {'ids': [50868], 'agg': 'max'},
    'Bicarbonate':      {'ids': [50882], 'agg': 'min'},
    'BUN':              {'ids': [51006], 'agg': 'max'},
    'Calcium':          {'ids': [50893], 'agg': 'min'},
    'Chloride':         {'ids': [50902, 50806], 'agg': 'min'},
    'Creatinine':       {'ids': [50912], 'agg': 'max'},
    'Glucose':          {'ids': [50931, 50809], 'agg': 'max'},
    'Sodium':           {'ids': [50983, 50824], 'agg': 'min'},
    'Potassium':        {'ids': [50971, 50822], 'agg': 'max'},
    'Lactate':          {'ids': [50813], 'agg': 'max'},
    'Hemoglobin':       {'ids': [51222], 'agg': 'min'},
    'Hematocrit':       {'ids': [51221], 'agg': 'min'},
    'Platelets':        {'ids': [51265], 'agg': 'min'},
    'WBC':              {'ids': [51301, 51300], 'agg': 'max'},
    'Lymphocytes':      {'ids': [51244, 51245], 'agg': 'min'},
    'Neutrophils':      {'ids': [51256], 'agg': 'max'},
    'INR':              {'ids': [51237], 'agg': 'max'},
    'PT':               {'ids': [51274], 'agg': 'max'},
    'PTT':              {'ids': [51275], 'agg': 'max'},
    'ALT':              {'ids': [50861], 'agg': 'max'},
    'AST':              {'ids': [50878], 'agg': 'max'},
    'Bilirubin_Total':  {'ids': [50885], 'agg': 'max'},
    'pH':               {'ids': [50820], 'agg': 'min'},
    'pO2':              {'ids': [50821], 'agg': 'min'},
    'pCO2':             {'ids': [50818], 'agg': 'max'},
    'BaseExcess':       {'ids': [50802], 'agg': 'min'},
    'Magnesium':        {'ids': [50960], 'agg': 'min'},
    'Phosphate':        {'ids': [50970], 'agg': 'min'},
    'LDH':              {'ids': [50954], 'agg': 'max'},
    'Troponin_T':       {'ids': [51003], 'agg': 'max'},
    'CRP':              {'ids': [50889], 'agg': 'max'},
}
VITAL_ITEMS = {
    'HR':     [220045],
    'RR':     [220210, 224690],
    'SBP':    [220179, 220050],
    'DBP':    [220180, 220051],
    'MAP':    [220052, 220181, 225312],
    'Temp':   [223762, 223761],
    'O2Sat':  [220277],
    'GCS':    [220739, 223900, 223901],
}
COMORBIDITIES = {
    'CCI_MI':        {'icd10': ['I21', 'I22', 'I252'], 'icd9': ['410', '412']},
    'CCI_CHF':       {'icd10': ['I50', 'I42'],          'icd9': ['428', '4254']},
    'CCI_PVD':       {'icd10': ['I71', 'I790'],         'icd9': ['4402', '4408', '4409']},
    'CCI_Stroke':    {'icd10': ['I60','I61','I62','I63','I64'], 'icd9': ['430','431','432','433','434','436']},
    'CCI_Dementia':  {'icd10': ['F00','F01','F02','F03'], 'icd9': ['290']},
    'CCI_COPD':      {'icd10': ['J40','J41','J42','J43','J44'], 'icd9': ['491','492','496']},
    'CCI_Renal':     {'icd10': ['N18', 'N19'],           'icd9': ['585', '586']},
    'CCI_Liver':     {'icd10': ['K70','K71','K72','K73','K74'], 'icd9': ['571']},
    'CCI_Cancer':    {'icd10': ['C0','C1','C2','C3','C4'], 'icd9': ['140','141','142','143','144','145','146','147','148','149','150']},
    'CCI_Diabetes':  {'icd10': ['E10','E11','E12','E13','E14'], 'icd9': ['250']},
    'CCI_HIV':       {'icd10': ['B20','B21','B22','B23','B24'], 'icd9': ['042','043','044']},
}

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



def load_bounds():
    bounds_path = os.path.join(OUT_DIR, 'bounds_approved.csv')
    if not os.path.exists(bounds_path):
        raise FileNotFoundError(f"Bounds file not found: {bounds_path}")
    
    bounds_df = pd.read_csv(bounds_path)
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
        raise ValueError(f"Variables missing lower or upper bounds in {bounds_path}: {missing_bounds}")
        
    return bounds_dict


def extract_base_cohort(flow_counts, extract_log):
    print("Step: base_cohort - Loading admissions and icustays")
    admissions_path = os.path.join(MIMIC_PATH, 'hosp', 'admissions.csv.gz')
    icustays_path = os.path.join(MIMIC_PATH, 'icu', 'icustays.csv.gz')
    patients_path = os.path.join(MIMIC_PATH, 'hosp', 'patients.csv.gz')
    
    admissions = pd.read_csv(admissions_path)
    icustays = pd.read_csv(icustays_path)
    patients = pd.read_csv(patients_path)
    
    base = icustays.merge(admissions[['hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag', 'discharge_location']], on='hadm_id', how='left')
    base = base.merge(patients[['subject_id', 'gender', 'anchor_age', 'anchor_year']], on='subject_id', how='left')
    
    # Calculate age
    base['admittime'] = pd.to_datetime(base['admittime'])
    base['intime'] = pd.to_datetime(base['intime'])
    base['outtime'] = pd.to_datetime(base['outtime'])
    base['dischtime'] = pd.to_datetime(base['dischtime'])
    base['deathtime'] = pd.to_datetime(base['deathtime'])
    
    base['age'] = base['anchor_age'] + (base['admittime'].dt.year - base['anchor_year'])
    capped_age = base['age'] > 90
    extract_log['capped_age_90'] = int(capped_age.sum())
    base.loc[capped_age, 'age'] = 90
    
    # Impute deathtime
    missing_death = (base['hospital_expire_flag'] == 1) & base['deathtime'].isna()
    extract_log['missing_deathtime_imputed_with_dischtime'] = int(missing_death.sum())
    base.loc[missing_death, 'deathtime'] = base.loc[missing_death, 'dischtime']
    base.loc[base['hospital_expire_flag'] == 0, 'deathtime'] = pd.NaT
    
    flow_counts['1_initial_icu_stays'] = len(base)
    
    # Adult filter
    base = base[base['age'] >= MIN_AGE]
    flow_counts['2_adults_>=18'] = len(base)
    
    # Time zero
    base['time_zero'] = base['intime'] + pd.Timedelta(hours=FEATURE_WINDOW_H)
    
    # Suspected infection recomputed
    presc_path = os.path.join(MIMIC_PATH, 'hosp', 'prescriptions.csv.gz')
    antibiotics = ['vancomycin', 'piperacillin', 'meropenem', 'cefepime', 'metronidazole', 'ciprofloxacin', 'ampicillin', 'gentamicin', 'azithromycin', 'levofloxacin', 'clindamycin', 'linezolid', 'daptomycin', 'ceftriaxone', 'ceftazidime']
    stay_hadm_ids = set(base['hadm_id'].dropna().unique())
    pattern = r'\b(' + '|'.join(antibiotics) + r')\b'
    
    presc_chunks = []
    for chunk in pd.read_csv(presc_path, chunksize=1000000, compression='gzip', low_memory=False):
        chunk = chunk[chunk['hadm_id'].isin(stay_hadm_ids)]
        if not chunk.empty:
            chunk = chunk[chunk['drug'].str.lower().str.contains(pattern, na=False, regex=True)]
            if not chunk.empty:
                presc_chunks.append(chunk[['subject_id', 'hadm_id', 'starttime']])
                
    presc_df = pd.concat(presc_chunks, ignore_index=True) if presc_chunks else pd.DataFrame(columns=['subject_id', 'hadm_id', 'starttime'])
    
    micro_path = os.path.join(MIMIC_PATH, 'hosp', 'microbiologyevents.csv.gz')
    micro_chunks = []
    for chunk in pd.read_csv(micro_path, chunksize=1000000, compression='gzip', low_memory=False):
        chunk = chunk[chunk['hadm_id'].isin(stay_hadm_ids)]
        if not chunk.empty:
            micro_chunks.append(chunk[['subject_id', 'hadm_id', 'charttime', 'chartdate']])
            
    micro_df = pd.concat(micro_chunks, ignore_index=True) if micro_chunks else pd.DataFrame(columns=['subject_id', 'hadm_id', 'charttime', 'chartdate'])
    
    presc_df['starttime'] = pd.to_datetime(presc_df['starttime'])
    micro_df['charttime'] = pd.to_datetime(micro_df['charttime'])
    micro_df['chartdate'] = pd.to_datetime(micro_df['chartdate'])
    micro_df['charttime'] = micro_df['charttime'].fillna(micro_df['chartdate'])
    
    abx_merge = presc_df.merge(base[['stay_id', 'hadm_id', 'intime', 'time_zero']], on='hadm_id')
    abx_merge = abx_merge[
        (abx_merge['starttime'] >= abx_merge['intime'] - pd.Timedelta(hours=24)) & 
        (abx_merge['starttime'] <= abx_merge['intime'] + pd.Timedelta(hours=24))
    ]
    
    if not abx_merge.empty:
        diff_h = (abx_merge['starttime'] - abx_merge['time_zero']).dt.total_seconds() / 3600.0
        extract_log.setdefault('temporal_audit', {})['prescriptions'] = {
            'min_h': float(diff_h.min()), 'max_h': float(diff_h.max()), 'rows': len(abx_merge)
        }
        assert diff_h.max() <= 0, f"prescriptions max_h > 0: {diff_h.max()}"

    first_abx = abx_merge.groupby('stay_id')['starttime'].min().reset_index().rename(columns={'starttime': 'antibiotic_time'})
    
    micro_merge = micro_df.merge(base[['stay_id', 'hadm_id', 'intime', 'time_zero']], on='hadm_id')
    micro_merge = micro_merge[
        (micro_merge['charttime'] >= micro_merge['intime'] - pd.Timedelta(hours=24)) & 
        (micro_merge['charttime'] <= micro_merge['intime'] + pd.Timedelta(hours=24))
    ]
    
    if not micro_merge.empty:
        diff_h = (micro_merge['charttime'] - micro_merge['time_zero']).dt.total_seconds() / 3600.0
        extract_log.setdefault('temporal_audit', {})['microbiologyevents'] = {
            'min_h': float(diff_h.min()), 'max_h': float(diff_h.max()), 'rows': len(micro_merge)
        }
        assert diff_h.max() <= 0, f"microbiologyevents max_h > 0: {diff_h.max()}"

    first_micro = micro_merge.groupby('stay_id')['charttime'].min().reset_index().rename(columns={'charttime': 'culture_time'})
    
    infection_df = base[['stay_id']].merge(first_abx, on='stay_id', how='left').merge(first_micro, on='stay_id', how='left')
    time_diff = (infection_df['culture_time'] - infection_df['antibiotic_time']).dt.total_seconds() / 3600.0
    is_suspected = (time_diff >= -72.0) & (time_diff <= 24.0)
    infection_df['suspected_infection_time'] = pd.NaT
    infection_df.loc[is_suspected, 'suspected_infection_time'] = infection_df.loc[is_suspected, ['antibiotic_time', 'culture_time']].min(axis=1)
    
    # sofa_total
    sepsis3_path = os.path.join(PROJECT_ROOT, 'Results', 'derived', 'sepsis3.csv')
    sepsis3 = pd.read_csv(sepsis3_path)
    extract_log['sofa_total_window'] = "[intime, intime + 24h] per generate_derived_tables.compute_sofa_scores()"
    extract_log['sepsis3_source'] = os.path.abspath(sepsis3_path)
    
    base = base.merge(infection_df[['stay_id', 'suspected_infection_time']], on='stay_id', how='left')
    base = base.merge(sepsis3[['stay_id', 'sofa_total', 'sepsis3_flag']], on='stay_id', how='left')
    
    new_sepsis3 = base['suspected_infection_time'].notna() & (base['sofa_total'] >= 2)
    base['sepsis3_new'] = new_sepsis3
    
    orig_1_new_1 = ((base['sepsis3_flag'] == 1) & (base['sepsis3_new'] == True)).sum()
    orig_1_new_0 = ((base['sepsis3_flag'] == 1) & (base['sepsis3_new'] == False)).sum()
    orig_0_new_1 = ((base['sepsis3_flag'] == 0) & (base['sepsis3_new'] == True)).sum()
    orig_0_new_0 = ((base['sepsis3_flag'] == 0) & (base['sepsis3_new'] == False)).sum()
    
    extract_log['sepsis3_crosstab'] = {
        'original_1': int((base['sepsis3_flag'] == 1).sum()),
        'new_1': int(base['sepsis3_new'].sum()),
        'orig_1_new_1': int(orig_1_new_1),
        'orig_1_new_0': int(orig_1_new_0),
        'orig_0_new_1': int(orig_0_new_1),
        'orig_0_new_0': int(orig_0_new_0)
    }
    
    base = base[base['sepsis3_new'] == True]
    flow_counts['3_sepsis3_by_time_zero'] = len(base)
    if len(base) == 0: raise ValueError("0 patients at step 3_sepsis3_by_time_zero")
    
    # First qualifying ICU stay
    base = base.sort_values(['subject_id', 'intime']).drop_duplicates(subset='subject_id', keep='first').reset_index(drop=True)
    flow_counts['4_first_icu_stay'] = len(base)
    
    # Alive at time zero
    alive_at_tz = base['deathtime'].isna() | (base['deathtime'] > base['time_zero'])
    base = base[alive_at_tz]
    flow_counts['5_alive_at_time_zero'] = len(base)
    
    # In ICU at time zero
    base = base[base['outtime'] > base['time_zero']]
    flow_counts['6_in_icu_at_time_zero'] = len(base)
    if len(base) == 0: raise ValueError("0 patients at step 6_in_icu_at_time_zero")
    
    return base


def extract_labs(base_cohort, bounds_dict, bounds_counts, extract_log):
    print("Step: cohort_with_labs - Extracting laboratory measurements")
    base_hadm_ids = set(base_cohort['hadm_id'].dropna().unique())
    base_subject_ids = set(base_cohort['subject_id'].unique())
    valid_patients = (
        (chunk['hadm_id'].isin(base_hadm_ids)) | 
        (chunk['hadm_id'].isna() & chunk['subject_id'].isin(base_subject_ids))
    ) if 'chunk' in locals() else None  # Dummy for syntax validation outside loop

    all_lab_item_ids = []
    item_to_lab = {}
    for name, info in LAB_ITEMS.items():
        for item_id in info['ids']:
            all_lab_item_ids.append(item_id)
            item_to_lab[item_id] = name

    lab_path = os.path.join(MIMIC_PATH, 'hosp', 'labevents.csv.gz')
    
    # 1. Log labels mapping
    lab_items_path = os.path.join(MIMIC_PATH, 'hosp', 'd_labitems.csv.gz')
    d_labitems = pd.read_csv(lab_items_path)
    label_map = d_labitems[d_labitems['itemid'].isin(all_lab_item_ids)].set_index('itemid')['label'].to_dict()
    nested_labels = {}
    for name, info in LAB_ITEMS.items():
        nested_labels[name] = {str(item_id): label_map.get(item_id, 'Unknown') for item_id in info['ids']}
    extract_log['lab_itemid_labels'] = nested_labels

    lab_rows = []
    specimen_rows = []
    
    for chunk in pd.read_csv(lab_path, chunksize=1000000, compression='gzip', low_memory=False):
        valid_patients = (
            (chunk['hadm_id'].isin(base_hadm_ids)) | 
            (chunk['hadm_id'].isna() & chunk['subject_id'].isin(base_subject_ids))
        )
        # Labs
        chunk_labs = chunk[chunk['itemid'].isin(all_lab_item_ids) & valid_patients]
        if not chunk_labs.empty:
            lab_rows.append(chunk_labs[['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum', 'valueuom', 'specimen_id']])
            
        # Specimen
        if ARTERIAL_ONLY_BG:
            chunk_spec = chunk[(chunk['itemid'] == SPECIMEN_TYPE_ITEMID) & valid_patients]
            if not chunk_spec.empty:
                specimen_rows.append(chunk_spec[['specimen_id', 'value']])

    if not lab_rows:
        df_labs = pd.DataFrame(columns=['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum', 'valueuom', 'specimen_id'])
    else:
        df_labs = pd.concat(lab_rows, ignore_index=True)

    df_labs_hadm = df_labs[df_labs['hadm_id'].notna()].merge(base_cohort[['hadm_id', 'stay_id', 'intime', 'time_zero']], on='hadm_id')
    df_labs_sub = df_labs[df_labs['hadm_id'].isna()].merge(base_cohort[['subject_id', 'stay_id', 'intime', 'time_zero']], on='subject_id')
    df_labs = pd.concat([df_labs_hadm, df_labs_sub], ignore_index=True)

    df_labs['charttime'] = pd.to_datetime(df_labs['charttime'])
    
    # Arterial BG Filter
    if ARTERIAL_ONLY_BG:
        spec_df = pd.concat(specimen_rows, ignore_index=True) if specimen_rows else pd.DataFrame(columns=['specimen_id', 'value'])
        spec_df = spec_df.dropna(subset=['specimen_id', 'value']).copy()
        spec_df['value'] = spec_df['value'].astype(str).str.strip()
        spec_df = spec_df.drop_duplicates(subset=['specimen_id', 'value'])
        extract_log[f'specimen_type_{SPECIMEN_TYPE_ITEMID}_raw_value_counts'] = {
            str(k): int(v) for k, v in spec_df['value'].value_counts().to_dict().items()
        }
        
        n_vals = spec_df.groupby('specimen_id')['value'].nunique()
        conflicting_ids = set(n_vals[n_vals > 1].index)
        extract_log[f'specimen_type_{SPECIMEN_TYPE_ITEMID}_conflicting_specimens'] = int(len(conflicting_ids))
        
        spec_df.loc[spec_df['specimen_id'].isin(conflicting_ids), 'value'] = 'other'
        spec_df = spec_df.drop_duplicates(subset=['specimen_id', 'value'])
        
        bg_itemids = {50820: 'pH', 50821: 'pO2', 50818: 'pCO2', 50802: 'BaseExcess'}
        is_bg = df_labs['itemid'].isin(bg_itemids.keys())
        bg_rows = df_labs[is_bg].copy()
        bg_with_spec = bg_rows.merge(spec_df, on='specimen_id', how='left')
        
        bg_with_spec['value'] = bg_with_spec['value'].fillna('no specimen type recorded')
        valid_cats = ['ART.', 'VEN.', 'MIX.', 'CENTRAL VENOUS.', 'other', 'no specimen type recorded']
        bg_with_spec['value'] = bg_with_spec['value'].apply(lambda x: x if x in valid_cats else 'other')
        
        bg_counts = {}
        bg_kept = {}
        bg_dropped = {}
        for itemid, varname in bg_itemids.items():
            var_rows = bg_with_spec[bg_with_spec['itemid'] == itemid]
            counts = {c: 0 for c in valid_cats}
            for k, v in var_rows['value'].value_counts().items():
                counts[k] = int(v)
            bg_counts[varname] = counts
            is_art = var_rows['value'] == 'ART.'
            bg_kept[str(itemid)] = int(is_art.sum())
            bg_dropped[str(itemid)] = int((~is_art).sum())
            
        extract_log['blood_gas_specimen_category_counts'] = bg_counts
        extract_log['arterial_bg_kept'] = bg_kept
        extract_log['arterial_bg_dropped'] = bg_dropped
        
        is_arterial = bg_with_spec['value'] == 'ART.'
        
        kept_bg = bg_with_spec[is_arterial].drop(columns=['value'])
        df_labs = pd.concat([df_labs[~is_bg], kept_bg], ignore_index=True)
        
        if len(df_labs) == 0:
            raise ValueError("0 lab rows remaining after arterial filter")

    # Time filter
    df_labs = df_labs[
        (df_labs['charttime'] >= df_labs['intime'] - pd.Timedelta(hours=LAB_PRE_WINDOW_H)) & 
        (df_labs['charttime'] <= df_labs['time_zero'])
    ]
    
    if not df_labs.empty:
        diff_h = (df_labs['charttime'] - df_labs['time_zero']).dt.total_seconds() / 3600.0
        extract_log.setdefault('temporal_audit', {})['labevents'] = {
            'min_h': float(diff_h.min()), 'max_h': float(diff_h.max()), 'rows': len(df_labs)
        }
        assert diff_h.max() <= 0, f"labevents max_h > 0: {diff_h.max()}"
        assert diff_h.min() >= -30, f"labevents min_h < -30: {diff_h.min()}"

    df_labs['feature_name'] = df_labs['itemid'].map(item_to_lab)
    
    # Unit check
    check_units(df_labs, 'feature_name', 'valueuom', extract_log, 'lab_unit_counts', 'lab_missing_units')
    
    # Bounds check
    df_labs = apply_bounds(df_labs, 'feature_name', 'valuenum', bounds_dict, bounds_counts)

    agg_dict = {name: info['agg'] for name, info in LAB_ITEMS.items() if name not in ['Sodium', 'Chloride']}
    grouped = df_labs.groupby(['stay_id', 'feature_name'])['valuenum'].agg(['min', 'max']).reset_index()

    def select_agg(row):
        if row['feature_name'] in agg_dict:
            return row[agg_dict[row['feature_name']]]
        return np.nan

    grouped['val'] = grouped.apply(select_agg, axis=1)
    pivoted_labs = grouped.pivot(index='stay_id', columns='feature_name', values='val').reset_index()
    
    sc_df = grouped[grouped['feature_name'].isin(['Sodium', 'Chloride'])]
    if not sc_df.empty:
        sc_pivot = sc_df.pivot(index='stay_id', columns='feature_name', values=['min', 'max'])
        sc_pivot.columns = [f"{col[1]}_{col[0]}" for col in sc_pivot.columns]
        pivoted_labs = pivoted_labs.merge(sc_pivot.reset_index(), on='stay_id', how='left')
    
    pivoted_labs = pivoted_labs.drop(columns=['Sodium', 'Chloride'], errors='ignore')

    expected_labs = [k for k in LAB_ITEMS.keys() if k not in ['Sodium', 'Chloride']] + ['Sodium_min', 'Sodium_max', 'Chloride_min', 'Chloride_max']
    for col in expected_labs:
        if col not in pivoted_labs.columns:
            pivoted_labs[col] = np.nan
            
    cohort_labs = base_cohort.merge(pivoted_labs[['stay_id'] + expected_labs], on='stay_id', how='left')
    return cohort_labs


def extract_vitals(cohort_labs, bounds_dict, bounds_counts, extract_log):
    print("Step: cohort_with_vitals - Extracting vital measurements")
    base_stay_ids = set(cohort_labs['stay_id'].unique())
    all_vital_item_ids = []
    item_to_vital = {}
    for name, ids in VITAL_ITEMS.items():
        for item_id in ids:
            all_vital_item_ids.append(item_id)
            item_to_vital[item_id] = name
            
    chart_path = os.path.join(MIMIC_PATH, 'icu', 'chartevents.csv.gz')
    vital_rows = []
    for chunk in pd.read_csv(chart_path, chunksize=1000000, compression='gzip', low_memory=False):
        chunk = chunk[chunk['itemid'].isin(all_vital_item_ids) & chunk['stay_id'].isin(base_stay_ids)]
        if not chunk.empty:
            vital_rows.append(chunk[['stay_id', 'charttime', 'itemid', 'value', 'valuenum', 'valueuom']])
            
    df_vitals = pd.concat(vital_rows, ignore_index=True) if vital_rows else pd.DataFrame(columns=['stay_id', 'charttime', 'itemid', 'value', 'valuenum', 'valueuom'])
    
    df_vitals = df_vitals.merge(cohort_labs[['stay_id', 'intime', 'time_zero']], on='stay_id')
    df_vitals['charttime'] = pd.to_datetime(df_vitals['charttime'])
    
    # Time filter [intime, time_zero]
    df_vitals = df_vitals[(df_vitals['charttime'] >= df_vitals['intime']) & (df_vitals['charttime'] <= df_vitals['time_zero'])]
    
    if not df_vitals.empty:
        diff_h = (df_vitals['charttime'] - df_vitals['time_zero']).dt.total_seconds() / 3600.0
        extract_log.setdefault('temporal_audit', {})['chartevents'] = {
            'min_h': float(diff_h.min()), 'max_h': float(diff_h.max()), 'rows': len(df_vitals)
        }
        assert diff_h.max() <= 0, f"chartevents max_h > 0: {diff_h.max()}"
        assert diff_h.min() >= -24, f"chartevents min_h < -24: {diff_h.min()}"

    
    if not df_vitals.empty:
        diff_h = (df_vitals['charttime'] - df_vitals['time_zero']).dt.total_seconds() / 3600.0
        extract_log.setdefault('temporal_audit', {})['chartevents'] = {
            'min_h': float(diff_h.min()), 'max_h': float(diff_h.max()), 'rows': len(df_vitals)
        }
        assert diff_h.max() <= 0, f"chartevents max_h > 0: {diff_h.max()}"
        assert diff_h.min() >= -24, f"chartevents min_h < -24: {diff_h.min()}"

    
    # Temperature F to C
    fahrenheit_mask = df_vitals['itemid'] == 223761
    extract_log['temp_converted_F_to_C'] = int(fahrenheit_mask.sum())
    df_vitals.loc[fahrenheit_mask, 'valuenum'] = (pd.to_numeric(df_vitals.loc[fahrenheit_mask, 'valuenum'], errors='coerce') - 32) * 5.0 / 9.0
    df_vitals.loc[fahrenheit_mask, 'valueuom'] = 'degC'
    
    df_vitals['vital_name'] = df_vitals['itemid'].map(item_to_vital)
    
    check_units(df_vitals, 'vital_name', 'valueuom', extract_log, 'vital_unit_counts', 'vital_missing_units')
    
    gcs_mask = df_vitals['vital_name'] == 'GCS'
    df_gcs = df_vitals[gcs_mask].copy()
    df_other = df_vitals[~gcs_mask].copy()
    
    df_other = apply_bounds(df_other, 'vital_name', 'valuenum', bounds_dict, bounds_counts)
    
    # GCS processing
    df_gcs['valuenum'] = pd.to_numeric(df_gcs['valuenum'], errors='coerce')
    
    verbal = df_gcs[df_gcs['itemid'] == GCS_ITEMIDS['verbal']]
    verbal_vc = verbal.groupby(['value', 'valuenum'], dropna=False).size().reset_index(name='count')
    extract_log['gcs_verbal_value_counts'] = [
        {'value': None if pd.isna(r['value']) else str(r['value']),
         'valuenum': None if pd.isna(r['valuenum']) else float(r['valuenum']),
         'count': int(r['count'])}
        for _, r in verbal_vc.sort_values('count', ascending=False).iterrows()
    ]
    
    # intubated_24h
    intubated_stays = verbal[verbal['value'] == INTUBATED_VERBAL_VALUE]['stay_id'].unique()
    extract_log['intubated_24h_rows_matched'] = int((verbal['value'] == INTUBATED_VERBAL_VALUE).sum())
    cohort_labs['intubated_24h'] = cohort_labs['stay_id'].isin(intubated_stays).astype(int)
    extract_log['intubated_24h_share'] = float(cohort_labs['intubated_24h'].mean())
    
    gcs_out_of_range = {}
    for comp, item_id in GCS_ITEMIDS.items():
        lo, hi = GCS_COMPONENT_RANGES[comp]
        comp_mask = df_gcs['itemid'] == item_id
        out_mask = comp_mask & ((df_gcs['valuenum'] < lo) | (df_gcs['valuenum'] > hi))
        gcs_out_of_range[comp] = int(out_mask.sum())
        df_gcs.loc[out_mask, 'valuenum'] = np.nan
    extract_log['gcs_out_of_range'] = gcs_out_of_range
    
    df_gcs = df_gcs.dropna(subset=['valuenum'])
    
    # Ensure one value per component per charttime
    dups = df_gcs.duplicated(subset=['stay_id', 'charttime', 'itemid'])
    extract_log['gcs_duplicate_component_rows'] = int(dups.sum())
    df_gcs = df_gcs[~dups]
    
    piv = df_gcs.pivot(index=['stay_id', 'charttime'], columns='itemid', values='valuenum').reset_index()
    
    eye = GCS_ITEMIDS['eye']
    ver = GCS_ITEMIDS['verbal']
    mot = GCS_ITEMIDS['motor']
    
    for c in [eye, ver, mot]:
        if c not in piv.columns: piv[c] = np.nan
        
    complete_mask = piv[eye].notna() & piv[ver].notna() & piv[mot].notna()
    extract_log['gcs_complete_charttimes'] = int(complete_mask.sum())
    extract_log['gcs_incomplete_charttimes'] = int((~complete_mask).sum())
    
    piv_complete = piv[complete_mask].copy()
    piv_complete['valuenum'] = piv_complete[eye] + piv_complete[ver] + piv_complete[mot]
    piv_complete['vital_name'] = 'GCS'
    
    df_gcs_sum = piv_complete[['stay_id', 'charttime', 'vital_name', 'valuenum']].copy()
    
    # Bounds on total GCS
    total_lo, total_hi = GCS_TOTAL_BOUNDS
    out_total = (df_gcs_sum['valuenum'] < total_lo) | (df_gcs_sum['valuenum'] > total_hi)
    extract_log['gcs_total_out_of_bounds'] = int(out_total.sum())
    df_gcs_sum.loc[out_total, 'valuenum'] = np.nan
    df_gcs_sum = df_gcs_sum.dropna(subset=['valuenum'])
    
    # BP Consistency Rule
    inv_sbp = 220050; inv_dbp = 220051; inv_map = [220052, 225312]
    ni_sbp = 220179; ni_dbp = 220180; ni_map = 220181
    
    def check_bp(df_sub, sbp_id, dbp_id, map_ids):
        df_sub = df_sub.copy()
        if not isinstance(map_ids, list): map_ids = [map_ids]
        
        piv = df_sub.pivot_table(index=['stay_id', 'charttime'], columns='itemid', values='valuenum').reset_index()
        
        # Ensure columns exist
        for col in [sbp_id, dbp_id] + map_ids:
            if col not in piv.columns: piv[col] = float('nan')
            
        map_col = piv[map_ids].mean(axis=1) if len(map_ids) > 1 else piv[map_ids[0]]
        
        cond_dbp_ge_sbp = (piv[sbp_id].notna()) & (piv[dbp_id].notna()) & (piv[dbp_id] >= piv[sbp_id])
        cond_map_gt_sbp = (map_col.notna()) & (piv[sbp_id].notna()) & (map_col > piv[sbp_id])
        cond_map_lt_dbp = (map_col.notna()) & (piv[dbp_id].notna()) & (map_col < piv[dbp_id])
        
        drop_mask = cond_dbp_ge_sbp | cond_map_gt_sbp | cond_map_lt_dbp
        drop_times = piv.loc[drop_mask, ['stay_id', 'charttime']]
        
        if drop_times.empty: return df_sub, 0
        
        drop_keys = set(tuple(x) for x in drop_times.values)
        mask_to_drop = df_sub[['stay_id', 'charttime']].apply(tuple, axis=1).isin(drop_keys) & df_sub['itemid'].isin([sbp_id, dbp_id] + map_ids)
        
        return df_sub[~mask_to_drop], int(len(drop_keys))

    df_other, dropped_inv = check_bp(df_other, inv_sbp, inv_dbp, inv_map)
    df_other, dropped_ni = check_bp(df_other, ni_sbp, ni_dbp, ni_map)
    extract_log['bp_consistency_dropped_invasive'] = dropped_inv
    extract_log['bp_consistency_dropped_noninvasive'] = dropped_ni

    df_vitals_agg = pd.concat([df_other[['stay_id', 'charttime', 'vital_name', 'valuenum']], df_gcs_sum], ignore_index=True)
    
    # Aggregation
    def p05(x): return x.quantile(0.05)
    def p95(x): return x.quantile(0.95)
    
    non_gcs = df_vitals_agg[df_vitals_agg['vital_name'] != 'GCS']
    gcs = df_vitals_agg[df_vitals_agg['vital_name'] == 'GCS']
    
    agg_non_gcs = non_gcs.groupby(['stay_id', 'vital_name'])['valuenum'].agg(['mean', p05, p95]).reset_index()
    agg_gcs = gcs.groupby(['stay_id', 'vital_name'])['valuenum'].agg(['mean', 'min', 'max']).reset_index()
    
    piv_mean = pd.concat([agg_non_gcs.pivot(index='stay_id', columns='vital_name', values='mean'),
                          agg_gcs.pivot(index='stay_id', columns='vital_name', values='mean')], axis=1).add_suffix('_mean')
                          
    piv_p05 = agg_non_gcs.pivot(index='stay_id', columns='vital_name', values='p05').add_suffix('_p05')
    piv_p95 = agg_non_gcs.pivot(index='stay_id', columns='vital_name', values='p95').add_suffix('_p95')
    piv_min = agg_gcs.pivot(index='stay_id', columns='vital_name', values='min').add_suffix('_min')
    piv_max = agg_gcs.pivot(index='stay_id', columns='vital_name', values='max').add_suffix('_max')
    
    pivoted_vitals = pd.concat([piv_mean, piv_p05, piv_p95, piv_min, piv_max], axis=1).reset_index()
    
    expected_vitals = []
    for v in VITAL_ITEMS.keys():
        if v == 'GCS':
            expected_vitals.extend([f"{v}_mean", f"{v}_min", f"{v}_max"])
        else:
            expected_vitals.extend([f"{v}_mean", f"{v}_p05", f"{v}_p95"])
    for col in expected_vitals:
        if col not in pivoted_vitals.columns:
            pivoted_vitals[col] = np.nan
            
    cohort_vitals = cohort_labs.merge(pivoted_vitals[['stay_id'] + expected_vitals], on='stay_id', how='left')
    return cohort_vitals


def extract_comorbidities(cohort_vitals, extract_log):
    print("Step: cohort_comorbidities - Extracting diagnoses")
    base_subject_ids = set(cohort_vitals['subject_id'].unique())
    diagnoses_path = os.path.join(MIMIC_PATH, 'hosp', 'diagnoses_icd.csv.gz')
    admissions_path = os.path.join(MIMIC_PATH, 'hosp', 'admissions.csv.gz')
    
    adms = pd.read_csv(admissions_path)[['subject_id', 'hadm_id', 'admittime']]
    adms['admittime'] = pd.to_datetime(adms['admittime'])
    
    # We need index admittime
    idx_adms = cohort_vitals[['subject_id', 'hadm_id']].merge(adms, on=['subject_id', 'hadm_id'])
    idx_adms = idx_adms.rename(columns={'admittime': 'index_admittime'})
    
    adms = adms.merge(idx_adms[['subject_id', 'index_admittime']], on='subject_id')
    # Strict mode: admittime BEFORE index admission
    valid_adms = adms[adms['admittime'] < adms['index_admittime']]['hadm_id'].unique()
    assert (adms.loc[adms['hadm_id'].isin(valid_adms), 'admittime'] < adms.loc[adms['hadm_id'].isin(valid_adms), 'index_admittime']).all(), "Not all admissions are strictly prior"
    extract_log.setdefault('temporal_audit', {})['diagnoses_icd'] = {'admissions_prior_checked': True}

    assert (adms.loc[adms['hadm_id'].isin(valid_adms), 'admittime'] < adms.loc[adms['hadm_id'].isin(valid_adms), 'index_admittime']).all(), "Not all admissions are strictly prior"
    extract_log.setdefault('temporal_audit', {})['diagnoses_icd'] = {'admissions_prior_checked': True}

    
    diag_chunks = []
    for chunk in pd.read_csv(diagnoses_path, chunksize=1000000, compression='gzip', low_memory=False):
        chunk = chunk[chunk['hadm_id'].isin(valid_adms)]
        if not chunk.empty:
            diag_chunks.append(chunk[['subject_id', 'icd_code', 'icd_version']])
            
    diagnoses = pd.concat(diag_chunks, ignore_index=True) if diag_chunks else pd.DataFrame(columns=['subject_id', 'icd_code', 'icd_version'])
    
    cohort = cohort_vitals.copy()
    for comob, prefixes in COMORBIDITIES.items():
        mask_icd9 = (diagnoses['icd_version'] == 9) & diagnoses['icd_code'].str.startswith(tuple(prefixes['icd9']), na=False)
        mask_icd10 = (diagnoses['icd_version'] == 10) & diagnoses['icd_code'].str.startswith(tuple(prefixes['icd10']), na=False)
        has_comob = diagnoses[mask_icd9 | mask_icd10]['subject_id'].unique()
        cohort[comob] = cohort['subject_id'].isin(has_comob).astype(int)
        
    return cohort


def main():
    parser = argparse.ArgumentParser(description="Extract MIMIC-IV sepsis cohort.")
    parser.add_argument("--mimic-dir", type=str, required=True, help="Path to raw MIMIC-IV folder")
    args = parser.parse_args()
    global MIMIC_PATH
    MIMIC_PATH = args.mimic_dir

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
            'COMORBIDITY_MODE': COMORBIDITY_MODE,
            'ARTERIAL_ONLY_BG': ARTERIAL_ONLY_BG,
            'SPECIMEN_TYPE_ITEMID': SPECIMEN_TYPE_ITEMID
        }
    }
    
    try:
        bounds_dict = load_bounds()
        base = extract_base_cohort(flow_counts, extract_log)
        cohort_l = extract_labs(base, bounds_dict, bounds_counts, extract_log)
        cohort_v = extract_vitals(cohort_l, bounds_dict, bounds_counts, extract_log)
        cohort_c = extract_comorbidities(cohort_v, extract_log)
        
        # Assertion: ensure deathtime, dischtime, hospital_expire_flag come ONLY from index admission
        adms_assert = pd.read_csv(os.path.join(MIMIC_PATH, 'hosp', 'admissions.csv.gz'), usecols=['hadm_id', 'deathtime', 'dischtime', 'hospital_expire_flag'])
        adms_assert['deathtime'] = pd.to_datetime(adms_assert['deathtime'])
        adms_assert['dischtime'] = pd.to_datetime(adms_assert['dischtime'])
        
        check_df = cohort_c[['hadm_id', 'deathtime', 'dischtime', 'hospital_expire_flag']].merge(
            adms_assert, on='hadm_id', suffixes=('', '_raw')
        )
        
        assert (check_df['dischtime'] == check_df['dischtime_raw']).all(), "dischtime mismatch with index admission"
        assert (check_df['hospital_expire_flag'] == check_df['hospital_expire_flag_raw']).all(), "hospital_expire_flag mismatch with index admission"
        
        expected_death = check_df['deathtime_raw'].copy()
        missing_death = (check_df['hospital_expire_flag_raw'] == 1) & expected_death.isna()
        expected_death.loc[missing_death] = check_df.loc[missing_death, 'dischtime_raw']
        expected_death.loc[check_df['hospital_expire_flag_raw'] == 0] = pd.NaT
        
        mask_both_nat = check_df['deathtime'].isna() & expected_death.isna()
        mask_equal = check_df['deathtime'] == expected_death
        assert (mask_both_nat | mask_equal).all(), "deathtime mismatch with index admission"
        
        # Outcome: death in (time_zero, time_zero + HORIZON_DAYS]
        death_after_tz = cohort_c['deathtime'] > cohort_c['time_zero']
        death_within_horizon = cohort_c['deathtime'] <= cohort_c['time_zero'] + pd.Timedelta(days=HORIZON_DAYS)
        cohort_c['event'] = (death_after_tz & death_within_horizon).astype(int)
        
        # Compute time_days
        cohort_c['time_days'] = HORIZON_DAYS
        cohort_c.loc[cohort_c['event'] == 1, 'time_days'] = (cohort_c.loc[cohort_c['event'] == 1, 'deathtime'] - cohort_c.loc[cohort_c['event'] == 1, 'time_zero']).dt.total_seconds() / 86400.0
        
        # Hospice masking
        hospice_mask = cohort_c['discharge_location'].str.contains('HOSPICE', na=False, case=False)
        extract_log['hospice_discharges'] = int(hospice_mask.sum())
        
        cohort_c['SOFA'] = cohort_c['sofa_total']
        
        # Output explicitly allowlisted columns only
        allowlist_features = ['age', 'gender'] + [k for k in LAB_ITEMS.keys() if k not in ['Sodium', 'Chloride']] + ['Sodium_min', 'Sodium_max', 'Chloride_min', 'Chloride_max'] + [f"{v}_{agg}" for v in VITAL_ITEMS.keys() if v != 'GCS' for agg in ['mean', 'p05', 'p95']] + ['GCS_mean', 'GCS_min', 'GCS_max'] + ['intubated_24h'] + list(COMORBIDITIES.keys()) + ['SOFA']
        missing_pct = cohort_c[allowlist_features].isna().mean()
        
        # 100% missingness guard
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
        
        final_cols = ['subject_id', 'hadm_id', 'stay_id', 'event', 'time_days'] + feature_cols
        
        present_not_allowlisted = [c for c in cohort_c.columns if c not in final_cols]
        print(f"Columns dropped because they are not in the allowlist: {present_not_allowlisted}")
        extract_log['dropped_not_in_allowlist'] = present_not_allowlisted
        
        final_cohort = cohort_c[final_cols].copy()
        
        # Assertions
        assert len(final_cohort) == final_cohort['subject_id'].nunique()
        assert (final_cohort['time_days'] > 0).all()
        assert (final_cohort['time_days'] <= HORIZON_DAYS).all()
        assert set(final_cohort['event'].unique()).issubset({0, 1})
        
        forbidden_words = ['time', 'offset', 'los', 'disch', 'death', 'dod', 'expire', 'intime', 'outtime', 'admit']
        for c in final_cohort.columns:
            if c in ['subject_id', 'hadm_id', 'stay_id', 'event', 'time_days']: continue
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
        os.makedirs(os.path.dirname(LOG_OUT), exist_ok=True)
        with open(LOG_OUT, 'w') as f:
            json.dump(sanitize(extract_log), f, indent=2)

if __name__ == "__main__":
    main()
