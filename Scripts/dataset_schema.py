import numpy as np
import pandas as pd
import warnings

DATASET_SCHEMAS = {
    'metabric': {
        'targets': ['event', 'survival_months'],
        'exclude': ['patientId', 'sampleId', 'cohort', 'gender'],
    },
    'mimic': {
        'targets': ['event', 'time_days'],
        'exclude': ['subject_id', 'hadm_id', 'stay_id'],
    },
    'eicu': {
        'targets': ['event', 'time_days'],
        'exclude': ['uniquepid', 'patientunitstayid', 'GCS_min', 'GCS_max'],
    },
    'synthetic': {
        'targets': ['event', 'time_days'],
        'exclude': ['patient_id'],
    },
}

def get_feature_columns(df, dataset_name):
    # Task 3: Operate on a copy
    df = df.copy()

    name_lower = dataset_name.lower()
    if 'metabric' in name_lower:
        schema = DATASET_SCHEMAS['metabric']
    elif 'eicu' in name_lower:
        schema = DATASET_SCHEMAS['eicu']
    elif 'mimic' in name_lower:
        schema = DATASET_SCHEMAS['mimic']
    elif 'synthetic' in name_lower:
        schema = DATASET_SCHEMAS['synthetic']
    else:
        raise ValueError(f"No schema defined in DATASET_SCHEMAS for dataset_name={dataset_name!r}. "
                          f"Add an explicit entry to DATASET_SCHEMAS before proceeding, "
                          f"do not fall back to another dataset's schema.")
                          
    if schema.get('targets') is None or schema.get('exclude') is None:
        raise ValueError(f"Schema for {dataset_name!r} is a placeholder and must be filled in before use.")

    # Apply Encodings before dropping
    if 'gender' in df.columns and 'gender' not in schema['exclude']:
        df['gender'] = df['gender'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})
    
    if 'menopausal_state' in df.columns and 'menopausal_state' not in schema['exclude']:
        df['menopausal_state'] = df['menopausal_state'].map({'PRE': 0, 'POST': 1})
        
    if 'cellularity' in df.columns and 'cellularity' not in schema['exclude']:
        df['cellularity'] = df['cellularity'].map({'Low': 1, 'Moderate': 2, 'High': 3})
        
    if 'histological_subtype' in df.columns and 'histological_subtype' not in schema['exclude']:
        missing_mask = df['histological_subtype'].isna()
        dummies = pd.get_dummies(df['histological_subtype'], prefix='hist', dummy_na=False)
        hist_cols = [c for c in FEATURE_TYPES['metabric']['binary'] if c.startswith('hist_')]
        for c in hist_cols:
            if c in dummies.columns:
                df[c] = dummies[c].astype(float)
            else:
                df[c] = 0.0
            df.loc[missing_mask, c] = np.nan
        df.drop(columns=['histological_subtype'], inplace=True)

    cols_to_drop = schema['targets'] + schema['exclude']
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df_filtered = df.drop(columns=existing_drops)
    
    # Task 3: Visibility Warning
    non_numeric_cols = df_filtered.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        warnings.warn(
            f"WARNING: The following non-numeric columns are not excluded by the schema "
            f"but will be silently dropped because they lack encoding: {non_numeric_cols}",
            UserWarning
        )

    feature_df = df_filtered.select_dtypes(include=[np.number])
    
    name_lower = dataset_name.lower()
    if name_lower in FEATURE_TYPES:
        ft = FEATURE_TYPES[name_lower]
        ordered_cols = ft.get('binary', []) + list(ft.get('categorical', {}).keys()) + ft.get('continuous', [])
        feature_df = feature_df[ordered_cols]
    
    return feature_df.columns, schema['targets'], feature_df

FEATURE_TYPES = {
    'metabric': {
        'categorical': {
            'grade': [1, 2, 3],
            'tumour_stage': [0, 1, 2, 3, 4],
            'cellularity': [1, 2, 3]
        },
        'binary': ['ER_STATUS_bin', 'PR_STATUS_bin', 'HER2_STATUS_bin', 'chemotherapy', 'hormone_therapy', 'radio_therapy', 'menopausal_state', 'hist_Ductal/NST', 'hist_Lobular', 'hist_Medullary', 'hist_Metaplastic', 'hist_Mixed', 'hist_Mucinous', 'hist_Other', 'hist_Tubular/ cribriform'],
        'continuous': ['age', 'tumor_size', 'lymph_nodes_positive', 'npi']
    },
    'mimic': {
        'categorical': {},
        'binary': ['gender', 'intubated_24h', 'CCI_MI', 'CCI_CHF', 'CCI_PVD', 'CCI_Stroke', 'CCI_Dementia', 'CCI_COPD', 'CCI_Renal', 'CCI_Liver', 'CCI_Cancer', 'CCI_Diabetes', 'CCI_HIV'],
        'continuous': ['age', 'Albumin', 'AnionGap', 'Bicarbonate', 'BUN', 'Calcium', 'Chloride_min', 'Chloride_max', 'Creatinine', 'Glucose', 'Sodium_min', 'Sodium_max', 'Potassium', 'Lactate', 'Hemoglobin', 'Hematocrit', 'Platelets', 'WBC', 'Lymphocytes', 'Neutrophils', 'INR', 'PT', 'PTT', 'ALT', 'AST', 'Bilirubin_Total', 'pH', 'pO2', 'pCO2', 'BaseExcess', 'Magnesium', 'Phosphate', 'LDH', 'Troponin_T', 'HR_mean', 'HR_p05', 'HR_p95', 'RR_mean', 'RR_p05', 'RR_p95', 'SBP_mean', 'SBP_p05', 'SBP_p95', 'DBP_mean', 'DBP_p05', 'DBP_p95', 'MAP_mean', 'MAP_p05', 'MAP_p95', 'Temp_mean', 'Temp_p05', 'Temp_p95', 'O2Sat_mean', 'O2Sat_p05', 'O2Sat_p95', 'GCS_mean', 'GCS_min', 'GCS_max', 'SOFA']
    },
    'eicu': {
        'categorical': {},
        'binary': ['gender', 'intubated_24h', 'CCI_MI', 'CCI_CHF', 'CCI_PVD', 'CCI_Stroke', 'CCI_Dementia', 'CCI_COPD', 'CCI_Renal', 'CCI_Liver', 'CCI_Cancer', 'CCI_Diabetes', 'CCI_HIV'],
        'continuous': ['age', 'admissionheight', 'admissionweight', 'Albumin', 'AnionGap', 'Bicarbonate', 'BUN', 'Calcium', 'Chloride_min', 'Chloride_max', 'Creatinine', 'Glucose', 'Sodium_min', 'Sodium_max', 'Potassium', 'Lactate', 'Hemoglobin', 'Hematocrit', 'Platelets', 'WBC', 'Lymphocytes', 'Neutrophils', 'INR', 'PT', 'PTT', 'ALT', 'AST', 'Bilirubin_Total', 'pH', 'pO2', 'pCO2', 'BaseExcess', 'Magnesium', 'Phosphate', 'HR_mean', 'HR_p05', 'HR_p95', 'RR_mean', 'RR_p05', 'RR_p95', 'SBP_mean', 'SBP_p05', 'SBP_p95', 'DBP_mean', 'DBP_p05', 'DBP_p95', 'MAP_mean', 'MAP_p05', 'MAP_p95', 'Temp_mean', 'Temp_p05', 'Temp_p95', 'O2Sat_mean', 'O2Sat_p05', 'O2Sat_p95', 'GCS_mean', 'SOFA']
    },
    'synthetic': {
        'categorical': {},
        'binary': [],
        'continuous': ['age', 'Lactate', 'Albumin', 'AnionGap', 'Bicarbonate', 'BUN', 'Calcium', 'Chloride_min', 'Chloride_max', 'Creatinine', 'Glucose', 'Sodium_min', 'Sodium_max', 'Potassium', 'Hemoglobin', 'Hematocrit', 'Platelets', 'WBC', 'Lymphocytes', 'Neutrophils', 'INR', 'PT', 'PTT', 'ALT', 'AST', 'Bilirubin_Total', 'pH', 'pO2', 'pCO2', 'BaseExcess', 'Magnesium', 'Phosphate', 'LDH', 'Troponin_T', 'HR_mean', 'HR_p05', 'HR_p95', 'RR_mean', 'RR_p05', 'RR_p95', 'SBP_mean', 'SBP_p05', 'SBP_p95', 'DBP_mean', 'DBP_p05', 'DBP_p95', 'MAP_mean', 'MAP_p05', 'MAP_p95', 'Temp_mean', 'Temp_p05', 'Temp_p95', 'O2Sat_mean', 'O2Sat_p05', 'O2Sat_p95', 'GCS_mean', 'GCS_min', 'GCS_max', 'SOFA']
    }
}

def get_feature_types(dataset_name, feature_cols):
    name_lower = dataset_name.lower()
    if 'metabric' in name_lower:
        base = FEATURE_TYPES['metabric']
    elif 'eicu' in name_lower:
        base = FEATURE_TYPES['eicu']
    elif 'mimic' in name_lower:
        base = FEATURE_TYPES['mimic']
    elif 'synthetic' in name_lower:
        base = FEATURE_TYPES['synthetic']
    else:
        raise ValueError(f"No feature types for {dataset_name}")

    types = {'binary': [], 'categorical': base.get('categorical', {}).copy(), 'continuous': []}
    
    for c in feature_cols:
        if c in base.get('binary', []):
            types['binary'].append(c)
        elif c in types['categorical']:
            continue
        elif c in base.get('continuous', []):
            types['continuous'].append(c)
        else:
            raise ValueError(f"Feature {c} not classified for {name_lower}")

    all_listed = set(types['binary']) | set(types['categorical'].keys()) | set(types['continuous'])
    assert len(types['binary']) + len(types['categorical']) + len(types['continuous']) == len(all_listed), "Overlap in types"
    
    missing_from_cols = all_listed - set(feature_cols)
    assert not missing_from_cols, f"Listed columns missing from feature_cols: {missing_from_cols}"
    
    missing_from_listed = set(feature_cols) - all_listed
    assert not missing_from_listed, f"feature_cols missing from types: {missing_from_listed}"

    return types
