import os
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# Configuration
MIMIC_PATH = None
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = os.path.join(BASE_DIR, 'Results')
DERIVED_PATH = os.path.join(OUTPUT_PATH, 'derived')
INTERMEDIATE_PATH = os.path.join(OUTPUT_PATH, 'intermediate')
FINAL_PATH = os.path.join(OUTPUT_PATH, 'final')
MIN_AGE = 18
ICU_WINDOW_HOURS = 24

def generate_suspected_infection(icustays):
    """Generate suspected infection table based on antibiotic and culture co-occurrence."""
    print("Step: suspected_infection - Loading prescriptions")
    presc_path = os.path.join(MIMIC_PATH, 'hosp', 'prescriptions.csv.gz')
    antibiotics = [
        'vancomycin', 'piperacillin', 'meropenem', 'cefepime', 'metronidazole', 
        'ciprofloxacin', 'ampicillin', 'gentamicin', 'azithromycin', 
        'levofloxacin', 'clindamycin', 'linezolid', 'daptomycin', 
        'ceftriaxone', 'ceftazidime'
    ]
    
    stay_hadm_ids = set(icustays['hadm_id'].dropna().unique())
    presc_chunks = []
    # Build regex pattern with word boundaries to avoid unexpected substring matching
    pattern = r'\b(' + '|'.join(antibiotics) + r')\b'
    for chunk in pd.read_csv(presc_path, chunksize=1000000, compression='gzip', low_memory=False):
        chunk = chunk[chunk['hadm_id'].isin(stay_hadm_ids)]
        if not chunk.empty:
            chunk = chunk[chunk['drug'].str.lower().str.contains(pattern, na=False, regex=True)]
            if not chunk.empty:
                presc_chunks.append(chunk[['subject_id', 'hadm_id', 'drug', 'starttime']])
                
    if not presc_chunks:
        print("Step: suspected_infection - No antibiotics matched")
        presc_df = pd.DataFrame(columns=['subject_id', 'hadm_id', 'drug', 'starttime'])
    else:
        presc_df = pd.concat(presc_chunks, ignore_index=True)
        
    print(f"Step: suspected_infection - Filtered prescriptions count: {len(presc_df)}")
    
    print("Step: suspected_infection - Loading microbiology events")
    micro_path = os.path.join(MIMIC_PATH, 'hosp', 'microbiologyevents.csv.gz')
    micro_chunks = []
    for chunk in pd.read_csv(micro_path, chunksize=1000000, compression='gzip', low_memory=False):
        chunk = chunk[chunk['hadm_id'].isin(stay_hadm_ids)]
        if not chunk.empty:
            micro_chunks.append(chunk[['subject_id', 'hadm_id', 'charttime']])
            
    if not micro_chunks:
        micro_df = pd.DataFrame(columns=['subject_id', 'hadm_id', 'charttime'])
    else:
        micro_df = pd.concat(micro_chunks, ignore_index=True)
        
    print(f"Step: suspected_infection - Filtered microbiology events count: {len(micro_df)}")
    
    presc_df['starttime'] = pd.to_datetime(presc_df['starttime'])
    micro_df['charttime'] = pd.to_datetime(micro_df['charttime'])
    
    abx_merge = presc_df.merge(icustays[['stay_id', 'hadm_id', 'intime', 'outtime']], on='hadm_id')
    abx_merge = abx_merge[
        (abx_merge['starttime'] >= pd.to_datetime(abx_merge['intime']) - pd.Timedelta(hours=24)) & 
        (abx_merge['starttime'] <= pd.to_datetime(abx_merge['outtime']) + pd.Timedelta(hours=24))
    ]
    first_abx = abx_merge.groupby('stay_id')['starttime'].min().reset_index().rename(columns={'starttime': 'antibiotic_time'})
    
    micro_merge = micro_df.merge(icustays[['stay_id', 'hadm_id', 'intime', 'outtime']], on='hadm_id')
    micro_merge = micro_merge[
        (micro_merge['charttime'] >= pd.to_datetime(micro_merge['intime']) - pd.Timedelta(hours=24)) & 
        (micro_merge['charttime'] <= pd.to_datetime(micro_merge['outtime']) + pd.Timedelta(hours=24))
    ]
    first_micro = micro_merge.groupby('stay_id')['charttime'].min().reset_index().rename(columns={'charttime': 'culture_time'})
    
    infection_df = icustays[['subject_id', 'hadm_id', 'stay_id']].copy()
    infection_df = infection_df.merge(first_abx, on='stay_id', how='left')
    infection_df = infection_df.merge(first_micro, on='stay_id', how='left')
    
    time_diff = (infection_df['culture_time'] - infection_df['antibiotic_time']).dt.total_seconds() / 3600.0
    is_suspected = (time_diff >= -72.0) & (time_diff <= 24.0)
    
    infection_df['suspected_infection_time'] = pd.NaT
    infection_df.loc[is_suspected, 'suspected_infection_time'] = infection_df.loc[is_suspected, ['antibiotic_time', 'culture_time']].min(axis=1)
    
    out_path = os.path.join(DERIVED_PATH, 'suspected_infection.csv')
    infection_df.to_csv(out_path, index=False)
    print(f"Step: suspected_infection - Saved suspected_infection table with {len(infection_df)} rows")
    return infection_df

def compute_sofa_scores(icustays):
    """Compute SOFA scores for each ICU stay within first 24 hours."""
    print("Step: sofa_scores - Beginning calculations")
    stay_ids = set(icustays['stay_id'].unique())
    stay_hadm_ids = set(icustays['hadm_id'].dropna().unique())
    
    print("Step: sofa_scores - Pre-filtering chartevents to temporary file")
    temp_chart_path = os.path.join(INTERMEDIATE_PATH, 'chartevents_sofa_temp.csv')
    chart_path = os.path.join(MIMIC_PATH, 'icu', 'chartevents.csv.gz')
    sofa_itemids = [223835, 220052, 220181, 225312, 220739, 223900, 223901]
    
    chart_chunks = []
    for chunk in pd.read_csv(chart_path, chunksize=1000000, compression='gzip', low_memory=False):
        chunk = chunk[chunk['itemid'].isin(sofa_itemids) & chunk['stay_id'].isin(stay_ids)]
        if not chunk.empty:
            chart_chunks.append(chunk[['stay_id', 'charttime', 'itemid', 'value', 'valuenum']])
            
    if not chart_chunks:
        temp_chart_df = pd.DataFrame(columns=['stay_id', 'charttime', 'itemid', 'value', 'valuenum'])
    else:
        temp_chart_df = pd.concat(chart_chunks, ignore_index=True)
        
    temp_chart_df.to_csv(temp_chart_path, index=False)
    print(f"Step: sofa_scores - Saved pre-filtered chartevents temp file with {len(temp_chart_df)} rows")
    
    # Load the temporary filtered chartevents once
    temp_chart = pd.read_csv(temp_chart_path, low_memory=False)
    
    print("Step: sofa_scores - Pre-filtering labevents to temporary file")
    temp_lab_path = os.path.join(INTERMEDIATE_PATH, 'labevents_sofa_temp.csv')
    lab_path = os.path.join(MIMIC_PATH, 'hosp', 'labevents.csv.gz')
    sofa_lab_itemids = [50821, 51265, 50885, 50912]
    
    lab_chunks = []
    for chunk in pd.read_csv(lab_path, chunksize=1000000, compression='gzip', low_memory=False):
        chunk = chunk[chunk['itemid'].isin(sofa_lab_itemids) & chunk['hadm_id'].isin(stay_hadm_ids)]
        if not chunk.empty:
            lab_chunks.append(chunk[['hadm_id', 'charttime', 'itemid', 'valuenum']])
            
    if not lab_chunks:
        temp_lab_df = pd.DataFrame(columns=['hadm_id', 'charttime', 'itemid', 'valuenum'])
    else:
        temp_lab_df = pd.concat(lab_chunks, ignore_index=True)
        
    temp_lab_df.to_csv(temp_lab_path, index=False)
    print(f"Step: sofa_scores - Saved pre-filtered labevents temp file with {len(temp_lab_df)} rows")
    
    # Load the temporary filtered labevents once
    temp_lab = pd.read_csv(temp_lab_path, low_memory=False)
    
    print("Step: sofa_scores - Extracting respiration parameters")
    po2_df = temp_lab[temp_lab['itemid'] == 50821][['hadm_id', 'charttime', 'valuenum']].copy()
    
    po2_df = po2_df.merge(icustays[['stay_id', 'hadm_id', 'intime']], on='hadm_id')
    po2_df['charttime'] = pd.to_datetime(po2_df['charttime'])
    po2_df['intime'] = pd.to_datetime(po2_df['intime'])
    po2_df = po2_df[(po2_df['charttime'] >= po2_df['intime']) & (po2_df['charttime'] <= po2_df['intime'] + pd.Timedelta(hours=24))]
    po2_min = po2_df.groupby('stay_id')['valuenum'].min().reset_index().rename(columns={'valuenum': 'po2_min'})
    
    # Read FiO2 from temp_chart
    fio2_df = temp_chart[temp_chart['itemid'] == 223835][['stay_id', 'charttime', 'valuenum']].copy()
    
    fio2_df = fio2_df.merge(icustays[['stay_id', 'intime']], on='stay_id')
    fio2_df['charttime'] = pd.to_datetime(fio2_df['charttime'])
    fio2_df['intime'] = pd.to_datetime(fio2_df['intime'])
    fio2_df = fio2_df[(fio2_df['charttime'] >= fio2_df['intime']) & (fio2_df['charttime'] <= fio2_df['intime'] + pd.Timedelta(hours=24))]
    fio2_max = fio2_df.groupby('stay_id')['valuenum'].max().reset_index().rename(columns={'valuenum': 'fio2_max'})
    
    resp_df = icustays[['stay_id']].merge(po2_min, on='stay_id', how='left').merge(fio2_max, on='stay_id', how='left')
    resp_df['fio2_fraction'] = resp_df['fio2_max'].apply(lambda x: x / 100.0 if x > 1.0 else x)
    resp_df['fio2_fraction'] = resp_df['fio2_fraction'].fillna(0.21)
    resp_df['pao2_fio2'] = resp_df['po2_min'] / resp_df['fio2_fraction']
    
    resp_score = np.zeros(len(resp_df))
    resp_score[resp_df['pao2_fio2'] < 400] = 1
    resp_score[resp_df['pao2_fio2'] < 300] = 2
    resp_score[resp_df['pao2_fio2'] < 200] = 3
    resp_score[resp_df['pao2_fio2'] < 100] = 4
    resp_score[resp_df['po2_min'].isna()] = 0
    resp_df['sofa_respiration'] = resp_score.astype(int)
    
    print("Step: sofa_scores - Extracting platelets (coagulation)")
    plt_df = temp_lab[temp_lab['itemid'] == 51265][['hadm_id', 'charttime', 'valuenum']].copy()
    
    plt_df = plt_df.merge(icustays[['stay_id', 'hadm_id', 'intime']], on='hadm_id')
    plt_df['charttime'] = pd.to_datetime(plt_df['charttime'])
    plt_df['intime'] = pd.to_datetime(plt_df['intime'])
    plt_df = plt_df[(plt_df['charttime'] >= plt_df['intime']) & (plt_df['charttime'] <= plt_df['intime'] + pd.Timedelta(hours=24))]
    plt_min = plt_df.groupby('stay_id')['valuenum'].min().reset_index().rename(columns={'valuenum': 'platelets_min'})
    
    coag_df = icustays[['stay_id']].merge(plt_min, on='stay_id', how='left')
    coag_score = np.zeros(len(coag_df))
    coag_score[coag_df['platelets_min'] < 150] = 1
    coag_score[coag_df['platelets_min'] < 100] = 2
    coag_score[coag_df['platelets_min'] < 50] = 3
    coag_score[coag_df['platelets_min'] < 20] = 4
    coag_score[coag_df['platelets_min'].isna()] = 0
    coag_df['sofa_coagulation'] = coag_score.astype(int)
    
    print("Step: sofa_scores - Extracting bilirubin (liver)")
    bili_df = temp_lab[temp_lab['itemid'] == 50885][['hadm_id', 'charttime', 'valuenum']].copy()
    
    bili_df = bili_df.merge(icustays[['stay_id', 'hadm_id', 'intime']], on='hadm_id')
    bili_df['charttime'] = pd.to_datetime(bili_df['charttime'])
    bili_df['intime'] = pd.to_datetime(bili_df['intime'])
    bili_df = bili_df[(bili_df['charttime'] >= bili_df['intime']) & (bili_df['charttime'] <= bili_df['intime'] + pd.Timedelta(hours=24))]
    bili_max = bili_df.groupby('stay_id')['valuenum'].max().reset_index().rename(columns={'valuenum': 'bilirubin_max'})
    
    liver_df = icustays[['stay_id']].merge(bili_max, on='stay_id', how='left')
    liver_score = np.zeros(len(liver_df))
    liver_score[liver_df['bilirubin_max'] >= 1.2] = 1
    liver_score[liver_df['bilirubin_max'] >= 2.0] = 2
    liver_score[liver_df['bilirubin_max'] >= 6.0] = 3
    liver_score[liver_df['bilirubin_max'] >= 12.0] = 4
    liver_score[liver_df['bilirubin_max'].isna()] = 0
    liver_df['sofa_liver'] = liver_score.astype(int)
    
    print("Step: sofa_scores - Extracting MAP and vasopressors (cardiovascular)")
    map_itemids = [220052, 220181, 225312]
    map_df = temp_chart[temp_chart['itemid'].isin(map_itemids)][['stay_id', 'charttime', 'valuenum']].copy()
    
    map_df = map_df.merge(icustays[['stay_id', 'intime']], on='stay_id')
    map_df['charttime'] = pd.to_datetime(map_df['charttime'])
    map_df['intime'] = pd.to_datetime(map_df['intime'])
    map_df = map_df[(map_df['charttime'] >= map_df['intime']) & (map_df['charttime'] <= map_df['intime'] + pd.Timedelta(hours=24))]
    map_min = map_df.groupby('stay_id')['valuenum'].min().reset_index().rename(columns={'valuenum': 'map_min'})
    
    input_path = os.path.join(MIMIC_PATH, 'icu', 'inputevents.csv.gz')
    vaso_itemids = [221906, 221289, 221662, 221653]
    vaso_chunks = []
    for chunk in pd.read_csv(input_path, chunksize=1000000, compression='gzip', low_memory=False):
        chunk = chunk[(chunk['itemid'].isin(vaso_itemids)) & (chunk['stay_id'].isin(stay_ids))]
        if not chunk.empty:
            vaso_chunks.append(chunk[['stay_id', 'starttime', 'endtime', 'itemid', 'rate']])
    vaso_df = pd.concat(vaso_chunks, ignore_index=True) if vaso_chunks else pd.DataFrame(columns=['stay_id', 'starttime', 'endtime', 'itemid', 'rate'])
    
    vaso_df = vaso_df.merge(icustays[['stay_id', 'intime']], on='stay_id')
    vaso_df['starttime'] = pd.to_datetime(vaso_df['starttime'])
    vaso_df['endtime'] = pd.to_datetime(vaso_df['endtime'])
    vaso_df['intime'] = pd.to_datetime(vaso_df['intime'])
    
    vaso_df = vaso_df[
        (vaso_df['starttime'] <= vaso_df['intime'] + pd.Timedelta(hours=24)) & 
        (vaso_df['endtime'] >= vaso_df['intime'])
    ]
    
    vaso_max = vaso_df.groupby(['stay_id', 'itemid'])['rate'].max().unstack(fill_value=np.nan)
    cardio_df = icustays[['stay_id']].merge(map_min, on='stay_id', how='left')
    cardio_df = cardio_df.merge(vaso_max, on='stay_id', how='left')
    
    cardio_score = np.zeros(len(cardio_df))
    cardio_score[cardio_df['map_min'] < 70] = 1
    
    # Audit Note: Rates for dobutamine (221653), dopamine (221906), epinephrine (221289),
    # and norepinephrine (221662) in MIMIC-IV inputevents are assumed to be in mcg/kg/min.
    # The scoring thresholds below correspond directly to these clinical units (e.g., dopamine <=5 vs >5 vs >15 mcg/kg/min;
    # epi/norepi <=0.1 vs >0.1 mcg/kg/min).
    if 221653 in cardio_df.columns:
        cardio_score = np.maximum(cardio_score, np.where(cardio_df[221653].notna(), 2, 0))
    if 221906 in cardio_df.columns:
        cardio_score = np.maximum(cardio_score, np.where(cardio_df[221906] > 15, 4,
                                               np.where(cardio_df[221906] > 5, 3,
                                               np.where(cardio_df[221906] > 0, 2, 0))))
    if 221289 in cardio_df.columns:
        cardio_score = np.maximum(cardio_score, np.where(cardio_df[221289] > 0.1, 4,
                                               np.where(cardio_df[221289] > 0, 3, 0)))
    if 221662 in cardio_df.columns:
        cardio_score = np.maximum(cardio_score, np.where(cardio_df[221662] > 0.1, 4,
                                               np.where(cardio_df[221662] > 0, 3, 0)))
        
    cardio_df['sofa_cardiovascular'] = cardio_score.astype(int)
    
    print("Step: sofa_scores - Extracting GCS components (CNS)")
    gcs_itemids = [220739, 223900, 223901]
    gcs_raw = temp_chart[temp_chart['itemid'].isin(gcs_itemids)][['stay_id', 'charttime', 'itemid', 'value']].copy()
    
    gcs_raw = gcs_raw.merge(icustays[['stay_id', 'intime']], on='stay_id')
    gcs_raw['charttime'] = pd.to_datetime(gcs_raw['charttime'])
    gcs_raw['intime'] = pd.to_datetime(gcs_raw['intime'])
    gcs_raw = gcs_raw[(gcs_raw['charttime'] >= gcs_raw['intime']) & (gcs_raw['charttime'] <= gcs_raw['intime'] + pd.Timedelta(hours=24))]
    
    def extract_first_digit(series):
        num = pd.to_numeric(series, errors='coerce')
        txt = series.astype(str).str.extract(r'^(\d)').astype(float)[0]
        return num.fillna(txt)
        
    gcs_raw['val_num'] = extract_first_digit(gcs_raw['value'])
    gcs_min_comp = gcs_raw.groupby(['stay_id', 'itemid'])['val_num'].min().unstack(fill_value=np.nan)
    
    cns_df = icustays[['stay_id']].copy()
    cns_df = cns_df.merge(gcs_min_comp, on='stay_id', how='left')
    for itemid in [220739, 223900, 223901]:
        if itemid not in cns_df.columns:
            cns_df[itemid] = np.nan
    cns_df['gcs_total'] = cns_df[[220739, 223900, 223901]].sum(axis=1, min_count=3)
    
    cns_score = np.zeros(len(cns_df))
    has_gcs = cns_df[[220739, 223900, 223901]].notna().all(axis=1)
    cns_score[has_gcs & (cns_df['gcs_total'] < 15)] = 1
    cns_score[has_gcs & (cns_df['gcs_total'] < 13)] = 2
    cns_score[has_gcs & (cns_df['gcs_total'] < 10)] = 3
    cns_score[has_gcs & (cns_df['gcs_total'] < 6)] = 4
    cns_df['sofa_cns'] = cns_score.astype(int)
    
    print("Step: sofa_scores - Extracting creatinine and urine output (renal)")
    cr_df = temp_lab[temp_lab['itemid'] == 50912][['hadm_id', 'charttime', 'valuenum']].copy()
    
    cr_df = cr_df.merge(icustays[['stay_id', 'hadm_id', 'intime']], on='hadm_id')
    cr_df['charttime'] = pd.to_datetime(cr_df['charttime'])
    cr_df['intime'] = pd.to_datetime(cr_df['intime'])
    cr_df = cr_df[(cr_df['charttime'] >= cr_df['intime']) & (cr_df['charttime'] <= cr_df['intime'] + pd.Timedelta(hours=24))]
    cr_max = cr_df.groupby('stay_id')['valuenum'].max().reset_index().rename(columns={'valuenum': 'creatinine_max'})
    
    output_path = os.path.join(MIMIC_PATH, 'icu', 'outputevents.csv.gz')
    uo_chunks = []
    for chunk in pd.read_csv(output_path, chunksize=1000000, compression='gzip', low_memory=False):
        chunk = chunk[chunk['stay_id'].isin(stay_ids)]
        if not chunk.empty:
            uo_chunks.append(chunk[['stay_id', 'charttime', 'value']])
    uo_df = pd.concat(uo_chunks, ignore_index=True) if uo_chunks else pd.DataFrame(columns=['stay_id', 'charttime', 'value'])
    
    uo_df = uo_df.merge(icustays[['stay_id', 'intime']], on='stay_id')
    uo_df['charttime'] = pd.to_datetime(uo_df['charttime'])
    uo_df['intime'] = pd.to_datetime(uo_df['intime'])
    uo_df = uo_df[(uo_df['charttime'] >= uo_df['intime']) & (uo_df['charttime'] <= uo_df['intime'] + pd.Timedelta(hours=24))]
    uo_total = uo_df.groupby('stay_id')['value'].sum().reset_index().rename(columns={'value': 'urine_total'})
    
    renal_df = icustays[['stay_id']].merge(cr_max, on='stay_id', how='left').merge(uo_total, on='stay_id', how='left')
    cr_score = np.zeros(len(renal_df))
    cr_score[renal_df['creatinine_max'] >= 1.2] = 1
    cr_score[renal_df['creatinine_max'] >= 2.0] = 2
    cr_score[renal_df['creatinine_max'] >= 3.5] = 3
    cr_score[renal_df['creatinine_max'] >= 5.0] = 4
    
    uo_score = np.zeros(len(renal_df))
    uo_score[renal_df['urine_total'] < 500] = 3
    uo_score[renal_df['urine_total'] < 200] = 4
    
    renal_df['sofa_renal'] = np.maximum(cr_score, uo_score).astype(int)
    
    sofa_df = icustays[['stay_id', 'hadm_id']].copy()
    sofa_df = sofa_df.merge(resp_df[['stay_id', 'sofa_respiration']], on='stay_id', how='left')
    sofa_df = sofa_df.merge(coag_df[['stay_id', 'sofa_coagulation']], on='stay_id', how='left')
    sofa_df = sofa_df.merge(liver_df[['stay_id', 'sofa_liver']], on='stay_id', how='left')
    sofa_df = sofa_df.merge(cardio_df[['stay_id', 'sofa_cardiovascular']], on='stay_id', how='left')
    sofa_df = sofa_df.merge(cns_df[['stay_id', 'sofa_cns']], on='stay_id', how='left')
    sofa_df = sofa_df.merge(renal_df[['stay_id', 'sofa_renal']], on='stay_id', how='left')
    
    for c in ['sofa_respiration', 'sofa_coagulation', 'sofa_liver', 'sofa_cardiovascular', 'sofa_cns', 'sofa_renal']:
        sofa_df[c] = sofa_df[c].fillna(0).astype(int)
        
    sofa_df['sofa_total'] = (
        sofa_df['sofa_respiration'] + 
        sofa_df['sofa_coagulation'] + 
        sofa_df['sofa_liver'] + 
        sofa_df['sofa_cardiovascular'] + 
        sofa_df['sofa_cns'] + 
        sofa_df['sofa_renal']
    )
    
    out_path = os.path.join(DERIVED_PATH, 'sofa_scores.csv')
    sofa_df.to_csv(out_path, index=False)
    print(f"Step: sofa_scores - Saved sofa_scores table with {len(sofa_df)} rows")
    if os.path.exists(temp_chart_path):
        os.remove(temp_chart_path)
    if os.path.exists(temp_lab_path):
        os.remove(temp_lab_path)
    return sofa_df

def generate_sepsis3_table(infection_df, sofa_df):
    """Generate sepsis3 flag table joining suspected infection and sofa scores."""
    print("Step: sepsis3 - Merging infection and sofa scores")
    sepsis3_df = infection_df[['subject_id', 'hadm_id', 'stay_id', 'suspected_infection_time']].copy()
    sepsis3_df = sepsis3_df.merge(sofa_df[['stay_id', 'sofa_total']], on='stay_id', how='inner')
    
    has_infection = sepsis3_df['suspected_infection_time'].notna()
    has_sofa = sepsis3_df['sofa_total'] >= 2
    sepsis3_df['sepsis3_flag'] = (has_infection & has_sofa).astype(int)
    
    out_path = os.path.join(DERIVED_PATH, 'sepsis3.csv')
    sepsis3_df.to_csv(out_path, index=False)
    print(f"Step: sepsis3 - Saved sepsis3 table with {len(sepsis3_df)} rows. Identified Sepsis-3 count: {sepsis3_df['sepsis3_flag'].sum()}")
    return sepsis3_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate derived MIMIC-IV tables.")
    parser.add_argument("--mimic-dir", type=str, required=True, help="Path to raw MIMIC-IV folder")
    args = parser.parse_args()
    MIMIC_PATH = args.mimic_dir

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(DERIVED_PATH, exist_ok=True)
    os.makedirs(INTERMEDIATE_PATH, exist_ok=True)
    os.makedirs(FINAL_PATH, exist_ok=True)
    
    print("Beginning generation of derived tables")
    
    icustays_path = os.path.join(MIMIC_PATH, 'icu', 'icustays.csv.gz')
    icustays_df = pd.read_csv(icustays_path, compression='gzip', low_memory=False)
    print(f"Loaded icustays with {len(icustays_df)} rows")
    
    inf_df = generate_suspected_infection(icustays_df)
    s_df = compute_sofa_scores(icustays_df)
    generate_sepsis3_table(inf_df, s_df)
    
    print("Completed generation of derived tables")
