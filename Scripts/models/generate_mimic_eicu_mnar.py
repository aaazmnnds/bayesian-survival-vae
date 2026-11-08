import os
import sys
import pandas as pd
import numpy as np
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mnar_injection import apply_threshold_interaction_mnar
from dataset_schema import get_feature_columns

def main():
    import os
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(PROJECT_ROOT, "Scripts", "models", "datasets_real")
    os.makedirs(out_dir, exist_ok=True)
    
    for dataset in ['mimic', 'eicu']:
        print(f"Processing {dataset}...")
        if dataset == 'mimic':
            data_dir = os.path.join(PROJECT_ROOT, "Results", "final")
            raw_path = os.path.join(data_dir, 'mimic_sepsis_cohort.csv')
            id_cols = ['subject_id', 'hadm_id', 'stay_id']
            targets = ['event', 'time_days']
        else:
            data_dir = os.path.join(PROJECT_ROOT, "Results", "eicu", "final")
            raw_path = os.path.join(data_dir, 'eicu_sepsis_cohort.csv')
            id_cols = ['uniquepid', 'patientunitstayid']
            targets = ['event', 'time_days']
            
        df_raw = pd.read_csv(raw_path)
        feature_cols, _, feature_df = get_feature_columns(df_raw, dataset)
        
        continuous_cols = []
        for col in feature_cols:
            unique_vals = feature_df[col].dropna().unique()
            if set(unique_vals).issubset({0.0, 1.0}):
                continue
            elif np.all(np.mod(unique_vals, 1) == 0) and len(unique_vals) > 2 and len(unique_vals) < 10:
                continue
            else:
                continuous_cols.append(col)
        
        print(f"Continuous cols: {len(continuous_cols)}")
        
        for severity in ['light', 'moderate', 'severe']:
            X_masked, M_eval = apply_threshold_interaction_mnar(
                df_raw, 
                severity=severity, 
                age_col='age', 
                lactate_col='Lactate',
                eligible_cols=continuous_cols
            )
            
            out_path = os.path.join(out_dir, f"{dataset}_mnar_{severity}.csv")
            X_masked.to_csv(out_path, index=False)
            
            mask_out_path = os.path.join(out_dir, f"{dataset}_mask_{severity}.csv")
            M_eval.to_csv(mask_out_path, index=False)
            
            total_masked = M_eval.sum().sum()
            print(f"  {severity}: {total_masked} masked cells (diff: {df_raw.isna().sum().sum() - X_masked.isna().sum().sum()})")
            
            pd.testing.assert_frame_equal(df_raw[id_cols], X_masked[id_cols], check_dtype=False)
            pd.testing.assert_frame_equal(df_raw[targets], X_masked[targets], check_dtype=False)
            # check non-masked cells are identical
            non_masked_mask = (M_eval == 0)
            pd.testing.assert_frame_equal(df_raw[non_masked_mask], X_masked[non_masked_mask], check_dtype=False)

if __name__ == "__main__":
    main()
