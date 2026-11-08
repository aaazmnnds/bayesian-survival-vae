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
    data_dir = os.path.join(PROJECT_ROOT, "Results", "metabric", "final")
    raw_path = os.path.join(data_dir, 'metabric_cohort.csv')
    df_raw = pd.read_csv(raw_path)
    id_cols = ['patientId', 'sampleId']
    targets = ['event', 'survival_months']
    
    out_dir = os.path.join(PROJECT_ROOT, "Scripts", "models", "datasets_real")
    os.makedirs(out_dir, exist_ok=True)
    
    splits_path = os.path.join(PROJECT_ROOT, "Results", "final", "cv_splits", "cv_splits_metabric.json")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Missing splits file: {splits_path}")
    with open(splits_path, 'r') as f:
        cv_splits = json.load(f)
    val_idx = cv_splits["fold_1"]["val"]
    
    feature_cols, _, feature_df = get_feature_columns(df_raw, 'metabric')
    continuous_cols = []
    for col in feature_cols:
        unique_vals = feature_df[col].dropna().unique()
        if set(unique_vals).issubset({0.0, 1.0}):
            continue
        elif np.all(np.mod(unique_vals, 1) == 0) and len(unique_vals) > 2 and len(unique_vals) < 10:
            continue
        else:
            continuous_cols.append(col)
            
    print(f"Continuous cols for masking: {continuous_cols}")
    
    for severity in ['light', 'moderate', 'severe']:
        X_masked, M_eval = apply_threshold_interaction_mnar(
            df_raw, 
            severity=severity, 
            age_col='age', 
            lactate_col='npi',
            eligible_cols=continuous_cols
        )
        
        out_path = os.path.join(out_dir, f"metabric_mnar_{severity}.csv")
        X_masked.to_csv(out_path, index=False)
        
        mask_out_path = os.path.join(out_dir, f"metabric_mask_{severity}.csv")
        M_eval.to_csv(mask_out_path, index=False)
        
        total_masked = M_eval.sum().sum()
        col_masked = M_eval.sum()[M_eval.sum() > 0]
        val_masked = M_eval.iloc[val_idx].sum().sum()
        
        print(f"--- Severity: {severity} ---")
        print(f"Total masked cells: {total_masked}")
        print(f"Masked cells in fold 0 val_idx (N={len(val_idx)}): {val_masked}")
        print("Per-column breakdown:")
        print(col_masked.to_string())
        print(f"Diff count vs raw: {df_raw.isna().sum().sum() - X_masked.isna().sum().sum()}")
        print()
        
        pd.testing.assert_frame_equal(df_raw[id_cols], X_masked[id_cols], check_dtype=False)
        pd.testing.assert_frame_equal(df_raw[targets], X_masked[targets], check_dtype=False)
        # check non-masked cells are identical
        non_masked_mask = (M_eval == 0)
        pd.testing.assert_frame_equal(df_raw[non_masked_mask], X_masked[non_masked_mask], check_dtype=False)

if __name__ == "__main__":
    main()
