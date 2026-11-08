import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

def generate_splits():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir = os.path.join(PROJECT_ROOT, "Results")
    
    datasets = {
        'metabric': os.path.join(base_dir, 'metabric/final/metabric_cohort.csv'),
        'mimic': os.path.join(base_dir, 'final/mimic_sepsis_cohort.csv'),
        'eicu': os.path.join(base_dir, 'eicu/final/eicu_sepsis_cohort.csv'),
        'synthetic': os.path.join(base_dir, 'synthetic/synthetic_cohort_observed.csv'),
    }
    
    id_map = {
        'metabric': 'patientId',
        'mimic': 'subject_id',
        'eicu': 'uniquepid',
        'synthetic': 'patient_id',
    }
    
    out_dir = os.path.join(base_dir, 'final', 'cv_splits')
    os.makedirs(out_dir, exist_ok=True)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    for name, path in datasets.items():
        print(f"--- Generating splits for {name} ---")
        df = pd.read_csv(path)
        pid = id_map[name]
        assert df[pid].nunique() == len(df), f"Expected one row per {pid}"
        
        event_col = 'event'
        events = df[event_col].values
        
        splits = {}
        all_test_idx = []
        for fold, (temp_idx, test_idx) in enumerate(skf.split(df, events)):
            # Now temp_idx is 80%, test_idx is 20%.
            # We split temp_idx into 80% train, 20% val of temp_idx (overall 64%, 16%)
            temp_events = events[temp_idx]
            train_sub_idx, val_sub_idx = next(sss.split(temp_idx, temp_events))
            
            train_idx = temp_idx[train_sub_idx]
            val_idx = temp_idx[val_sub_idx]
            
            # verify disjointness
            assert len(set(train_idx).intersection(set(val_idx))) == 0
            assert len(set(train_idx).intersection(set(test_idx))) == 0
            assert len(set(val_idx).intersection(set(test_idx))) == 0
            assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)
            
            all_test_idx.extend(test_idx)
            
            splits[f"fold_{fold+1}"] = {
                "train": [int(i) for i in train_idx],
                "val": [int(i) for i in val_idx],
                "test": [int(i) for i in test_idx],
                "train_ids": [int(x) if isinstance(x, np.integer) else x for x in df.iloc[train_idx][pid]],
                "val_ids": [int(x) if isinstance(x, np.integer) else x for x in df.iloc[val_idx][pid]],
                "test_ids": [int(x) if isinstance(x, np.integer) else x for x in df.iloc[test_idx][pid]]
            }
            
            train_ev = events[train_idx].mean()
            val_ev = events[val_idx].mean()
            test_ev = events[test_idx].mean()
            
            print(f"Fold {fold+1}: Train N={len(train_idx)} ({train_ev:.3f}), "
                  f"Val N={len(val_idx)} ({val_ev:.3f}), "
                  f"Test N={len(test_idx)} ({test_ev:.3f})")
            
        assert len(set(all_test_idx)) == len(df), "The five test sets do not partition all rows"
            
        out_path = os.path.join(out_dir, f"cv_splits_{name}.json")
        with open(out_path, 'w') as f:
            json.dump(splits, f, indent=2)
            
        print(f"Saved to {out_path}\n")

if __name__ == "__main__":
    generate_splits()
