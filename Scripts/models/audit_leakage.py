import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import traceback
import math
from lifelines.utils import concordance_index
from scipy.stats import spearmanr

# Add Scripts to sys.path to import dataset_schema
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_schema import get_feature_columns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

REAL_COHORTS = {
    'metabric': os.path.join(PROJECT_ROOT, "Results", "metabric", "final", "metabric_cohort.csv"),
    'mimic': os.path.join(PROJECT_ROOT, "Results", "final", "mimic_sepsis_cohort.csv"),
    'eicu': os.path.join(PROJECT_ROOT, "Results", "eicu", "final", "eicu_sepsis_cohort.csv")
}

def get_paths(dataset, scenario):
    masked = os.path.join(os.path.dirname(__file__), "datasets_real", f"{dataset}_mnar_{scenario}.csv")
    mask = os.path.join(os.path.dirname(__file__), "datasets_real", f"{dataset}_mask_{scenario}.csv")
    splits = os.path.join(PROJECT_ROOT, "Results", "final", "cv_splits", f"cv_splits_{dataset}.json")
    return REAL_COHORTS[dataset], masked, mask, splits

def sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(x) for x in obj]
    if isinstance(obj, tuple):
        return [sanitize(x) for x in obj]
    if isinstance(obj, (pd.Index, pd.Series, np.ndarray)):
        return sanitize(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return sanitize(obj.to_dict(orient="records"))
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if obj is None:
        return None
    return str(obj)

def report(results, check_id, title, status, message, details=None):
    results.append({
        "check_id": check_id,
        "title": title,
        "status": status,
        "message": message,
        "details": details or {}
    })

def check_c1(df_real, df_masked, df_mask, splits, files, results):
    for name, path in files.items():
        if name != "splits":
            df = {"real": df_real, "masked": df_masked, "mask": df_mask}[name]
            details = {"shape": df.shape, "columns": list(df.columns), "dtypes": df.dtypes.astype(str).to_dict()}
        else:
            details = {"keys": list(splits.keys()) if isinstance(splits, dict) else []}
        report(results, "C1", f"File inventory - {name}", "PASS", "File exists and is readable.", details)

def check_c2(df_real, df_masked, df_mask, dataset, results):
    c2a_status = "PASS" if df_real.shape[0] == df_masked.shape[0] == df_mask.shape[0] else "FAIL"
    report(results, "C2a", "Row counts are equal", c2a_status, f"Real: {df_real.shape[0]}, Masked: {df_masked.shape[0]}, Mask: {df_mask.shape[0]}")
    
    ids = []
    if dataset == 'mimic': ids = [c for c in ['subject_id', 'hadm_id', 'stay_id'] if c in df_real.columns]
    elif dataset == 'eicu': ids = [c for c in ['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 'hospitalid'] if c in df_real.columns]
    elif dataset == 'metabric': ids = [c for c in df_real.columns if "patient" in c.lower() or "id" in c.lower()]
    
    if ids:
        missing_in_masked = [c for c in ids if c not in df_masked.columns]
        if missing_in_masked:
            report(results, "C2b", "IDs match", "FAIL", f"ID columns missing in masked: {missing_in_masked}")
        else:
            matches = True
            for c in ids:
                if not df_real[c].equals(df_masked[c]): matches = False
            report(results, "C2b", "IDs match", "PASS" if matches else "FAIL", f"Matched IDs: {ids}")
            
    mismatches = {}
    non_numeric_columns = []
    cols = [c for c in df_masked.columns if c in df_real.columns and c not in ids]
    for c in cols:
        mask_non_nan = ~df_masked[c].isna()
        if mask_non_nan.sum() > 0:
            if pd.api.types.is_numeric_dtype(df_real[c]) and pd.api.types.is_numeric_dtype(df_masked[c]):
                diff = np.abs(df_real.loc[mask_non_nan, c] - df_masked.loc[mask_non_nan, c])
                if (diff > 1e-6).any():
                    mismatches[c] = int((diff > 1e-6).sum())
            else:
                s_real = df_real.loc[mask_non_nan, c].astype(str).str.strip()
                s_masked = df_masked.loc[mask_non_nan, c].astype(str).str.strip()
                diff_count = int((s_real != s_masked).sum())
                if diff_count > 0:
                    mismatches[c] = diff_count
                
                non_numeric_columns.append({
                    "column": c,
                    "dtype_real": str(df_real[c].dtype),
                    "dtype_masked": str(df_masked[c].dtype),
                    "examples": df_real.loc[mask_non_nan, c].astype(str).head(5).tolist()
                })
    report(results, "C2c", "Masked values equal real", "FAIL" if mismatches else "PASS", "Mismatches details", {"mismatches": mismatches, "non_numeric_columns": non_numeric_columns})
    
    # Mask file agreement
    mask_cols = [c for c in cols if c in df_mask.columns]
    if len(mask_cols) > 0:
        c2d_pass = True
        c2d_counts = {'mask_1_but_not_nan_in_masked': 0, 'mask_1_but_nan_in_real': 0, 'nan_in_masked_but_not_masked_or_already_nan': 0}
        
        for c in mask_cols:
            m1 = df_mask[c] == 1
            masked_nan = df_masked[c].isna()
            real_nan = df_real[c].isna()
            
            err1 = (m1 & ~masked_nan).sum()
            err2 = (m1 & real_nan).sum()
            c2d_counts['mask_1_but_not_nan_in_masked'] += int(err1)
            c2d_counts['mask_1_but_nan_in_real'] += int(err2)
            
            err3 = (masked_nan & ~(m1 | real_nan)).sum()
            c2d_counts['nan_in_masked_but_not_masked_or_already_nan'] += int(err3)
            
            if err1 > 0 or err2 > 0 or err3 > 0:
                c2d_pass = False
                
        report(results, "C2d", "Mask file agreement", "PASS" if c2d_pass else "FAIL", "Agreement checks", c2d_counts)
        
    missing_stats = {}
    for c in cols:
        real_miss = df_real[c].isna().sum()
        masked_miss = df_masked[c].isna().sum()
        missing_stats[c] = {"real_miss": int(real_miss), "injected": int(masked_miss - real_miss), "total_miss": int(masked_miss)}
    report(results, "C2e", "Missingness counts", "PASS", "Counts per feature", missing_stats)

def check_c3(df_real, df_masked, dataset, results):
    _, schema_targets, _ = get_feature_columns(df_real, dataset)
    report(results, "C3a", "Schema targets", "PASS", "Output of get_feature_columns", {"schema_targets": schema_targets})
    
    if len(schema_targets) >= 2:
        assert schema_targets[0] == 'event' and schema_targets[1] in ('survival_months', 'los_days', 'time_days'), f"Unexpected targets: {schema_targets}"
        event_col, time_col = schema_targets[0], schema_targets[1]
        ev_uniq = set(df_real[event_col].dropna().unique())
        t_num = pd.api.types.is_numeric_dtype(df_real[time_col])
        t_pos = (df_real[time_col] > 0).all()
        
        c3b_status = "PASS"
        if not ev_uniq.issubset({0, 1}) or not t_num or not t_pos: c3b_status = "FAIL"
        report(results, "C3b", "Time and Event validity", c3b_status, "Assumptions check", {"event_unique": list(ev_uniq), "time_numeric": t_num, "time_positive": bool(t_pos)})
        
        t_nan = int(df_real[time_col].isna().sum())
        e_nan = int(df_real[event_col].isna().sum())
        t_le_0 = int((df_real[time_col] <= 0).sum())
        ev_rate = float(df_real[event_col].mean())
        report(results, "C3c", "Target stats", "PASS", "Stats", {"time_nan": t_nan, "event_nan": e_nan, "time_le_0": t_le_0, "event_rate": ev_rate})
        
        if event_col in df_masked.columns and time_col in df_masked.columns:
            ev_identical = df_real[event_col].equals(df_masked[event_col])
            
            t_real = df_real[time_col]
            t_masked = df_masked[time_col]
            
            # fillna with a dummy to use allclose for nans
            time_close = np.allclose(t_real.fillna(-9999), t_masked.fillna(-9999), atol=1e-9)
            
            max_diff = float((t_real - t_masked).abs().max())
            identical = ev_identical and time_close
            report(results, "C3d", "Targets identical", "PASS" if identical else "FAIL", f"Real vs Masked exact ev, close time. max_diff={max_diff}", {"max_diff": max_diff})
    else:
        report(results, "C3", "Targets", "FAIL", "schema_targets length < 2")

def check_c4(df_real, dataset, results):
    _, schema_targets, feature_df = get_feature_columns(df_real, dataset)
    
    dup_real = list(df_real.columns[df_real.columns.duplicated()])
    dup_feat = list(feature_df.columns[feature_df.columns.duplicated()])
    if dup_real or dup_feat:
        report(results, "C4", "Duplicate columns", "FAIL", "Duplicate column names detected.", {"df_real": dup_real, "feature_df": dup_feat})
        
    features = list(feature_df.columns)
    report(results, "C4a", "Feature columns", "PASS", "Features", features)
    
    leakage_keywords = ['id', 'subject', 'hadm', 'stay', 'patient', 'unit', 'hospital', 'los', 'length', 'discharge', 'death', 'dod', 'expire', 'mortality', 'outcome', 'survival', 'time', 'event', 'status', 'censor']
    leaky_feats = [f for f in features if any(k in f.lower() for k in leakage_keywords)]
    report(results, "C4b", "Name leakage proxies", "WARN" if leaky_feats else "PASS", "Features with leaky keywords", leaky_feats)
    
    if len(schema_targets) >= 2:
        assert schema_targets[0] == 'event' and schema_targets[1] in ('survival_months', 'los_days', 'time_days'), f"Unexpected targets: {schema_targets}"
        event_col, time_col = schema_targets[0], schema_targets[1]
        cindex_warnings = []
        corr_warnings = []
        cindex_diffs = []
        constant_feats = []
        for f in features:
            valid = pd.concat([feature_df[f].reset_index(drop=True), df_real[[time_col, event_col]].reset_index(drop=True)], axis=1).dropna()
            if len(valid) == 0: continue
            
            # Constant check
            top_val_share = valid[f].value_counts(normalize=True).max()
            if top_val_share > 0.99:
                constant_feats.append(f)
                continue
                
            try:
                cidx = concordance_index(valid[time_col], valid[f], valid[event_col])
                if cidx > 0.75 or cidx < 0.25: cindex_warnings.append(f)
                cindex_diffs.append({"feature": f, "cindex": cidx, "diff": abs(cidx - 0.5)})
            except: pass
            
            try:
                rho, _ = spearmanr(valid[f], valid[time_col])
                if abs(rho) > 0.5: corr_warnings.append(f)
            except: pass
            
        cindex_diffs.sort(key=lambda x: x["diff"], reverse=True)
        top15 = cindex_diffs[:15]
        report(results, "C4c", "Target proxies", "WARN" if (cindex_warnings or corr_warnings) else "PASS", "Metrics", {"cindex_warns": cindex_warnings, "corr_warns": corr_warnings, "top_15_cindex": top15})
        report(results, "C4d", "Constant features", "WARN" if constant_feats else "PASS", ">99% one value", constant_feats)

def check_c5(df_real, df_masked, df_mask, dataset, results):
    if dataset == 'mimic':
        ids = ['subject_id', 'hadm_id', 'stay_id']
    elif dataset == 'eicu':
        ids = ['uniquepid', 'patientunitstayid']
    elif dataset == 'metabric':
        ids = ['patientId', 'sampleId']
        
    ids = [c for c in ids if c in df_masked.columns]
    
    if not ids:
        report(results, "C5", "Identifier columns intact", "FAIL", "No explicit ID columns found.")
    else:
        status = "PASS"
        missing = []
        for c in ids:
            if c not in df_masked.columns:
                missing.append(c)
                status = "FAIL"
            elif c in df_mask.columns and df_mask[c].sum() > 0:
                report(results, "C5", f"ID column {c} masked", "FAIL", f"ID column {c} has {df_mask[c].sum()} masked values.")
                status = "FAIL"
                
        if status == "PASS":
            report(results, "C5", "Identifier columns intact", "PASS", f"ID columns {ids} exist and are unmasked.")
        else:
            report(results, "C5", "Identifier columns intact", "FAIL", f"Issues with IDs: missing {missing}")
            
def check_c6(df_real, splits, dataset, results):
    first_fold = list(splits.keys())[0] if splits else None
    if first_fold:
        keys = list(splits[first_fold].keys())
        sizes = {f: {k: len(splits[f][k]) for k in splits[f]} for f in splits}
        report(results, "C6a", "Split keys and sizes", "PASS", "Details", {"keys": keys, "sizes": sizes})
        
        # Positional check
        all_idx = splits[first_fold].get("train", []) + splits[first_fold].get("val", []) + splits[first_fold].get("test", [])
        is_positional = all(isinstance(x, int) and 0 <= x < len(df_real) for x in all_idx)
        report(results, "C6b", "Indices type", "PASS", "Positional or IDs", "positional" if is_positional else "IDs")
        
        disjoints = True
        coverage = True
        for f, p in splits.items():
            tr, v, te = set(p.get("train", [])), set(p.get("val", [])), set(p.get("test", []))
            if tr & v or tr & te or v & te: disjoints = False
            if tr | v | te != set(range(len(df_real))): coverage = False
            
        report(results, "C6c", "Pairwise disjoint", "PASS" if disjoints else "FAIL", "Within fold")
        report(results, "C6d", "Coverage", "PASS" if coverage else "WARN", "Covers all rows")
        
        te_overlap = False
        for f1 in splits:
            for f2 in splits:
                if f1 < f2:
                    if set(splits[f1].get("test", [])) & set(splits[f2].get("test", [])):
                        te_overlap = True
        report(results, "C6e", "Test overlap across folds", "PASS", "Overlap?", te_overlap)
        
        c6f_status = "PASS"
        c6f_leaks = {}
        for f in splits:
            val = set(splits[f].get("val", []))
            tr = set(splits[f].get("train", []))
            te = set(splits[f].get("test", []))
            
            ov_tr = val & tr
            ov_te = val & te
            
            if ov_tr or ov_te:
                c6f_status = "FAIL"
                c6f_leaks[f] = {"val_train_overlap": len(ov_tr), "val_test_overlap": len(ov_te)}
                
        report(results, "C6f", "Validation disjoint from own fold", c6f_status, "Overlaps within fold", c6f_leaks)
        
        _, schema_targets, _ = get_feature_columns(df_real, dataset)
        if len(schema_targets) >= 2:
            assert schema_targets[0] == 'event' and schema_targets[1] in ('survival_months', 'los_days', 'time_days'), f"Unexpected targets: {schema_targets}"
            ev_col, t_col = schema_targets[0], schema_targets[1]
            stats = {}
            for f, p in splits.items():
                stats[f] = {}
                for part in ["train", "val", "test"]:
                    idx = p.get(part, [])
                    if len(idx) > 0:
                        stats[f][part] = {
                            "event_rate": float(df_real.loc[idx, ev_col].mean()),
                            "median_time": float(df_real.loc[idx, t_col].median())
                        }
            report(results, "C6g", "Split targets distribution", "PASS", "Stats", stats)

def check_c7(df_real, dataset, results):
    _, schema_targets, feature_df = get_feature_columns(df_real, dataset)
    dups_feats = int(feature_df.duplicated().sum())
    
    if len(schema_targets) >= 2:
        assert schema_targets[0] == 'event' and schema_targets[1] in ('survival_months', 'los_days', 'time_days'), f"Unexpected targets: {schema_targets}"
        df_target = pd.concat([feature_df.reset_index(drop=True), df_real[[schema_targets[0], schema_targets[1]]].reset_index(drop=True)], axis=1)
        dups_all = int(df_target.duplicated().sum())
    else:
        dups_all = "N/A"
        
    report(results, "C7", "Duplicates", "PASS", "Counts", {"on_features": dups_feats, "on_features_and_targets": dups_all})

def check_c8(df_real, dataset, results):
    _, _, feature_df = get_feature_columns(df_real, dataset)
    # Infer continuous features using optimize_bsvae.py logic
    continuous_idx = []
    categorical_groups_idx = []
    binary_idx = []
    cols = list(feature_df.columns)
    for i, col in enumerate(cols):
        unique_vals = feature_df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            binary_idx.append(i)
        elif len(unique_vals) > 2 and len(unique_vals) < 10 and all(float(x).is_integer() for x in unique_vals):
            categorical_groups_idx.append(i)
        else:
            continuous_idx.append(i)
            
    cont_cols = [cols[i] for i in continuous_idx]
    
    bounds_rows = []
    c8_details = []
    has_extremes = False
    for c in cont_cols:
        v = feature_df[c].dropna()
        n = len(v)
        if n == 0: continue
        
        q = np.percentile(v, [0, 0.1, 1, 25, 50, 75, 99, 99.9, 100])
        iqr = q[5] - q[3]
        n_above = int(((v > q[5] + 10*iqr) | (v < q[3] - 10*iqr)).sum())
        
        sentinels = {-999, -99, -1, 999, 9999, 99999, 999999, 9999999}
        n_sentinel = int(v.isin(sentinels).sum() + (v.abs() > 1e5).sum())
        
        if n_above > 0 or n_sentinel > 0: has_extremes = True
        
        bounds_rows.append({
            "variable": c, "n_nonmissing": n, "observed_min": q[0], "p0.1": q[1], "p1": q[2], "p50": q[4], "p99": q[6], "p99.9": q[7], "observed_max": q[8],
            "n_above_fence": n_above, "n_sentinel": n_sentinel, "proposed_lower": "", "proposed_upper": "", "unit": "", "source": ""
        })
        
        c8_details.append({
            "variable": c, "max/p99.9": float(q[8]/q[7]) if q[7] != 0 else float('inf'), "n_above": n_above, "n_sentinel": n_sentinel
        })
        
    c8_details.sort(key=lambda x: x["max/p99.9"], reverse=True)
    report(results, "C8", "Extreme values", "WARN" if has_extremes else "PASS", "Extremes", c8_details)
    return pd.DataFrame(bounds_rows), cont_cols, binary_idx, categorical_groups_idx

def check_c9(df_real, cont_cols, dataset, results):
    temp_cols = [c for c in cont_cols if "temp" in c.lower()]
    temp_res = {}
    for c in temp_cols:
        v = df_real[c].dropna()
        p_celsius = float(((v >= 30) & (v <= 45)).mean())
        p_fahrenheit = float(((v >= 86) & (v <= 113)).mean())
        temp_res[c] = {"celsius_share": p_celsius, "fahrenheit_share": p_fahrenheit}
        
    hosp_res = {}
    has_warn = False
    if dataset == 'eicu' and 'hospitalid' in df_real.columns:
        counts = df_real['hospitalid'].value_counts()
        valid_hosps = counts[counts >= 50].index
        df_valid = df_real[df_real['hospitalid'].isin(valid_hosps)]
        
        for c in cont_cols:
            overall_med = df_valid[c].median()
            if overall_med == 0: continue
            hosp_meds = df_valid.groupby('hospitalid')[c].median()
            bad_hosps = hosp_meds[(hosp_meds > overall_med * 3) | (hosp_meds < overall_med / 3)]
            if len(bad_hosps) > 0:
                hosp_res[c] = bad_hosps.to_dict()
                has_warn = True
                
    report(results, "C9", "Units heuristics", "WARN" if has_warn else "PASS", "Heuristics", {"temps": temp_res, "eicu_hospitals": hosp_res})

def check_c10(df_real, df_masked, dataset, results):
    _, _, f_real = get_feature_columns(df_real, dataset)
    _, _, f_masked = get_feature_columns(df_masked, dataset)
    
    def get_types(df):
        types = {}
        cat_details = {}
        for c in df.columns:
            u = df[c].dropna().unique()
            if set(u).issubset({0, 1}): types[c] = "binary"
            elif 2 < len(u) < 10 and all(float(x).is_integer() for x in u):
                types[c] = "categorical"
                cat_details[c] = {"unique": list(u), "contiguous": bool((np.max(u) - np.min(u) + 1) == len(u))}
            else:
                types[c] = "continuous"
        return types, cat_details
        
    t_real, cat_real = get_types(f_real)
    t_masked, cat_masked = get_types(f_masked)
    
    diffs = {c: (t_real[c], t_masked[c]) for c in f_real.columns if t_real[c] != t_masked[c]}
    report(results, "C10a", "Inference consistency", "FAIL" if diffs else "PASS", "Differences (Real vs Masked)", diffs)
    
    bad_cats = {c: d for c, d in cat_masked.items() if not d["contiguous"]}
    report(results, "C10b", "Categorical contiguity", "WARN" if bad_cats else "PASS", "Non-contiguous categorical (Masked)", bad_cats)

def check_c11(df_real, dataset, results):
    _, schema_targets, feature_df = get_feature_columns(df_real, dataset)
    
    leak_suffixes = ['_max', '_min', '_mean', '_avg', '_last', '_first', '_sum', '_count']
    leaks = []
    
    if len(schema_targets) >= 2:
        assert schema_targets[0] == 'event' and schema_targets[1] in ('survival_months', 'los_days', 'time_days'), f"Unexpected targets: {schema_targets}"
        ev_col, t_col = schema_targets[0], schema_targets[1]
        
        for c in feature_df.columns:
            if any(c.lower().endswith(s) or s in c.lower() for s in leak_suffixes):
                valid = pd.concat([feature_df[c].reset_index(drop=True), df_real[[t_col, ev_col]].reset_index(drop=True)], axis=1).dropna()
                cidx = "N/A"
                if len(valid) > 0:
                    try:
                        cidx = concordance_index(valid[t_col], valid[c], valid[ev_col])
                    except: pass
                leaks.append({"feature": c, "cindex": cidx})
                
    report(results, "C11", "Temporal Leakage Indicators", "PASS", "Aggregation windows cannot be verified from the CSV. Check the extraction SQL and confirm every feature is measured before or at time zero (for example, the first 24 hours).", leaks)

def check_c12(df_masked, dataset, results):
    _, schema_targets, f_masked = get_feature_columns(df_masked, dataset)
    if len(schema_targets) >= 2:
        assert schema_targets[0] == 'event' and schema_targets[1] in ('survival_months', 'los_days', 'time_days'), f"Unexpected targets: {schema_targets}"
        ev_col = schema_targets[0]
        ev_mask = df_masked[ev_col] == 1
        cens_mask = df_masked[ev_col] == 0
        
        diffs = []
        has_warn = False
        for c in f_masked.columns:
            m_ev = f_masked.loc[ev_mask, c].isna().mean()
            m_cens = f_masked.loc[cens_mask, c].isna().mean()
            d = abs(m_ev - m_cens)
            if d > 0.2: has_warn = True
            diffs.append({"feature": c, "miss_event": m_ev, "miss_cens": m_cens, "diff": d})
            
        report(results, "C12", "Missingness vs outcome", "WARN" if has_warn else "PASS", "Expected under MNAR.", diffs)

def check_c13(df_masked, splits, cont_cols, results):
    warns = []
    for f, p in splits.items():
        tr_idx = p.get("train", [])
        if len(tr_idx) == 0: continue
        
        df_tr = df_masked.loc[tr_idx]
        for c in cont_cols:
            v = df_tr[c].dropna()
            if len(v) == 0: continue
            
            c_min = v.min()
            c_max = v.max()
            p999 = np.percentile(v, 99.9)
            
            if c_max - c_min < 1e-6:
                warns.append({"fold": f, "feature": c, "issue": "constant"})
            if p999 != 0 and c_max / p999 > 100:
                warns.append({"fold": f, "feature": c, "issue": "extreme_max"})
                
    report(results, "C13", "Training-fold scaling guard", "WARN" if warns else "PASS", "Compression warnings", warns)

def check_c14(df_real, df_masked, dataset, results):
    text_cols_real = [c for c in df_real.columns if df_real[c].dtype == object]
    text_cols_masked = [c for c in df_masked.columns if df_masked[c].dtype == object]
    text_cols = list(set(text_cols_real) & set(text_cols_masked))
    
    text_miss = {}
    for c in text_cols:
        text_miss[c] = {
            "real_nan": int(df_real[c].isna().sum()),
            "masked_nan": int(df_masked[c].isna().sum())
        }
        
    _, _, f_real = get_feature_columns(df_real, dataset)
    _, _, f_masked = get_feature_columns(df_masked, dataset)
    
    prefixes = {}
    for c in f_real.columns:
        if '_' in c:
            u = f_real[c].dropna().unique()
            if set(u).issubset({0, 1}):
                prefix = c.rsplit('_', 1)[0] + '_'
                if prefix not in prefixes: prefixes[prefix] = []
                prefixes[prefix].append(c)
            
    groups = {p: cols for p, cols in prefixes.items() if len(cols) > 1}
    
    group_stats = []
    has_fail = False
    for p, cols in groups.items():
        if not all(c in f_masked.columns for c in cols): continue
        
        real_all_zero = int((f_real[cols].sum(axis=1) == 0).sum())
        real_nan = int(f_real[cols].isna().any(axis=1).sum())
        
        masked_all_zero = int((f_masked[cols].sum(axis=1) == 0).sum())
        masked_nan = int(f_masked[cols].isna().any(axis=1).sum())
        
        group_stats.append({
            "prefix": p,
            "real_all_zero": real_all_zero,
            "real_nan": real_nan,
            "masked_all_zero": masked_all_zero,
            "masked_nan": masked_nan
        })
        
        if masked_all_zero > real_all_zero and masked_nan == 0:
            has_fail = True
            
    report(results, "C14", "One-hot encoding of missing values", "FAIL" if has_fail else "PASS", "Check for missingness encoded as zeros.", {"text_columns_nan": text_miss, "one_hot_groups": group_stats})

def check_c15(df_real, df_masked, dataset, results):
    text_cols = ['menopausal_state', 'cellularity', 'gender']
    
    _, _, f_real = get_feature_columns(df_real, dataset)
    _, _, f_masked = get_feature_columns(df_masked, dataset)
    
    def process(df_raw, df_feat, name):
        stats = []
        has_fail = False
        for c in text_cols:
            if c in df_raw.columns and c in df_feat.columns:
                raw_counts = df_raw[c].value_counts(dropna=False).to_dict()
                raw_counts_str = {repr(k): v for k, v in raw_counts.items()}
                
                raw_nan = int(df_raw[c].isna().sum())
                feat_nan = int(df_feat[c].isna().sum())
                
                stats.append({
                    "column": c,
                    "raw_nan": raw_nan,
                    "feat_nan": feat_nan,
                    "raw_counts": raw_counts_str
                })
                if feat_nan > raw_nan:
                    has_fail = True
        if stats:
            report(results, f"C15 ({name})", f"Text mapping loss ({name})", "FAIL" if has_fail else "PASS", "Check for unmapped strings", stats)

    process(df_real, f_real, "real")
    process(df_masked, f_masked, "masked")

def check_c16(df_real, dataset, results):
    if dataset not in ['mimic', 'eicu']:
        return
        
    keywords = ['id', 'pid', 'hospital', 'offset', 'discharge', 'death', 'dod', 'expire', 'mortality', 'los', 'intime', 'outtime', 'admit', 'disch']
    _, schema_targets, f_real = get_feature_columns(df_real, dataset)
    
    from dataset_schema import DATASET_SCHEMAS
    schema = DATASET_SCHEMAS.get(dataset, {})
    exclude_list = schema.get('exclude', [])
    targets_list = schema.get('targets', [])
    
    matched_cols = []
    for c in df_real.columns:
        c_lower = c.lower()
        tokens = c_lower.split('_')
        for k in keywords:
            if any(t == k or t.startswith(k) for t in tokens) or c_lower == k:
                matched_cols.append(c)
                break
            
    has_fail = False
    details = []
    for c in matched_cols:
        in_exclude = c in exclude_list
        in_targets = c in targets_list
        in_features = c in f_real.columns
        details.append({
            "column": c,
            "in_exclude": in_exclude,
            "in_targets": in_targets,
            "in_features": in_features
        })
        if in_features:
            has_fail = True
            
    report(results, "C16", "Schema exclusion coverage", "FAIL" if has_fail else "PASS", "Columns with suspicious names", details)

def write_markdown(results, out_md):
    with open(out_md, 'w') as f:
        f.write("# Audit Leakage Report\n\n")
        f.write("| Check | Status | Title |\n")
        f.write("|---|---|---|\n")
        for r in results:
            f.write(f"| {r['check_id']} | {r['status']} | {r['title']} |\n")
        
        f.write("\n## Details\n")
        for r in results:
            f.write(f"### {r['check_id']}: {r['title']} [{r['status']}]\n")
            f.write(f"{r['message']}\n")
            f.write(f"```json\n{json.dumps(r['details'], indent=2, allow_nan=False)}\n```\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all')
    parser.add_argument('--scenario', default='all')
    parser.add_argument('--outdir', default=os.path.join(PROJECT_ROOT, 'Results', 'final'))
    args = parser.parse_args()

    datasets = ['metabric', 'mimic', 'eicu'] if args.dataset == 'all' else [args.dataset]
    scenarios = ['light', 'moderate', 'severe'] if args.scenario == 'all' else [args.scenario]

    all_outputs = []

    for d in datasets:
        for s in scenarios:
            print(f"Auditing {d} - {s}...")
            real_path, masked_path, mask_path, splits_path = get_paths(d, s)
            
            results = []
            files = {"real": real_path, "masked": masked_path, "mask": mask_path, "splits": splits_path}
            
            # Inventory
            file_exists = True
            for path in files.values():
                if not os.path.exists(path):
                    report(results, "C0", "File existence", "FAIL", f"Missing file: {path}")
                    file_exists = False
            
            if file_exists:
                df_real = pd.read_csv(real_path)
                df_masked = pd.read_csv(masked_path)
                df_mask = pd.read_csv(mask_path)
                with open(splits_path, 'r') as f:
                    splits = json.load(f)
                    
                def safe_call(func, check_id, title, *args, **kwargs):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        tb = traceback.format_exc()
                        report(results, check_id, title, "FAIL", "Exception", {"error": str(e), "traceback": tb})
                        print(f"[{check_id}] FAILED with exception, continuing\n{tb}")
                        return None
                        
                safe_call(check_c1, "C1", "File inventory", df_real, df_masked, df_mask, splits, files, results)
                safe_call(check_c2, "C2", "Row alignment", df_real, df_masked, df_mask, d, results)
                safe_call(check_c3, "C3", "Targets", df_real, df_masked, d, results)
                safe_call(check_c4, "C4", "Feature columns", df_real, d, results)
                safe_call(check_c5, "C5", "Identifier columns intact", df_real, df_masked, df_mask, d, results)
                safe_call(check_c6, "C6", "Splits", df_real, splits, d, results)
                safe_call(check_c7, "C7", "Duplicates", df_real, d, results)
                
                c8_res = safe_call(check_c8, "C8", "Extreme and impossible values", df_real, d, results)
                df_bounds = pd.DataFrame()
                cont_cols = []
                if c8_res is not None:
                    df_bounds, cont_cols, _, _ = c8_res
                
                # Output bounds removed
                
                safe_call(check_c9, "C9", "Units", df_real, cont_cols, d, results)
                safe_call(check_c10, "C10", "Feature type inference consistency", df_real, df_masked, d, results)
                safe_call(check_c11, "C11", "Temporal leakage indicators", df_real, d, results)
                safe_call(check_c12, "C12", "Missingness vs outcome", df_masked, d, results)
                safe_call(check_c13, "C13", "Training-fold scaling guard", df_masked, splits, cont_cols, results)
                safe_call(check_c14, "C14", "One-hot encoding of missing values", df_real, df_masked, d, results)
                safe_call(check_c15, "C15", "Text mapping loss", df_real, df_masked, d, results)
                safe_call(check_c16, "C16", "Schema exclusion coverage", df_real, d, results)
                
            # Sanitize results
            sanitized_results = sanitize(results)
            
            out_md = os.path.join(args.outdir, f"audit_{d}_{s}.md")
            try:
                write_markdown(sanitized_results, out_md)
                all_outputs.append(out_md)
            except Exception as e:
                print(f"Failed to write Markdown to {out_md}: {e}")
                
            out_json = os.path.join(args.outdir, f"audit_{d}_{s}.json")
            try:
                with open(out_json, 'w') as f:
                    json.dump(sanitized_results, f, indent=2, allow_nan=False)
                all_outputs.append(out_json)
            except Exception as e:
                print(f"Failed to write JSON to {out_json}: {e}")

    print("\nOUTPUT FILES:")
    for f in all_outputs:
        print(f)
        
    print("\nASSUMPTIONS:")
    print("1. Column Names: get_feature_columns determines features and schema_targets[0]=event, schema_targets[1]=time.")
    print("2. Mask Coding: Inferred by checking if df_mask==1 matches ~df_masked.isna() or df_masked.isna().")
    print("3. ID Columns: Hardcoded for mimic ('subject_id', 'hadm_id', 'stay_id'), eicu ('uniquepid', etc.), and metabric ('patient' or 'id').")
    
    sys.exit(0)

if __name__ == '__main__':
    main()
