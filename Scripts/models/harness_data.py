import os
import json
import numpy as np
import pandas as pd

from dataset_schema import get_feature_columns, get_feature_types

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class FoldData:
    def __init__(self, dataset, scenario, fold, is_test=False):
        self.dataset = dataset
        self.scenario = scenario
        self.fold = fold
        self.is_test = is_test
        
        name_lower = dataset.lower()
        if 'metabric' in name_lower:
            real_path = os.path.join(PROJECT_ROOT, "Results", "metabric", "final", "metabric_cohort.csv")
            pid = 'patientId'
        elif 'eicu' in name_lower:
            real_path = os.path.join(PROJECT_ROOT, "Results", "eicu", "final", "eicu_sepsis_cohort.csv")
            pid = 'uniquepid'
        elif 'synthetic' in name_lower:
            real_path = os.path.join(PROJECT_ROOT, "Results", "synthetic", "synthetic_cohort_observed.csv")
            pid = 'patient_id'
        else:
            real_path = os.path.join(PROJECT_ROOT, "Results", "final", "mimic_sepsis_cohort.csv")
            pid = 'subject_id'
            
        mnar_path = os.path.join(PROJECT_ROOT, "Scripts", "models", "datasets_real", f"{dataset}_mnar_{scenario}.csv")
        mask_path = os.path.join(PROJECT_ROOT, "Scripts", "models", "datasets_real", f"{dataset}_mask_{scenario}.csv")
        splits_path = os.path.join(PROJECT_ROOT, "Results", "final", "cv_splits", f"cv_splits_{dataset}.json")
        
        print(f"Data paths resolved:")
        print(f"  Real: {real_path}")
        print(f"  MNAR: {mnar_path}")
        print(f"  Mask: {mask_path}")
        
        for p in [real_path, mnar_path, mask_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required data file missing: {p}")
        
        df_real = pd.read_csv(real_path)
        df_mnar = pd.read_csv(mnar_path)
        df_mask = pd.read_csv(mask_path)
        
        f_cols, targets, X_true_df_full = get_feature_columns(df_real, dataset)
        _, _, X_obs_df_full = get_feature_columns(df_mnar, dataset)
        
        with open(splits_path, 'r') as f:
            splits = json.load(f)
            
        fold_key = f"fold_{fold}"
        split_dict = splits[fold_key]
        
        if self.is_test:
            self.cohort_indices = {'test': split_dict['test']}
            indices = split_dict['test']
        else:
            self.cohort_indices = {'train': split_dict['train'], 'val': split_dict['val']}
            indices = split_dict['train'] + split_dict['val']
            
            tr = set(split_dict['train'])
            va = set(split_dict['val'])
            te = set(split_dict['test'])
            assert len(tr & va) == 0 and len(tr & te) == 0 and len(va & te) == 0, "Splits are not disjoint"
            
        self.df_real = df_real.iloc[indices].reset_index(drop=True)
        self.df_mnar = df_mnar.iloc[indices].reset_index(drop=True)
        self.df_mask = df_mask.iloc[indices].reset_index(drop=True)
        
        assert self.df_real[pid].nunique() == len(self.df_real), f"Expected one row per ID in FoldData"

        assert self.df_real[targets[0]].equals(self.df_mnar[targets[0]]), "Event target mismatch"
        assert np.allclose(self.df_real[targets[1]].fillna(-9999), self.df_mnar[targets[1]].fillna(-9999), atol=1e-9), "Time target mismatch"

        self.feature_names = list(f_cols)
        self.feature_types = get_feature_types(dataset, f_cols)
        
        self.T = self.df_real[targets[1]].values
        self.E = self.df_real[targets[0]].values
        
        self.X_true_raw = X_true_df_full.iloc[indices][f_cols].values
        self.X_obs_raw = X_obs_df_full.iloc[indices][f_cols].values
        
        self.M_inj = np.zeros((len(self.df_mask), len(f_cols)))
        for i, c in enumerate(f_cols):
            if c in self.df_mask.columns:
                self.M_inj[:, i] = self.df_mask[c].values
            else:
                if c in self.feature_types['continuous']:
                    raise ValueError(f"Continuous feature {c} is missing from the mask file.")
                self.M_inj[:, i] = 0
                
        non_cont_cols = self.feature_types['binary'] + list(self.feature_types['categorical'].keys())
        for c in non_cont_cols:
            idx = list(f_cols).index(c)
            assert np.all(self.M_inj[:, idx] == 0), f"M_inj is non-zero for non-continuous feature {c}"
            
        self.M_obs = (~np.isnan(self.X_obs_raw)).astype(float)
        
        m_inj_mask = self.M_inj == 1
        assert np.all(np.isnan(self.X_obs_raw[m_inj_mask])), "M_inj=1 cells must be NaN in X_obs"
        assert np.all(~np.isnan(self.X_true_raw[m_inj_mask])), "M_inj=1 cells must be observed in X_true"
        
        self.ids = self.df_real[pid].values
        
        self.scaler_mean = None
        self.scaler_std = None
        self.X_obs = None
        self.X_true = None
        
    def fit_scaler(self):
        if self.is_test: return
        
        cont_cols = self.feature_types['continuous']
        cont_idxs = [list(self.feature_names).index(c) for c in cont_cols]
        
        if len(cont_idxs) > 0:
            tr_len = len(self.cohort_indices['train'])
            train_obs = self.X_obs_raw[:tr_len, cont_idxs]
            self.scaler_mean = np.nanmean(train_obs, axis=0)
            self.scaler_std = np.nanstd(train_obs, axis=0)
            self.scaler_std[self.scaler_std == 0] = 1.0
            self.scaler_std[np.isnan(self.scaler_std)] = 1.0
            self.scaler_mean[np.isnan(self.scaler_mean)] = 0.0
            
    def apply_scaling(self, mean, std):
        self.scaler_mean = mean
        self.scaler_std = std
        
        X_obs_sc = self.X_obs_raw.copy()
        X_true_sc = self.X_true_raw.copy()
        
        cont_cols = self.feature_types['continuous']
        cont_idxs = [list(self.feature_names).index(c) for c in cont_cols]
        
        if len(cont_idxs) > 0:
            X_obs_sc[:, cont_idxs] = (X_obs_sc[:, cont_idxs] - self.scaler_mean) / self.scaler_std
            X_true_sc[:, cont_idxs] = (X_true_sc[:, cont_idxs] - self.scaler_mean) / self.scaler_std
            
        cat_dict = self.feature_types['categorical']
        for col, codes in cat_dict.items():
            idx = list(self.feature_names).index(col)
            for i, arr in enumerate([X_obs_sc, X_true_sc]):
                valid_mask = ~np.isnan(arr[:, idx])
                vals = arr[valid_mask, idx]
                invalid = set(vals) - set(codes)
                if invalid:
                    raise ValueError(f"Invalid categorical codes for {col}: {invalid}")
                
                mapped = np.zeros_like(vals)
                for k, code in enumerate(codes):
                    mapped[vals == code] = k
                arr[valid_mask, idx] = mapped
                
        self.X_obs = X_obs_sc
        self.X_true = X_true_sc
        
    def inverse_transform(self, X):
        X_inv = X.copy()
        cont_cols = self.feature_types['continuous']
        cont_idxs = [list(self.feature_names).index(c) for c in cont_cols]
        if len(cont_idxs) > 0:
            X_inv[:, cont_idxs] = X_inv[:, cont_idxs] * self.scaler_std + self.scaler_mean
        return X_inv
        
    def part(self, name):
        if self.is_test and name != 'test':
            raise ValueError("This is a test FoldData object.")
        if not self.is_test and name not in ['train', 'val']:
            raise ValueError("This is a train/val FoldData object.")
            
        if name == 'test':
            start, end = 0, len(self.df_real)
        elif name == 'train':
            start, end = 0, len(self.cohort_indices['train'])
        elif name == 'val':
            start, end = len(self.cohort_indices['train']), len(self.df_real)
            
        return {
            'X_obs': self.X_obs[start:end],
            'X_true': self.X_true[start:end],
            'M_obs': self.M_obs[start:end],
            'M_inj': self.M_inj[start:end],
            'T': self.T[start:end],
            'E': self.E[start:end],
            'ids': self.ids[start:end],
            'cohort_index': self.cohort_indices[name],
            'feature_names': list(self.feature_names),
            'feature_types': self.feature_types
        }

def fill_for_input(X_obs):
    X_filled = X_obs.copy()
    X_filled[np.isnan(X_filled)] = 0.0
    return X_filled

def load_fold(dataset, scenario, fold):
    fd = FoldData(dataset, scenario, fold, is_test=False)
    fd.fit_scaler()
    fd.apply_scaling(fd.scaler_mean, fd.scaler_std)
    return fd

def load_test(dataset, scenario, fold, fd_train):
    fd = FoldData(dataset, scenario, fold, is_test=True)
    fd.apply_scaling(fd_train.scaler_mean, fd_train.scaler_std)
    return fd
