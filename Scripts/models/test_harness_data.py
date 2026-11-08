import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from harness_data import load_fold, load_test, FoldData

@pytest.mark.parametrize("dataset", ["metabric", "mimic", "eicu"])
def test_harness_basics(dataset):
    scenario = "light"
    fold = 1
    
    fd_train = load_fold(dataset, scenario, fold)
    fd_test = load_test(dataset, scenario, fold, fd_train)
    
    # disjoint test
    tr_ids = fd_train.part('train')['ids']
    va_ids = fd_train.part('val')['ids']
    te_ids = fd_test.part('test')['ids']
    assert len(np.intersect1d(tr_ids, va_ids)) == 0
    assert len(np.intersect1d(tr_ids, te_ids)) == 0
    assert len(np.intersect1d(va_ids, te_ids)) == 0
    
    tr_idx = fd_train.part('train')['cohort_index']
    va_idx = fd_train.part('val')['cohort_index']
    te_idx = fd_test.part('test')['cohort_index']
    assert len(set(tr_idx) & set(va_idx)) == 0
    assert len(set(tr_idx) & set(te_idx)) == 0
    assert len(set(va_idx) & set(te_idx)) == 0
    
    # Scaler logic test
    cont_cols = fd_train.feature_types['continuous']
    cont_idxs = [list(fd_train.feature_names).index(c) for c in cont_cols]
    if len(cont_idxs) > 0:
        tr_raw = fd_train.X_obs_raw[:len(tr_idx), cont_idxs]
        expected_mean = np.nanmean(tr_raw, axis=0)
        expected_std = np.nanstd(tr_raw, axis=0)
        expected_std[expected_std == 0] = 1.0
        expected_std[np.isnan(expected_std)] = 1.0
        expected_mean[np.isnan(expected_mean)] = 0.0
        
        assert np.allclose(fd_train.scaler_mean, expected_mean)
        assert np.allclose(fd_train.scaler_std, expected_std)
        
        # Test differs from train+val+test
        all_raw = np.concatenate([fd_train.X_obs_raw[:, cont_idxs], fd_test.X_obs_raw[:, cont_idxs]], axis=0)
        all_mean = np.nanmean(all_raw, axis=0)
        all_mean[np.isnan(all_mean)] = 0.0
        # Not necessarily failing for every single feature if means are accidentally equal, but generally:
        assert not np.array_equal(fd_train.scaler_mean, all_mean)

@pytest.mark.parametrize("dataset", ["metabric", "mimic", "eicu"])
def test_feature_order_all_folds(dataset):
    scenario = "light"
    # Get reference expected order from dataset_schema
    from dataset_schema import FEATURE_TYPES
    ft = FEATURE_TYPES[dataset]
    expected_order = ft.get('binary', []) + list(ft.get('categorical', {}).keys()) + ft.get('continuous', [])
    
    for fold in range(1, 6):
        fd_train = load_fold(dataset, scenario, fold)
        fd_test = load_test(dataset, scenario, fold, fd_train)
        
        # Test feature names equal the reference order
        assert list(fd_train.feature_names) == expected_order
        assert list(fd_test.feature_names) == expected_order
