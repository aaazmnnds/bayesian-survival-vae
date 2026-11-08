import numpy as np
import pytest
import torch
from harness_metrics import (rmse, mae, picp, mpiw, ece, breslow_baseline, survival_matrix,
                             eval_time_grid, cindex, ibs, td_auc_unweighted, d_calibration,
                             survival_at_event_times, interval_coverage)
from sksurv.metrics import concordance_index_censored
import os, sys

sys.path.append(os.path.dirname(__file__))
from baseline_hazard import compute_breslow_baseline_hazard

np.random.seed(42)

def test_imputation_metrics():
    # Small example
    x_true = np.array([1.0, 2.0, 3.0, 4.0])
    mu = np.array([1.5, 2.0, 2.5, 5.0])
    M_inj = np.array([1, 0, 1, 1])
    
    # Cells with M_inj == 1: indices 0, 2, 3
    # True: 1, 3, 4. Mu: 1.5, 2.5, 5.0.
    # diffs: -0.5, 0.5, -1.0. abs: 0.5, 0.5, 1.0. mae: (0.5+0.5+1.0)/3 = 2.0/3
    # sq diffs: 0.25, 0.25, 1.0. mse: 1.5/3 = 0.5. rmse: sqrt(0.5)
    
    assert np.isclose(mae(x_true, mu, M_inj), 2.0 / 3.0)
    assert np.isclose(rmse(x_true, mu, M_inj), np.sqrt(0.5))

def test_imputation_coverage():
    # Large n for probabilistic tests
    n = 200000
    mu = np.random.randn(n) * 2
    sigma = np.random.uniform(0.5, 2.0, n)
    x_true = np.random.normal(mu, sigma)
    M_inj = np.ones(n)
    
    p = picp(x_true, mu, sigma, M_inj, alpha=0.05)
    assert 0.945 <= p <= 0.955
    
    e = ece(x_true, mu, sigma, M_inj, B=10)
    assert e < 0.01

def test_breslow_baseline():
    T = np.random.randint(1, 100, size=200).astype(float)
    E = np.random.randint(0, 2, size=200)
    eta = np.random.randn(200)
    
    times = np.unique(T)
    
    # Python numpy implementation
    unique_t, cum_hazard = breslow_baseline(eta, T, E)
    
    # PyTorch existing implementation
    eta_t = torch.tensor(eta, dtype=torch.float32)
    T_t = torch.tensor(T, dtype=torch.float32)
    E_t = torch.tensor(E, dtype=torch.float32)
    eval_t = torch.tensor(unique_t, dtype=torch.float32)
    H0_eval = compute_breslow_baseline_hazard(eta_t, T_t, E_t, eval_t).numpy()
    
    assert np.allclose(cum_hazard, H0_eval, rtol=1e-5, atol=1e-7)

def test_cindex():
    T = np.random.exponential(10, size=200).astype(float)
    E = np.random.randint(0, 2, size=200)
    eta = np.random.randn(200)
    
    res1 = cindex(T, E, eta)
    res2 = concordance_index_censored(E.astype(bool), T, eta)[0]
    
    assert np.isclose(res1, res2, atol=1e-6)

def test_cindex_brute_force():
    T = np.array([2.0, 2.0, 3.0, 4.0, 4.0])
    E = np.array([1, 0, 1, 1, 0])
    eta = np.array([1.5, 0.5, 2.0, 2.0, 1.0])
    
    concordant = 0.0
    comparable = 0.0
    for i in range(len(T)):
        if E[i] == 1:
            for j in range(len(T)):
                if T[i] < T[j]:
                    comparable += 1.0
                    if eta[i] > eta[j]:
                        concordant += 1.0
                    elif eta[i] == eta[j]:
                        concordant += 0.5
                        
    brute_force_c = concordant / comparable if comparable > 0 else 0.0
    func_c = cindex(T, E, eta)
    
    assert np.isclose(func_c, brute_force_c)

def test_td_auc_unweighted():
    T = np.array([1, 2, 3, 4, 5], dtype=float)
    E = np.array([1, 0, 1, 1, 0])
    eta = np.array([0.1, -0.5, 1.2, 0.8, -0.2])
    times = [2.5]
    
    _, auc = td_auc_unweighted(T, E, eta, times)
    
    # Manual compute for t = 2.5:
    # Cases: T <= 2.5 & E == 1 => idx 0 (T=1, E=1)
    # Controls: T > 2.5 => idx 2, 3, 4
    # eta case: 0.1
    # eta controls: 1.2, 0.8, -0.2
    # case > control: (0.1 > 1.2 => 0), (0.1 > 0.8 => 0), (0.1 > -0.2 => 1)
    # AUC = 1 / 3
    assert np.isclose(auc, 1.0 / 3.0)

def test_d_calibration():
    # Simulate known exponential model
    np.random.seed(42)
    n = 2000
    lam = np.random.uniform(0.1, 1.0, n)
    T_true = np.random.exponential(1.0 / lam)
    C = np.random.exponential(1.0, n)
    T = np.minimum(T_true, C)
    E = (T_true <= C).astype(int)
    
    # S(t) = exp(-lam * t)
    S_at_T = np.exp(-lam * T)
    _, pval, _ = d_calibration(S_at_T, E, B=10)
    assert pval > 0.01
    
    # Miscalibrated
    lam_wrong = lam * 3
    S_wrong = np.exp(-lam_wrong * T)
    _, pval_wrong, _ = d_calibration(S_wrong, E, B=10)
    assert pval_wrong < 0.001

def test_eval_time_grid():
    T_train = np.array([1, 100])
    T_test = np.array([10, 80])
    E_test = np.ones(2)
    times = eval_time_grid(T_train, T_test, E_test, n=10)
    
    # t_min = 1st percentile of [10, 80] = 10.7
    # t_max = min(max train, max test) * 0.999 = 80 * 0.999 = 79.92
    assert times[0] >= 10.7
    assert times[-1] <= 80

def test_cindex_chunked_vs_unchunked():
    T = np.random.exponential(10, size=5000).astype(float)
    E = np.random.randint(0, 2, size=5000)
    eta = np.random.randn(5000)
    
    # chunked from our library
    c_chunked = cindex(T, E, eta)
    
    # unchunked
    case_mask = E == 1
    T_cases = T[case_mask]
    eta_cases = eta[case_mask]
    
    comp_matrix = T_cases[:, None] < T[None, :]
    eta_diff = eta_cases[:, None] - eta[None, :]
    conc_matrix = (eta_diff > 0).astype(float) + 0.5 * (eta_diff == 0).astype(float)
    
    comparable = np.sum(comp_matrix)
    concordant = np.sum(conc_matrix * comp_matrix)
    c_unchunked = concordant / comparable if comparable > 0 else 0.0
    
    assert np.isclose(c_chunked, c_unchunked)

def test_ibs_truncates_test_times_exceeding_train_max():
    T_train = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    E_train = np.array([1, 0, 1, 1, 0])
    
    # Test set includes a patient with observed time 30.0 exceeding max(T_train)=25.0
    T_test = np.array([6.0, 12.0, 30.0])
    E_test = np.array([1, 0, 1])
    
    # Manually truncated test times
    T_test_manually_truncated = np.array([6.0, 12.0, 25.0])
    
    # Time grid strictly below max follow-up
    times = np.array([7.0, 14.0])
    
    # Mock survival matrix: shape (n_test_samples, n_times)
    S_matrix = np.array([
        [0.9, 0.7],
        [0.85, 0.65],
        [0.8, 0.5]
    ])
    
    val_auto = ibs(T_train, E_train, T_test, E_test, S_matrix, times)
    val_manual = ibs(T_train, E_train, T_test_manually_truncated, E_test, S_matrix, times)
    
    assert np.isfinite(val_auto)
    assert np.isclose(val_auto, val_manual)

