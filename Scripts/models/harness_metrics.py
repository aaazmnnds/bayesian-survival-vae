import numpy as np
import scipy.stats as stats
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc, concordance_index_ipcw
from sksurv.nonparametric import kaplan_meier_estimator

def rmse(x_true, mu, M_inj):
    mask = M_inj == 1
    if not np.any(mask):
        return np.nan
    return np.sqrt(np.mean((x_true[mask] - mu[mask])**2))

def mae(x_true, mu, M_inj):
    mask = M_inj == 1
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs(x_true[mask] - mu[mask]))

def picp(x_true, mu, sigma_T, M_inj, alpha=0.05):
    mask = M_inj == 1
    if not np.any(mask):
        return np.nan
    z = stats.norm.ppf(1 - alpha/2)
    lower = mu[mask] - z * sigma_T[mask]
    upper = mu[mask] + z * sigma_T[mask]
    return np.mean((x_true[mask] >= lower) & (x_true[mask] <= upper))

def mpiw(sigma_T, M_inj, alpha=0.05):
    mask = M_inj == 1
    if not np.any(mask):
        return np.nan
    z = stats.norm.ppf(1 - alpha/2)
    return np.mean(2 * z * sigma_T[mask])

def ece(x_true, mu, sigma_T, M_inj, B=10):
    mask = M_inj == 1
    if not np.any(mask):
        return np.nan
    p_b_levels = np.linspace(0.1, 0.9, B)
    eces = []
    x_t = x_true[mask]
    m_t = mu[mask]
    s_t = sigma_T[mask]
    for p_b in p_b_levels:
        z_b = stats.norm.ppf((1 + p_b) / 2)
        lower = m_t - z_b * s_t
        upper = m_t + z_b * s_t
        coverage = np.mean((x_t >= lower) & (x_t <= upper))
        eces.append(np.abs(coverage - p_b))
    return np.mean(eces)

def breslow_baseline(eta_train, T_train, E_train):
    order = np.argsort(T_train)
    t = T_train[order]
    e = E_train[order]
    exp_eta = np.exp(eta_train[order])
    
    unique_t = np.unique(t)
    d = np.zeros_like(unique_t, dtype=float)
    
    t_idx = np.searchsorted(unique_t, t)
    np.add.at(d, t_idx, e)
    
    risk_sum_all = np.cumsum(exp_eta[::-1])[::-1]
    first_occurrence = np.searchsorted(t, unique_t)
    risk_set = risk_sum_all[first_occurrence]
    
    hazard = d / risk_set
    cum_hazard = np.cumsum(hazard)
    
    mask = d > 0
    if not np.any(mask):
        return np.array([]), np.array([])
        
    return unique_t[mask], cum_hazard[mask]

def survival_matrix(baseline, eta_eval, times):
    unique_t, cum_hazard = baseline
    if len(unique_t) == 0:
        return np.ones((len(eta_eval), len(times)))
        
    idx = np.searchsorted(unique_t, times, side='right') - 1
    H0 = np.zeros_like(times, dtype=float)
    valid = idx >= 0
    H0[valid] = cum_hazard[idx[valid]]
    
    exp_eta = np.exp(eta_eval)
    S = np.exp(-np.outer(exp_eta, H0))
    return S

def eval_time_grid(T_train, T_test, E_test, n=50):
    t_min = np.percentile(T_test, 1)
    t_max = min(np.max(T_train), np.max(T_test)) * 0.999
    return np.linspace(t_min, t_max, n)

def cindex(T, E, eta):
    n = len(T)
    concordant = 0.0
    comparable = 0.0
    
    chunk_size = 2000
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        
        T_i = T[i:end]
        E_i = E[i:end]
        eta_i = eta[i:end]
        
        case_mask = E_i == 1
        if not np.any(case_mask):
            continue
            
        T_cases = T_i[case_mask]
        eta_cases = eta_i[case_mask]
        
        comp_matrix = T_cases[:, None] < T[None, :]
        comparable += np.sum(comp_matrix)
        
        eta_diff = eta_cases[:, None] - eta[None, :]
        conc_matrix = (eta_diff > 0).astype(float) + 0.5 * (eta_diff == 0).astype(float)
        
        concordant += np.sum(conc_matrix * comp_matrix)
        
    if comparable == 0:
        return 0.0
    return concordant / comparable

def ibs(T_train, E_train, T_test, E_test, S_matrix, times):
    dt = np.dtype([('event', bool), ('time', float)])
    train_struct = np.empty(len(T_train), dtype=dt)
    train_struct['event'] = E_train.astype(bool)
    train_struct['time'] = T_train
    
    max_train_time = np.max(T_train)
    T_test_trunc = np.minimum(np.copy(T_test), max_train_time)
    
    test_struct = np.empty(len(T_test), dtype=dt)
    test_struct['event'] = E_test.astype(bool)
    test_struct['time'] = T_test_trunc
    
    return integrated_brier_score(train_struct, test_struct, S_matrix, times)

def td_auc_unweighted(T, E, eta, times):
    aucs = []
    valid_times = []
    for t in times:
        cases = (T <= t) & (E == 1)
        controls = (T > t)
        
        n_cases = cases.sum()
        n_controls = controls.sum()
        if n_cases == 0 or n_controls == 0:
            continue
            
        eta_cases = eta[cases]
        eta_controls = eta[controls]
        
        diff = eta_cases[:, None] - eta_controls[None, :]
        auc_t = np.mean((diff > 0).astype(float) + 0.5 * (diff == 0).astype(float))
        aucs.append(auc_t)
        valid_times.append(t)
        
    if not aucs:
        return np.array([]), np.nan
    return np.array(aucs), np.mean(aucs)

def d_calibration(S_at_T, E, B=10):
    w = 1.0 / B
    bins = np.zeros(B)
    
    for s, e in zip(S_at_T, E):
        if e == 1:
            idx = int(np.clip(s // w, 0, B-1))
            bins[idx] += 1
        else:
            q = s
            if q <= 0:
                continue
                
            n_full = int(q // w)
            for i in range(n_full):
                if i < B:
                    bins[i] += w / q
            
            rem = q - n_full * w
            if n_full < B and rem > 0:
                bins[n_full] += rem / q
                
    expected = len(S_at_T) / B
    chi2 = np.sum((bins - expected)**2 / expected)
    pval = stats.chi2.sf(chi2, B - 1)
    
    return chi2, pval, bins

def survival_at_event_times(baseline, eta, T):
    unique_t, cum_hazard = baseline
    if len(unique_t) == 0:
        return np.ones_like(T, dtype=float)
        
    idx = np.searchsorted(unique_t, T, side='right') - 1
    H0 = np.zeros_like(T, dtype=float)
    valid = idx >= 0
    H0[valid] = cum_hazard[idx[valid]]
    
    exp_eta = np.exp(eta)
    S_at_T = np.exp(-H0 * exp_eta)
    return S_at_T

def interval_coverage(S_true, S_lower, S_upper):
    in_interval = (S_true >= S_lower) & (S_true <= S_upper)
    per_time = np.mean(in_interval, axis=0)
    overall = np.mean(in_interval)
    return per_time, overall


def td_auc_ipcw(T_train, E_train, T_test, E_test, eta, times):
    dt = np.dtype([('event', bool), ('time', float)])
    train = np.empty(len(T_train), dtype=dt)
    train['event'] = E_train.astype(bool)
    train['time'] = T_train

    # clipping above the largest grid time does not change the estimates.
    T_test_c = np.minimum(T_test, np.nextafter(np.max(T_train), 0))
    test = np.empty(len(T_test), dtype=dt)
    test['event'] = E_test.astype(bool)
    test['time'] = T_test_c

    times = np.asarray(times)
    t_min = np.min(T_test)
    t_max = np.max(T_test)

    censored = (E_train == 0)
    if not np.any(censored):
        g_times = np.ones(len(times), dtype=float)
    else:
        try:
            km_times, km_surv = kaplan_meier_estimator(censored, T_train)
            idx = np.searchsorted(km_times, times, side='right') - 1
            g_times = np.ones(len(times), dtype=float)
            valid = idx >= 0
            g_times[valid] = km_surv[idx[valid]]
        except Exception:
            order = np.argsort(T_train)
            t_sorted = T_train[order]
            c_sorted = censored[order].astype(float)
            unique_t = np.unique(t_sorted)
            d_c = np.zeros_like(unique_t, dtype=float)
            t_idx = np.searchsorted(unique_t, t_sorted)
            np.add.at(d_c, t_idx, c_sorted)
            n_at_risk = len(T_train) - np.searchsorted(t_sorted, unique_t)
            prob_step = 1.0 - (d_c / n_at_risk)
            cum_g = np.cumprod(prob_step)
            idx = np.searchsorted(unique_t, times, side='right') - 1
            g_times = np.ones(len(times), dtype=float)
            valid = idx >= 0
            g_times[valid] = cum_g[idx[valid]]

    mask = (times >= t_min) & (times < t_max) & (times < np.max(T_train)) & (g_times > 0)
    kept_times = times[mask]
    n_dropped = int(len(times) - len(kept_times))

    if len(kept_times) == 0:
        return np.array([]), np.nan, n_dropped

    auc_array, _ = cumulative_dynamic_auc(train, test, eta, kept_times)
    return auc_array, float(np.mean(auc_array)), n_dropped


def cindex_uno(T_train, E_train, T_test, E_test, eta, tau):
    dt = np.dtype([('event', bool), ('time', float)])
    train = np.empty(len(T_train), dtype=dt)
    train['event'] = E_train.astype(bool)
    train['time'] = T_train

    # clipping above the largest grid time does not change the estimates.
    T_test_c = np.minimum(T_test, np.nextafter(np.max(T_train), 0))
    test = np.empty(len(T_test), dtype=dt)
    test['event'] = E_test.astype(bool)
    test['time'] = T_test_c

    result = concordance_index_ipcw(train, test, eta, tau=tau)
    return result[0]

