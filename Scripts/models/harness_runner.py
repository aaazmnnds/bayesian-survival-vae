"""
Plugin interface:
A plug-in is a class with:
- name: str
- directions: list of 'minimize'/'maximize' for its tuning objectives
- pareto: bool
- search_space(trial): takes an Optuna trial, returns a params dict
- fit(train, val, params, seed): train and val are part() dicts. 
  Returns a state object. The plug-in may use val only for early stopping 
  or checkpointing; it never receives test data.
- predict(state, part, n_passes): returns a dict with:
    'mu': n x p imputed values, standardized scale
    'sigma_E2' (optional): n x p
    'sigma_A2' (optional): n x p
    'eta': n, MC mean risk score
    'eta_samples' (optional): n x n_passes
- objectives(pred_val, val): returns tuple of floats in the order of directions.
"""

import os
import json
import time
import random
import traceback
import hashlib
import inspect
import subprocess
import numpy as np
import torch
import optuna

from harness_data import load_fold, load_test, fill_for_input
from harness_metrics import (rmse, mae, picp, mpiw, ece, cindex, 
                             breslow_baseline, survival_matrix, eval_time_grid, 
                             ibs, td_auc_unweighted, d_calibration, 
                             survival_at_event_times)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_runs_dir():
    return os.path.join(PROJECT_ROOT, 'Results', 'final', 'runs_v2')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_config_hash(plugin, n_trials, seed):
    source = inspect.getsource(plugin.__class__)
    hasher = hashlib.md5()
    hasher.update(plugin.name.encode('utf-8'))
    hasher.update(source.encode('utf-8'))
    hasher.update(str(n_trials).encode('utf-8'))
    hasher.update(str(seed).encode('utf-8'))
    
    files_to_hash = ['losses.py', 'train_utils.py', 'bsvae.py', 'encoder.py', 
                     'decoder.py', 'cox_predictor.py', 'harness_data.py', 'harness_metrics.py']
    for f in files_to_hash:
        fpath = os.path.join(os.path.dirname(__file__), f)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as file_obj:
                hasher.update(file_obj.read())
                
    return hasher.hexdigest()

def tune_and_select(plugin, dataset, scenario, fold, n_trials, seed):
    set_seed(seed)
    
    study_name = f"{plugin.name}_{dataset}_{scenario}_fold{fold}_v2"
    runs_dir = get_runs_dir()
    optuna_dir = os.path.join(runs_dir, "optuna")
    os.makedirs(optuna_dir, exist_ok=True)
    storage_url = f"sqlite:///{os.path.join(optuna_dir, study_name + '.db')}"
    
    cfg_hash = get_config_hash(plugin, n_trials, seed)
    hash_file = os.path.join(optuna_dir, f"{study_name}_hash.txt")
    
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            existing_hash = f.read().strip()
        if existing_hash != cfg_hash:
            raise ValueError("Study exists with a different config hash. Aborting.")
    else:
        with open(hash_file, 'w') as f:
            f.write(cfg_hash)
            
    sampler = optuna.samplers.TPESampler(seed=seed)
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_url, 
        load_if_exists=True,
        directions=plugin.directions,
        sampler=sampler
    )
    
    fd = load_fold(dataset, scenario, fold)
    train_part = fd.part('train')
    val_part = fd.part('val')
    
    def objective(trial):
        params = plugin.search_space(trial)
        try:
            state = plugin.fit(train_part, val_part, params, seed)
            pred_val = plugin.predict(state, val_part, n_passes=20)
            return plugin.objectives(pred_val, val_part)
        except Exception as e:
            print(f"Trial {trial.number} failed with exception:")
            traceback.print_exc()
            raise e
            
    while True:
        states = [t.state for t in study.trials]
        completed = states.count(optuna.trial.TrialState.COMPLETE)
        failed = states.count(optuna.trial.TrialState.FAIL)
        total = completed + failed
        
        if total > 0 and failed / total > 0.2 and total >= 5:
            raise RuntimeError(f"More than 20% of trials failed. Aborting.")
            
        if completed >= n_trials:
            break
            
        study.optimize(objective, n_trials=1, catch=(Exception,))

    best_trial = select(study, plugin)
    
    out_dir = os.path.join(runs_dir, plugin.name)
    os.makedirs(out_dir, exist_ok=True)
    
    if plugin.pareto:
        front = [{'number': t.number, 'params': t.params, 'values': t.values} for t in study.best_trials]
    else:
        front = [{'number': study.best_trial.number, 'params': study.best_trial.params, 'values': study.best_trial.values}]
        
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except:
        git_hash = "unknown"
        
    def hash_file(fpath):
        if not os.path.exists(fpath): return ""
        with open(fpath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
            
    runner_hash = hash_file(__file__)
    data_hash = hash_file(os.path.join(os.path.dirname(__file__), "harness_data.py"))
    metrics_hash = hash_file(os.path.join(os.path.dirname(__file__), "harness_metrics.py"))
    
    out_json = {
        'selected_trial': best_trial.number,
        'params': best_trial.params,
        'validation_objectives': best_trial.values,
        'pareto_front': front,
        'seed': seed,
        'n_trials': n_trials,
        'git_hash': git_hash,
        'file_hashes': {'runner': runner_hash, 'data': data_hash, 'metrics': metrics_hash}
    }
    
    json_path = os.path.join(out_dir, f"{dataset}_{scenario}_fold{fold}_selected.json")
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=2)
        
    return best_trial

def select(study, plugin):
    if plugin.pareto:
        front = study.best_trials
        if len(front) == 1:
            return front[0]
            
        objs = np.array([t.values for t in front])
        normed = np.zeros_like(objs)
        for i, d in enumerate(plugin.directions):
            min_val = np.min(objs[:, i])
            max_val = np.max(objs[:, i])
            if max_val > min_val:
                normed[:, i] = (objs[:, i] - min_val) / (max_val - min_val)
            if d == 'maximize':
                normed[:, i] = 1.0 - normed[:, i]
                
        dists = np.sqrt(np.sum(normed**2, axis=1))
        best_idx = np.argmin(dists)
        return front[best_idx]
    else:
        return study.best_trial

def reselect(source_name, target_name, dataset, scenario, fold, objective_indices, directions):
    study_name = f"{source_name}_{dataset}_{scenario}_fold{fold}_v2"
    runs_dir = get_runs_dir()
    optuna_dir = os.path.join(runs_dir, "optuna")
    storage_url = f"sqlite:///{os.path.join(optuna_dir, study_name + '.db')}"
    
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise ValueError(f"No complete trials found in study {study_name}.")
        
    filtered_data = []
    for t in completed_trials:
        vals = [t.values[i] for i in objective_indices]
        filtered_data.append((t, vals))
        
    front_tuples = []
    for t, vals in filtered_data:
        dominated = False
        for other_t, other_vals in filtered_data:
            if other_t.number == t.number:
                continue
            at_least_as_good = True
            strictly_better = False
            for v_other, v_curr, d in zip(other_vals, vals, directions):
                if d == 'minimize':
                    if v_other > v_curr:
                        at_least_as_good = False
                        break
                    elif v_other < v_curr:
                        strictly_better = True
                elif d == 'maximize':
                    if v_other < v_curr:
                        at_least_as_good = False
                        break
                    elif v_other > v_curr:
                        strictly_better = True
                else:
                    raise ValueError(f"Unknown direction {d}")
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            front_tuples.append((t, vals))
            
    # Sort front tuples by trial number for deterministic tie-breaking
    front_tuples.sort(key=lambda x: x[0].number)
    
    if len(front_tuples) == 1:
        selected_trial, selected_vals = front_tuples[0]
    else:
        objs = np.array([vals for _, vals in front_tuples])
        normed = np.zeros_like(objs)
        for i, d in enumerate(directions):
            min_val = np.min(objs[:, i])
            max_val = np.max(objs[:, i])
            if max_val > min_val:
                normed[:, i] = (objs[:, i] - min_val) / (max_val - min_val)
            if d == 'maximize':
                normed[:, i] = 1.0 - normed[:, i]
                
        dists = np.sqrt(np.sum(normed**2, axis=1))
        min_dist = np.min(dists)
        # On ties take lowest trial number
        tie_indices = [idx for idx, d in enumerate(dists) if np.isclose(d, min_dist)]
        best_idx = min(tie_indices, key=lambda idx: front_tuples[idx][0].number)
        selected_trial, selected_vals = front_tuples[best_idx]
        
    source_dir = os.path.join(runs_dir, source_name)
    source_json_path = os.path.join(source_dir, f"{dataset}_{scenario}_fold{fold}_selected.json")
    if not os.path.exists(source_json_path):
        raise FileNotFoundError(f"Missing {source_json_path} to read seed.")
    with open(source_json_path, 'r') as f:
        source_json = json.load(f)
    seed = source_json['seed']
    
    front_dicts = [
        {'number': t.number, 'params': t.params, 'values': vals}
        for t, vals in front_tuples
    ]
    
    out_json = {
        'seed': seed,
        'params': selected_trial.params,
        'validation_objectives': selected_vals,
        'selected_trial': selected_trial.number,
        'source_study': study_name,
        'objective_indices': objective_indices,
        'pareto_front': front_dicts
    }
    
    target_dir = os.path.join(runs_dir, target_name)
    os.makedirs(target_dir, exist_ok=True)
    target_json_path = os.path.join(target_dir, f"{dataset}_{scenario}_fold{fold}_selected.json")
    with open(target_json_path, 'w') as f:
        json.dump(out_json, f, indent=2)
        
    return out_json

def _cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (ImportError, Exception):
        pass

def final_evaluate(plugin, dataset, scenario, fold, timing=False):
    runs_dir = get_runs_dir()
    out_dir = os.path.join(runs_dir, plugin.name)
    
    selected_path = os.path.join(out_dir, f"{dataset}_{scenario}_fold{fold}_selected.json")
    if not os.path.exists(selected_path):
        raise FileNotFoundError(f"Missing {selected_path}. Run tune_and_select first.")
        
    with open(selected_path, 'r') as f:
        selected = json.load(f)
        
    seed = selected['seed']
    params = selected['params']
    set_seed(seed)
    
    t0 = time.time()
    fd_train = load_fold(dataset, scenario, fold)
    train_part = fd_train.part('train')
    val_part = fd_train.part('val')
    
    if timing:
        _cuda_sync()
        t_fit_0 = time.perf_counter()
        state = plugin.fit(train_part, val_part, params, seed)
        _cuda_sync()
        fit_seconds = time.perf_counter() - t_fit_0
    else:
        state = plugin.fit(train_part, val_part, params, seed)
    
    fd_test = load_test(dataset, scenario, fold, fd_train)
    test_part = fd_test.part('test')
    
    if timing:
        _cuda_sync()
        t_infer_0 = time.perf_counter()
        pred_test = plugin.predict(state, test_part, n_passes=100)
        _cuda_sync()
        inference_seconds = time.perf_counter() - t_infer_0
    else:
        pred_test = plugin.predict(state, test_part, n_passes=100)
    
    has_own_surv = hasattr(plugin, 'survival_curves') and hasattr(plugin, 'survival_at_own_times')
    if not has_own_surv:
        if timing:
            _cuda_sync()
            t_base_0 = time.perf_counter()

        pred_train = plugin.predict(state, train_part, n_passes=100)
        baseline = breslow_baseline(pred_train['eta'], train_part['T'], train_part['E'])

        if timing:
            _cuda_sync()
            fit_seconds += (time.perf_counter() - t_base_0)
    
    M_inj = test_part['M_inj']
    x_true = test_part['X_true']
    mu = pred_test['mu']
    
    metrics = {}
    metrics['rmse_std'] = rmse(x_true, mu, M_inj)
    metrics['mae_std'] = mae(x_true, mu, M_inj)
    
    mu_orig = fd_test.inverse_transform(mu)
    x_true_orig = fd_test.inverse_transform(x_true)
    metrics['rmse_orig'] = rmse(x_true_orig, mu_orig, M_inj)
    metrics['mae_orig'] = mae(x_true_orig, mu_orig, M_inj)
    
    if 'sigma_E2' in pred_test and 'sigma_A2' in pred_test:
        sigma_T = np.sqrt(pred_test['sigma_E2'] + pred_test['sigma_A2'])
        metrics['picp'] = picp(x_true, mu, sigma_T, M_inj)
        metrics['mpiw'] = mpiw(sigma_T, M_inj)
        metrics['ece'] = ece(x_true, mu, sigma_T, M_inj)
        
    T_test = test_part['T']
    E_test = test_part['E']
    eta_test = pred_test['eta']
    metrics['cindex'] = cindex(T_test, E_test, eta_test)
    
    T_train = train_part['T']
    E_train = train_part['E']
    times = eval_time_grid(T_train, T_test, E_test)
    
    save_dict = {
        'mu': pred_test['mu'],
        'eta': pred_test['eta'],
        'cohort_index': test_part['cohort_index'],
        'time_grid': times
    }
    
    if has_own_surv:
        if timing:
            _cuda_sync()
            t_surv_0 = time.perf_counter()
            surv_dict = plugin.survival_curves(state, test_part, times, n_passes=100)
            S_at_T = plugin.survival_at_own_times(state, test_part, n_passes=100)
            _cuda_sync()
            inference_seconds += (time.perf_counter() - t_surv_0)
        else:
            surv_dict = plugin.survival_curves(state, test_part, times, n_passes=100)
            S_at_T = plugin.survival_at_own_times(state, test_part, n_passes=100)
        S_matrix = surv_dict['S']
        if 'S_samples' in surv_dict:
            save_dict['S_lower'] = np.percentile(surv_dict['S_samples'], 2.5, axis=0)
            save_dict['S_upper'] = np.percentile(surv_dict['S_samples'], 97.5, axis=0)
    elif 'eta_samples' in pred_test:
        if timing:
            _cuda_sync()
            t_surv_0 = time.perf_counter()

        n_passes = pred_test['eta_samples'].shape[1]
        S_passes = []
        S_at_T_passes = []
        for i in range(n_passes):
            S_p = survival_matrix(baseline, pred_test['eta_samples'][:, i], times)
            S_passes.append(S_p)
            S_at_T_p = survival_at_event_times(baseline, pred_test['eta_samples'][:, i], T_test)
            S_at_T_passes.append(S_at_T_p)
            
        S_passes = np.array(S_passes)
        S_matrix = np.mean(S_passes, axis=0)
        S_at_T = np.mean(np.array(S_at_T_passes), axis=0)
        
        save_dict['S_lower'] = np.percentile(S_passes, 2.5, axis=0)
        save_dict['S_upper'] = np.percentile(S_passes, 97.5, axis=0)

        if timing:
            _cuda_sync()
            inference_seconds += (time.perf_counter() - t_surv_0)
    else:
        if timing:
            _cuda_sync()
            t_surv_0 = time.perf_counter()

        S_matrix = survival_matrix(baseline, eta_test, times)
        S_at_T = survival_at_event_times(baseline, eta_test, T_test)

        if timing:
            _cuda_sync()
            inference_seconds += (time.perf_counter() - t_surv_0)
        
    metrics['ibs'] = ibs(T_train, E_train, T_test, E_test, S_matrix, times)
    
    auc_times_vals, mean_auc = td_auc_unweighted(T_test, E_test, eta_test, times)
    metrics['td_auc_mean'] = mean_auc
    
    chi2, pval, bins = d_calibration(S_at_T, E_test)
    metrics['dcal_chi2'] = chi2
    metrics['dcal_pval'] = pval
    
    save_dict['td_auc'] = auc_times_vals
    save_dict['dcal_bins'] = bins
    
    if hasattr(plugin, 'extra_test_metrics'):
        extra_metrics = plugin.extra_test_metrics(state, test_part)
        metrics.update(extra_metrics)
    
    if 'eta_samples' in pred_test: save_dict['eta_samples'] = pred_test['eta_samples']
    if 'sigma_E2' in pred_test: save_dict['sigma_E2'] = pred_test['sigma_E2']
    if 'sigma_A2' in pred_test: save_dict['sigma_A2'] = pred_test['sigma_A2']
    
    if timing:
        import socket
        if hasattr(plugin, 'device'):
            dev_str = str(plugin.device)
            device_name = "cuda" if "cuda" in dev_str else "cpu"
        else:
            device_name = "cpu"
            
        timing_json = {
            'fit_seconds': fit_seconds,
            'inference_seconds': inference_seconds,
            'n_train': len(T_train),
            'n_test': len(T_test),
            'hostname': socket.gethostname(),
            'device': device_name,
            'metrics': metrics
        }
        
        timing_path = os.path.join(out_dir, f"timing_{plugin.name}_{dataset}_{scenario}_fold{fold}.json")
        with open(timing_path, 'w') as f:
            json.dump(timing_json, f, indent=2)
            
        return timing_json
    
    t1 = time.time()
    
    out_json = {
        'params': params,
        'metrics': metrics,
        'n_train': len(T_train),
        'n_val': len(val_part['T']),
        'n_test': len(T_test),
        'seed': seed,
        'runtime': t1 - t0,
    }
    
    json_path = os.path.join(out_dir, f"{dataset}_{scenario}_fold{fold}_test.json")
    npz_path = os.path.join(out_dir, f"{dataset}_{scenario}_fold{fold}_test.npz")
    
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=2)
        
    np.savez_compressed(npz_path, **save_dict)
    return out_json

def summarize(plugin_name, dataset, scenario):
    runs_dir = get_runs_dir()
    out_dir = os.path.join(runs_dir, plugin_name)
    
    results = []
    for fold in range(1, 6):
        json_path = os.path.join(out_dir, f"{dataset}_{scenario}_fold{fold}_test.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing test result for fold {fold}: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
            results.append(data['metrics'])
            
    summary = {}
    keys = [k for k, v in results[0].items() if isinstance(v, (int, float))]
    
    for k in keys:
        vals = [r[k] for r in results]
        summary[k] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)),
            'folds': vals
        }
        
    summary_path = os.path.join(out_dir, f"{dataset}_{scenario}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

