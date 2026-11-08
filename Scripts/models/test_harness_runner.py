import os
import json
import numpy as np
import pytest

from harness_runner import tune_and_select, select, final_evaluate, summarize
from plugin_reference import MeanImputeCox
import harness_data

def test_harness_runner(tmp_path, monkeypatch):
    def mock_get_runs_dir():
        return str(tmp_path)
    import harness_runner
    monkeypatch.setattr(harness_runner, 'get_runs_dir', mock_get_runs_dir)
    
    plugin = MeanImputeCox()
    dataset = "metabric"
    scenario = "light"
    fold = 1
    n_trials = 3
    seed = 0
    
    splits_path = os.path.join(os.path.dirname(__file__), "..", "..", "Results", "final", "cv_splits", f"cv_splits_{dataset}.json")
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    test_indices = set(splits[f"fold_{fold}"]['test'])
    
    original_fit = plugin.fit
    original_predict = plugin.predict
    
    seen_indices = set()
    
    def spy_fit(train, val, params, seed):
        seen_indices.update(train['cohort_index'])
        seen_indices.update(val['cohort_index'])
        return original_fit(train, val, params, seed)
        
    def spy_predict(state, part, n_passes):
        seen_indices.update(part['cohort_index'])
        return original_predict(state, part, n_passes)
        
    monkeypatch.setattr(plugin, 'fit', spy_fit)
    monkeypatch.setattr(plugin, 'predict', spy_predict)
    
    best_trial = tune_and_select(plugin, dataset, scenario, fold, n_trials, seed)
    
    assert len(seen_indices & test_indices) == 0, "Test data leaked during tune"
    
    # Test rerunning with different config raises ValueError
    with pytest.raises(ValueError):
        tune_and_select(plugin, dataset, scenario, fold, n_trials + 1, seed)
    
    monkeypatch.undo()
    monkeypatch.setattr(harness_runner, 'get_runs_dir', mock_get_runs_dir)
    
    original_load_test = harness_runner.load_test
    load_test_calls = 0
    def spy_load_test(d, s, f, fd_train):
        nonlocal load_test_calls
        load_test_calls += 1
        return original_load_test(d, s, f, fd_train)
        
    monkeypatch.setattr(harness_runner, 'load_test', spy_load_test)
    
    out1 = final_evaluate(plugin, dataset, scenario, fold)
    
    assert load_test_calls == 1, "load_test should be called exactly once in final_evaluate"
    
    out_dir = os.path.join(mock_get_runs_dir(), plugin.name)
    test_json = os.path.join(out_dir, f"{dataset}_{scenario}_fold{fold}_test.json")
    test_npz = os.path.join(out_dir, f"{dataset}_{scenario}_fold{fold}_test.npz")
    
    assert os.path.exists(test_json)
    assert os.path.exists(test_npz)
    
    with open(test_json, 'r') as f:
        res = json.load(f)
        
    required_metrics = ['rmse_std', 'mae_std', 'rmse_orig', 'mae_orig', 'cindex', 'ibs', 'td_auc_mean', 'dcal_chi2', 'dcal_pval']
    for m in required_metrics:
        assert m in res['metrics']
        
    npz = np.load(test_npz)
    assert 'mu' in npz
    assert 'eta' in npz
    assert 'cohort_index' in npz
    assert 'time_grid' in npz
    assert 'td_auc' in npz
    assert 'dcal_bins' in npz
    
    # Test final_evaluate missing json
    bad_fold = 2
    with pytest.raises(FileNotFoundError):
        final_evaluate(plugin, dataset, scenario, bad_fold)
        
    out2 = final_evaluate(plugin, dataset, scenario, fold)
    
    for k in out1['metrics']:
        assert np.isclose(out1['metrics'][k], out2['metrics'][k]), f"Metric {k} differs on re-run"

def test_select_pareto():
    class DummyPlugin:
        def __init__(self):
            self.pareto = True
            self.directions = ['maximize', 'minimize', 'maximize']
            
    plugin = DummyPlugin()
    
    class DummyTrial:
        def __init__(self, values, params):
            self.values = values
            self.params = params
            
    t1 = DummyTrial([10.0, 5.0, 100.0], {'id': 1})
    t2 = DummyTrial([20.0, 1.0, 50.0], {'id': 2})
    t3 = DummyTrial([30.0, 10.0, 150.0], {'id': 3})
    
    class DummyStudy:
        def __init__(self):
            self.best_trials = [t1, t2, t3]
            
    study = DummyStudy()
    
    best = select(study, plugin)
    assert best.params['id'] == 3

def test_config_hash_file_dependency(monkeypatch):
    import harness_runner
    from plugin_reference import MeanImputeCox
    import builtins
    
    plugin = MeanImputeCox()
    
    hash1 = harness_runner.get_config_hash(plugin, 3, 0)
    
    original_open = builtins.open
    def mock_open(fpath, mode='r', *args, **kwargs):
        if 'losses.py' in str(fpath) and 'b' in mode:
            from io import BytesIO
            return BytesIO(b"modified_content")
        return original_open(fpath, mode, *args, **kwargs)
        
    monkeypatch.setattr(builtins, 'open', mock_open)
    
    hash2 = harness_runner.get_config_hash(plugin, 3, 0)
    assert hash1 != hash2, "Config hash must change when hashed files are modified"


def test_final_evaluate_plugin_with_own_survival_curves(tmp_path, monkeypatch):
    import harness_runner
    from plugin_reference import MeanImputeCox
    
    monkeypatch.setattr(harness_runner, 'get_runs_dir', lambda: str(tmp_path))
    
    def breslow_should_not_be_called(*args, **kwargs):
        raise AssertionError("breslow_baseline must NEVER be called for plugins providing their own survival curves!")
        
    monkeypatch.setattr(harness_runner, 'breslow_baseline', breslow_should_not_be_called)
    
    class OwnSurvivalDummyPlugin(MeanImputeCox):
        def survival_curves(self, state, part, times, n_passes):
            n = len(part['X_obs'])
            n_t = len(times)
            # return constant survival 0.5 with samples
            S_samples = np.full((n_passes, n, n_t), 0.5, dtype=np.float64)
            return {
                'S': np.full((n, n_t), 0.5, dtype=np.float64),
                'S_samples': S_samples
            }
            
        def survival_at_own_times(self, state, part, n_passes):
            n = len(part['X_obs'])
            return np.full((n,), 0.5, dtype=np.float64)
            
    plugin = OwnSurvivalDummyPlugin()
    dataset = "metabric"
    scenario = "light"
    fold = 1
    
    tune_and_select(plugin, dataset, scenario, fold, n_trials=1, seed=0)
    out = final_evaluate(plugin, dataset, scenario, fold)
    
    assert 'ibs' in out['metrics'] and np.isfinite(out['metrics']['ibs'])
    assert 'dcal_pval' in out['metrics'] and np.isfinite(out['metrics']['dcal_pval'])
    
    test_npz = os.path.join(str(tmp_path), plugin.name, f"{dataset}_{scenario}_fold{fold}_test.npz")
    npz = np.load(test_npz)
    assert 'S_lower' in npz and npz['S_lower'].shape == (len(npz['cohort_index']), len(npz['time_grid']))
    assert 'S_upper' in npz and npz['S_upper'].shape == (len(npz['cohort_index']), len(npz['time_grid']))
    assert np.allclose(npz['S_lower'], 0.5)
    assert np.allclose(npz['S_upper'], 0.5)
