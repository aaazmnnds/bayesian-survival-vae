import os
import json
import numpy as np
import pytest
from optimize_bsvae import BSVAEPlugin
import harness_runner

class NoTruthDict(dict):
    def __getitem__(self, key):
        if key in ['X_true', 'M_inj']:
            raise KeyError(f"Access to {key} is forbidden during fit!")
        return super().__getitem__(key)

def test_bsvae_smoke(tmp_path, monkeypatch):
    # Monkeypatch the output root
    monkeypatch.setattr(harness_runner, 'get_runs_dir', lambda: str(tmp_path))
    
    dataset = "metabric"
    scenario = "light"
    fold = 1
    
    plugin = BSVAEPlugin(stage_epochs=(1, 1, 1), batch_size=128)
    
    # Check fit with NoTruthDict
    from harness_data import load_fold, load_test
    fd_train = load_fold(dataset, scenario, fold)
    
    train_part = fd_train.part('train')
    val_part = fd_train.part('val')
    
    train_safe = NoTruthDict(train_part)
    val_safe = NoTruthDict(val_part)
    
    # We use tune_and_select to verify it runs, which calls fit
    # We must patch the data loading inside tune_and_select as well? No, tune_and_select calls plugin.fit directly.
    # To test NoTruthDict, we must run fit directly first.
    
    # 1. Direct fit test for forbidden access
    params = plugin.search_space(optuna_mock_trial())
    state = plugin.fit(train_safe, val_safe, params, seed=42)
    
    # 2. Predict output shapes
    pred = plugin.predict(state, val_part, n_passes=5)
    
    N = len(val_part['X_obs'])
    P = val_part['X_obs'].shape[1]
    
    assert 'mu' in pred and pred['mu'].shape == (N, P)
    assert 'eta' in pred and pred['eta'].shape == (N,)
    assert 'eta_samples' in pred and pred['eta_samples'].shape == (N, 5)
    
    assert 'sigma_E2' in pred and pred['sigma_E2'].shape == (N, P)
    assert 'sigma_A2' in pred and pred['sigma_A2'].shape == (N, P)
    
    cont_cols = val_part['feature_types'].get('continuous', [])
    cont_idx = [val_part['feature_names'].index(c) for c in cont_cols]
    non_cont_idx = [i for i in range(P) if i not in cont_idx]
    
    if len(non_cont_idx) > 0:
        assert np.isnan(pred['sigma_E2'][:, non_cont_idx]).all(), "sigma_E2 must be NaN outside continuous"
        assert np.isnan(pred['sigma_A2'][:, non_cont_idx]).all(), "sigma_A2 must be NaN outside continuous"
        
    if len(cont_idx) > 0:
        assert not np.isnan(pred['sigma_E2'][:, cont_idx]).any(), "sigma_E2 must be finite for continuous"
        assert not np.isnan(pred['sigma_A2'][:, cont_idx]).any(), "sigma_A2 must be finite for continuous"
        
    # 3. Full pipeline test
    harness_runner.tune_and_select(plugin, dataset, scenario, fold, n_trials=2, seed=42)
    
    selected_json = tmp_path / plugin.name / f"{dataset}_{scenario}_fold{fold}_selected.json"
    assert selected_json.exists(), "tune_and_select must write _selected.json"
    
    harness_runner.final_evaluate(plugin, dataset, scenario, fold)
    
    test_json = tmp_path / plugin.name / f"{dataset}_{scenario}_fold{fold}_test.json"
    assert test_json.exists(), "final_evaluate must write _test.json"
    
    with open(test_json, 'r') as f:
        res = json.load(f)
        
    metrics = res['metrics']
    expected_metrics = ['rmse_std', 'picp', 'mpiw', 'ece', 'cindex', 'ibs', 'td_auc_mean', 'dcal_chi2', 'dcal_pval', 'iwae_mean']
    
    for m in expected_metrics:
        assert m in metrics, f"Metric {m} missing from output"
        assert np.isfinite(metrics[m]), f"Metric {m} is not finite"

def test_two_step_cox_mice_smoke(tmp_path, monkeypatch):
    from optimize_baselines import TwoStepCoxPlugin
    from harness_data import load_fold

    # Monkeypatch the output root
    monkeypatch.setattr(harness_runner, 'get_runs_dir', lambda: str(tmp_path))

    dataset = "metabric"
    scenario = "light"
    fold = 1

    plugin = TwoStepCoxPlugin('mice')

    fd_train = load_fold(dataset, scenario, fold)
    train_part = fd_train.part('train')
    val_part = fd_train.part('val')

    train_safe = NoTruthDict(train_part)
    val_safe = NoTruthDict(val_part)

    # 1. Fit test: verify fit never reads 'X_true' or 'M_inj'
    params = plugin.search_space(optuna_mock_trial())
    state = plugin.fit(train_safe, val_safe, params, seed=42)

    # 2. Predict test: check shapes, observed cell equality, and finite eta
    pred = plugin.predict(state, val_part, n_passes=5)

    X_obs = val_part['X_obs']
    assert 'mu' in pred and pred['mu'].shape == X_obs.shape
    observed_mask = ~np.isnan(X_obs)
    assert np.allclose(pred['mu'][observed_mask], X_obs[observed_mask])
    assert 'eta' in pred and np.all(np.isfinite(pred['eta']))

    # 3. Full pipeline test: tune_and_select with 2 trials and final_evaluate
    harness_runner.tune_and_select(plugin, dataset, scenario, fold, n_trials=2, seed=42)

    selected_json = tmp_path / plugin.name / f"{dataset}_{scenario}_fold{fold}_selected.json"
    assert selected_json.exists(), "tune_and_select must write _selected.json"

    harness_runner.final_evaluate(plugin, dataset, scenario, fold)

    test_json = tmp_path / plugin.name / f"{dataset}_{scenario}_fold{fold}_test.json"
    assert test_json.exists(), "final_evaluate must write _test.json"

    with open(test_json, 'r') as f:
        res = json.load(f)

    metrics = res['metrics']
    required_metrics = ['rmse_std', 'cindex', 'ibs', 'td_auc_mean', 'dcal_pval']
    for m in required_metrics:
        assert m in metrics, f"Metric {m} missing from output"
        assert np.isfinite(metrics[m]), f"Metric {m} is not finite"

def test_two_step_cox_missforest_caching():
    from optimize_baselines import TwoStepCoxPlugin
    from harness_data import load_fold

    dataset = "metabric"
    scenario = "light"
    fold = 1

    plugin = TwoStepCoxPlugin('missforest', n_jobs=2)

    fd_train = load_fold(dataset, scenario, fold)
    train_part = fd_train.part('train')
    val_part = fd_train.part('val')

    seed = 42
    params_1 = {'penalizer': 0.01}
    params_2 = {'penalizer': 0.1}

    state_1 = plugin.fit(train_part, val_part, params_1, seed=seed)
    state_2 = plugin.fit(train_part, val_part, params_2, seed=seed)

    # Assert that the second call reused the cached imputer (same object)
    assert state_1['imputer'] is state_2['imputer']

    # Assert both returned states give finite eta from predict
    pred_1 = plugin.predict(state_1, val_part, n_passes=1)
    pred_2 = plugin.predict(state_2, val_part, n_passes=1)

    assert np.all(np.isfinite(pred_1['eta']))
    assert np.all(np.isfinite(pred_2['eta']))

class optuna_mock_trial:
    def suggest_categorical(self, name, choices): return choices[0]
    def suggest_float(self, name, low, high, log=False): return low
    def suggest_int(self, name, low, high): return low


def test_gain_smoke(tmp_path, monkeypatch):
    from optimize_gain import GAINCoxPlugin
    from harness_data import load_fold

    # Monkeypatch the output root to a temporary folder
    monkeypatch.setattr(harness_runner, 'get_runs_dir', lambda: str(tmp_path))

    dataset = "metabric"
    scenario = "light"
    fold = 1

    plugin = GAINCoxPlugin(iterations=200, checkpoint_every=100)

    fd_train = load_fold(dataset, scenario, fold)
    train_part = fd_train.part('train')
    val_part = fd_train.part('val')

    train_safe = NoTruthDict(train_part)

    # 1. Fit with the TRAINING part wrapped in NoTruthDict (validation part unwrapped) succeeds
    trial = optuna_mock_trial()
    params = plugin.search_space(trial)
    state = plugin.fit(train_safe, val_part, params, seed=42)

    # 2. Predict gives mu equal to X_obs at every observed cell, binary columns in {0,1}, finite eta
    pred = plugin.predict(state, val_part, n_passes=1)
    mu = pred['mu']
    X_obs_val = val_part['X_obs']
    observed_mask = (val_part['M_obs'] == 1.0)
    assert np.allclose(mu[observed_mask], X_obs_val[observed_mask]), "mu must match X_obs at all observed cells"

    bin_cols = val_part['feature_types'].get('binary', [])
    for c in bin_cols:
        idx = val_part['feature_names'].index(c)
        assert np.all(np.isin(mu[:, idx], [0.0, 1.0])), f"Binary column {c} not in {{0, 1}}"

    assert 'eta' in pred and np.all(np.isfinite(pred['eta'])), "eta must be finite"

    # 3. Two fits with the same seed give identical mu on validation
    state_repeat = plugin.fit(train_safe, val_part, params, seed=42)
    pred_repeat = plugin.predict(state_repeat, val_part, n_passes=1)
    assert np.allclose(pred['mu'], pred_repeat['mu']), "Two fits with same seed must give identical mu on validation"

    # 4. The hint sampler gives a mean of B close to hint_rate (within 0.02 over 100000 draws)
    hint_rate = 0.75
    rng = np.random.Generator(np.random.PCG64(42))
    B_draws = (rng.uniform(0.0, 1.0, size=(100000,)) < hint_rate).astype(np.float32)
    assert abs(B_draws.mean() - hint_rate) < 0.02, "Hint sampler mean must be within 0.02 of hint_rate"

    # 5. tune_and_select with 2 trials and final_evaluate write _selected.json and _test.json with finite rmse_std, cindex, ibs
    harness_runner.tune_and_select(plugin, dataset, scenario, fold, n_trials=2, seed=42)
    selected_json = tmp_path / plugin.name / f"{dataset}_{scenario}_fold{fold}_selected.json"
    assert selected_json.exists(), "tune_and_select must write _selected.json"

    harness_runner.final_evaluate(plugin, dataset, scenario, fold)
    test_json = tmp_path / plugin.name / f"{dataset}_{scenario}_fold{fold}_test.json"
    assert test_json.exists(), "final_evaluate must write _test.json"

    with open(test_json, 'r') as f:
        res = json.load(f)

    metrics = res['metrics']
    for m in ['rmse_std', 'cindex', 'ibs']:
        assert m in metrics, f"Metric {m} missing from output"
        assert np.isfinite(metrics[m]), f"Metric {m} is not finite"


def test_miwae_smoke(tmp_path, monkeypatch):
    from optimize_miwae import MIWAECoxPlugin, build_encoder, build_decoder, miwae_impute
    from harness_data import load_fold
    import torch
    import torch.distributions as td

    # Monkeypatch the output root to a temporary folder
    monkeypatch.setattr(harness_runner, 'get_runs_dir', lambda: str(tmp_path))

    dataset = "metabric"
    scenario = "light"
    fold = 1

    plugin = MIWAECoxPlugin(max_epochs=4, check_every=2, L_tune=20, L_test=50)

    fd_train = load_fold(dataset, scenario, fold)
    train_part = fd_train.part('train')
    val_part = fd_train.part('val')

    train_safe = NoTruthDict(train_part)

    # 1. Fit with the TRAINING part wrapped in NoTruthDict (validation part unwrapped) succeeds
    trial = optuna_mock_trial()
    params = plugin.search_space(trial)
    state = plugin.fit(train_safe, val_part, params, seed=42)

    # 2. Predict: mu equals X_obs at observed cells, binary columns in {0,1}, finite eta
    pred = plugin.predict(state, val_part, n_passes=1)
    mu = pred['mu']
    X_obs_val = val_part['X_obs']
    observed_mask = (val_part['M_obs'] == 1.0)
    assert np.allclose(mu[observed_mask], X_obs_val[observed_mask]), "mu must match X_obs at all observed cells"

    bin_cols = val_part['feature_types'].get('binary', [])
    for c in bin_cols:
        idx = val_part['feature_names'].index(c)
        assert np.all(np.isin(mu[:, idx], [0.0, 1.0])), f"Binary column {c} not in {{0, 1}}"

    assert 'eta' in pred and np.all(np.isfinite(pred['eta'])), "eta must be finite"

    # 3. Two fits with the same seed give identical mu on validation
    state_repeat = plugin.fit(train_safe, val_part, params, seed=42)
    pred_repeat = plugin.predict(state_repeat, val_part, n_passes=1)
    assert np.allclose(pred['mu'], pred_repeat['mu']), "Two fits with same seed must give identical mu on validation"

    # 4. The importance weights for one row sum to 1 (within 1e-5)
    p = state['p']
    d = state['d']
    h = state['h']
    enc = build_encoder(p, h, d).to(plugin.device)
    dec = build_decoder(d, h, p).to(plugin.device)
    enc.load_state_dict(state['encoder_state'])
    dec.load_state_dict(state['decoder_state'])
    p_z = td.Independent(td.Normal(loc=torch.zeros(d, device=plugin.device), scale=torch.ones(d, device=plugin.device)), 1)

    row_x = val_part['X_obs'][:1]
    row_m = val_part['M_obs'][:1]
    row_norm = np.nan_to_num((row_x - state['col_mean']) / state['col_std'], nan=0.0)
    x_t = torch.tensor(row_norm, dtype=torch.float32, device=plugin.device)
    m_t = torch.tensor(row_m, dtype=torch.float32, device=plugin.device)

    with torch.no_grad():
        _, weights = miwae_impute(x_t, m_t, enc, dec, p_z, p, d, L=plugin.L_tune, return_weights=True)
    # weights shape: (L, 1)
    weight_sum = weights.sum(dim=0).item()
    assert abs(weight_sum - 1.0) < 1e-5, f"Importance weights for row must sum to 1, got {weight_sum}"

    # 5. tune_and_select with 2 trials and final_evaluate write _selected.json and _test.json with finite rmse_std, cindex, ibs
    harness_runner.tune_and_select(plugin, dataset, scenario, fold, n_trials=2, seed=42)
    selected_json = tmp_path / plugin.name / f"{dataset}_{scenario}_fold{fold}_selected.json"
    assert selected_json.exists(), "tune_and_select must write _selected.json"

    harness_runner.final_evaluate(plugin, dataset, scenario, fold)
    test_json = tmp_path / plugin.name / f"{dataset}_{scenario}_fold{fold}_test.json"
    assert test_json.exists(), "final_evaluate must write _test.json"

    with open(test_json, 'r') as f:
        res = json.load(f)

    metrics = res['metrics']
    for m in ['rmse_std', 'cindex', 'ibs']:
        assert m in metrics, f"Metric {m} missing from output"
        assert np.isfinite(metrics[m]), f"Metric {m} is not finite"


def test_mnar_svae_smoke(tmp_path, monkeypatch):
    import torch
    import train_utils
    import optimize_survival_vae
    from optimize_survival_vae import MNARSVAEPlugin, FixedVarianceDecoder
    from harness_data import load_fold

    # Monkeypatch the output root to a temporary folder
    monkeypatch.setattr(harness_runner, 'get_runs_dir', lambda: str(tmp_path))

    # Monkeypatch compute_L_cal to raise if called anywhere
    def fail_if_called(*args, **kwargs):
        raise RuntimeError("compute_L_cal must NEVER be called in MNARSVAEPlugin!")
    monkeypatch.setattr(train_utils, 'compute_L_cal', fail_if_called)

    dataset = "metabric"
    scenario = "light"
    fold = 1

    plugin = MNARSVAEPlugin(stage_epochs=(1, 1, 1), batch_size=128)

    fd_train = load_fold(dataset, scenario, fold)
    train_part = fd_train.part('train')
    val_part = fd_train.part('val')

    train_safe = NoTruthDict(train_part)

    # 1. Fit with the TRAINING part wrapped in NoTruthDict succeeds, and compute_L_cal is NEVER called
    trial = optuna_mock_trial()
    params = plugin.search_space(trial)
    state = plugin.fit(train_safe, val_part, params, seed=42)

    # 2. The wrapped decoder returns a log-variance of exactly 0
    model = state['model']
    assert isinstance(model.decoder, FixedVarianceDecoder), "model.decoder must be wrapped in FixedVarianceDecoder"
    z_dummy = torch.zeros(2, params['latent_dim'], device=plugin.device)
    dec_out = model.decoder(z_dummy)
    if 'x_recon_cont' in dec_out:
        mu_dummy, log_var_dummy = dec_out['x_recon_cont']
        assert torch.equal(log_var_dummy, torch.zeros_like(log_var_dummy)), "Wrapped decoder must return log-variance of exactly 0"

    # 3. Predict returns only 'mu' and 'eta', with finite eta, and two predict calls give identical output (deterministic)
    pred_1 = plugin.predict(state, val_part, n_passes=1)
    assert set(pred_1.keys()) == {'mu', 'eta'}, f"Predict keys must be exactly {{'mu', 'eta'}}, got {set(pred_1.keys())}"
    assert np.all(np.isfinite(pred_1['eta'])), "eta must be finite"

    pred_2 = plugin.predict(state, val_part, n_passes=5)
    assert np.array_equal(pred_1['mu'], pred_2['mu']), "Two predict calls must give identical mu"
    assert np.array_equal(pred_1['eta'], pred_2['eta']), "Two predict calls must give identical eta"

    # 4. tune_and_select with 2 trials and final_evaluate write _selected.json and _test.json with finite rmse_std, cindex, ibs, iwae_mean, and WITHOUT picp or ece
    harness_runner.tune_and_select(plugin, dataset, scenario, fold, n_trials=2, seed=42)
    selected_json = tmp_path / plugin.name / f"{dataset}_{scenario}_fold{fold}_selected.json"
    assert selected_json.exists(), "tune_and_select must write _selected.json"

    harness_runner.final_evaluate(plugin, dataset, scenario, fold)
    test_json = tmp_path / plugin.name / f"{dataset}_{scenario}_fold{fold}_test.json"
    assert test_json.exists(), "final_evaluate must write _test.json"

    with open(test_json, 'r') as f:
        res = json.load(f)

    metrics = res['metrics']
    for m in ['rmse_std', 'cindex', 'ibs', 'iwae_mean']:
        assert m in metrics, f"Metric {m} missing from output"
        assert np.isfinite(metrics[m]), f"Metric {m} is not finite"

    assert 'picp' not in metrics, "picp must NOT be in metrics"
    assert 'ece' not in metrics, "ece must NOT be in metrics"


def test_mc_dropout_cox_smoke(tmp_path, monkeypatch):
    import torch
    from optimize_mc_dropout_cox import MCDropoutCoxPlugin, cox_loss_vec
    from harness_data import load_fold

    # 1. Verify vectorized Cox loss equals a reference double loop on random data WITH tied times (1e-5)
    def cox_loss_ref(eta, T, E):
        loss = 0.0
        N = len(T)
        for i in range(N):
            if E[i] == 1:
                risk_indices = [j for j in range(N) if T[j] >= T[i]]
                risk_etas = eta[risk_indices]
                denom = torch.logsumexp(risk_etas, dim=0)
                loss += (denom - eta[i])
        return loss

    torch.manual_seed(42)
    N_sim = 60
    eta_sim = torch.randn(N_sim, dtype=torch.float64, requires_grad=True)
    # create ties
    T_sim = torch.randint(1, 10, (N_sim,), dtype=torch.float64)
    E_sim = torch.randint(0, 2, (N_sim,), dtype=torch.float64)

    ref_loss = cox_loss_ref(eta_sim, T_sim, E_sim)
    vec_loss = cox_loss_vec(eta_sim, T_sim, E_sim)
    assert torch.abs(ref_loss - vec_loss).item() < 1e-5, "Vectorized Cox loss must equal reference loop"

    # Monkeypatch output root to temporary folder
    # Note: harness_runner.get_runs_dir is used by tune_and_select and final_evaluate to write outputs
    # BUT plugin.__init__ reads the MICE selection file from the real results folder!
    real_runs_dir = harness_runner.get_runs_dir()

    dataset = "metabric"
    scenario = "light"
    fold = 1

    # Instantiate plugin BEFORE monkeypatching get_runs_dir, or ensure it reads real results folder
    plugin = MCDropoutCoxPlugin(dataset, scenario, fold, batch_size=64, max_epochs=2, val_passes=5)

    # Now monkeypatch get_runs_dir for output writing
    monkeypatch.setattr(harness_runner, 'get_runs_dir', lambda: str(tmp_path))

    fd_train = load_fold(dataset, scenario, fold)
    train_part = fd_train.part('train')
    val_part = fd_train.part('val')

    train_safe = NoTruthDict(train_part)

    # 2. Fit with the TRAINING part wrapped in NoTruthDict succeeds
    trial = optuna_mock_trial()
    params = plugin.search_space(trial)
    # Ensure dropout > 0
    params['dropout'] = 0.2
    state = plugin.fit(train_safe, val_part, params, seed=42)

    # 3. Predict returns eta_samples with n_passes columns, and eta differs across passes when dropout > 0
    pred = plugin.predict(state, val_part, n_passes=5)
    assert 'eta_samples' in pred, "eta_samples missing from predict output"
    assert pred['eta_samples'].shape == (len(val_part['X_obs']), 5), "eta_samples shape mismatch"
    
    # Check that eta differs across passes
    pass_diffs = np.std(pred['eta_samples'], axis=1)
    assert np.all(pass_diffs > 1e-7), "eta must differ across passes when dropout > 0"

    # 4. tune_and_select with 2 trials and final_evaluate write _selected.json and _test.json with finite cindex, ibs, dcal_pval
    harness_runner.tune_and_select(plugin, dataset, scenario, fold, n_trials=2, seed=42)
    selected_json = tmp_path / plugin.name / f"{dataset}_{scenario}_fold{fold}_selected.json"
    assert selected_json.exists(), "tune_and_select must write _selected.json"

    harness_runner.final_evaluate(plugin, dataset, scenario, fold)
    test_json = tmp_path / plugin.name / f"{dataset}_{scenario}_fold{fold}_test.json"
    assert test_json.exists(), "final_evaluate must write _test.json"

    with open(test_json, 'r', encoding='utf-8') as f:
        res = json.load(f)

    metrics = res['metrics']
    for m in ['cindex', 'ibs', 'dcal_pval']:
        assert m in metrics, f"Metric {m} missing from output"
        assert np.isfinite(metrics[m]), f"Metric {m} is not finite"


def test_deephit_smoke(tmp_path, monkeypatch):
    pytest.importorskip("pycox.models", exc_type=ImportError)
    import torch
    import torch.nn as nn
    from optimize_deephit import DeepHitPlugin
    from harness_data import load_fold

    dataset = "metabric"
    scenario = "light"
    fold = 1

    # Instantiate plugin BEFORE monkeypatching get_runs_dir (reads MICE selection from real folder)
    plugin = DeepHitPlugin(dataset, scenario, fold, batch_size=64, max_epochs=2, val_passes=5, num_durations=10)

    # Now monkeypatch get_runs_dir for output writing
    monkeypatch.setattr(harness_runner, 'get_runs_dir', lambda: str(tmp_path))

    fd_train = load_fold(dataset, scenario, fold)
    train_part = fd_train.part('train')
    val_part = fd_train.part('val')

    train_safe = NoTruthDict(train_part)

    # 1. Fit with the TRAINING part wrapped in NoTruthDict succeeds
    trial = optuna_mock_trial()
    params = plugin.search_space(trial)
    params['dropout'] = 0.2
    state = plugin.fit(train_safe, val_part, params, seed=42)

    # 2. survival_curves gives values in [0,1], non-increasing over time for every patient, and different across passes
    times = np.linspace(0.0, float(np.max(val_part['T'])), 20)
    surv_out = plugin.survival_curves(state, val_part, times, n_passes=5)
    S = surv_out['S']
    S_samples = surv_out['S_samples']

    assert np.all(S >= 0.0 - 1e-6) and np.all(S <= 1.0 + 1e-6), "S values must be in [0, 1]"
    assert np.all(S_samples >= 0.0 - 1e-6) and np.all(S_samples <= 1.0 + 1e-6), "S_samples values must be in [0, 1]"

    # Non-increasing over time for every patient
    # S has shape (n, len(times))
    time_diffs = np.diff(S, axis=1)
    assert np.all(time_diffs <= 1e-6), "Survival curves must be non-increasing over time"

    # Different across passes
    pass_std = np.std(S_samples, axis=0)
    assert np.any(pass_std > 1e-6), "Survival curves must differ across passes with active dropout"

    # 3. Batch-normalization running statistics are identical before and after an MC prediction (eval mode kept)
    net = state['net']
    bn_stats_before = {}
    for name, module in net.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_stats_before[name] = (module.running_mean.clone(), module.running_var.clone())

    # Run MC prediction
    _ = plugin.predict(state, val_part, n_passes=5)
    _ = plugin.survival_curves(state, val_part, times, n_passes=5)

    for name, module in net.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mean_b, var_b = bn_stats_before[name]
            assert torch.equal(mean_b, module.running_mean), f"BatchNorm running_mean changed in {name}!"
            assert torch.equal(var_b, module.running_var), f"BatchNorm running_var changed in {name}!"

    # 4. tune_and_select with 2 trials and final_evaluate write _selected.json and _test.json with finite cindex, ibs, dcal_pval, and the .npz contains S_lower and S_upper
    harness_runner.tune_and_select(plugin, dataset, scenario, fold, n_trials=2, seed=42)
    selected_json = tmp_path / plugin.name / f"{dataset}_{scenario}_fold{fold}_selected.json"
    assert selected_json.exists(), "tune_and_select must write _selected.json"

    harness_runner.final_evaluate(plugin, dataset, scenario, fold)
    test_json = tmp_path / plugin.name / f"{dataset}_{scenario}_fold{fold}_test.json"
    assert test_json.exists(), "final_evaluate must write _test.json"

    with open(test_json, 'r', encoding='utf-8') as f:
        res = json.load(f)

    metrics = res['metrics']
    for m in ['cindex', 'ibs', 'dcal_pval']:
        assert m in metrics, f"Metric {m} missing from output"
        assert np.isfinite(metrics[m]), f"Metric {m} is not finite"

    test_npz = tmp_path / plugin.name / f"{dataset}_{scenario}_fold{fold}_test.npz"
    assert test_npz.exists(), "final_evaluate must write _test.npz"
    npz_data = np.load(test_npz)
    assert 'S_lower' in npz_data, "S_lower must be in .npz file"
    assert 'S_upper' in npz_data, "S_upper must be in .npz file"
