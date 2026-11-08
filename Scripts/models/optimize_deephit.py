import os
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torchtuples as tt
import pycox
from pycox.models import DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

import harness_metrics
import harness_runner
from optimize_mc_dropout_cox import MCDropoutCoxPlugin

def pad_col(input, val=0, where='end'):
    """
    Adds a column of val at the end of input (quoted from pycox.models.utils.pad_col).
    """
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == 'end':
        return torch.cat([input, pad], dim=1)
    elif where == 'start':
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need where to be 'start' or 'end', got {where}")

def raw_to_surv(preds_t):
    """
    Computes survival at discrete cut points from raw network output exactly as pycox PMFBase does:
    pmf = pad_col(preds).softmax(1)[:, :-1]
    surv = 1 - pmf.cumsum(1)
    """
    pmf = pad_col(preds_t, val=0, where='end').softmax(1)[:, :-1]
    surv = 1.0 - pmf.cumsum(1)
    return surv

class DeepHitPlugin(MCDropoutCoxPlugin):
    def __init__(self, dataset, scenario, fold, batch_size=256, max_epochs=100, val_passes=20, num_durations=10, device=None):
        super().__init__(dataset, scenario, fold, batch_size=batch_size, max_epochs=max_epochs, val_passes=val_passes, device=device)
        self.name = 'deephit'
        self.directions = ['maximize']
        self.pareto = False
        self.num_durations = num_durations

    def search_space(self, trial):
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.05, 0.5),
            'hidden': trial.suggest_categorical('hidden', ['32_32', '64_64', '128_128'])
        }

    def _hidden_layers(self, hidden_str):
        if hidden_str == '32_32':
            return [32, 32]
        elif hidden_str == '64_64':
            return [64, 64]
        elif hidden_str == '128_128':
            return [128, 128]
        elif isinstance(hidden_str, (list, tuple)):
            return list(hidden_str)
        else:
            return [int(x) for x in str(hidden_str).split('_')]

    def fit(self, train, val, params, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # MICE imputation exactly as the parent (fit on train X_obs ONLY)
        imputer = self._build_mice_imputer(seed)
        imputer.fit(train['X_obs'])

        X_train_imp = self._impute_part(imputer, train).astype('float32')
        X_val_imp = self._impute_part(imputer, val).astype('float32')

        P = X_train_imp.shape[1]

        # Discretize event times on TRAINING T and E only
        labtrans = LabTransDiscreteTime(self.num_durations)
        y_train = labtrans.fit_transform(train['T'], train['E'])
        cuts = labtrans.cuts

        hidden_layers = self._hidden_layers(params['hidden'])
        dropout = params['dropout']
        lr = params['learning_rate']

        net = tt.practical.MLPVanilla(
            in_features=P,
            num_nodes=hidden_layers,
            out_features=labtrans.out_features,
            batch_norm=True,
            dropout=dropout
        )

        model = DeepHitSingle(
            net,
            tt.optim.Adam(lr),
            alpha=0.2,
            sigma=0.1,
            duration_index=cuts,
            device=self.device
        )

        T_val = val['T']
        E_val = val['E']
        X_v_t = torch.tensor(X_val_imp, dtype=torch.float32, device=self.device)

        best_cindex = -float('inf')
        best_net_state = copy.deepcopy(model.net.state_dict())

        for epoch in range(self.max_epochs):
            model.fit(X_train_imp, y_train, self.batch_size, epochs=1, verbose=False)

            # Validation C-index evaluation using val_passes MC passes
            net.eval()
            for m in net.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

            passes_S = []
            with torch.no_grad():
                for _ in range(self.val_passes):
                    preds_raw = net(X_v_t)
                    surv_pass = raw_to_surv(preds_raw).cpu().numpy()
                    passes_S.append(surv_pass)

            S_mean_val = np.mean(passes_S, axis=0) # (N_val, num_durations)
            diffs = np.diff(cuts) # (num_durations - 1,)
            rmst_val = np.sum(S_mean_val[:, :-1] * diffs, axis=1) # (N_val,)
            val_eta = -rmst_val

            val_c = harness_metrics.cindex(T_val, E_val, val_eta)
            if np.isnan(val_c):
                val_c = 0.0

            if val_c > best_cindex:
                best_cindex = val_c
                best_net_state = copy.deepcopy(model.net.state_dict())

        net.load_state_dict(best_net_state)

        return {
            'imputer': imputer,
            'net': net,
            'cuts': cuts,
            'P': P,
            'params': params
        }

    def _mc_surv_cuts(self, state, part, n_passes):
        imputer = state['imputer']
        net = state['net']
        mu = self._impute_part(imputer, part)

        X_t = torch.tensor(mu, dtype=torch.float32, device=self.device)

        net.eval()
        for m in net.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        passes_S = []
        with torch.no_grad():
            for _ in range(n_passes):
                preds_raw = net(X_t)
                surv_pass = raw_to_surv(preds_raw).cpu().numpy()
                passes_S.append(surv_pass)

        # S_samples_cuts: (n_passes, n, num_durations)
        S_samples_cuts = np.array(passes_S, dtype=np.float64)
        return mu, S_samples_cuts

    def predict(self, state, part, n_passes):
        mu, S_samples_cuts = self._mc_surv_cuts(state, part, n_passes)
        cuts = state['cuts']
        S_mean_cuts = np.mean(S_samples_cuts, axis=0) # (n, num_durations)

        diffs = np.diff(cuts)
        rmst = np.sum(S_mean_cuts[:, :-1] * diffs, axis=1)
        eta = -rmst

        return {
            'mu': mu,
            'eta': eta
        }

    def survival_curves(self, state, part, times, n_passes):
        mu, S_samples_cuts = self._mc_surv_cuts(state, part, n_passes)
        cuts = state['cuts']
        # S_samples_cuts: (n_passes, n, num_durations)
        # S(t) step function on cuts: S at largest cut <= t, 1 if t < cuts[0]
        idx = np.searchsorted(cuts, times, side='right') - 1
        below = (idx == -1)

        n_p, n, _ = S_samples_cuts.shape
        n_times = len(times)
        S_samples = np.empty((n_p, n, n_times), dtype=np.float64)

        S_samples[:, :, below] = 1.0
        S_samples[:, :, ~below] = S_samples_cuts[:, :, idx[~below]]

        S_mean = np.mean(S_samples, axis=0)

        return {
            'S': S_mean,
            'S_samples': S_samples
        }

    def survival_at_own_times(self, state, part, n_passes):
        mu, S_samples_cuts = self._mc_surv_cuts(state, part, n_passes)
        cuts = state['cuts']
        T_part = part['T']

        idx = np.searchsorted(cuts, T_part, side='right') - 1
        below = (idx == -1)

        n_p, n, _ = S_samples_cuts.shape
        S_at_T_passes = np.empty((n_p, n), dtype=np.float64)

        S_at_T_passes[:, below] = 1.0
        row_indices = np.arange(n)[~below]
        S_at_T_passes[:, ~below] = S_samples_cuts[:, row_indices, idx[~below]]

        return np.mean(S_at_T_passes, axis=0)

    def objectives(self, pred_val, val):
        val_cindex = harness_metrics.cindex(val['T'], val['E'], pred_val['eta'])
        return (val_cindex,)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--fold', type=str, required=True)
    parser.add_argument('--trials', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    if args.fold == 'all':
        folds = [1, 2, 3, 4, 5]
    else:
        folds = [int(args.fold)]

    for f in folds:
        plugin = DeepHitPlugin(args.dataset, args.scenario, f)
        harness_runner.tune_and_select(plugin, args.dataset, args.scenario, f, args.trials, args.seed)
