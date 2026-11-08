import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import harness_metrics
import harness_runner
from optimize_baselines import fit_cox, cox_eta


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class NetG(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(NetG, self).__init__()
        self.fc1 = nn.Linear(dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        inputs = torch.cat([x, m], dim=1)
        out = self.relu(self.fc1(inputs))
        out = self.relu(self.fc2(out))
        return self.sigmoid(self.fc3(out))


class NetD(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(NetD, self).__init__()
        self.fc1 = nn.Linear(dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        inputs = torch.cat([x, h], dim=1)
        out = self.relu(self.fc1(inputs))
        out = self.relu(self.fc2(out))
        return self.sigmoid(self.fc3(out))


class GAINCoxPlugin:
    def __init__(self, iterations=10000, batch_size=128, checkpoint_every=500, device=None):
        self.name = 'gain'
        self.directions = ['minimize', 'maximize']
        self.pareto = True
        self.iterations = iterations
        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def search_space(self, trial):
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'alpha': trial.suggest_float('alpha', 1.0, 1000.0, log=True),
            'hint_rate': trial.suggest_float('hint_rate', 0.5, 0.99),
            'penalizer': trial.suggest_float('penalizer', 1e-3, 1.0, log=True)
        }

    def _impute_part(self, G_model, part, col_min, col_range, feature_types, feature_names, seed):
        X_obs = part['X_obs']
        M = part['M_obs']
        p = X_obs.shape[1]

        # Normalize with training min/range
        X_norm = (X_obs - col_min) / col_range
        X_norm = np.nan_to_num(X_norm, nan=0.0)

        # Deterministic Z sampling using seed + 1
        rng_impute = np.random.Generator(np.random.PCG64(seed + 1))
        Z = rng_impute.uniform(0.0, 0.01, size=X_obs.shape)

        X_in = M * X_norm + (1.0 - M) * Z

        G_model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_in, dtype=torch.float32, device=self.device)
            m_t = torch.tensor(M, dtype=torch.float32, device=self.device)
            g_sample = G_model(x_t, m_t).cpu().numpy()

        out_norm = M * X_norm + (1.0 - M) * g_sample

        # Denormalize with training min/range
        X_denorm = out_norm * col_range + col_min

        # Round binary columns to {0, 1}
        bin_cols = feature_types.get('binary', [])
        for c in bin_cols:
            if c in feature_names:
                idx = feature_names.index(c)
                X_denorm[:, idx] = np.clip(np.round(X_denorm[:, idx]), 0.0, 1.0)

        # Round categorical columns to nearest valid code
        cat_dict = feature_types.get('categorical', {})
        for c, codes in cat_dict.items():
            if c in feature_names:
                idx = feature_names.index(c)
                col_vals = X_denorm[:, idx, None]
                codes_arr = np.array(codes).reshape(1, -1)
                dists = np.abs(col_vals - codes_arr)
                nearest_idx = np.argmin(dists, axis=1)
                X_denorm[:, idx] = codes_arr[0, nearest_idx]

        # Finally set every observed cell exactly to part['X_obs']
        observed_mask = (M == 1.0)
        X_denorm[observed_mask] = X_obs[observed_mask]

        return X_denorm

    def fit(self, train, val, params, seed):
        # Seed numpy, torch (CPU and CUDA) with seed; dedicated Generator for sampling
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.Generator(np.random.PCG64(seed))

        X_obs = train['X_obs']
        M_obs = train['M_obs']
        N, p = X_obs.shape
        feature_names = train['feature_names']
        feature_types = train['feature_types']

        # Min-max normalization per column from OBSERVED TRAINING values only
        col_min = np.nanmin(X_obs, axis=0)
        col_max = np.nanmax(X_obs, axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0.0] = 1.0

        # Normalized NaN -> 0
        X_train_norm = (X_obs - col_min) / col_range
        X_train_norm = np.nan_to_num(X_train_norm, nan=0.0)

        # Networks: G and D Linear(2p, h) - ReLU - Linear(h, h) - ReLU - Linear(h, p) - Sigmoid, h = p
        G = NetG(p, p).to(self.device)
        D = NetD(p, p).to(self.device)
        G.apply(xavier_init)
        D.apply(xavier_init)

        opt_G = optim.Adam(G.parameters(), lr=params['learning_rate'])
        opt_D = optim.Adam(D.parameters(), lr=params['learning_rate'])

        alpha = params['alpha']
        hint_rate = params['hint_rate']
        batch_size = min(self.batch_size, N)

        best_rmse = float('inf')
        best_G_state = copy.deepcopy(G.state_dict())

        def evaluate_checkpoint():
            nonlocal best_rmse, best_G_state
            X_val_imp = self._impute_part(G, val, col_min, col_range, feature_types, feature_names, seed)
            val_rmse = harness_metrics.rmse(val['X_true'], X_val_imp, val['M_inj'])
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_G_state = copy.deepcopy(G.state_dict())

        for it in range(1, self.iterations + 1):
            G.train()
            D.train()

            # Sample batch_size row indices without replacement
            batch_idx = rng.choice(N, size=batch_size, replace=False)
            X_b = X_train_norm[batch_idx]
            M_b = M_obs[batch_idx]

            # Z ~ Uniform(0, 0.01)
            Z_b = rng.uniform(0.0, 0.01, size=(batch_size, p))
            X_in_b = M_b * X_b + (1.0 - M_b) * Z_b

            # B = Bernoulli(hint_rate) with P(B=1) = hint_rate; H = M*B
            B_b = (rng.uniform(0.0, 1.0, size=(batch_size, p)) < hint_rate).astype(np.float32)
            H_b = M_b * B_b

            X_in_t = torch.tensor(X_in_b, dtype=torch.float32, device=self.device)
            M_t = torch.tensor(M_b, dtype=torch.float32, device=self.device)
            H_t = torch.tensor(H_b, dtype=torch.float32, device=self.device)

            # D step
            opt_D.zero_grad()
            with torch.no_grad():
                G_sample = G(X_in_t, M_t)
            X_hat_D = M_t * X_in_t + (1.0 - M_t) * G_sample
            D_prob = D(X_hat_D, H_t)
            D_loss = -torch.mean(M_t * torch.log(D_prob + 1e-8) + (1.0 - M_t) * torch.log(1.0 - D_prob + 1e-8))
            D_loss.backward()
            opt_D.step()

            # G step
            opt_G.zero_grad()
            G_sample = G(X_in_t, M_t)
            X_hat_G = M_t * X_in_t + (1.0 - M_t) * G_sample
            D_prob = D(X_hat_G, H_t)
            G_loss_adv = -torch.mean((1.0 - M_t) * torch.log(D_prob + 1e-8))
            MSE_train_loss = torch.mean((M_t * X_in_t - M_t * G_sample) ** 2) / (torch.mean(M_t) + 1e-8)
            G_loss = G_loss_adv + alpha * MSE_train_loss
            G_loss.backward()
            opt_G.step()

            if it % self.checkpoint_every == 0:
                evaluate_checkpoint()

        evaluate_checkpoint()

        # Load the best G
        G.load_state_dict(best_G_state)

        # Impute the training part
        X_train_imp = self._impute_part(G, train, col_min, col_range, feature_types, feature_names, seed)

        # Fit the Cox model on it
        cox_state = fit_cox(X_train_imp, train['T'], train['E'], params['penalizer'])

        return {
            'G_state': copy.deepcopy(G.state_dict()),
            'p': p,
            'col_min': col_min,
            'col_range': col_range,
            'feature_types': feature_types,
            'feature_names': feature_names,
            'cox_state': cox_state,
            'seed': seed
        }

    def predict(self, state, part, n_passes):
        p = state['p']
        G = NetG(p, p).to(self.device)
        G.load_state_dict(state['G_state'])

        mu = self._impute_part(
            G,
            part,
            state['col_min'],
            state['col_range'],
            state['feature_types'],
            state['feature_names'],
            state['seed']
        )
        eta = cox_eta(state['cox_state'], mu)
        return {'mu': mu, 'eta': eta}

    def objectives(self, pred_val, val):
        val_rmse = harness_metrics.rmse(val['X_true'], pred_val['mu'], val['M_inj'])
        val_cindex = harness_metrics.cindex(val['T'], val['E'], pred_val['eta'])
        return val_rmse, val_cindex


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--fold', type=str, required=True)
    parser.add_argument('--trials', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    plugin = GAINCoxPlugin()

    if args.fold == 'all':
        folds = [1, 2, 3, 4, 5]
    else:
        folds = [int(args.fold)]

    for f in folds:
        harness_runner.tune_and_select(plugin, args.dataset, args.scenario, f, args.trials, args.seed)

