import os
import json
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

import harness_metrics
import harness_runner

def cox_loss_vec(eta, T, E):
    """
    Vectorized Breslow Cox partial likelihood loss summed over events.
    eta: (N, 1) or (N,)
    T: (N,)
    E: (N,)
    """
    eta_flat = eta.squeeze(-1) if eta.dim() > 1 else eta
    if E.sum() == 0:
        return (eta_flat * 0).sum()
        
    idx_sort = torch.argsort(T, descending=True)
    T_sort = T[idx_sort]
    E_sort = E[idx_sort]
    eta_sort = eta_flat[idx_sort]
    
    risk_set_log_sum = torch.logcumsumexp(eta_sort, dim=0)
    unique_t, inv_idx, counts = torch.unique_consecutive(T_sort, return_inverse=True, return_counts=True)
    last_indices = counts.cumsum(0) - 1
    tied_risk_set_log_sum = risk_set_log_sum[last_indices[inv_idx]]
    
    event_mask = (E_sort == 1)
    eta_events = eta_sort[event_mask]
    risk_set_events = tied_risk_set_log_sum[event_mask]
    
    return -torch.sum(eta_events - risk_set_events)

class MCDropoutCoxModel(nn.Module):
    def __init__(self, in_features, dropout_prob):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(in_features, 1, bias=False)
        
    def forward(self, x):
        return self.linear(self.dropout(x))

class MCDropoutCoxPlugin:
    def __init__(self, dataset, scenario, fold, batch_size=256, max_epochs=100, val_passes=20, device=None):
        self.name = 'mc_dropout_cox'
        self.directions = ['maximize']
        self.pareto = False
        
        self.dataset = dataset
        self.scenario = scenario
        self.fold = int(fold)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.val_passes = val_passes
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        runs_dir = harness_runner.get_runs_dir()
        mice_sel_path = os.path.join(runs_dir, 'mice', f"{dataset}_{scenario}_fold{self.fold}_selected.json")
        if not os.path.exists(mice_sel_path):
            raise FileNotFoundError(
                f"Missing required MICE selection file: {mice_sel_path}. "
                f"Run MICE optimization for {dataset} {scenario} fold {self.fold} first."
            )
            
        with open(mice_sel_path, 'r', encoding='utf-8') as f:
            mice_sel = json.load(f)
            
        self.mice_params = mice_sel['params']

    def search_space(self, trial):
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.05, 0.5),
            'l2': trial.suggest_float('l2', 1e-6, 1e-1, log=True)
        }

    def _build_mice_imputer(self, seed):
        n_nf = None if self.mice_params['n_nearest_features'] == 'all' else self.mice_params['n_nearest_features']
        return IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.mice_params['max_iter'],
            n_nearest_features=n_nf,
            random_state=seed
        )

    def _impute_part(self, imputer, part):
        X_obs = part['X_obs']
        X_imp = imputer.transform(X_obs)
        observed_mask = ~np.isnan(X_obs)
        X_imp[observed_mask] = X_obs[observed_mask]
        return X_imp

    def fit(self, train, val, params, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # Fit MICE imputer on train['X_obs'] ONLY
        imputer = self._build_mice_imputer(seed)
        imputer.fit(train['X_obs'])
        
        X_train_imp = self._impute_part(imputer, train)
        X_val_imp = self._impute_part(imputer, val)
        
        N_train, P = X_train_imp.shape
        model = MCDropoutCoxModel(P, params['dropout']).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2'])
        
        X_tr_t = torch.tensor(X_train_imp, dtype=torch.float32, device=self.device)
        T_tr_t = torch.tensor(train['T'], dtype=torch.float32, device=self.device)
        E_tr_t = torch.tensor(train['E'], dtype=torch.float32, device=self.device)
        
        X_v_t = torch.tensor(X_val_imp, dtype=torch.float32, device=self.device)
        T_val = val['T']
        E_val = val['E']
        
        rng = np.random.Generator(np.random.PCG64(seed))
        
        best_cindex = -float('inf')
        best_state_dict = copy.deepcopy(model.state_dict())
        
        model.train()
        for epoch in range(self.max_epochs):
            perm = rng.permutation(N_train)
            model.train()
            for start_idx in range(0, N_train, self.batch_size):
                b_idx = perm[start_idx : start_idx + self.batch_size]
                b_idx_t = torch.tensor(b_idx, dtype=torch.long, device=self.device)
                
                b_x = X_tr_t[b_idx_t]
                b_t = T_tr_t[b_idx_t]
                b_e = E_tr_t[b_idx_t]
                
                optimizer.zero_grad()
                b_eta = model(b_x)
                loss = cox_loss_vec(b_eta, b_t, b_e)
                loss.backward()
                optimizer.step()
                
            # Validation C-index evaluation with MC dropout active
            model.train() # Keep dropout active
            with torch.no_grad():
                val_passes_eta = []
                for _ in range(self.val_passes):
                    pass_eta = model(X_v_t).squeeze(-1).cpu().numpy()
                    val_passes_eta.append(pass_eta)
                mean_val_eta = np.mean(val_passes_eta, axis=0)
                
            val_c = harness_metrics.cindex(T_val, E_val, mean_val_eta)
            if np.isnan(val_c):
                val_c = 0.0
                
            if val_c > best_cindex:
                best_cindex = val_c
                best_state_dict = copy.deepcopy(model.state_dict())
                
        model.load_state_dict(best_state_dict)
        
        return {
            'imputer': imputer,
            'model': model,
            'params': params,
            'P': P
        }

    def predict(self, state, part, n_passes):
        imputer = state['imputer']
        model = state['model']
        
        mu = self._impute_part(imputer, part)
        
        X_t = torch.tensor(mu, dtype=torch.float32, device=self.device)
        model.train() # Keep dropout active
        
        passes = []
        with torch.no_grad():
            for _ in range(n_passes):
                out = model(X_t).squeeze(-1).cpu().numpy()
                passes.append(out)
                
        # eta_samples: (n, n_passes)
        eta_samples = np.column_stack(passes)
        eta_mean = np.mean(eta_samples, axis=1)
        
        return {
            'mu': mu,
            'eta': eta_mean,
            'eta_samples': eta_samples
        }

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
        plugin = MCDropoutCoxPlugin(args.dataset, args.scenario, f)
        harness_runner.tune_and_select(plugin, args.dataset, args.scenario, f, args.trials, args.seed)
