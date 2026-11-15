import argparse
import gc
import hashlib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from lifelines import CoxPHFitter

import harness_metrics
import harness_runner

def fit_cox(X_imp, T, E, penalizer):
    keep = np.where(X_imp.std(axis=0) > 1e-8)[0]
    _, first = np.unique(X_imp[:, keep], axis=1, return_index=True)
    cols = keep[np.sort(first)]
    names = [f'f{i}' for i in cols]
    
    df = pd.DataFrame(X_imp[:, cols], columns=names)
    df['T'] = T
    df['E'] = E
    
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col='T', event_col='E')
    
    return {
        'cox_model': cph,
        'cols': cols,
        'names': names
    }

def cox_eta(cox_state, X_imp):
    df = pd.DataFrame(X_imp[:, cox_state['cols']], columns=cox_state['names'])
    eta = np.log(cox_state['cox_model'].predict_partial_hazard(df).values + 1e-12)
    return eta

class TwoStepCoxPlugin:
    def __init__(self, method, n_jobs=8, mi_draws=None):
        if method not in {'mice', 'missforest'}:
            raise ValueError(f"Unknown method {method}, expected 'mice' or 'missforest'")
        if mi_draws is not None and method != 'mice':
            raise ValueError("mi_draws is allowed only for method 'mice'")
        self.method = method
        self.n_jobs = n_jobs
        self.mi_draws = mi_draws
        self.name = 'mice_mi' if mi_draws is not None else method
        self.directions = ['minimize', 'maximize']
        self.pareto = True
        self._cache = {}

    def search_space(self, trial):
        if self.method == 'mice':
            return {
                'max_iter': trial.suggest_int('max_iter', 5, 20),
                'n_nearest_features': trial.suggest_categorical('n_nearest_features', [10, 20, 40, 'all']),
                'penalizer': trial.suggest_float('penalizer', 1e-3, 1.0, log=True)
            }
        elif self.method == 'missforest':
            return {
                'penalizer': trial.suggest_float('penalizer', 1e-3, 1.0, log=True)
            }

    def fit(self, train, val, params, seed):
        if self.method == 'mice':
            n_nf = None if params['n_nearest_features'] == 'all' else params['n_nearest_features']
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=params['max_iter'],
                n_nearest_features=n_nf,
                random_state=seed
            )
            imputer.fit(train['X_obs'])
            X_imp = imputer.transform(train['X_obs'])
        elif self.method == 'missforest':
            cache_key = (hashlib.sha1(train['X_obs'].tobytes()).hexdigest(), seed)
            if cache_key in self._cache:
                imputer, X_imp = self._cache[cache_key]
            else:
                self._cache.clear()
                gc.collect()
                P = train['X_obs'].shape[1]
                # Reproduce defaults of missForest 1.6.1 with ranger 0.18 backend
                imputer = IterativeImputer(
                    estimator=RandomForestRegressor(
                        n_estimators=100,
                        max_features=max(1, int(np.floor(np.sqrt(P)))),   # P = train['X_obs'].shape[1]; missForest mtry = floor(sqrt(ncol(xmis)))
                        min_samples_split=5,                              # ranger min.node.size = 5 for regression
                        min_samples_leaf=1,
                        max_depth=None,
                        bootstrap=True,
                        random_state=seed,
                        n_jobs=self.n_jobs
                    ),
                    max_iter=10,
                    random_state=seed
                )
                imputer.fit(train['X_obs'])
                X_imp = imputer.transform(train['X_obs'])
                self._cache[cache_key] = (imputer, X_imp)
        
        cox_state = fit_cox(X_imp, train['T'], train['E'], params['penalizer'])
        
        state = {
            'imputer': imputer,
            'cox_state': cox_state,
            'cox_model': cox_state['cox_model'],
            'cols': cox_state['cols'],
            'names': cox_state['names']
        }
        if self.mi_draws is not None:
            _st = np.random.get_state()
            mi_imputers = []
            for k in range(1, self.mi_draws + 1):
                imp = IterativeImputer(
                    estimator=BayesianRidge(),
                    max_iter=params['max_iter'],
                    n_nearest_features=n_nf,
                    sample_posterior=True,
                    random_state=seed + k
                )
                imp.fit(train['X_obs'])
                mi_imputers.append(imp)
            np.random.set_state(_st)
            state['mi_imputers'] = mi_imputers

        return state

    def predict(self, state, part, n_passes):
        X_obs = part['X_obs']
        mu = state['imputer'].transform(X_obs)
        observed_mask = ~np.isnan(X_obs)
        mu[observed_mask] = X_obs[observed_mask]
        
        cox_state = state.get('cox_state', state)
        eta = cox_eta(cox_state, mu)

        if self.mi_draws is not None and 'mi_imputers' in state:
            _st = np.random.get_state()
            draws = []
            for imp in state['mi_imputers']:
                draw = imp.transform(X_obs)
                draw[observed_mask] = X_obs[observed_mask]
                draws.append(draw)
            draws = np.stack(draws, axis=0)
            mu_mi = np.mean(draws, axis=0)
            sigma_E2 = np.var(draws, axis=0, ddof=1)
            sigma_A2 = np.zeros_like(sigma_E2)
            np.random.set_state(_st)
            return {
                'mu': mu_mi,
                'eta': eta,
                'sigma_E2': sigma_E2,
                'sigma_A2': sigma_A2
            }

        return {'mu': mu, 'eta': eta}

    def objectives(self, pred_val, val):
        val_rmse = harness_metrics.rmse(val['X_true'], pred_val['mu'], val['M_inj'])
        val_cindex = harness_metrics.cindex(val['T'], val['E'], pred_val['eta'])
        return val_rmse, val_cindex

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=['mice', 'missforest'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--fold', type=str, required=True)
    parser.add_argument('--trials', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--n_jobs', type=int, default=8)
    args = parser.parse_args()

    plugin = TwoStepCoxPlugin(args.method, n_jobs=args.n_jobs)

    if args.fold == 'all':
        folds = [1, 2, 3, 4, 5]
    else:
        folds = [int(args.fold)]

    for f in folds:
        harness_runner.tune_and_select(plugin, args.dataset, args.scenario, f, args.trials, args.seed)


