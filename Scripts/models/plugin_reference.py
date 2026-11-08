"""
MeanImputeCox is a harness test and reference model, not a manuscript baseline.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from harness_data import fill_for_input
from harness_metrics import cindex

class MeanImputeCox:
    """Harness test and reference model, not a manuscript baseline."""
    def __init__(self):
        self.name = "MeanImputeCox"
        self.directions = ['maximize']
        self.pareto = False

    def search_space(self, trial):
        return {'penalizer': trial.suggest_float('penalizer', 1e-3, 1.0, log=True)}

    def _design(self, part):
        X = fill_for_input(part['X_obs'])
        return np.concatenate([X, part['M_obs']], axis=1)

    def fit(self, train, val, params, seed):
        X = self._design(train)
        keep = np.where(X.std(axis=0) > 1e-8)[0]
        _, first = np.unique(X[:, keep], axis=1, return_index=True)
        cols = keep[np.sort(first)]
        names = [f'f{i}' for i in cols]
        df = pd.DataFrame(X[:, cols], columns=names)
        df['T'] = train['T']
        df['E'] = train['E']
        cph = CoxPHFitter(penalizer=params['penalizer'])
        cph.fit(df, duration_col='T', event_col='E')
        return {'model': cph, 'cols': cols, 'names': names}

    def predict(self, state, part, n_passes):
        X_obs = part['X_obs']
        mu = np.where(np.isnan(X_obs), 0.0, X_obs)
        X = self._design(part)
        df = pd.DataFrame(X[:, state['cols']], columns=state['names'])
        eta = np.log(state['model'].predict_partial_hazard(df).values + 1e-12)
        return {'mu': mu, 'eta': eta}

    def objectives(self, pred_val, val):
        return (cindex(val['T'], val['E'], pred_val['eta']),)
