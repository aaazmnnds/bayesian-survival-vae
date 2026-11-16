import argparse
import os
import shutil
import json
import hashlib
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

import harness_data
import harness_runner
from optimize_baselines import TwoStepCoxPlugin, fit_cox

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=['mice', 'missforest'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--folds', type=str, default="1,2,3,4,5", help="Comma-separated list of folds to evaluate")
    parser.add_argument('--timing', action='store_true', default=False, help="Measure fit and inference timing")
    parser.add_argument('--mi-draws', type=int, default=None, help="Number of posterior sampling draws for MICE")
    parser.add_argument('--indicators', action='store_true', default=False, help="Evaluate MICE with missingness indicators")
    args = parser.parse_args()

    if args.mi_draws is not None and args.method != 'mice':
        parser.error("--mi-draws is valid only with --method mice")
    if args.indicators and args.method != 'mice':
        parser.error("--indicators is valid only with --method mice")
    if args.indicators and args.mi_draws is not None:
        parser.error("--indicators and --mi-draws cannot be used together")

    folds = [int(f.strip()) for f in args.folds.split(',') if f.strip()]
    if args.indicators:
        plugin = TwoStepCoxPlugin("mice", add_indicators=True)
        runs_dir = harness_runner.get_runs_dir()
        dst_dir = os.path.join(runs_dir, 'mice_ind')
        os.makedirs(dst_dir, exist_ok=True)
        penalizers = np.logspace(-3, 0, 20)

        for fold in folds:
            src = os.path.join(runs_dir, 'mice', f"{args.dataset}_{args.scenario}_fold{fold}_selected.json")
            if not os.path.exists(src):
                raise FileNotFoundError(f"Missing {src}")
            with open(src, 'r') as f:
                selected_data = json.load(f)
            params = selected_data['params']
            seed = selected_data['seed']

            fd_train = harness_data.load_fold(args.dataset, args.scenario, fold)
            train_part = fd_train.part('train')
            val_part = fd_train.part('val')

            n_nf = None if params['n_nearest_features'] == 'all' else params['n_nearest_features']
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=params['max_iter'],
                n_nearest_features=n_nf,
                random_state=seed
            )
            imputer.fit(train_part['X_obs'])
            X_imp = imputer.transform(train_part['X_obs'])

            I_train = np.isnan(train_part['X_obs']).astype(float)
            kept_cols = np.where(I_train.std(axis=0) > 1e-8)[0]
            X_cox_train = np.hstack([X_imp, I_train[:, kept_cols]])

            cache_key = ('mice', hashlib.sha1(train_part['X_obs'].tobytes()).hexdigest(), params['max_iter'], str(n_nf), seed)
            plugin._cache[cache_key] = (imputer, X_imp)

            best_penalizer = None
            best_val_cindex = -float('inf')

            for penalizer in penalizers:
                try:
                    cox_state = fit_cox(X_cox_train, train_part['T'], train_part['E'], penalizer)
                    state = {
                        'imputer': imputer,
                        'cox_state': cox_state,
                        'cox_model': cox_state['cox_model'],
                        'cols': cox_state['cols'],
                        'names': cox_state['names'],
                        'indicator_cols': kept_cols
                    }
                    pred_val = plugin.predict(state, val_part, n_passes=1)
                    val_rmse, val_cindex = plugin.objectives(pred_val, val_part)
                    if val_cindex > best_val_cindex:
                        best_val_cindex = val_cindex
                        best_penalizer = float(penalizer)
                except Exception:
                    continue

            dst_json_path = os.path.join(dst_dir, f"{args.dataset}_{args.scenario}_fold{fold}_selected.json")
            selected_ind = dict(selected_data)
            selected_ind['params'] = dict(selected_data['params'])
            selected_ind['params']['penalizer'] = best_penalizer
            with open(dst_json_path, 'w') as f:
                json.dump(selected_ind, f, indent=2)

            harness_runner.final_evaluate(plugin, args.dataset, args.scenario, fold, timing=args.timing)

            test_json_path = os.path.join(dst_dir, f"{args.dataset}_{args.scenario}_fold{fold}_test.json")
            with open(test_json_path, 'r') as f:
                test_res = json.load(f)
            test_cindex = test_res['metrics']['cindex']

            print(f"{args.dataset}, {args.scenario}, fold {fold}: chosen penalizer={best_penalizer:.6e}, validation C-index={best_val_cindex:.4f}, test C-index={test_cindex:.4f}")

        if not args.timing and set(folds) >= {1, 2, 3, 4, 5}:
            harness_runner.summarize(plugin.name, args.dataset, args.scenario)

    else:
        if args.mi_draws is not None:
            plugin = TwoStepCoxPlugin("mice", mi_draws=args.mi_draws)
        else:
            plugin = TwoStepCoxPlugin(args.method)

        for fold in folds:
            if args.mi_draws is not None:
                runs_dir = harness_runner.get_runs_dir()
                src = os.path.join(runs_dir, 'mice', f"{args.dataset}_{args.scenario}_fold{fold}_selected.json")
                dst_dir = os.path.join(runs_dir, 'mice_mi')
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, f"{args.dataset}_{args.scenario}_fold{fold}_selected.json")
                if not os.path.exists(dst):
                    shutil.copyfile(src, dst)

            harness_runner.final_evaluate(plugin, args.dataset, args.scenario, fold, timing=args.timing)

        if not args.timing and set(folds) >= {1, 2, 3, 4, 5}:
            harness_runner.summarize(plugin.name, args.dataset, args.scenario)

