import argparse
import os
import shutil
import harness_runner
from optimize_baselines import TwoStepCoxPlugin

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=['mice', 'missforest'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--folds', type=str, default="1,2,3,4,5", help="Comma-separated list of folds to evaluate")
    parser.add_argument('--timing', action='store_true', default=False, help="Measure fit and inference timing")
    parser.add_argument('--mi-draws', type=int, default=None, help="Number of posterior sampling draws for MICE")
    args = parser.parse_args()

    if args.mi_draws is not None and args.method != 'mice':
        parser.error("--mi-draws is valid only with --method mice")

    folds = [int(f.strip()) for f in args.folds.split(',') if f.strip()]
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

