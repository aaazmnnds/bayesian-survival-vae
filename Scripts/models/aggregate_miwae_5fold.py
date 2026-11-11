import argparse
import harness_runner
from optimize_miwae import MIWAECoxPlugin

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--folds', type=str, default="1,2,3,4,5", help="Comma-separated list of folds to evaluate")
    parser.add_argument('--timing', action='store_true', default=False, help="Measure fit and inference timing")
    args = parser.parse_args()

    folds = [int(f.strip()) for f in args.folds.split(',') if f.strip()]
    plugin = MIWAECoxPlugin()

    for fold in folds:
        harness_runner.final_evaluate(plugin, args.dataset, args.scenario, fold, timing=args.timing)

    if not args.timing and set(folds) >= {1, 2, 3, 4, 5}:
        harness_runner.summarize(plugin.name, args.dataset, args.scenario)

