import argparse
from optimize_bsvae import BSVAEPlugin
import harness_runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--folds', type=str, default="1,2,3,4,5", help="Comma-separated list of folds to evaluate")
    args = parser.parse_args()

    folds = [int(f.strip()) for f in args.folds.split(',') if f.strip()]
    plugin = BSVAEPlugin()

    for fold in folds:
        harness_runner.final_evaluate(plugin, args.dataset, args.scenario, fold)

    if set(folds) >= {1, 2, 3, 4, 5}:
        harness_runner.summarize(plugin.name, args.dataset, args.scenario)
