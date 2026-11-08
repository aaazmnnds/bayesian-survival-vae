import argparse
import harness_runner
from optimize_gain import GAINCoxPlugin

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    args = parser.parse_args()

    plugin = GAINCoxPlugin()

    for fold in range(1, 6):
        harness_runner.final_evaluate(plugin, args.dataset, args.scenario, fold)

    harness_runner.summarize(plugin.name, args.dataset, args.scenario)

