import os
import json
import argparse
from optimize_bsvae import BSVAEPlugin
import harness_runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--folds', type=str, default="1,2,3,4,5", help="Comma-separated list of folds to evaluate")
    parser.add_argument('--selection', type=str, choices=['default', 'rmse_cindex', 'check3'], default='default', help="Model selection criterion")
    args = parser.parse_args()

    folds = [int(f.strip()) for f in args.folds.split(',') if f.strip()]
    plugin = BSVAEPlugin()

    if args.selection == 'check3':
        runs_dir = harness_runner.get_runs_dir()
        bsvae_dir = os.path.join(runs_dir, "bsvae")
        for fold in folds:
            reselect_out = harness_runner.reselect(
                source_name="bsvae",
                target_name="bsvae_check3",
                dataset=args.dataset,
                scenario=args.scenario,
                fold=fold,
                objective_indices=[0, 1, 2],
                directions=plugin.directions
            )
            orig_json_path = os.path.join(bsvae_dir, f"{args.dataset}_{args.scenario}_fold{fold}_selected.json")
            if not os.path.exists(orig_json_path):
                raise FileNotFoundError(f"Missing {orig_json_path}")
            with open(orig_json_path, 'r') as f:
                orig_json = json.load(f)
            
            if reselect_out['selected_trial'] == orig_json['selected_trial']:
                print(f"fold {fold}: match")
            else:
                print(f"fold {fold}: MISMATCH")

    elif args.selection == 'rmse_cindex':
        plugin.name = "bsvae_sel_rmse_c"
        for fold in folds:
            harness_runner.reselect(
                source_name="bsvae",
                target_name="bsvae_sel_rmse_c",
                dataset=args.dataset,
                scenario=args.scenario,
                fold=fold,
                objective_indices=[0, 2],
                directions=["minimize", "maximize"]
            )
            harness_runner.final_evaluate(plugin, args.dataset, args.scenario, fold)

        if set(folds) >= {1, 2, 3, 4, 5}:
            harness_runner.summarize(plugin.name, args.dataset, args.scenario)

    else:
        for fold in folds:
            harness_runner.final_evaluate(plugin, args.dataset, args.scenario, fold)

        if set(folds) >= {1, 2, 3, 4, 5}:
            harness_runner.summarize(plugin.name, args.dataset, args.scenario)
