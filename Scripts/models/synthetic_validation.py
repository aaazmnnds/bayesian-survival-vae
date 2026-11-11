import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

# Path setup relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

from harness_data import load_fold, load_test
import harness_metrics
from optimize_bsvae import BSVAEPlugin


def run_subsample(variant=None):
    dataset = "synthetic"
    scenario = "moderate"
    fold = 1

    method_name = "bsvae_nolcal" if variant == "nolcal" else "bsvae"
    prefix = "nolcal_" if variant == "nolcal" else ""

    selected_json_path = os.path.join(
        PROJECT_ROOT,
        "Results",
        "final",
        "runs_v2",
        method_name,
        f"{dataset}_{scenario}_fold{fold}_selected.json",
    )
    if not os.path.exists(selected_json_path):
        raise FileNotFoundError(f"Missing selected configuration file: {selected_json_path}")

    with open(selected_json_path, "r", encoding="utf-8") as f:
        selected_data = json.load(f)
    params = selected_data["params"]

    output_dir = os.path.join(PROJECT_ROOT, "Results", "synthetic")
    os.makedirs(output_dir, exist_ok=True)

    fd_train = load_fold(dataset, scenario, fold)
    full_train_part = fd_train.part("train")
    val_part = fd_train.part("val")

    fd_test = load_test(dataset, scenario, fold, fd_train)
    test_part = fd_test.part("test")

    n_full_train = len(full_train_part["T"])
    sizes = [19200, 6400, 3200, 1280]

    for size in sizes:
        for r in range(5):
            out_npz_path = os.path.join(output_dir, f"{prefix}subsample_n{size}_r{r}.npz")
            if os.path.exists(out_npz_path):
                print(f"Skipping existing run: {out_npz_path}")
                continue

            # Subsample indices from training part
            rng = np.random.default_rng(1000 + r)
            if size == n_full_train:
                sub_idx = np.arange(n_full_train)
            else:
                sub_idx = rng.choice(n_full_train, size=size, replace=False)

            train_part_sub = {}
            for k, v in full_train_part.items():
                if isinstance(v, np.ndarray):
                    train_part_sub[k] = v[sub_idx]
                elif isinstance(v, list) and len(v) == n_full_train:
                    train_part_sub[k] = [v[i] for i in sub_idx]
                else:
                    train_part_sub[k] = v

            plugin = BSVAEPlugin(variant=variant)
            state = plugin.fit(train_part_sub, val_part, params, seed=r)

            pred_test = plugin.predict(state, test_part, n_passes=100)

            np.savez_compressed(
                out_npz_path,
                sigma_E2=pred_test["sigma_E2"],
                sigma_A2=pred_test["sigma_A2"],
                mu=pred_test["mu"],
                eta=pred_test["eta"],
                cohort_index=test_part["cohort_index"],
                M_inj=test_part["M_inj"],
            )
            print(f"Completed subsample run: size={size}, r={r} -> {out_npz_path}")


def run_analyze(variant=None):
    dataset = "synthetic"
    scenario = "moderate"
    fold = 1

    method_name = "bsvae_nolcal" if variant == "nolcal" else "bsvae"
    prefix = "nolcal_" if variant == "nolcal" else ""
    summary_filename = "synthetic_validation_summary_nolcal.json" if variant == "nolcal" else "synthetic_validation_summary.json"

    synth_dir = os.path.join(PROJECT_ROOT, "Results", "synthetic")
    test_npz_path = os.path.join(
        PROJECT_ROOT,
        "Results",
        "final",
        "runs_v2",
        method_name,
        f"{dataset}_{scenario}_fold{fold}_test.npz",
    )
    if not os.path.exists(test_npz_path):
        raise FileNotFoundError(f"Missing test.npz file: {test_npz_path}")

    obs_csv_path = os.path.join(synth_dir, "synthetic_cohort_observed.csv")
    true_csv_path = os.path.join(synth_dir, "synthetic_cohort_true.csv")
    true_vars_path = os.path.join(synth_dir, "true_noise_variances.npy")
    weibull_path = os.path.join(synth_dir, "weibull_params.json")

    for p in [obs_csv_path, true_csv_path, true_vars_path, weibull_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required synthetic file: {p}")

    test_npz = np.load(test_npz_path)
    df_obs = pd.read_csv(obs_csv_path)
    df_true = pd.read_csv(true_csv_path)
    true_noise_vars = np.load(true_vars_path)
    with open(weibull_path, "r", encoding="utf-8") as f:
        weibull_params = json.load(f)

    v_shape = weibull_params["v"]

    fd_train = load_fold(dataset, scenario, fold)
    fd_test = load_test(dataset, scenario, fold, fd_train)
    test_part = fd_test.part("test")

    cohort_idx = test_npz["cohort_index"]
    test_M_inj = test_part["M_inj"]
    s_std = fd_train.scaler_std

    # (i) Aleatoric recovery
    sigma_A2_test = test_npz["sigma_A2"]
    est_aleatoric_per_feat = np.nanmean(sigma_A2_test, axis=0)
    true_aleatoric_std = true_noise_vars / (s_std**2)

    aleatoric_mse = float(np.mean((est_aleatoric_per_feat - true_aleatoric_std) ** 2))
    ratios = est_aleatoric_per_feat / true_aleatoric_std
    aleatoric_median_ratio = float(np.median(ratios))

    # (ii) Epistemic by training size
    sizes = [19200, 6400, 3200, 1280]
    epistemic_by_size = {}

    mask_inj = test_M_inj == 1

    for size in sizes:
        e_runs = []
        a_runs = []
        for r in range(5):
            sub_npz_path = os.path.join(synth_dir, f"{prefix}subsample_n{size}_r{r}.npz")
            if not os.path.exists(sub_npz_path):
                raise FileNotFoundError(f"Missing subsample file: {sub_npz_path}")
            sub_npz = np.load(sub_npz_path)
            e_runs.append(float(np.mean(sub_npz["sigma_E2"][mask_inj])))
            a_runs.append(float(np.mean(sub_npz["sigma_A2"][mask_inj])))

        epistemic_by_size[str(size)] = {
            "sigma_E2_mean": float(np.mean(e_runs)),
            "sigma_E2_std": float(np.std(e_runs)),
            "sigma_A2_mean": float(np.mean(a_runs)),
            "sigma_A2_std": float(np.std(a_runs)),
        }

    # (iii) Survival interval coverage
    time_grid = test_npz["time_grid"]
    S_lower = test_npz["S_lower"]
    S_upper = test_npz["S_upper"]

    # Match patients by cohort_index to get weibull_scale
    if "weibull_scale" in df_true.columns:
        test_weibull_scales = df_true.iloc[cohort_idx]["weibull_scale"].values
    else:
        # compute scale from formula if weibull_scale column missing
        beta = np.array(weibull_params["beta"])
        lam = weibull_params["lambda"]
        feat_cols = [c for c in fd_train.feature_names]
        X_test_scaled = test_part["X_true"]
        eta = np.dot(X_test_scaled, beta)
        test_weibull_scales = lam * np.exp(-eta / v_shape)

    # true S_i(t) = exp(-(t / scale_i)^v) for each (i, t)
    # shape: (N_test, len(time_grid))
    t_broadcast = time_grid[None, :]
    scale_broadcast = test_weibull_scales[:, None]
    true_S = np.exp(-((t_broadcast / scale_broadcast) ** v_shape))

    in_interval = (true_S >= S_lower) & (true_S <= S_upper)
    overall_coverage = float(np.mean(in_interval))
    coverage_per_time = [float(c) for c in np.mean(in_interval, axis=0)]

    # (iv) Ratio near the MNAR threshold
    median_age = float(df_obs["age"].median())
    lactate_mean = float(df_obs["Lactate"].mean())
    lactate_std = float(df_obs["Lactate"].std())

    test_obs_df = df_obs.iloc[cohort_idx].reset_index(drop=True)
    a_indic = (test_obs_df["age"].values > median_age).astype(float)
    z_lactate = (test_obs_df["Lactate"].values - lactate_mean) / lactate_std
    p_missing = 1.0 / (1.0 + np.exp(-(3.0 * a_indic * z_lactate - 0.39)))

    # Per patient, ratio = mean(sigma_E2 / sigma_A2) over that patient's masked cells
    sigma_E2_patient = test_npz["sigma_E2"]
    sigma_A2_patient = test_npz["sigma_A2"]

    patient_ratios = []
    valid_p = []
    valid_age_indic = []

    for i in range(len(cohort_idx)):
        row_mask = test_M_inj[i] == 1
        if np.any(row_mask):
            e_vals = sigma_E2_patient[i, row_mask]
            a_vals = sigma_A2_patient[i, row_mask]
            ratio_i = np.mean(e_vals / a_vals)
            patient_ratios.append(float(ratio_i))
            valid_p.append(float(p_missing[i]))
            valid_age_indic.append(float(a_indic[i]))

    patient_ratios = np.array(patient_ratios)
    valid_p = np.array(valid_p)
    valid_age_indic = np.array(valid_age_indic)

    # Below median age: single group
    mask_below_age = valid_age_indic == 0
    ratio_below_median_age = float(np.mean(patient_ratios[mask_below_age]))
    p_below_median_age = float(valid_p[mask_below_age][0]) if np.any(mask_below_age) else float("nan")

    # Above median age: compute deciles of p ONLY among patients above the median age
    mask_above_age = valid_age_indic == 1
    p_above = valid_p[mask_above_age]
    ratios_above = patient_ratios[mask_above_age]

    # Use pd.qcut to get 10 quantiles without empty bins
    decile_series, bin_edges = pd.qcut(p_above, q=10, retbins=True, duplicates="drop")
    ratio_per_decile_above = []
    decile_info = []
    for cat in decile_series.categories:
        m = decile_series == cat
        mean_ratio_bin = float(np.mean(ratios_above[m]))
        ratio_per_decile_above.append(mean_ratio_bin)
        decile_info.append({
            "p_range": [float(cat.left), float(cat.right)],
            "mean_ratio": mean_ratio_bin,
            "count": int(np.sum(m)),
        })

    ratio_above_median_age = float(np.mean(ratios_above))

    summary = {
        "aleatoric_recovery": {
            "mse": aleatoric_mse,
            "median_ratio": aleatoric_median_ratio,
        },
        "epistemic_by_training_size": epistemic_by_size,
        "survival_interval_coverage": {
            "overall": overall_coverage,
            "time_grid": [float(t) for t in time_grid],
            "per_time": coverage_per_time,
        },
        "ratio_near_mnar_threshold": {
            "below_median_age": {
                "p": p_below_median_age,
                "mean_ratio": ratio_below_median_age,
                "count": int(np.sum(mask_below_age)),
            },
            "above_median_age": {
                "mean_ratio": ratio_above_median_age,
                "count": int(np.sum(mask_above_age)),
                "deciles": decile_info,
                "per_decile_ratio": ratio_per_decile_above,
            },
        },
    }

    summary_path = os.path.join(synth_dir, summary_filename)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Synthetic Validation Analysis Results ===")
    print(
        f"(i) Aleatoric Recovery: MSE={aleatoric_mse:.6f}, Median Ratio={aleatoric_median_ratio:.4f}"
    )
    print("(ii) Epistemic / Aleatoric by Training Size:")
    for sz, vals in epistemic_by_size.items():
        print(
            f"   N={sz}: sigma_E2={vals['sigma_E2_mean']:.5f} (SD={vals['sigma_E2_std']:.5f}), "
            f"sigma_A2={vals['sigma_A2_mean']:.5f} (SD={vals['sigma_A2_std']:.5f})"
        )
    print(f"(iii) Survival Interval Coverage: Overall={overall_coverage:.4f}")
    print(
        f"(iv) Ratio (sigma_E2 / sigma_A2): Below Age Median={ratio_below_median_age:.4f}, "
        f"Above Age Median={ratio_above_median_age:.4f}"
    )
    print(f"Written summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic validation for B-SVAE.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--subsample", action="store_true", help="Run training size subsampling experiments."
    )
    group.add_argument(
        "--analyze", action="store_true", help="Analyze results and produce summary."
    )
    parser.add_argument(
        "--variant", type=str, choices=["none", "nolcal"], default="none", help="Model variant."
    )
    args = parser.parse_args()

    variant = None if args.variant == "none" else args.variant

    if args.subsample:
        run_subsample(variant=variant)
    elif args.analyze:
        run_analyze(variant=variant)


if __name__ == "__main__":
    main()
