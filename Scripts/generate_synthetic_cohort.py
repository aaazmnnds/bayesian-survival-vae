import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import weibull_min

# Local imports
from dataset_schema import FEATURE_TYPES
from mnar_injection import apply_threshold_interaction_mnar


def generate_synthetic_cohort(
    N: int = 30000,
    target_event_rate: float = 0.1637,
    random_state: int = 42,
    output_dir: str = None,
    mimic_file: str = None,
):
    """
    Generates a synthetic dataset for B-SVAE validation based on MIMIC-IV correlation structures.
    """
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "Results", "synthetic")
    if mimic_file is None:
        mimic_file = os.path.join(PROJECT_ROOT, "Results", "final", "mimic_sepsis_cohort.csv")

    rng = np.random.default_rng(random_state)

    print("Calibrating covariance matrix from MIMIC-IV...")
    if not os.path.exists(mimic_file):
        raise FileNotFoundError(f"Cannot find {mimic_file} for covariance calibration.")

    df_mimic = pd.read_csv(mimic_file)

    # Features: type 'continuous' for 'mimic' from dataset_schema.FEATURE_TYPES
    schema_continuous = list(FEATURE_TYPES["mimic"]["continuous"])
    rest_cols = [c for c in schema_continuous if c not in ("age", "Lactate")]
    ordered_cols = ["age", "Lactate"] + rest_cols
    assert len(ordered_cols) == 59, f"Expected 59 continuous features, found {len(ordered_cols)}"
    P = len(ordered_cols)

    print(f"Generating synthetic cohort with N={N}, P={P}...")

    # Pairwise correlation of these columns in the cohort, NaN set to 0, symmetrized
    corr_matrix = df_mimic[ordered_cols].corr().fillna(0).values
    corr_matrix = np.nan_to_num(corr_matrix, posinf=0.0, neginf=0.0)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2.0

    # Make positive definite by clipping eigenvalues at 1e-6, then rescale to unit diagonal
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    eigvals_clipped = np.clip(eigvals, 1e-6, None)
    cov_pd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    d = np.sqrt(np.diag(cov_pd))
    corr_matrix = cov_pd / np.outer(d, d)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2.0

    # 18-factor latent model from correlation matrix C
    eigvals_c, eigvecs_c = np.linalg.eigh(corr_matrix)
    sort_idx = np.argsort(eigvals_c)[::-1]
    top18_idx = sort_idx[:18]
    eigvals_18 = eigvals_c[top18_idx]
    V_18 = eigvecs_c[:, top18_idx]
    W = V_18 * np.sqrt(eigvals_18)  # shape (59, 18)

    sigma2_j = 1.0 - np.sum(W**2, axis=1)
    assert np.all(sigma2_j > 0), f"Expected all noise variances > 0, found min={np.min(sigma2_j)}"

    z = rng.normal(0.0, 1.0, size=(N, 18))
    signal = z @ W.T
    noise = rng.normal(0.0, np.sqrt(sigma2_j), size=(N, P))
    x_standardized = signal + noise

    # Map from standard normal to real marginal distributions
    means = df_mimic[ordered_cols].mean().fillna(0).values
    stds = df_mimic[ordered_cols].std().replace(0, 1.0).fillna(1.0).values

    X_true = signal * stds + means
    X_observed = x_standardized * stds + means
    true_noise_vars = sigma2_j * (stds**2)

    df_obs = pd.DataFrame(X_observed, columns=ordered_cols)
    df_true = pd.DataFrame(X_true, columns=ordered_cols)

    # Survival: beta ~ N(0, 0.2), eta from standardized X_true, shape v = 1.5, scale_i = lambda * exp(-eta_i / v)
    beta = rng.normal(0, 0.2, size=P)
    X_true_scaled = (X_true - means) / stds
    eta = np.dot(X_true_scaled, beta)

    v = 1.5
    horizon = 28.0

    def expected_event_rate_diff(lam):
        scale_i = lam * np.exp(-eta / v)
        expected_rate = np.mean(1.0 - np.exp(-((horizon / scale_i) ** v)))
        return expected_rate - target_event_rate

    print("Calibrating Weibull scale parameter to target event rate...")
    lambda_opt = brentq(expected_event_rate_diff, 0.001, 1000.0)

    scale_i = lambda_opt * np.exp(-eta / v)
    # Draw T_true using the script's rng
    U = rng.uniform(0.0, 1.0, size=N)
    T_true = scale_i * (-np.log(1.0 - U)) ** (1.0 / v)

    # Administrative censoring at day 28: event = T_true <= 28, T_obs = min(T_true, 28)
    event_indicator = (T_true <= horizon).astype(int)
    T_obs = np.minimum(T_true, horizon)

    # Inject MNAR missingness: eligible_cols = all features except 'age', severity='moderate'
    eligible_cols = [c for c in ordered_cols if c != "age"]
    print("Applying threshold-interaction MNAR...")
    df_masked, mask_eval = apply_threshold_interaction_mnar(
        df_obs,
        severity="moderate",
        random_state=random_state,
        age_col="age",
        lactate_col="Lactate",
        eligible_cols=eligible_cols,
    )

    # Prepend patient_id as the first column
    patient_ids = np.arange(N)
    df_obs.insert(0, "patient_id", patient_ids)
    df_masked.insert(0, "patient_id", patient_ids)
    df_true.insert(0, "patient_id", patient_ids)
    mask_eval.insert(0, "patient_id", 0)

    # Append survival targets
    df_obs["event"] = event_indicator
    df_obs["time_days"] = T_obs

    df_masked["event"] = event_indicator
    df_masked["time_days"] = T_obs

    df_true["weibull_scale"] = scale_i
    df_true["event"] = event_indicator
    df_true["time_days"] = T_obs

    mask_eval["event"] = 0
    mask_eval["time_days"] = 0

    # Save the dataset
    os.makedirs(output_dir, exist_ok=True)
    df_obs.to_csv(os.path.join(output_dir, "synthetic_cohort_observed.csv"), index=False)
    df_masked.to_csv(os.path.join(output_dir, "synthetic_cohort_masked.csv"), index=False)
    df_true.to_csv(os.path.join(output_dir, "synthetic_cohort_true.csv"), index=False)
    mask_eval.to_csv(os.path.join(output_dir, "synthetic_mask.csv"), index=False)

    # Write to Scripts/models/datasets_real
    models_datasets_dir = os.path.join(PROJECT_ROOT, "Scripts", "models", "datasets_real")
    os.makedirs(models_datasets_dir, exist_ok=True)
    df_masked.to_csv(os.path.join(models_datasets_dir, "synthetic_mnar_moderate.csv"), index=False)
    mask_eval.to_csv(os.path.join(models_datasets_dir, "synthetic_mask_moderate.csv"), index=False)

    np.save(os.path.join(output_dir, "true_noise_variances.npy"), true_noise_vars)
    np.save(os.path.join(output_dir, "true_loadings.npy"), W)
    with open(os.path.join(output_dir, "weibull_params.json"), "w") as f:
        json.dump(
            {
                "v": v,
                "lambda": lambda_opt,
                "horizon": horizon,
                "beta": beta.tolist(),
                "n_factors": 18,
            },
            f,
            indent=2,
        )
    np.save(os.path.join(output_dir, "true_beta.npy"), beta)

    achieved_event_rate = float(np.mean(event_indicator))
    masked_eligible_cells = mask_eval[eligible_cols].sum().sum()
    total_eligible_cells = len(df_masked) * len(eligible_cols)
    share_masked_eligible = float(masked_eligible_cells / total_eligible_cells)

    print(
        f"P={P}, Achieved event rate={achieved_event_rate:.4f}, Share of masked cells among eligible={share_masked_eligible:.4f}"
    )


if __name__ == "__main__":
    generate_synthetic_cohort()

