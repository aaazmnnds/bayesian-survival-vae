# Bayesian Survival Variational Autoencoder (B-SVAE)

Code for the study "Bayesian Survival Variational Autoencoder: Uncertainty-Aware Missing Data Imputation for Clinical Survival Analysis" (manuscript in preparation). B-SVAE extends MNAR-SVAE with a heteroscedastic decoder, Monte Carlo dropout, and a calibration regularizer, and is compared with seven methods on METABRIC, MIMIC-IV, and eICU under simulated threshold-interaction MNAR missingness, and on a synthetic dataset.

## Data

No data are included. MIMIC-IV (version 2.2, https://doi.org/10.13026/6mm1-ek67) and eICU-CRD (version 2.0, https://doi.org/10.13026/C2WM1R) require credentialed access through PhysioNet. METABRIC clinical data are downloaded from cBioPortal (https://www.cbioportal.org/study/summary?id=brca_metabric) by the extraction script. Cohorts derived from MIMIC-IV and eICU must not be shared outside the PhysioNet data use agreement.

The only data file in this repository is `Results/final/bounds_approved.csv`, the physiological plausibility bounds used in cohort cleaning. It contains no patient data.

## Structure

- `Scripts/`: cohort extraction, Sepsis-3 derivation, MNAR injection, synthetic data generation.
- `Scripts/models/`: evaluation harness, B-SVAE, comparison methods, tuning, final evaluation, results tables, tests.
- `requirements/`: package versions of the two computing environments (A: B-SVAE, MNAR-SVAE ablation, MICE, MIWAE; B: missForest, GAIN, MC Dropout Cox, DeepHit).

## Run order

Run from the repository root with `export PYTHONPATH=Scripts:Scripts/models`.

1. Sepsis-3 tables (MIMIC-IV): `python Scripts/generate_derived_tables.py --mimic-dir <path>`
2. Cohorts: `python Scripts/extract_metabric_clinical.py`, `python Scripts/extract_mimic_sepsis_cohort.py --mimic-dir <path>`, `python Scripts/extract_eicu_cohort.py --eicu-dir <path>`
3. Splits and masks: `python Scripts/models/generate_real_splits.py`, `python Scripts/models/generate_mimic_eicu_mnar.py`, `python Scripts/models/generate_metabric_mnar.py`
4. Leakage audit: `python Scripts/models/audit_leakage.py --dataset <d> --scenario <s>`
5. Tuning, per dataset (`metabric`, `mimic`, `eicu`), scenario (`light`, `moderate`, `severe`), and fold (1 to 5):
   `python Scripts/models/optimize_bsvae.py --dataset <d> --scenario <s> --fold <f> --trials 60 --seed 0`, and likewise `optimize_survival_vae.py`, `optimize_gain.py`, `optimize_miwae.py`, `optimize_mc_dropout_cox.py`, `optimize_deephit.py`, and `optimize_baselines.py --method mice` or `--method missforest`. MC Dropout Cox and DeepHit require the MICE selections.
6. Final evaluation: `python Scripts/models/aggregate_bsvae_5fold.py --dataset <d> --scenario <s>`, and likewise the other `aggregate_*_5fold.py` scripts (`aggregate_baselines_5fold.py` takes `--method`).
7. Synthetic study: `python Scripts/generate_synthetic_cohort.py`, `python Scripts/models/generate_real_splits.py`, tuning with `optimize_bsvae.py --dataset synthetic --scenario moderate --fold 1`, `aggregate_bsvae_5fold.py --dataset synthetic --scenario moderate --folds 1`, then `python Scripts/models/synthetic_validation.py --subsample` and `--analyze`.
8. Tables: `python Scripts/models/make_results_tables.py`
9. Tests: `python -m pytest Scripts/models`

`Scripts/make_bounds_template.py --mimic-dir <path> --eicu-dir <path>` creates an empty bounds template; it is not needed to reproduce the study.

## License

MIT License. See `LICENSE`.
