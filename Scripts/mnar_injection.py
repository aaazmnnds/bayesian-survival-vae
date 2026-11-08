import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def apply_threshold_interaction_mnar(X: pd.DataFrame, 
                                     severity: str = 'moderate',
                                     random_state: int = 42,
                                     age_col: str = 'Age',
                                     lactate_col: str = 'Lactate',
                                     eligible_cols: list = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Injects threshold-interaction MNAR missingness into continuous features of X.
    
    The probability of missingness depends on an interaction between age (thresholded at median)
    and normalized lactate.
    
    Args:
        X: Fully observed feature matrix (or partially observed real data).
        severity: 'light', 'moderate', or 'severe'.
        random_state: Random seed for reproducibility.
        age_col: Name of the Age column.
        lactate_col: Name of the Lactate column.
        
    Returns:
        X_masked: Feature matrix with injected NaNs.
        M_eval: Binary mask indicating which values were artificially hidden 
                (1 if artificially hidden, 0 otherwise).
    """
    np.random.seed(random_state)
    
    # MNAR parameters per severity
    scenarios = {
        'light':    {'alpha': 1.0, 'lambda': -1.44},
        'moderate': {'alpha': 3.0, 'lambda': -0.39},
        'severe':   {'alpha': 5.0, 'lambda': 0.75}
    }
    params = scenarios.get(severity, scenarios['moderate'])
    alpha = params['alpha']
    lam = params['lambda']
    
    X_masked = X.copy()
    M_eval = pd.DataFrame(0, index=X.index, columns=X.columns)
    
    # Thresholding covariate (X_A)
    median_age = X[age_col].median()
    age_indicator = (X[age_col] > median_age).astype(float)
    
    # Interaction covariate (X_B)
    lactate_vals = X[lactate_col].copy()
    lactate_mean = lactate_vals.mean()
    lactate_std = lactate_vals.std()
    lactate_filled = lactate_vals.fillna(lactate_mean)
    lactate_norm = (lactate_filled - lactate_mean) / lactate_std
    
    interaction_term = age_indicator * lactate_norm
    
    # Probability of missingness
    prob_missing = sigmoid(alpha * interaction_term + lam)
    
    print(f"--- MNAR Scenario: {severity} ---")
    print(f"Mechanism parameters: alpha={alpha}, lambda={lam}")
    print(f"Constants computed: median_age={median_age:.2f}, {lactate_col}_mean={lactate_mean:.4f}, {lactate_col}_std={lactate_std:.4f}")
    
    if eligible_cols is not None:
        continuous_cols = [c for c in eligible_cols if c in X.columns]
    else:
        continuous_cols = X.select_dtypes(include=[np.number]).columns
    
    for var in continuous_cols:
        # In the original MNAR-SVAE codebase, Age was excluded from the target variables,
        # but Lactate was included in the target variables and eligible for masking.
        if var == age_col:
            continue
            
        observed_indices = X[var].notna()
        
        random_draw = np.random.rand(len(X))
        should_be_missing = random_draw < prob_missing
        
        to_mask = should_be_missing & observed_indices
        
        X_masked.loc[to_mask, var] = np.nan
        M_eval.loc[to_mask, var] = 1
        
    print(f"Total missingness injected: {M_eval.sum().sum()} cells")
    total_eligible = M_eval.shape[0] * len(continuous_cols)
    print(f"Overall injected rate (among eligible): {M_eval.sum().sum() / total_eligible:.4f}")
    
    for var in continuous_cols:
        if var == age_col: continue
        rate = M_eval[var].sum() / len(M_eval)
        print(f"  {var}: {rate:.4f}")
        
    return X_masked, M_eval
