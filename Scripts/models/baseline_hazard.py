import torch

def compute_breslow_baseline_hazard(eta, event_times, event_indicators, eval_times):
    """
    Computes the Breslow-weighted cumulative baseline hazard H_0(t) at eval_times.

    Args:
        eta: risk scores, shape [N], detached (runs under no_grad)
        event_times: shape [N]
        event_indicators: shape [N], 1 if event, 0 if censored
        eval_times: the time points t_1, ..., t_K to evaluate H_0

    Returns:
        H_0: shape [K], cumulative baseline hazard at each eval time
    """
    device = eta.device
    
    # Find unique event times and sort them
    is_event = event_indicators == 1
    event_times_only = event_times[is_event]
    
    if len(event_times_only) == 0:
        return torch.zeros_like(eval_times, dtype=torch.float32)
        
    distinct_times = torch.unique(event_times_only).sort()[0]
    
    H0_j = []
    current_H0 = 0.0
    exp_eta = torch.exp(eta)
    
    # Iterate over distinct event times
    for t_j in distinct_times:
        # d_j: number of events EXACTLY at t_j
        d_j = ((event_times == t_j) & (event_indicators == 1)).sum().float()
        
        # Risk set denominator: sum(exp(eta_k)) for all k where T_k >= t_j
        risk_set_mask = event_times >= t_j
        denom = exp_eta[risk_set_mask].sum()
        
        # Breslow update (handling ties naturally through d_j)
        if denom > 0:
            current_H0 += (d_j / denom).item()
            
        H0_j.append(current_H0)
        
    distinct_times = distinct_times.to(device)
    H0_j = torch.tensor(H0_j, device=device, dtype=torch.float32)
    
    # Interpolate for eval_times (step function right-continuous)
    H0_eval = torch.zeros_like(eval_times, dtype=torch.float32)
    for i, t in enumerate(eval_times):
        valid_idx = (distinct_times <= t).nonzero(as_tuple=True)[0]
        if len(valid_idx) > 0:
            H0_eval[i] = H0_j[valid_idx[-1]]
        else:
            H0_eval[i] = 0.0
            
    return H0_eval
