import torch
import numpy as np
from harness_metrics import breslow_baseline

def on_epoch_start(model, x_filled_all, m_all, event_times_all, event_indicators_all, eval_times, n_bins=10):
    model.set_mc_dropout(False)
    
    with torch.no_grad():
        out = model(x_filled_all, m_all, sample_z=False)
        eta_all = out['eta']
        
        quantiles = torch.linspace(0, 1, n_bins + 1, device=x_filled_all.device)
        bin_edges = torch.quantile(eta_all, quantiles)
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5
        
        bin_assignments = torch.bucketize(eta_all, bin_edges[1:-1])
        
        S_b_KM = torch.zeros((n_bins, len(eval_times)), device=x_filled_all.device)
        for b in range(n_bins):
            b_mask = (bin_assignments == b)
            if b_mask.sum() == 0:
                S_b_KM[b] = 1.0
                continue
                
            b_times = event_times_all[b_mask]
            b_events = event_indicators_all[b_mask]
            
            is_event_b = (b_events == 1)
            event_times_b = b_times[is_event_b]
            
            if len(event_times_b) > 0:
                distinct_t = torch.unique(event_times_b).sort()[0]
                current_S = 1.0
                S_j = []
                for t_j in distinct_t:
                    d_j = ((b_times == t_j) & (b_events == 1)).sum().float()
                    n_j = (b_times >= t_j).sum().float()
                    if n_j > 0:
                        current_S *= (1.0 - d_j / n_j).item()
                    S_j.append(current_S)
                    
                distinct_t = distinct_t.to(x_filled_all.device)
                S_j = torch.tensor(S_j, device=x_filled_all.device, dtype=torch.float32)
                
                for i, t in enumerate(eval_times):
                    valid_idx = (distinct_t <= t).nonzero(as_tuple=True)[0]
                    if len(valid_idx) > 0:
                        S_b_KM[b, i] = S_j[valid_idx[-1]]
                    else:
                        S_b_KM[b, i] = 1.0
            else:
                S_b_KM[b] = 1.0

        eta_np = eta_all.cpu().numpy()
        T_np = event_times_all.cpu().numpy()
        E_np = event_indicators_all.cpu().numpy()
        
        unique_t, cum_hazard = breslow_baseline(eta_np, T_np, E_np)
        
        if len(unique_t) == 0:
            H0_eval = torch.zeros(len(eval_times), dtype=torch.float32, device=x_filled_all.device)
        else:
            idx = np.searchsorted(unique_t, eval_times.cpu().numpy(), side='right') - 1
            H0_eval_np = np.zeros_like(eval_times.cpu().numpy(), dtype=float)
            valid = idx >= 0
            H0_eval_np[valid] = cum_hazard[idx[valid]]
            H0_eval = torch.tensor(H0_eval_np, dtype=torch.float32, device=x_filled_all.device)
        
    model.train()
    
    return {
        'bin_assignments': bin_assignments.detach(),
        'S_b_KM': S_b_KM.detach(),
        'H_0': H0_eval.detach()
    }

def compute_L_cal(model, x_filled, m_mask, bin_assignments_for_batch, S_b_KM_cached, H_0_cached, 
                  eval_times, alpha=0.05, n_passes=10, lambda_val=10.0, steepness=100.0):
    
    n_batch = x_filled.shape[0]
    K = len(eval_times)
    device = x_filled.device
    
    model.set_mc_dropout(True)
    
    S_i_m_tk = torch.zeros((n_passes, n_batch, K), device=device)
    
    for p in range(n_passes):
        out = model(x_filled, m_mask, sample_z=False)
        eta_p = out['eta']
        
        H_0_expanded = H_0_cached.unsqueeze(0).expand(n_batch, K)
        eta_expanded = eta_p.unsqueeze(1).expand(n_batch, K)
        
        S_i = torch.ones_like(H_0_expanded)
        mask_gt0 = H_0_expanded > 0
        if mask_gt0.any():
            log_H0 = torch.log(H_0_expanded[mask_gt0])
            clamped = torch.clamp(log_H0 + eta_expanded[mask_gt0], max=50.0)
            S_i[mask_gt0] = torch.exp(-torch.exp(clamped))
            
        S_i_m_tk[p] = S_i
        
    q_low = alpha / 2.0
    q_high = 1.0 - (alpha / 2.0)
    
    lower_bound = torch.quantile(S_i_m_tk, q_low, dim=0)
    upper_bound = torch.quantile(S_i_m_tk, q_high, dim=0)
    
    target_S = S_b_KM_cached[bin_assignments_for_batch]
    
    covered = torch.sigmoid(steepness * (target_S - lower_bound)) * torch.sigmoid(steepness * (upper_bound - target_S))
    
    width_per_k = ((upper_bound - lower_bound) * covered).sum(dim=0) / (covered.sum(dim=0) + 1e-8)
    width = width_per_k.mean()
    
    PICP_k = covered.mean(dim=0)
    
    scaling_factor = n_batch / (alpha * (1.0 - alpha))
    penalty = torch.relu((1.0 - alpha) - PICP_k).pow(2).mean()
    
    L_cal = width + lambda_val * scaling_factor * penalty
    return L_cal
