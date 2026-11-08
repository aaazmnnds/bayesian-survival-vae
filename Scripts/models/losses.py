import torch
import torch.nn.functional as F

def compute_base_losses(x_true, m, decoder_out, mu_z, log_var_z, eta, T, E, 
                        cont_idx, bin_idx, cat_groups):
    # NaN safe replacement before any arithmetic
    x_safe = torch.where(torch.isnan(x_true), torch.zeros_like(x_true), x_true)
    
    L_NLL_recon = 0.0
    
    # Continuous
    if cont_idx and 'x_recon_cont' in decoder_out:
        mu_x, log_var_x = decoder_out['x_recon_cont']
        x_cont = x_safe[:, cont_idx]
        m_cont = m[:, cont_idx]
        
        nll_cont = 0.5 * (torch.log(torch.tensor(2 * torch.pi)) + log_var_x + 
                          ((x_cont - mu_x)**2) / torch.exp(log_var_x))
        L_NLL_recon += (nll_cont * m_cont).sum()
        
    # Binary
    if bin_idx and 'x_recon_bin' in decoder_out:
        logits_bin = decoder_out['x_recon_bin']
        x_bin = x_safe[:, bin_idx]
        m_bin = m[:, bin_idx]
        
        bce_bin = F.binary_cross_entropy_with_logits(logits_bin, x_bin, reduction='none')
        L_NLL_recon += (bce_bin * m_bin).sum()
        
    # Categorical
    if cat_groups and 'x_recon_cat' in decoder_out:
        logits_cat = decoder_out['x_recon_cat']
        
        current_offset = 0
        for feat_idx, K in cat_groups:
            group_logits = logits_cat[:, current_offset : current_offset + K]
            targets = x_safe[:, feat_idx].long()
            
            ce_loss = F.cross_entropy(group_logits, targets, reduction='none')
            m_group = m[:, feat_idx]
            
            L_NLL_recon += (ce_loss * m_group).sum()
            current_offset += K
            
    # KL Divergence
    kl_div = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp())
    L_latent = kl_div
    
    # Cox Loss
    # Vectorized Breslow using descending sort and logcumsumexp
    if E.sum() == 0:
        L_Cox = (eta * 0).sum()
    else:
        # Sort descending by T
        idx_sort = torch.argsort(T, descending=True)
        T_sort = T[idx_sort]
        E_sort = E[idx_sort]
        eta_sort = eta[idx_sort]
        
        # log sum_{j: T_j >= T_i} exp(eta_j)
        # Because we sorted descending, the cumulative sum from index 0 to i is exactly T_j >= T_i.
        risk_set_log_sum = torch.logcumsumexp(eta_sort, dim=0)
        
        # To handle tied times correctly: the risk set sum for all tied subjects must include all of them.
        # So we want to replace risk_set_log_sum[i] with the value from the last occurrence of T_i.
        
        # Find unique T and their max index (last occurrence in descending sort)
        unique_t, inv_idx, counts = torch.unique_consecutive(T_sort, return_inverse=True, return_counts=True)
        
        # The last index of each group in the descending sorted array
        # counts.cumsum(0) - 1 gives the end index of each block of ties
        last_indices = counts.cumsum(0) - 1
        
        # Gather the corrected risk_set_log_sum
        tied_risk_set_log_sum = risk_set_log_sum[last_indices[inv_idx]]
        
        # Only compute the loss for subjects with an event
        event_mask = E_sort == 1
        eta_events = eta_sort[event_mask]
        risk_set_events = tied_risk_set_log_sum[event_mask]
        
        L_Cox = -torch.sum(eta_events - risk_set_events)
        
    return L_NLL_recon, L_latent, L_Cox
