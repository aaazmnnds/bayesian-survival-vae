import torch
import torch.nn.functional as F
import pytest
from bsvae import BSVAE
from losses import compute_base_losses

def test_nan_masking_gradients():
    batch_size = 4
    num_continuous = 2
    num_binary = 1
    cat_groups = [(3, 3), (4, 5)] # (feat_idx, K) -> indices 3 and 4
    
    model = BSVAE(num_continuous, num_binary, [3, 5], [16], [16], 8)
    
    # x_true: inputs are standard continuous/binary/categorical-codes
    # P = 2(cont) + 1(bin) + 2(cat) = 5 features total
    x_true = torch.zeros(batch_size, 5)
    x_true[:, 0:2] = torch.randn(batch_size, 2)
    x_true[:, 2] = torch.randint(0, 2, (batch_size,)).float()
    x_true[:, 3] = torch.randint(0, 3, (batch_size,)).float()
    x_true[:, 4] = torch.randint(0, 5, (batch_size,)).float()
    
    # Mask
    m = torch.ones(batch_size, 5)
    
    # Introduce NaNs in unobserved cells
    m[0, 0] = 0.0
    x_true[0, 0] = float('nan')
    
    m[1, 2] = 0.0
    x_true[1, 2] = float('nan')
    
    m[2, 3] = 0.0
    x_true[2, 3] = float('nan')
    
    x_filled = torch.where(torch.isnan(x_true), torch.zeros_like(x_true), x_true)
    
    out = model(x_filled, m, sample_z=True)
    
    T = torch.tensor([10.0, 5.0, 20.0, 15.0])
    E = torch.tensor([1, 1, 0, 1])
    
    cont_idx = [0, 1]
    bin_idx = [2]
    cat_groups = cat_groups
    
    L_NLL, L_latent, L_Cox = compute_base_losses(
        x_true, m, out['decoder_out'], out['mu_z'], out['log_var_z'], out['eta'],
        T, E, cont_idx, bin_idx, cat_groups
    )
    
    loss = L_NLL + L_latent + L_Cox
    assert torch.isfinite(loss), "Loss must be finite with NaNs in x_true"
    
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Gradient of {name} has NaNs"

def test_categorical_logits_mapping():
    batch_size = 2
    cat_groups = [(0, 3), (1, 5)]
    
    x_true = torch.tensor([[1.0, 4.0], [2.0, 0.0]])
    m = torch.ones(2, 2)
    
    logits_cat = torch.randn(2, 8) # 3 + 5
    decoder_out = {'x_recon_cat': logits_cat}
    
    L_NLL, _, _ = compute_base_losses(
        x_true, m, decoder_out, torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2),
        torch.tensor([1.0, 1.0]), torch.tensor([0, 0]), [], [], cat_groups
    )
    
    # Check against manual F.cross_entropy
    group1_logits = logits_cat[:, 0:3]
    group1_targets = x_true[:, 0].long()
    ce1 = F.cross_entropy(group1_logits, group1_targets, reduction='sum')
    
    group2_logits = logits_cat[:, 3:8]
    group2_targets = x_true[:, 1].long()
    ce2 = F.cross_entropy(group2_logits, group2_targets, reduction='sum')
    
    expected_ce = ce1 + ce2
    assert torch.isclose(L_NLL, expected_ce), "Categorical logits mapping failed"
