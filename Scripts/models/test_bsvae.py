import torch
import pytest
from bsvae import BSVAE
from losses import compute_base_losses
import math

def test_bsvae_cox_loss():
    # Vectorized Cox loss equals a reference Python double loop WITH tied times (to 1e-5)
    T = torch.tensor([10.0, 10.0, 5.0, 15.0, 10.0, 2.0, 25.0, 12.0])
    E = torch.tensor([1, 0, 1, 1, 1, 1, 1, 0])
    eta = torch.tensor([0.1, 0.5, -0.2, 1.2, 0.0, -1.0, 0.8, 0.3])
    
    # Reference Python double loop
    ref_loss = 0.0
    for i in range(len(T)):
        if E[i] == 1:
            risk_set_sum = 0.0
            for j in range(len(T)):
                if T[j] >= T[i]:
                    risk_set_sum += math.exp(eta[j].item())
            ref_loss -= (eta[i].item() - math.log(risk_set_sum))
            
    # Vectorized compute_base_losses call
    # We only need L_Cox, so dummy out everything else
    _, _, L_Cox = compute_base_losses(
        x_true=torch.zeros(len(T), 1),
        m=torch.zeros(len(T), 1),
        decoder_out={},
        mu_z=torch.zeros(len(T), 1),
        log_var_z=torch.zeros(len(T), 1),
        eta=eta,
        T=T,
        E=E,
        cont_idx=[],
        bin_idx=[],
        cat_groups=[]
    )
    
    assert abs(L_Cox.item() - ref_loss) < 1e-5, f"Vectorized {L_Cox.item()} != Ref {ref_loss}"

def test_bsvae_dropout_mc():
    model = BSVAE(num_continuous=2, num_binary=1, categorical_groups=[2],
                  hidden_dims_enc=[16], hidden_dims_dec=[16], latent_dim=8, dropout_prob=0.3)
    
    x = torch.randn(4, 4)
    m = torch.ones(4, 4)
    
    # With set_mc_dropout(True), two forward passes give different eta
    model.set_mc_dropout(True)
    out1 = model(x, m, sample_z=False)
    out2 = model(x, m, sample_z=False)
    assert not torch.allclose(out1['eta'], out2['eta']), "Eta should differ with MC dropout"
    
    # With set_mc_dropout(False), they are identical
    model.set_mc_dropout(False)
    out3 = model(x, m, sample_z=False)
    out4 = model(x, m, sample_z=False)
    assert torch.allclose(out3['eta'], out4['eta']), "Eta should be identical without MC dropout"

def test_bsvae_forward_sample_z():
    model = BSVAE(num_continuous=2, num_binary=1, categorical_groups=[2],
                  hidden_dims_enc=[16], hidden_dims_dec=[16], latent_dim=8, dropout_prob=0.0)
    
    x = torch.randn(4, 4)
    m = torch.ones(4, 4)
    
    out = model(x, m, sample_z=False)
    assert torch.allclose(out['z'], out['mu_z']), "forward(sample_z=False) returns z == mu_z"

def test_bsvae_iwae_bound():
    torch.manual_seed(0)
    model = BSVAE(num_continuous=2, num_binary=0, categorical_groups=[],
                  hidden_dims_enc=[16], hidden_dims_dec=[16], latent_dim=8, dropout_prob=0.0)
    
    x_filled = torch.randn(256, 2)
    m = torch.ones(256, 2)
    x_target = x_filled.clone()
    
    cont_idx = [0, 1]
    bin_idx = []
    cat_groups = []
    
    model.set_mc_dropout(False)
    
    k1_vals = []
    for _ in range(10):
        iwae_k1 = model.iwae_bound(x_filled, m, x_target, cont_idx, bin_idx, cat_groups, K=1)
        assert torch.isfinite(iwae_k1).all()
        k1_vals.append(iwae_k1.mean().item())
        
    k50_vals = []
    for _ in range(10):
        iwae_k50 = model.iwae_bound(x_filled, m, x_target, cont_idx, bin_idx, cat_groups, K=50)
        assert torch.isfinite(iwae_k50).all()
        k50_vals.append(iwae_k50.mean().item())
        
    mean_k1 = sum(k1_vals) / 10
    mean_k50 = sum(k50_vals) / 10
    
    assert mean_k50 >= mean_k1 - 1e-3, f"Mean IWAE should not decrease from K=1 to K=50 (K=1: {mean_k1}, K=50: {mean_k50})"
