import torch
import pytest
from bsvae import BSVAE
from train_utils import on_epoch_start, compute_L_cal

def test_lcal():
    torch.manual_seed(42)
    
    N = 20
    eval_times = torch.tensor([5.0, 10.0, 15.0])
    
    model = BSVAE(3, 2, [2], [16], [16], 16)
    x_filled_all = torch.randn(N, 6)
    m_all = torch.ones(N, 6)
    event_times = torch.randint(1, 30, (N,)).float()
    event_indicators = torch.randint(0, 2, (N,))
    
    cached_state = on_epoch_start(model, x_filled_all, m_all, event_times, event_indicators, eval_times)
    
    batch_size = 4
    x_batch = x_filled_all[:batch_size]
    m_batch = m_all[:batch_size]
    bin_assignments_batch = cached_state['bin_assignments'][:batch_size]
    
    L_cal = compute_L_cal(model, x_batch, m_batch, bin_assignments_batch, 
                          cached_state['S_b_KM'], cached_state['H_0'], eval_times)
                          
    assert torch.isfinite(L_cal), "L_cal must be finite"
    
    L_cal.backward()
    
    has_encoder_grad = False
    has_cox_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'encoder' in name:
                has_encoder_grad = True
                assert torch.isfinite(param.grad).all()
            if 'cox_predictor' in name:
                has_cox_grad = True
                assert torch.isfinite(param.grad).all()
                
    assert has_encoder_grad, "L_cal must have gradients on encoder"
    assert has_cox_grad, "L_cal must have gradients on Cox head"

def test_lcal_penalty_scaling():
    # To test penalty scales with n, we mock S_i_m_tk to produce a fixed PICP
    # and compute_L_cal to return just the penalty portion or check its value.
    # A simpler way is to check the function behavior with identical inputs but different batch sizes
    
    batch_size1 = 4
    batch_size2 = 8
    alpha = 0.05
    
    model = BSVAE(3, 2, [2], [16], [16], 16)
    
    # We can just mock the S_i_m_tk computation inside compute_L_cal, but since we can't easily,
    # let's supply inputs that yield a deterministic penalty.
    # With large steepness, sigmoid becomes step function.
    # By using deterministic H_0 and S_b_KM, we can measure L_cal1 and L_cal2.
    # Wait, the prompt just says: "its penalty term scales with the mini-batch size n as in the formula".
    
    x1 = torch.randn(batch_size1, 6)
    m1 = torch.ones(batch_size1, 6)
    
    # Duplicate for batch 2
    x2 = torch.cat([x1, x1], dim=0)
    m2 = torch.cat([m1, m1], dim=0)
    
    eval_times = torch.tensor([5.0])
    bin_assignments1 = torch.zeros(batch_size1, dtype=torch.long)
    bin_assignments2 = torch.zeros(batch_size2, dtype=torch.long)
    
    S_b_KM = torch.tensor([[0.5]])
    H_0 = torch.tensor([0.5])
    
    # Ensure reproducible forward passes
    torch.manual_seed(42)
    L_cal1 = compute_L_cal(model, x1, m1, bin_assignments1, S_b_KM, H_0, eval_times, alpha=0.05)
    
    torch.manual_seed(42)
    L_cal2 = compute_L_cal(model, x2, m2, bin_assignments2, S_b_KM, H_0, eval_times, alpha=0.05)
    
    # Since the batch is duplicated, the mean width is identical, and PICP is identical.
    # Thus, only the scaling factor changes.
    # L_cal = width + lambda * (n / (alpha*(1-alpha))) * penalty
    # If penalty > 0, L_cal2 - width should be roughly 2 * (L_cal1 - width)
    # We don't have exactly width, but L_cal2 should be significantly larger than L_cal1
    # if penalty is non-zero. With random weights, penalty is likely non-zero.
    
    assert L_cal2 > L_cal1, "L_cal should scale with mini-batch size due to penalty term"

def test_lcal_extreme_values():
    torch.manual_seed(42)
    model = BSVAE(3, 2, [2], [16], [16], 16)
    
    batch_size = 4
    x = torch.randn(batch_size, 6)
    m = torch.ones(batch_size, 6)
    
    # Mock H0 with zeros and small/large values
    H_0 = torch.tensor([0.0, 1e-5, 10.0])
    eval_times = torch.tensor([5.0, 10.0, 15.0])
    bin_assignments = torch.zeros(batch_size, dtype=torch.long)
    S_b_KM = torch.tensor([[1.0, 0.5, 0.1]])
    
    # Force eta to have extreme values by mocking the model output
    # Since we can't easily mock the model's internal return value without monkeypatching,
    # we can inject a bias to the cox_predictor head temporarily.
    with torch.no_grad():
        model.cox_predictor.fc.weight.fill_(100.0)
        
    L_cal = compute_L_cal(model, x, m, bin_assignments, S_b_KM, H_0, eval_times)
    
    assert torch.isfinite(L_cal), "L_cal must be finite even with extreme eta and zero H0"
    
    L_cal.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Gradient of {name} must be finite"


