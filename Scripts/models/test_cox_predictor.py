import torch
import pytest
from cox_predictor import CoxPredictor

def test_cox_predictor_dropout():
    latent_dim = 16
    batch_size = 4
    
    predictor = CoxPredictor(latent_dim, dropout_prob=0.0)
    predictor.train()
    
    z = torch.randn(batch_size, latent_dim)
    
    eta1 = predictor(z)
    eta2 = predictor(z)
    
    assert torch.allclose(eta1, eta2), "Passes should be identical with dropout_prob = 0"
    assert hasattr(predictor, 'dropout'), "Head must contain dropout"
    assert predictor.fc.bias is None, "Bias must be None for identifiability"
    assert eta1.shape == (batch_size,)
