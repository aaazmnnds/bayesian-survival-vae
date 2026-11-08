import torch
import torch.nn as nn

class CoxPredictor(nn.Module):
    def __init__(self, latent_dim: int, dropout_prob: float = 0.0):
        """
        Cox Predictor for the B-SVAE, mapping latent z to log-hazard ratio eta.
        
        Args:
            latent_dim: Dimension of the latent space z.
            dropout_prob: Dropout probability before the linear layer.
        """
        super().__init__()
        
        # Cox partial likelihood is invariant to additive shifts in eta,
        # so bias=False ensures identifiability.
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(latent_dim, 1, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Latent tensor of shape [batch_size, latent_dim]
            
        Returns:
            eta: Unbounded log-hazard-ratio risk score of shape [batch_size]
        """
        # Shape: [batch_size, 1]
        eta = self.fc(self.dropout(z))
        
        # Squeeze to shape [batch_size] for convenience in downstream loss and KM bins
        return eta.squeeze(-1)
