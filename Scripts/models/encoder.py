import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, dropout_prob: float = 0.2):
        """
        B-SVAE Encoder parsing the concatenated [x, m] vector into latent distribution parameters.
        
        Args:
            input_dim: Dimension of the input (typically 2 * P, where P is the total feature count).
            hidden_dims: List of hidden layer dimensions for the shared MLP.
            latent_dim: Dimension of the latent space z.
            dropout_prob: Dropout probability (must be active for MC Dropout inference).
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            in_dim = h_dim
            
        self.shared_mlp = nn.Sequential(*layers)
        
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_log_var = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the encoder.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim], typically [x_obs, m].
            
        Returns:
            Tuple containing:
            - mu_z: Mean of the latent distribution.
            - log_var_z: Log-variance of the latent distribution, clamped to [-10, 10].
        """
        h = self.shared_mlp(x)
        
        mu_z = self.fc_mu(h)
        log_var_z = self.fc_log_var(h)
        
        # Clamp log_var_z for numerical stability
        log_var_z = torch.clamp(log_var_z, min=-10.0, max=10.0)
        
        return mu_z, log_var_z
