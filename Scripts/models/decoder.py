import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list, 
                 num_continuous: int, num_binary: int, categorical_groups: list,
                 dropout_prob: float = 0.2):
        """
        B-SVAE Decoder parsing the latent space into three distinct feature type slices.
        
        Args:
            latent_dim: Dimension of the latent space z.
            hidden_dims: List of hidden layer dimensions for the shared MLP.
            num_continuous: Number of continuous features (each gets mu and log_var).
            num_binary: Number of binary features (each gets 1 logit).
            categorical_groups: List containing the number of classes for each categorical feature.
            dropout_prob: Dropout probability (active during MC Dropout inference).
        """
        super().__init__()
        
        self.num_continuous = num_continuous
        self.num_binary = num_binary
        self.categorical_groups = categorical_groups
        self.total_categorical_dims = sum(categorical_groups) if categorical_groups else 0
        
        # Shared MLP base
        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            in_dim = h_dim
            
        self.shared_mlp = nn.Sequential(*layers)
        
        # Branching Heads
        # Continuous head predicts (mu, log_var) for each continuous feature
        if self.num_continuous > 0:
            self.head_continuous = nn.Linear(in_dim, self.num_continuous * 2)
            
        # Binary head predicts a single logit per binary feature
        if self.num_binary > 0:
            self.head_binary = nn.Linear(in_dim, self.num_binary)
            
        # Categorical head predicts C logits per categorical feature
        if self.total_categorical_dims > 0:
            self.head_categorical = nn.Linear(in_dim, self.total_categorical_dims)

    def forward(self, z: torch.Tensor):
        """
        Forward pass for the decoder.
        
        Returns:
            Dict containing:
            - 'x_recon_cont': (mu, log_var) for continuous features if they exist.
            - 'x_recon_bin': logits for binary features if they exist.
            - 'x_recon_cat': logits for categorical features if they exist.
        """
        h = self.shared_mlp(z)
        
        outputs = {}
        
        if self.num_continuous > 0:
            cont_out = self.head_continuous(h)
            # Shape: [batch_size, num_continuous * 2]
            # Split into mu and log_var (log_sigma2). 
            # We predict log_var directly to match the exact mathematical formulation 
            # used in the NLL reconstruction loss and MC propagation.
            mu = cont_out[:, :self.num_continuous]
            log_var = cont_out[:, self.num_continuous:]
            
            # Clamp log_var for numerical stability
            log_var = torch.clamp(log_var, min=-10.0, max=10.0)
            
            outputs['x_recon_cont'] = (mu, log_var)
            
        if self.num_binary > 0:
            # Predict raw logits for BCEWithLogitsLoss
            outputs['x_recon_bin'] = self.head_binary(h)
            
        if self.total_categorical_dims > 0:
            # Predict raw logits for grouped CrossEntropyLoss
            outputs['x_recon_cat'] = self.head_categorical(h)
            
        return outputs
