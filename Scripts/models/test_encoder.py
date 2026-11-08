import torch
from encoder import Encoder

# Small batch size
batch_size = 4
input_dim = 140  # 2 * 70 for MIMIC-IV-sized input
latent_dim = 16
hidden_dims = [32, 64]

encoder = Encoder(input_dim, hidden_dims, latent_dim)

# Generate dummy input
x = torch.randn(batch_size, input_dim)

# Forward pass
mu_z, log_var_z = encoder(x)

print("Output shapes:")
print(f"mu_z: {mu_z.shape}")
print(f"log_var_z: {log_var_z.shape}")

print(f"log_var_z range: min={log_var_z.min().item():.4f}, max={log_var_z.max().item():.4f}")
