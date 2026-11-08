import torch
from decoder import Decoder

# Small batch size
batch_size = 4
latent_dim = 16
hidden_dims = [32, 64]
num_continuous = 58
num_binary = 11
categorical_groups = [3, 4]  # Two categorical features, one with 3 classes, one with 4

decoder = Decoder(latent_dim, hidden_dims, num_continuous, num_binary, categorical_groups)

# Generate dummy latent
z = torch.randn(batch_size, latent_dim)

# Forward pass
outputs = decoder(z)

mu, log_var = outputs['x_recon_cont']
bin_logits = outputs['x_recon_bin']
cat_logits = outputs['x_recon_cat']

print("Output shapes:")
print(f"mu: {mu.shape}")
print(f"log_var: {log_var.shape}")
print(f"bin_logits: {bin_logits.shape}")
print(f"cat_logits: {cat_logits.shape}")

print(f"log_var range: min={log_var.min().item():.4f}, max={log_var.max().item():.4f}")
