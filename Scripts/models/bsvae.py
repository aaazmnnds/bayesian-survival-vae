import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from encoder import Encoder
from decoder import Decoder
from cox_predictor import CoxPredictor

class BSVAE(nn.Module):
    def __init__(self, num_continuous: int, num_binary: int, categorical_groups: list,
                 hidden_dims_enc: list, hidden_dims_dec: list, latent_dim: int,
                 dropout_prob: float = 0.2):
        """
        Bayesian Survival Variational Autoencoder (B-SVAE).
        """
        super().__init__()
        
        self.num_continuous = num_continuous
        self.num_binary = num_binary
        self.categorical_groups = categorical_groups
        self.P = num_continuous + num_binary + len(categorical_groups)
        
        self.encoder = Encoder(
            input_dim=2 * self.P,
            hidden_dims=hidden_dims_enc,
            latent_dim=latent_dim,
            dropout_prob=dropout_prob
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims_dec,
            num_continuous=num_continuous,
            num_binary=num_binary,
            categorical_groups=categorical_groups,
            dropout_prob=dropout_prob
        )
        
        self.cox_predictor = CoxPredictor(latent_dim=latent_dim, dropout_prob=dropout_prob)

    def reparameterize(self, mu_z: torch.Tensor, log_var_z: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var_z)
        eps = torch.randn_like(std)
        return mu_z + eps * std

    def set_mc_dropout(self, active: bool):
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train(mode=active)

    @torch.no_grad()
    def iwae_bound(self, x_filled, m, x_target, cont_idx, bin_idx, cat_groups, K=50):
        self.set_mc_dropout(False)
        
        encoder_input = torch.cat([x_filled, m], dim=-1)
        mu_z, log_var_z = self.encoder(encoder_input)
        
        batch_size = x_filled.size(0)
        latent_dim = mu_z.size(1)
        
        # We draw K samples for each subject
        std_z = torch.exp(0.5 * log_var_z)
        
        # Expand for K samples
        mu_z_k = mu_z.unsqueeze(1).expand(batch_size, K, latent_dim)
        std_z_k = std_z.unsqueeze(1).expand(batch_size, K, latent_dim)
        eps = torch.randn_like(std_z_k)
        z_k = mu_z_k + eps * std_z_k
        
        # log q(z|x,m)
        log_q_z = -0.5 * (math.log(2 * math.pi) + log_var_z.unsqueeze(1) + eps.pow(2))
        log_q_z = log_q_z.sum(dim=-1) # [batch_size, K]
        
        # log p(z)
        log_p_z = -0.5 * (math.log(2 * math.pi) + z_k.pow(2))
        log_p_z = log_p_z.sum(dim=-1) # [batch_size, K]
        
        # decode all z_k
        z_k_flat = z_k.view(batch_size * K, latent_dim)
        decoder_out = self.decoder(z_k_flat)
        
        # Compute log p(x_obs|z_k)
        log_p_x = torch.zeros(batch_size * K, device=x_filled.device)
        
        # x_target has NaNs in unobserved cells. We MUST replace them so there are no NaNs propagating.
        x_target_safe = torch.where(torch.isnan(x_target), torch.zeros_like(x_target), x_target)
        x_target_exp = x_target_safe.repeat_interleave(K, dim=0)
        m_exp = m.repeat_interleave(K, dim=0)
        
        if cont_idx and 'x_recon_cont' in decoder_out:
            mu_x, log_var_x = decoder_out['x_recon_cont']
            x_cont = x_target_exp[:, cont_idx]
            m_cont = m_exp[:, cont_idx]
            
            # log N(x|mu, sigma^2)
            log_p_cont = -0.5 * (math.log(2 * math.pi) + log_var_x + ((x_cont - mu_x).pow(2) / torch.exp(log_var_x)))
            log_p_x += (log_p_cont * m_cont).sum(dim=-1)
            
        if bin_idx and 'x_recon_bin' in decoder_out:
            logits_bin = decoder_out['x_recon_bin']
            x_bin = x_target_exp[:, bin_idx]
            m_bin = m_exp[:, bin_idx]
            
            log_p_bin = -F.binary_cross_entropy_with_logits(logits_bin, x_bin, reduction='none')
            log_p_x += (log_p_bin * m_bin).sum(dim=-1)
            
        if cat_groups and 'x_recon_cat' in decoder_out:
            logits_cat = decoder_out['x_recon_cat']
            
            current_offset = 0
            for group_feat_idx, num_classes in cat_groups:
                group_logits = logits_cat[:, current_offset : current_offset + num_classes]
                
                targets = x_target_exp[:, group_feat_idx].long()
                
                ce_loss = F.cross_entropy(group_logits, targets, reduction='none')
                m_group = m_exp[:, group_feat_idx]
                
                log_p_x += (-ce_loss * m_group)
                
                current_offset += num_classes
                
        log_p_x = log_p_x.view(batch_size, K)
        
        log_w_k = log_p_x + log_p_z - log_q_z
        iwae = torch.logsumexp(log_w_k, dim=1) - math.log(K)
        
        return iwae

    def forward(self, x_filled: torch.Tensor, m: torch.Tensor, sample_z: bool = True) -> dict:
        encoder_input = torch.cat([x_filled, m], dim=-1)
        
        mu_z, log_var_z = self.encoder(encoder_input)
        if sample_z:
            z = self.reparameterize(mu_z, log_var_z)
        else:
            z = mu_z
        
        decoder_out = self.decoder(z)
        eta = self.cox_predictor(z)
        
        return {
            'mu_z': mu_z,
            'log_var_z': log_var_z,
            'z': z,
            'decoder_out': decoder_out,
            'eta': eta
        }
