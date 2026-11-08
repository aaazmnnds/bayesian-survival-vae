import argparse
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from harness_data import fill_for_input
from bsvae import BSVAE
from losses import compute_base_losses
from optimize_bsvae import BSVAEPlugin
import harness_metrics
import harness_runner


class FixedVarianceDecoder(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, z: torch.Tensor) -> dict:
        out = self.inner(z)
        if 'x_recon_cont' in out:
            mu, log_var = out['x_recon_cont']
            out['x_recon_cont'] = (mu, torch.zeros_like(log_var))
        return out


class MNARSVAEPlugin(BSVAEPlugin):
    def __init__(self, stage_epochs=(30, 20, 50), batch_size=256, train_mc_passes=10, val_mc_passes=20, device=None):
        super().__init__(stage_epochs=stage_epochs, batch_size=batch_size, train_mc_passes=train_mc_passes, val_mc_passes=val_mc_passes, device=device)
        self.name = 'mnar_svae'
        self.directions = ['minimize', 'maximize']
        self.pareto = True

    def search_space(self, trial):
        return {
            'beta': trial.suggest_float('beta', 1e-4, 10, log=True),
            'gamma': trial.suggest_float('gamma', 1e-2, 10, log=True),
            'latent_dim': trial.suggest_categorical('latent_dim', [8, 16, 32, 64]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'dropout_prob': trial.suggest_float('dropout_prob', 0.05, 0.5),
            'hidden': trial.suggest_categorical("hidden", ["64_32", "128_64", "128_64_32"])
        }

    def fit(self, train, val, params, seed):
        feature_names = train['feature_names']
        feature_types = train['feature_types']
        
        cont_cols = feature_types.get('continuous', [])
        bin_cols = feature_types.get('binary', [])
        cat_dict = feature_types.get('categorical', {})
        
        cont_idx = [feature_names.index(c) for c in cont_cols]
        bin_idx = [feature_names.index(c) for c in bin_cols]
        cat_groups = [(feature_names.index(c), len(cat_dict[c])) for c in cat_dict]
        
        K_list = [K for _, K in cat_groups]
        
        hidden_dims_enc = [int(x) for x in params['hidden'].split("_")]
        hidden_dims_dec = hidden_dims_enc[::-1]
        
        model = BSVAE(
            num_continuous=len(cont_idx),
            num_binary=len(bin_idx),
            categorical_groups=K_list,
            hidden_dims_enc=hidden_dims_enc,
            hidden_dims_dec=hidden_dims_dec,
            latent_dim=params['latent_dim'],
            dropout_prob=params['dropout_prob']
        ).to(self.device)
        model.decoder = FixedVarianceDecoder(model.decoder)
        
        x_filled = torch.tensor(fill_for_input(train['X_obs']), dtype=torch.float32, device=self.device)
        m = torch.tensor(train['M_obs'], dtype=torch.float32, device=self.device)
        target = torch.tensor(train['X_obs'], dtype=torch.float32, device=self.device)
        T = torch.tensor(train['T'], dtype=torch.float32, device=self.device)
        E = torch.tensor(train['E'], dtype=torch.float32, device=self.device)
        
        optimizer = Adam(model.parameters(), lr=params['learning_rate'])
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        
        N = len(x_filled)
        
        best_state = None
        best_cindex = -float('inf')
        
        current_epoch = 0
        
        # Stage 1: Cox head frozen, loss = L_recon + beta * L_latent
        for param in model.cox_predictor.parameters():
            param.requires_grad = False
            
        for ep in range(self.stage_epochs[0]):
            model.train()
            indices = torch.randperm(N, generator=gen, device=self.device)
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch_idx = indices[start:end]
                
                out = model(x_filled[batch_idx], m[batch_idx], sample_z=True)
                
                L_NLL, L_latent, _ = compute_base_losses(
                    target[batch_idx], m[batch_idx], out['decoder_out'], out['mu_z'], out['log_var_z'],
                    out['eta'], T[batch_idx], E[batch_idx], cont_idx, bin_idx, cat_groups
                )
                
                loss = L_NLL + params['beta'] * L_latent
                
                if not torch.isfinite(loss):
                    max_eta = out['eta'].abs().max().item()
                    comps = []
                    if not torch.isfinite(L_NLL): comps.append('L_NLL')
                    if not torch.isfinite(L_latent): comps.append('L_latent')
                    raise ValueError(f"Loss is {loss.item()} in stage 1, epoch {ep}. Not finite: {', '.join(comps)}. max |eta|: {max_eta}")
                    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            current_epoch += 1

        # Stage 2: Encoder and decoder frozen, loss = gamma * L_Cox
        for param in model.encoder.parameters(): param.requires_grad = False
        for param in model.decoder.parameters(): param.requires_grad = False
        for param in model.cox_predictor.parameters(): param.requires_grad = True
        
        for ep in range(self.stage_epochs[1]):
            model.train()
            indices = torch.randperm(N, generator=gen, device=self.device)
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch_idx = indices[start:end]
                
                out = model(x_filled[batch_idx], m[batch_idx], sample_z=True)
                
                _, _, L_Cox = compute_base_losses(
                    target[batch_idx], m[batch_idx], out['decoder_out'], out['mu_z'], out['log_var_z'],
                    out['eta'], T[batch_idx], E[batch_idx], cont_idx, bin_idx, cat_groups
                )
                
                loss = params['gamma'] * L_Cox
                
                if not torch.isfinite(loss):
                    max_eta = out['eta'].abs().max().item()
                    comps = []
                    if not torch.isfinite(L_Cox): comps.append('L_Cox')
                    raise ValueError(f"Loss is {loss.item()} in stage 2, epoch {ep}. Not finite: {', '.join(comps)}. max |eta|: {max_eta}")
                    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            current_epoch += 1
            
            # Checkpoint
            pred_val = self.predict({'model': model, 'cont_idx': cont_idx, 'bin_idx': bin_idx, 'cat_groups': cat_groups}, val, self.val_mc_passes)
            cidx = harness_metrics.cindex(val['T'], val['E'], pred_val['eta'])
            if cidx > best_cindex:
                best_cindex = cidx
                best_state = copy.deepcopy(model.state_dict())

        # Stage 3: All unfrozen, loss = L_recon + beta * L_latent + gamma * L_Cox
        for param in model.parameters(): param.requires_grad = True
        
        for ep in range(self.stage_epochs[2]):
            model.train()
            indices = torch.randperm(N, generator=gen, device=self.device)
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch_idx = indices[start:end]
                
                out = model(x_filled[batch_idx], m[batch_idx], sample_z=True)
                
                L_NLL, L_latent, L_Cox = compute_base_losses(
                    target[batch_idx], m[batch_idx], out['decoder_out'], out['mu_z'], out['log_var_z'],
                    out['eta'], T[batch_idx], E[batch_idx], cont_idx, bin_idx, cat_groups
                )
                
                loss = L_NLL + params['beta'] * L_latent + params['gamma'] * L_Cox
                
                if not torch.isfinite(loss):
                    max_eta = out['eta'].abs().max().item()
                    comps = []
                    if not torch.isfinite(L_NLL): comps.append('L_NLL')
                    if not torch.isfinite(L_latent): comps.append('L_latent')
                    if not torch.isfinite(L_Cox): comps.append('L_Cox')
                    raise ValueError(f"Loss is {loss.item()} in stage 3, epoch {ep}. Not finite: {', '.join(comps)}. max |eta|: {max_eta}")
                    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            current_epoch += 1
            
            # Checkpoint
            pred_val = self.predict({'model': model, 'cont_idx': cont_idx, 'bin_idx': bin_idx, 'cat_groups': cat_groups}, val, self.val_mc_passes)
            cidx = harness_metrics.cindex(val['T'], val['E'], pred_val['eta'])
            if cidx > best_cindex:
                best_cindex = cidx
                best_state = copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)
            
        return {
            'model': model,
            'cont_idx': cont_idx,
            'bin_idx': bin_idx,
            'cat_groups': cat_groups
        }

    @torch.no_grad()
    def predict(self, state, part, n_passes):
        model = state['model']
        model.set_mc_dropout(False)
        
        x_filled = torch.tensor(fill_for_input(part['X_obs']), dtype=torch.float32, device=self.device)
        m = torch.tensor(part['M_obs'], dtype=torch.float32, device=self.device)
        
        cont_idx = state['cont_idx']
        bin_idx = state['bin_idx']
        cat_groups = state['cat_groups']
        
        N = len(x_filled)
        P = x_filled.shape[1]
        
        out = model(x_filled, m, sample_z=False)
        decoder_out = out['decoder_out']
        
        mu = torch.zeros(N, P, device=self.device)
        
        if cont_idx and 'x_recon_cont' in decoder_out:
            mu_x, _ = decoder_out['x_recon_cont']
            mu[:, cont_idx] = mu_x
            
        if bin_idx and 'x_recon_bin' in decoder_out:
            mu[:, bin_idx] = torch.sigmoid(decoder_out['x_recon_bin'])
            
        if cat_groups and 'x_recon_cat' in decoder_out:
            logits_cat = decoder_out['x_recon_cat']
            probs_cat = torch.softmax(logits_cat, dim=-1)
            current_offset = 0
            for feat_idx, K in cat_groups:
                group_probs = probs_cat[:, current_offset : current_offset + K]
                mu[:, feat_idx] = group_probs.argmax(dim=-1).float()
                current_offset += K
                
        eta = out['eta'].cpu().numpy()
        
        return {
            'mu': mu.cpu().numpy(),
            'eta': eta
        }

    def objectives(self, pred_val, val):
        val_rmse = harness_metrics.rmse(val['X_true'], pred_val['mu'], val['M_inj'])
        val_cindex = harness_metrics.cindex(val['T'], val['E'], pred_val['eta'])
        return val_rmse, val_cindex

    def extra_test_metrics(self, state, test_part):
        model = state['model']
        model.set_mc_dropout(False)
        
        x_obs = torch.tensor(fill_for_input(test_part['X_obs']), dtype=torch.float32, device=self.device)
        m_obs = torch.tensor(test_part['M_obs'], dtype=torch.float32, device=self.device)
        x_target = torch.tensor(test_part['X_obs'], dtype=torch.float32, device=self.device)
        
        iwae = model.iwae_bound(x_obs, m_obs, x_target, state['cont_idx'], state['bin_idx'], state['cat_groups'], K=50)
        
        return {'iwae_mean': iwae.mean().item()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--fold', type=str, required=True)
    parser.add_argument('--trials', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    
    plugin = MNARSVAEPlugin()
    
    if args.fold == 'all':
        folds = [1, 2, 3, 4, 5]
    else:
        folds = [int(args.fold)]
        
    for f in folds:
        harness_runner.tune_and_select(plugin, args.dataset, args.scenario, f, args.trials, args.seed)

