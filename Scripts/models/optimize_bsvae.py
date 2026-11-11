import argparse
import copy
import math
import numpy as np
import torch
from torch.optim import Adam

from harness_data import fill_for_input
from bsvae import BSVAE
from losses import compute_base_losses
from train_utils import on_epoch_start, compute_L_cal
import harness_runner

class BSVAEPlugin:
    def __init__(self, stage_epochs=(30, 20, 50), batch_size=256, train_mc_passes=10, val_mc_passes=20, device=None, variant=None):
        self.variant = variant
        self.name = 'bsvae_nolcal' if variant == 'nolcal' else 'bsvae'
        self.directions = ['minimize', 'minimize', 'maximize']
        self.pareto = True
        self.stage_epochs = stage_epochs
        self.batch_size = batch_size
        self.train_mc_passes = train_mc_passes
        self.val_mc_passes = val_mc_passes
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def search_space(self, trial):
        space = {
            'beta': trial.suggest_float('beta', 1e-4, 10, log=True),
            'gamma': trial.suggest_float('gamma', 1e-2, 10, log=True),
            'latent_dim': trial.suggest_categorical('latent_dim', [8, 16, 32, 64]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'dropout_prob': trial.suggest_float('dropout_prob', 0.05, 0.5),
            'hidden': trial.suggest_categorical("hidden", ["64_32", "128_64", "128_64_32"])
        }
        if self.variant == 'nolcal':
            space['lambda'] = 0.0
        else:
            space['lambda'] = trial.suggest_float('lambda', 1e-2, 1e2, log=True)
        return space

    def fit(self, train, val, params, seed):
        if self.variant == "nolcal":
            params = {**params, 'lambda': 0.0}
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
            cidx = pred_val['cindex']
            if cidx > best_cindex:
                best_cindex = cidx
                best_state = copy.deepcopy(model.state_dict())

        # Stage 3: All unfrozen, loss = L_recon + beta * L_latent + gamma * L_Cox + lambda * L_cal
        for param in model.parameters(): param.requires_grad = True
        
        for ep in range(self.stage_epochs[2]):
            if params['lambda'] > 0:
                eval_times = torch.linspace(T.min(), T.max(), steps=10, device=self.device)
                epoch_cache = on_epoch_start(model, x_filled, m, T, E, eval_times)
            else:
                epoch_cache = None
            
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
                
                if params['lambda'] > 0:
                    L_cal = compute_L_cal(
                        model, x_filled[batch_idx], m[batch_idx], epoch_cache['bin_assignments'][batch_idx],
                        epoch_cache['S_b_KM'], epoch_cache['H_0'], eval_times, n_passes=self.train_mc_passes
                    )
                    loss = L_NLL + params['beta'] * L_latent + params['gamma'] * L_Cox + params['lambda'] * L_cal
                else:
                    L_cal = None
                    loss = L_NLL + params['beta'] * L_latent + params['gamma'] * L_Cox
                
                if not torch.isfinite(loss):
                    max_eta = out['eta'].abs().max().item()
                    comps = []
                    if not torch.isfinite(L_NLL): comps.append('L_NLL')
                    if not torch.isfinite(L_latent): comps.append('L_latent')
                    if not torch.isfinite(L_Cox): comps.append('L_Cox')
                    if L_cal is not None and not torch.isfinite(L_cal): comps.append('L_cal')
                    raise ValueError(f"Loss is {loss.item()} in stage 3, epoch {ep}. Not finite: {', '.join(comps)}. max |eta|: {max_eta}")
                    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            current_epoch += 1
            
            # Checkpoint
            pred_val = self.predict({'model': model, 'cont_idx': cont_idx, 'bin_idx': bin_idx, 'cat_groups': cat_groups}, val, self.val_mc_passes)
            cidx = pred_val['cindex']
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
        model.set_mc_dropout(True)
        
        x_filled = torch.tensor(fill_for_input(part['X_obs']), dtype=torch.float32, device=self.device)
        m = torch.tensor(part['M_obs'], dtype=torch.float32, device=self.device)
        
        cont_idx = state['cont_idx']
        bin_idx = state['bin_idx']
        cat_groups = state['cat_groups']
        
        N = len(x_filled)
        P = x_filled.shape[1]
        
        mu_passes = torch.zeros(n_passes, N, P, device=self.device)
        var_passes = torch.zeros(n_passes, N, P, device=self.device)
        eta_passes = torch.zeros(n_passes, N, device=self.device)
        
        cat_probs_sum = None
        
        for i in range(n_passes):
            out = model(x_filled, m, sample_z=False)
            decoder_out = out['decoder_out']
            
            if cont_idx and 'x_recon_cont' in decoder_out:
                mu_x, log_var_x = decoder_out['x_recon_cont']
                mu_passes[i, :, cont_idx] = mu_x
                var_passes[i, :, cont_idx] = torch.exp(log_var_x)
                
            if bin_idx and 'x_recon_bin' in decoder_out:
                mu_passes[i, :, bin_idx] = torch.sigmoid(decoder_out['x_recon_bin'])
                var_passes[i, :, bin_idx] = float('nan')
                
            if cat_groups and 'x_recon_cat' in decoder_out:
                logits_cat = decoder_out['x_recon_cat']
                probs_cat = torch.softmax(logits_cat, dim=-1)
                if cat_probs_sum is None:
                    cat_probs_sum = probs_cat.clone()
                else:
                    cat_probs_sum += probs_cat
                    
            eta_passes[i] = out['eta']
            
        eta_samples = eta_passes.T.cpu().numpy()
        eta = eta_samples.mean(axis=1)
        
        mu = torch.zeros(N, P, device=self.device)
        sigma_E2 = torch.full((N, P), float('nan'), device=self.device)
        sigma_A2 = torch.full((N, P), float('nan'), device=self.device)
        
        if cont_idx:
            mu[:, cont_idx] = mu_passes[:, :, cont_idx].mean(dim=0)
            sigma_E2[:, cont_idx] = mu_passes[:, :, cont_idx].var(dim=0, unbiased=True)
            sigma_A2[:, cont_idx] = var_passes[:, :, cont_idx].mean(dim=0)
            
        if bin_idx:
            mu[:, bin_idx] = mu_passes[:, :, bin_idx].mean(dim=0)
            
        if cat_groups:
            cat_probs_mean = cat_probs_sum / n_passes
            current_offset = 0
            for feat_idx, K in cat_groups:
                group_probs = cat_probs_mean[:, current_offset : current_offset + K]
                mu[:, feat_idx] = group_probs.argmax(dim=-1).float()
                current_offset += K

        # also C-index on val needs C-index return inside pred_val here or in objectives?
        # The prompt says: checkpoint uses val_mc_passes, and expects cidx = pred_val['cindex'] ?? No, pred_val does not return cindex.
        # Wait, the prompt says "compute the validation C-index from the mean eta".
        import harness_metrics
        cidx = harness_metrics.cindex(part['T'], part['E'], eta)
        
        return {
            'mu': mu.cpu().numpy(),
            'sigma_E2': sigma_E2.cpu().numpy(),
            'sigma_A2': sigma_A2.cpu().numpy(),
            'eta': eta,
            'eta_samples': eta_samples,
            'cindex': cidx
        }

    def objectives(self, pred_val, val):
        import harness_metrics
        x_true = val['X_true']
        M_inj = val['M_inj']
        mu = pred_val['mu']
        
        rmse_val = harness_metrics.rmse(x_true, mu, M_inj)
        
        sigma_T = np.sqrt(pred_val['sigma_E2'] + pred_val['sigma_A2'])
        ece_val = harness_metrics.ece(x_true, mu, sigma_T, M_inj)
        
        return rmse_val, ece_val, pred_val['cindex']

    def extra_test_metrics(self, state, test_part):
        model = state['model']
        model.set_mc_dropout(False)
        
        x_obs = torch.tensor(fill_for_input(test_part['X_obs']), dtype=torch.float32, device=self.device)
        m_obs = torch.tensor(test_part['M_obs'], dtype=torch.float32, device=self.device)
        
        # x_target is X_obs according to prompt (X_obs with M_obs)
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
    parser.add_argument('--variant', type=str, choices=['none', 'nolcal'], default='none')
    args = parser.parse_args()
    
    variant = None if args.variant == 'none' else args.variant
    plugin = BSVAEPlugin(variant=variant)
    
    if args.fold == 'all':
        folds = [1, 2, 3, 4, 5]
    else:
        folds = [int(args.fold)]
        
    for f in folds:
        harness_runner.tune_and_select(plugin, args.dataset, args.scenario, f, args.trials, args.seed)
