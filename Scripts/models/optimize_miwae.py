# QUOTES FROM OFFICIAL NOTEBOOK:
# /tmp/miwae/Pytorch notebooks/MIWAE_Pytorch_exercises_demo_ProbAI.ipynb
#
# 1. Prior p(z) (Cell 29, Line 1):
#    p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)
#
# 2. Decoder network (Cell 31, Lines 1-7):
#    decoder = nn.Sequential(
#        torch.nn.Linear(d, h),
#        torch.nn.ReLU(),
#        torch.nn.Linear(h, h),
#        torch.nn.ReLU(),
#        torch.nn.Linear(h, 3*p),  # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
#    )
#
# 3. Encoder network (Cell 34, Lines 1-7):
#    encoder = nn.Sequential(
#        torch.nn.Linear(p, h),
#        torch.nn.ReLU(),
#        torch.nn.Linear(h, h),
#        torch.nn.ReLU(),
#        torch.nn.Linear(h, 2*d),  # the encoder will output both the mean and the diagonal covariance
#    )
#
# 4. Weights initialization (Cell 45, Lines 1-2):
#    def weights_init(layer):
#      if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
#
# 5. MIWAE bound computation (Cell 39, Lines 1-26):
#    def miwae_loss(iota_x,mask):
#      batch_size = iota_x.shape[0]
#      out_encoder = encoder(iota_x)
#      q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)
#      
#      zgivenx = q_zgivenxobs.rsample([K])
#      zgivenx_flat = zgivenx.reshape([K*batch_size,d])
#      
#      out_decoder = decoder(zgivenx_flat)
#      all_means_obs_model = out_decoder[..., :p]
#      all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
#      all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3
#      
#      data_flat = torch.Tensor.repeat(iota_x,[K,1]).reshape([-1,1])
#      tiledmask = torch.Tensor.repeat(mask,[K,1])
#      
#      all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
#      all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,p])
#      
#      logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size])
#      logpz = p_z.log_prob(zgivenx)
#      logq = q_zgivenxobs.log_prob(zgivenx)
#      
#      neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))
#      
#      return neg_bound
#
# 6. MIWAE Impute (Cell 43, Lines 1-32):
#    def miwae_impute(iota_x,mask,L):
#      batch_size = iota_x.shape[0]
#      out_encoder = encoder(iota_x)
#      q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)
#      
#      zgivenx = q_zgivenxobs.rsample([L])
#      zgivenx_flat = zgivenx.reshape([L*batch_size,d])
#      
#      out_decoder = decoder(zgivenx_flat)
#      all_means_obs_model = out_decoder[..., :p]
#      all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
#      all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3
#      
#      data_flat = torch.Tensor.repeat(iota_x,[L,1]).reshape([-1,1]).cuda()
#      tiledmask = torch.Tensor.repeat(mask,[L,1]).cuda()
#      
#      all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
#      all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L*batch_size,p])
#      
#      logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([L,batch_size])
#      logpz = p_z.log_prob(zgivenx)
#      logq = q_zgivenxobs.log_prob(zgivenx)
#      
#      xgivenz = td.Independent(td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model),1)
#
#      imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
#      xms = xgivenz.sample().reshape([L,batch_size,p])
#      xm=torch.einsum('ki,kij->ij', imp_weights, xms) 
#      
#      return xm

import argparse
import copy
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as td
import torch.optim as optim

import harness_metrics
import harness_runner
from optimize_baselines import fit_cox, cox_eta


def weights_init(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.orthogonal_(layer.weight)


def build_encoder(p, h, d):
    return nn.Sequential(
        torch.nn.Linear(p, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, 2 * d),
    )


def build_decoder(d, h, p):
    return nn.Sequential(
        torch.nn.Linear(d, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, 3 * p),
    )


def miwae_loss(iota_x, mask, encoder, decoder, p_z, p, d, K):
    batch_size = iota_x.shape[0]
    out_encoder = encoder(iota_x)
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d], scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])), 1)

    zgivenx = q_zgivenxobs.rsample([K])
    zgivenx_flat = zgivenx.reshape([K * batch_size, d])

    out_decoder = decoder(zgivenx_flat)
    all_means_obs_model = out_decoder[..., :p]
    all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
    all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3

    data_flat = torch.Tensor.repeat(iota_x, [K, 1]).reshape([-1, 1])
    tiledmask = torch.Tensor.repeat(mask, [K, 1])

    all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1, 1]), scale=all_scales_obs_model.reshape([-1, 1]), df=all_degfreedom_obs_model.reshape([-1, 1])).log_prob(data_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K * batch_size, p])

    logpxobsgivenz = torch.sum(all_log_pxgivenz * tiledmask, 1).reshape([K, batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)

    neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq, 0))

    return neg_bound


def miwae_impute(iota_x, mask, encoder, decoder, p_z, p, d, L, return_weights=False):
    batch_size = iota_x.shape[0]
    out_encoder = encoder(iota_x)
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d], scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])), 1)

    zgivenx = q_zgivenxobs.rsample([L])
    zgivenx_flat = zgivenx.reshape([L * batch_size, d])

    out_decoder = decoder(zgivenx_flat)
    all_means_obs_model = out_decoder[..., :p]
    all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
    all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3

    data_flat = torch.Tensor.repeat(iota_x, [L, 1]).reshape([-1, 1])
    tiledmask = torch.Tensor.repeat(mask, [L, 1])

    all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1, 1]), scale=all_scales_obs_model.reshape([-1, 1]), df=all_degfreedom_obs_model.reshape([-1, 1])).log_prob(data_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L * batch_size, p])

    logpxobsgivenz = torch.sum(all_log_pxgivenz * tiledmask, 1).reshape([L, batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)

    xgivenz = td.Independent(td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model), 1)

    imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq, 0) # these are w_1,....,w_L for all observations in the batch
    xms = xgivenz.sample().reshape([L, batch_size, p])
    xm = torch.einsum('ki,kij->ij', imp_weights, xms)

    if return_weights:
        return xm, imp_weights
    return xm


class MIWAECoxPlugin:
    def __init__(self, max_epochs=600, batch_size=64, K=20, L_tune=1000, L_test=10000, check_every=10, patience=5, device=None):
        self.name = 'miwae'
        self.directions = ['minimize', 'maximize']
        self.pareto = True
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.K = K
        self.L_tune = L_tune
        self.L_test = L_test
        self.check_every = check_every
        self.patience = patience
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def search_space(self, trial):
        return {
            'latent': trial.suggest_categorical('latent', [8, 10, 16, 32, 'half']),
            'hidden': trial.suggest_categorical('hidden', [64, 128, 256]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'penalizer': trial.suggest_float('penalizer', 1e-3, 1.0, log=True)
        }

    def _impute_part(self, encoder, decoder, p_z, p, d, L, part, col_mean, col_std, feature_types, feature_names, seed, chunk_size=64, return_weights=False):
        X_obs = part['X_obs']
        M_obs = part['M_obs']
        N = X_obs.shape[0]

        # Standardize with training statistics, missing -> 0
        X_norm = (X_obs - col_mean) / col_std
        X_norm = np.nan_to_num(X_norm, nan=0.0)

        encoder.eval()
        decoder.eval()

        devices = [self.device] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 1)

            xm_chunks = []
            weights_list = []
            with torch.no_grad():
                for start in range(0, N, chunk_size):
                    end = min(start + chunk_size, N)
                    x_chunk = torch.tensor(X_norm[start:end], dtype=torch.float32, device=self.device)
                    m_chunk = torch.tensor(M_obs[start:end], dtype=torch.float32, device=self.device)
                    if return_weights:
                        xm_chunk, w_chunk = miwae_impute(x_chunk, m_chunk, encoder, decoder, p_z, p, d, L, return_weights=True)
                        weights_list.append(w_chunk.cpu())
                    else:
                        xm_chunk = miwae_impute(x_chunk, m_chunk, encoder, decoder, p_z, p, d, L, return_weights=False)
                    xm_chunks.append(xm_chunk.cpu().numpy())

        out_norm = np.concatenate(xm_chunks, axis=0)

        # Destandardize
        X_destd = out_norm * col_std + col_mean

        # Round binary columns to {0, 1}
        bin_cols = feature_types.get('binary', [])
        for c in bin_cols:
            if c in feature_names:
                idx = feature_names.index(c)
                X_destd[:, idx] = np.clip(np.round(X_destd[:, idx]), 0.0, 1.0)

        # Round categorical columns to nearest valid code
        cat_dict = feature_types.get('categorical', {})
        for c, codes in cat_dict.items():
            if c in feature_names:
                idx = feature_names.index(c)
                col_vals = X_destd[:, idx, None]
                codes_arr = np.array(codes).reshape(1, -1)
                dists = np.abs(col_vals - codes_arr)
                nearest_idx = np.argmin(dists, axis=1)
                X_destd[:, idx] = codes_arr[0, nearest_idx]

        # Set every observed cell exactly to part['X_obs']
        observed_mask = (M_obs == 1.0)
        X_destd[observed_mask] = X_obs[observed_mask]

        if return_weights:
            all_weights = torch.cat(weights_list, dim=1) # (L, N)
            return X_destd, all_weights
        return X_destd

    def fit(self, train, val, params, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.Generator(np.random.PCG64(seed))

        X_obs = train['X_obs']
        M_obs = train['M_obs']
        N, p = X_obs.shape
        feature_names = train['feature_names']
        feature_types = train['feature_types']

        # Latent dimension d
        latent_cfg = params['latent']
        if latent_cfg == 'half':
            d = max(1, int(np.floor(p / 2)))
        else:
            d = int(latent_cfg)

        h = int(params['hidden'])

        # Standardize EVERY column with mean and SD of OBSERVED TRAINING values only
        col_mean = np.nanmean(X_obs, axis=0)
        col_std = np.nanstd(X_obs, axis=0)
        col_std[col_std == 0.0] = 1.0
        col_std[np.isnan(col_std)] = 1.0
        col_mean[np.isnan(col_mean)] = 0.0

        # Missing -> 0
        X_train_norm = (X_obs - col_mean) / col_std
        X_train_norm = np.nan_to_num(X_train_norm, nan=0.0)

        # Networks
        encoder = build_encoder(p, h, d).to(self.device)
        decoder = build_decoder(d, h, p).to(self.device)
        encoder.apply(weights_init)
        decoder.apply(weights_init)

        p_z = td.Independent(td.Normal(loc=torch.zeros(d, device=self.device), scale=torch.ones(d, device=self.device)), 1)
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=params['learning_rate'])

        X_t = torch.tensor(X_train_norm, dtype=torch.float32, device=self.device)
        M_t = torch.tensor(M_obs, dtype=torch.float32, device=self.device)

        best_rmse = float('inf')
        best_encoder_state = copy.deepcopy(encoder.state_dict())
        best_decoder_state = copy.deepcopy(decoder.state_dict())
        no_improve_checks = 0

        for ep in range(1, self.max_epochs + 1):
            encoder.train()
            decoder.train()

            perm = rng.permutation(N)
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                idx_t = torch.as_tensor(perm[start:end], device=self.device)

                b_data = X_t[idx_t]
                b_mask = M_t[idx_t]

                optimizer.zero_grad()
                loss = miwae_loss(b_data, b_mask, encoder, decoder, p_z, p, d, self.K)
                loss.backward()
                optimizer.step()

            if ep % self.check_every == 0:
                X_val_imp = self._impute_part(encoder, decoder, p_z, p, d, self.L_tune, val, col_mean, col_std, feature_types, feature_names, seed)
                val_rmse = harness_metrics.rmse(val['X_true'], X_val_imp, val['M_inj'])

                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_encoder_state = copy.deepcopy(encoder.state_dict())
                    best_decoder_state = copy.deepcopy(decoder.state_dict())
                    no_improve_checks = 0
                else:
                    no_improve_checks += 1
                    if no_improve_checks >= self.patience:
                        break

        # Load best state
        encoder.load_state_dict(best_encoder_state)
        decoder.load_state_dict(best_decoder_state)

        # Impute training part with L_tune
        X_train_imp = self._impute_part(encoder, decoder, p_z, p, d, self.L_tune, train, col_mean, col_std, feature_types, feature_names, seed)

        # Fit Cox model
        cox_state = fit_cox(X_train_imp, train['T'], train['E'], params['penalizer'])

        return {
            'encoder_state': copy.deepcopy(encoder.state_dict()),
            'decoder_state': copy.deepcopy(decoder.state_dict()),
            'p': p,
            'd': d,
            'h': h,
            'col_mean': col_mean,
            'col_std': col_std,
            'feature_types': feature_types,
            'feature_names': feature_names,
            'cox_state': cox_state,
            'seed': seed
        }

    def predict(self, state, part, n_passes):
        p = state['p']
        d = state['d']
        h = state['h']

        encoder = build_encoder(p, h, d).to(self.device)
        decoder = build_decoder(d, h, p).to(self.device)
        encoder.load_state_dict(state['encoder_state'])
        decoder.load_state_dict(state['decoder_state'])
        p_z = td.Independent(td.Normal(loc=torch.zeros(d, device=self.device), scale=torch.ones(d, device=self.device)), 1)

        mu = self._impute_part(
            encoder,
            decoder,
            p_z,
            p,
            d,
            self.L_test,
            part,
            state['col_mean'],
            state['col_std'],
            state['feature_types'],
            state['feature_names'],
            state['seed']
        )
        eta = cox_eta(state['cox_state'], mu)
        return {'mu': mu, 'eta': eta}

    def objectives(self, pred_val, val):
        val_rmse = harness_metrics.rmse(val['X_true'], pred_val['mu'], val['M_inj'])
        val_cindex = harness_metrics.cindex(val['T'], val['E'], pred_val['eta'])
        return val_rmse, val_cindex


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--fold', type=str, required=True)
    parser.add_argument('--trials', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    plugin = MIWAECoxPlugin()

    if args.fold == 'all':
        folds = [1, 2, 3, 4, 5]
    else:
        folds = [int(args.fold)]

    for f in folds:
        harness_runner.tune_and_select(plugin, args.dataset, args.scenario, f, args.trials, args.seed)

