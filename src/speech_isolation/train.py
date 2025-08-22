import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from .models import ConvTasNetSmall
import numpy as np


def si_sdr_torch(est, ref, eps=1e-8):
    # est, ref: (B, T)
    if est.dim() == 3:
        est = est.squeeze(1)
    if ref.dim() == 3:
        ref = ref.squeeze(1)
    # zero-mean
    est_z = est - est.mean(dim=1, keepdim=True)
    ref_z = ref - ref.mean(dim=1, keepdim=True)
    s_target = (torch.sum(est_z * ref_z, dim=1, keepdim=True) * ref_z) / (torch.sum(ref_z * ref_z, dim=1, keepdim=True) + eps)
    e_noise = est_z - s_target
    si = 10 * torch.log10((torch.sum(s_target**2, dim=1) + eps) / (torch.sum(e_noise**2, dim=1) + eps))
    return si.mean()


def train(cfg_path="configs/config.yaml", quick_run=True):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() and cfg.get('device', 'auto') != 'cpu' else 'cpu')
    model = ConvTasNetSmall().to(device)
    opt_cfg = cfg.get('training', {}).get('optimizer', {}) or {}
    try:
        lr = float(opt_cfg.get('lr', 1e-3))
    except Exception:
        lr = 1e-3
    try:
        weight_decay = float(opt_cfg.get('weight_decay', 0.0))
    except Exception:
        weight_decay = 0.0
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Quick synthetic data: mixture = speech + background
    batch_size = min(4, cfg.get('training', {}).get('batch_size', 8))
    n_iters = 8 if quick_run else 200
    T = 16000  # 1 second signals for smoke test

    print('Starting quick training run on device:', device)
    model.train()
    for it in range(n_iters):
        # create synthetic speech (sine) and background (noise)
        t = torch.linspace(0, 1, T, device=device).unsqueeze(0).repeat(batch_size, 1)
        freqs = torch.rand(batch_size, 1, device=device) * 400 + 100
        speech = 0.6 * torch.sin(2 * np.pi * freqs * t)
        background = 0.2 * torch.randn(batch_size, T, device=device)
        mixture = speech + background

        pred = model(mixture)  # (B, 2, T)
        # assume source 0 is speech (by construction)
        pred_speech = pred[:, 0, :]

        loss = -si_sdr_torch(pred_speech, speech)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (it + 1) % 4 == 0:
            print(f'iter {it+1}/{n_iters} loss {loss.item():.4f}')

    print('Quick training run finished')
    return model
