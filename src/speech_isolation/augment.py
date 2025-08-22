import numpy as np
import torch

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except Exception:
    _HAS_TORCHAUDIO = False


def random_gain(x: torch.Tensor, min_gain=0.5, max_gain=1.5):
    g = torch.empty((x.shape[0], 1), device=x.device).uniform_(min_gain, max_gain)
    return x * g


def add_noise(x: torch.Tensor, snr_db=30.0):
    # x: (B, T)
    if snr_db is None:
        return x
    sig_pow = (x**2).mean(dim=1, keepdim=True) + 1e-9
    snr = 10 ** (snr_db / 10.0)
    noise_pow = sig_pow / snr
    noise = torch.randn_like(x) * torch.sqrt(noise_pow)
    return x + noise


def lowpass(x: torch.Tensor, sr: int, cutoff: float):
    if not _HAS_TORCHAUDIO:
        return x
    return torchaudio.functional.lowpass_biquad(x.unsqueeze(1), sr, cutoff).squeeze(1)


def random_eq(x: torch.Tensor, sr: int):
    # Randomly apply a lowpass or highpass filter
    if not _HAS_TORCHAUDIO or torch.rand(1).item() < 0.5:
        return x
    if torch.rand(1).item() < 0.5:
        cutoff = torch.empty(1).uniform_(2000, sr//2 - 100).item()
        return lowpass(x, sr, cutoff)
    else:
        cutoff = torch.empty(1).uniform_(80, 400).item()
        return torchaudio.functional.highpass_biquad(x.unsqueeze(1), sr, cutoff).squeeze(1)


def apply_augment(x: torch.Tensor, sr: int):
    x = random_gain(x)
    x = add_noise(x, snr_db=30.0)
    x = random_eq(x, sr)
    return x
