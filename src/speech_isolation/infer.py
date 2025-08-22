import os
try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except Exception:
    from scipy.io import wavfile
    _HAS_SOUNDFILE = False

import torch
import numpy as np
from .models import IdentitySeparator


def infer(model, input_path, out_dir, device='cpu'):
    os.makedirs(out_dir, exist_ok=True)
    if _HAS_SOUNDFILE:
        x, sr = sf.read(input_path)
        if x.ndim>1:
            x = np.mean(x, axis=1)
    else:
        sr, x = wavfile.read(input_path)
        x = x.astype(np.float32)
        if x.ndim>1:
            x = np.mean(x, axis=1)
    x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
    model.to(device).eval()
    with torch.no_grad():
        s, b = model(x_t)
    s = s.squeeze().cpu().numpy()
    b = b.squeeze().cpu().numpy()
    if _HAS_SOUNDFILE:
        sf.write(os.path.join(out_dir, 'speech.wav'), s, sr)
        sf.write(os.path.join(out_dir, 'background.wav'), b, sr)
    else:
        wavfile.write(os.path.join(out_dir, 'speech.wav'), sr, s)
        wavfile.write(os.path.join(out_dir, 'background.wav'), sr, b)
    return out_dir
