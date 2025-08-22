"""Use a finetuned checkpoint to separate FLACs into vocals/accompaniment and write to out_finetuned."""
import os
import sys
import numpy as np
import torch
import torchaudio

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from speech_isolation.models import ConvTasNetSmall


def load_mono(path, sr_target):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sr_target:
        wav = torchaudio.functional.resample(wav, sr, sr_target)
    return wav.squeeze(0), sr_target


def infer_enc_channels(ckpt):
    # infer encoder channels from weight shape
    w = ckpt.get('encoder.conv.weight', None)
    if w is None:
        # state_dict may not be wrapped, iterate keys
        for k, v in ckpt.items():
            if k.endswith('encoder.conv.weight'):
                w = v
                break
    if w is None:
        return 256
    return int(w.shape[0])


def main(ckpt_path='.models/finetune_best.pth', sr=16000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc_channels = 256
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        enc_channels = infer_enc_channels(ckpt)
        model = ConvTasNetSmall(enc_channels=enc_channels).to(device)
        model.load_state_dict(ckpt)
    else:
        model = ConvTasNetSmall(enc_channels=enc_channels).to(device)
    model.eval()
    in_dir = 'input_flacs'
    out_dir = 'out_finetuned'
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(in_dir):
        if not fn.lower().endswith('.flac'):
            continue
        path = os.path.join(in_dir, fn)
        x, _ = load_mono(path, sr)
        with torch.no_grad():
            out = model(x.unsqueeze(0).to(device))
        out = out.squeeze(0).cpu().numpy()
        vocals = out[0]
        acc = out[1] if out.shape[0] > 1 else np.zeros_like(vocals)
        base = os.path.splitext(fn)[0]
        torchaudio.save(os.path.join(out_dir, base + '_vocals_finetuned.wav'), torch.from_numpy(vocals).unsqueeze(0), sr)
        torchaudio.save(os.path.join(out_dir, base + '_accompaniment_finetuned.wav'), torch.from_numpy(acc).unsqueeze(0), sr)

if __name__ == '__main__':
    main()
