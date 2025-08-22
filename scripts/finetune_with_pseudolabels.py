"""Finetune small separator on pseudo-labels from Demucs vocals.
- Loads Demucs vocals as target (reference) and mixtures from original FLAC.
- Trains ConvTasNetSmall with checkpointing, early stopping, simple augmentations.
"""
import os
import sys
import glob
import time
import numpy as np
import torch
import torchaudio

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from speech_isolation.models import ConvTasNetSmall
from speech_isolation.train import si_sdr_torch
from speech_isolation.augment import apply_augment


def list_pairs():
    demucs_root = os.path.join('out_demucs', 'htdemucs')
    pairs = []
    for track in os.listdir(demucs_root):
        v = os.path.join(demucs_root, track, 'vocals.wav')
        mix = os.path.join('input_flacs', track + '.flac')
        if os.path.exists(v) and os.path.exists(mix):
            pairs.append((mix, v))
    return pairs


def load_mono(path, sr_target):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != sr_target:
        wav = torchaudio.functional.resample(wav, sr, sr_target)
    return wav.squeeze(0)


def finetune(epochs=3, batch_size=1, sr=16000, patience=2, out_dir='.models', seg_len=64000, enc_channels=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvTasNetSmall(enc_channels=enc_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    pairs = list_pairs()
    if not pairs:
        print('No pairs found for finetuning. Run Demucs first.')
        return None

    best_loss = float('inf')
    best_path = os.path.join(out_dir, 'finetune_best.pth')
    os.makedirs(out_dir, exist_ok=True)
    bad_epochs = 0

    for epoch in range(1, epochs+1):
        np.random.shuffle(pairs)
        total = 0.0
        steps = 0
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            mixes, targets = [], []
            for mix_p, v_p in batch:
                mix = load_mono(mix_p, sr)
                voc = load_mono(v_p, sr)
                n = min(mix.shape[-1], voc.shape[-1])
                # choose a random segment to reduce memory
                if n > seg_len:
                    s = np.random.randint(0, n - seg_len)
                    e = s + seg_len
                    mix = mix[s:e]
                    voc = voc[s:e]
                else:
                    mix = mix[:n]
                    voc = voc[:n]
                mixes.append(mix)
                targets.append(voc)
            x = torch.stack(mixes).to(device)
            y = torch.stack(targets).to(device)
            # augment mixture to improve robustness
            x_aug = apply_augment(x, sr)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(x_aug)[:, 0, :]  # predict vocals -> (B, T_pred)
                # align lengths (stride/padding may cause off-by-few samples)
                t = min(pred.shape[-1], y.shape[-1])
                pred = pred[:, :t]
                y = y[:, :t]
                loss = -si_sdr_torch(pred, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item(); steps += 1
            if steps % 10 == 0:
                print(f'epoch {epoch} step {steps} loss {loss.item():.4f}')
        avg = total / max(steps, 1)
        print(f'epoch {epoch} avg_loss {avg:.4f}')
        # early stopping on train loss (simple)
        if avg < best_loss - 1e-3:
            best_loss = avg
            torch.save(model.state_dict(), best_path)
            bad_epochs = 0
            print('Saved best to', best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print('Early stopping.')
                break
        return best_path if os.path.exists(best_path) else None


if __name__ == '__main__':
    p = finetune(epochs=2, batch_size=1)
    print('Best checkpoint:', p)


if __name__ == '__main__':
    p = finetune(epochs=2, batch_size=1)
    print('Best checkpoint:', p)
