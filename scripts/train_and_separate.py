"""Train a small separator (quick run) and run it on all FLACs in input_flacs/.
Outputs go to out_separated_learned/ with suffixes _vocals_learned.wav and _accompaniment_learned.wav
"""
import os
import sys
import tempfile
import shutil
import subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from speech_isolation.train import train
from speech_isolation.models import ConvTasNetSmall

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except Exception:
    _HAS_SOUNDFILE = False

try:
    from scipy.io import wavfile
    _HAS_SCIPY_WAV = True
except Exception:
    _HAS_SCIPY_WAV = False

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except Exception:
    _HAS_TORCHAUDIO = False

import torch


def read_audio(path):
    # Try torchaudio first; some builds may not have a FLAC backend, so catch and fallback to ffmpeg
    if _HAS_TORCHAUDIO:
        try:
            wav, sr = torchaudio.load(path)
            data = wav.numpy().T
            return data, int(sr)
        except Exception:
            pass
    if _HAS_SOUNDFILE:
        data, sr = sf.read(path, always_2d=False)
        data = np.asarray(data, dtype=np.float32)
        if data.dtype.kind == 'i':
            maxv = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / maxv
        return data, sr
    # fallback: use ffmpeg to convert FLAC -> WAV and load with torchaudio (or scipy if available)
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('No available reader for FLAC: torchaudio/soundfile missing a backend and ffmpeg not found')
    tmpwav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmpwav.close()
    try:
        cmd = ['ffmpeg', '-y', '-i', path, tmpwav.name]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if _HAS_TORCHAUDIO:
            wav, sr = torchaudio.load(tmpwav.name)
            data = wav.numpy().T
            return data, int(sr)
        if _HAS_SCIPY_WAV:
            sr, data = wavfile.read(tmpwav.name)
            data = np.asarray(data)
            if data.dtype.kind == 'i':
                maxv = np.iinfo(data.dtype).max
                data = data.astype(np.float32) / maxv
            else:
                data = data.astype(np.float32)
            return data, sr
        raise RuntimeError('Converted to WAV but no reader (torchaudio/scipy) available')
    finally:
        try:
            os.remove(tmpwav.name)
        except Exception:
            pass
    # fallback to scipy or ffmpeg
    if _HAS_SCIPY_WAV:
        try:
            sr, data = wavfile.read(path)
            data = np.asarray(data)
            if data.dtype.kind == 'i':
                maxv = np.iinfo(data.dtype).max
                data = data.astype(np.float32) / maxv
            else:
                data = data.astype(np.float32)
            return data, sr
        except Exception:
            pass
    # try ffmpeg -> wav
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('No soundfile or scipy available and ffmpeg not found; cannot read FLAC')
    tmpwav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmpwav.close()
    try:
        cmd = ['ffmpeg', '-y', '-i', path, tmpwav.name]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not _HAS_SCIPY_WAV:
            raise RuntimeError('scipy required to read temporary WAV fallback')
        sr, data = wavfile.read(tmpwav.name)
        data = np.asarray(data)
        if data.dtype.kind == 'i':
            maxv = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / maxv
        else:
            data = data.astype(np.float32)
        return data, sr
    finally:
        try:
            os.remove(tmpwav.name)
        except Exception:
            pass


def write_wav(path, data, sr):
    # prefer torchaudio if available
    if _HAS_TORCHAUDIO:
        tensor = torch.from_numpy(np.asarray(data))
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.T
        # tensor shape now (channels, samples)
        torchaudio.save(path, tensor, sr)
        return
    if _HAS_SOUNDFILE:
        sf.write(path, data, sr)
        return
    if not _HAS_SCIPY_WAV:
        raise RuntimeError('No soundfile, torchaudio, or scipy available to write WAV.')
    clipped = np.clip(data, -1.0, 1.0)
    int16 = (clipped * 32767).astype(np.int16)
    wavfile.write(path, sr, int16)


def run():
    # quick train
    print('Running quick training to obtain a model...')
    model = train('configs/config.yaml', quick_run=True)
    # save checkpoint
    ckpt_dir = '.models'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'quick_ckpt.pth')
    torch.save(model.state_dict(), ckpt_path)
    print('Saved quick checkpoint to', ckpt_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    in_dir = 'input_flacs'
    out_dir = 'out_separated_learned'
    os.makedirs(out_dir, exist_ok=True)

    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.lower().endswith('.flac')]
    if not files:
        print('No FLAC files found in', in_dir)
        return

    for path in files:
        print('Processing', path)
        data, sr = read_audio(path)
        # if stereo, average to mono
        if data.ndim == 2:
            data_m = np.mean(data, axis=1)
        else:
            data_m = data
        # normalize to -1..1 if needed
        if data_m.dtype.kind == 'i':
            data_m = data_m.astype(np.float32) / np.iinfo(data.dtype).max
        # to torch
        x = torch.from_numpy(data_m).float().unsqueeze(0).to(device)  # (1, T)
        with torch.no_grad():
            out = model(x)  # (1, n_sources, T)
        out = out.squeeze(0).cpu().numpy()
        # choose source 0 as vocals, rest summed as accompaniment
        vocals = out[0]
        if out.shape[0] > 1:
            accompaniment = out[1:].sum(axis=0)
        else:
            accompaniment = np.zeros_like(vocals)
        # scale outputs to -1..1
        maxv = max(np.max(np.abs(vocals)), np.max(np.abs(accompaniment)), 1e-9)
        vocals = vocals / maxv
        accompaniment = accompaniment / maxv

        base = os.path.splitext(os.path.basename(path))[0]
        out_v = os.path.join(out_dir, base + '_vocals_learned.wav')
        out_a = os.path.join(out_dir, base + '_accompaniment_learned.wav')
        write_wav(out_v, vocals, sr)
        write_wav(out_a, accompaniment, sr)
        print('Wrote', out_v, out_a)

    print('Done. Outputs in', out_dir)

if __name__ == '__main__':
    run()
