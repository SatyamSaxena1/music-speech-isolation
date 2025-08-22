"""Batch process FLAC files in input_flacs/ and produce vocals + accompaniment using mid/side.

This script preserves the original sample rate (maximum bandwidth of the FLAC) and writes two WAVs per input:
  <basename>_vocals.wav  (mid)
  <basename>_accompaniment.wav  (side)

Method: mid = (L+R)/2 -> centered content (commonly vocals), side = (L-R)/2 -> stereo information (often accompaniment).

If a file is mono, the script will copy it to vocals and create a zero accompaniment (can't remove vocals from mono reliably).

Fallbacks: tries to use soundfile; if not installed, attempts to use ffmpeg to convert to WAV and reads with scipy.io.wavfile.
"""
import os
import argparse
import tempfile
import shutil
import subprocess
import numpy as np

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except Exception:
    _HAS_SOUNDFILE = False

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except Exception:
    _HAS_TORCHAUDIO = False

try:
    from scipy.io import wavfile
    _HAS_SCIPY_WAV = True
except Exception:
    _HAS_SCIPY_WAV = False


def read_audio(path):
    """Return tuple (data_float32, sr) where data is 1D or 2D numpy float32 in range [-1,1]."""
    # prefer torchaudio which handles FLAC and preserves sample rate
    if _HAS_TORCHAUDIO:
        wav, sr = torchaudio.load(path)
        # wav: (channels, samples)
        data = wav.numpy().T
        if data.ndim == 1:
            data = data.astype(np.float32)
        return data, int(sr)
    if _HAS_SOUNDFILE:
        data, sr = sf.read(path, always_2d=False)
        data = np.asarray(data, dtype=np.float32)
        # soundfile returns floats in range -1..1 (for floats) or ints; ensure float32
        if data.dtype.kind == 'i':
            # convert integers to float32
            maxv = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / maxv
        return data, sr

    # fallback to ffmpeg -> wav -> scipy
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('soundfile not installed and ffmpeg not found; cannot read FLAC.')
    tmpwav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmpwav.close()
    try:
        cmd = ['ffmpeg', '-y', '-i', path, '-ar', '0', '-ac', '0', tmpwav.name]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not _HAS_SCIPY_WAV:
            raise RuntimeError('scipy is required to read WAV fallback. Please install scipy or soundfile.')
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
    # write float32 via soundfile if available, else convert to int16 and use scipy
    if _HAS_SOUNDFILE:
        sf.write(path, data, sr)
        return
    if not _HAS_SCIPY_WAV:
        raise RuntimeError('No soundfile and no scipy available to write WAV.')
    # convert to int16
    clipped = np.clip(data, -1.0, 1.0)
    int16 = (clipped * 32767).astype(np.int16)
    wavfile.write(path, sr, int16)


def process_file(path, out_dir):
    print(f'Processing: {path}')
    data, sr = read_audio(path)
    # ensure shape (N, C) or (N,)
    if data.ndim == 1:
        # mono
        vocals = data
        accompaniment = np.zeros_like(data)
        note = 'mono: cannot separate vocals reliably; returning copy as vocals.'
    else:
        # take first two channels for stereo. If more channels, average beyond first two.
        if data.shape[1] >= 2:
            L = data[:, 0].astype(np.float32)
            R = data[:, 1].astype(np.float32)
        else:
            L = data[:, 0].astype(np.float32)
            R = L
        # mid/side
        mid = 0.5 * (L + R)
        side = 0.5 * (L - R)
        vocals = mid
        accompaniment = side
        note = 'stereo mid/side separation applied.'

    base = os.path.splitext(os.path.basename(path))[0]
    out_v = os.path.join(out_dir, f'{base}_vocals.wav')
    out_a = os.path.join(out_dir, f'{base}_accompaniment.wav')
    os.makedirs(out_dir, exist_ok=True)
    write_wav(out_v, vocals, sr)
    write_wav(out_a, accompaniment, sr)
    print(f'Wrote: {out_v} ({sr} Hz)')
    print(f'Wrote: {out_a} ({sr} Hz)')
    print('Note:', note)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', default='input_flacs')
    p.add_argument('--out_dir', default='out_separated')
    p.add_argument('--ext', default='.flac')
    args = p.parse_args()

    files = []
    for fn in os.listdir(args.in_dir):
        if fn.lower().endswith(args.ext.lower()):
            files.append(os.path.join(args.in_dir, fn))
    if not files:
        print('No FLAC files found in', args.in_dir)
        return

    for f in files:
        try:
            process_file(f, args.out_dir)
        except Exception as e:
            print('Failed to process', f, '->', e)

    print('Done. Outputs in', args.out_dir)

if __name__ == '__main__':
    main()
