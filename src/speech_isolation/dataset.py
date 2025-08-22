import os
import soundfile as sf
import numpy as np

# Minimal dataset helpers - placeholders for real dataset loading

def load_wav(path, target_sr=16000):
    data, sr = sf.read(path)
    if sr != target_sr:
        raise RuntimeError(f"Expected {target_sr} but got {sr}")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), sr

class SimpleMixtureDataset:
    def __init__(self, mixture_files):
        self.files = mixture_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x, sr = load_wav(self.files[idx])
        return x, sr
