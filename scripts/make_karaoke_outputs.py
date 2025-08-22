import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import soundfile as sf
except Exception as e:
    sf = None


def _read_wav(path: Path) -> Tuple[np.ndarray, int]:
    if sf is None:
        raise RuntimeError("soundfile is required for this script. Please `pip install soundfile`.")
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.dtype != np.float32 and audio.dtype != np.float64:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    if audio.ndim == 1:
        audio = audio[:, None]
    return audio, sr


def _write_wav(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure float32 and stereo/mono shape [T, C]
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    # Safety: peak-normalize to avoid clipping
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / (peak + 1e-8)
    sf.write(str(path), audio, sr)


def ensure_demucs_stems(track_path: Path, model: str, out_root: Path):
    base = track_path.stem
    stems_dir = out_root / model / base
    vocals = stems_dir / "vocals.wav"
    if vocals.exists():
        return stems_dir
    # Run Demucs to generate stems
    out_root.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        model,
        "-o",
        str(out_root),
        str(track_path),
    ]
    print(f"[demucs] Generating stems for {track_path.name} → {stems_dir} …")
    subprocess.run(cmd, check=True)
    return stems_dir


def make_outputs_for_track(track_path: Path, out_dir: Path, model: str, stems_root: Path, overwrite: bool):
    base = track_path.stem
    out_vocals = out_dir / f"{base}_vocals.wav"
    out_instr = out_dir / f"{base}_instrumental.wav"
    if out_vocals.exists() and out_instr.exists() and not overwrite:
        print(f"[skip] Outputs already exist for {base}")
        return

    stems_dir = ensure_demucs_stems(track_path, model=model, out_root=stems_root)

    # Read stems and build outputs
    want = {
        "vocals": stems_dir / "vocals.wav",
        "drums": stems_dir / "drums.wav",
        "bass": stems_dir / "bass.wav",
        "other": stems_dir / "other.wav",
    }
    missing = [k for k, p in want.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing stems {missing} under {stems_dir}")

    v, sr = _read_wav(want["vocals"])  # [T, C]
    d, sr_d = _read_wav(want["drums"])  # [T, C]
    b, sr_b = _read_wav(want["bass"])   # [T, C]
    o, sr_o = _read_wav(want["other"])  # [T, C]
    assert sr == sr_d == sr_b == sr_o, "Mismatched sample rates across stems"

    # Align lengths to minimum
    T = min(v.shape[0], d.shape[0], b.shape[0], o.shape[0])
    v, d, b, o = v[:T], d[:T], b[:T], o[:T]

    # Instrumental = drums + bass + other
    instr = d + b + o

    # Optional safety limiter: scale if peak > 1.0
    peak = float(np.max(np.abs(instr))) if instr.size else 0.0
    if peak > 1.0:
        instr = instr / (peak + 1e-8)

    _write_wav(out_vocals, v, sr)
    _write_wav(out_instr, instr, sr)
    print(f"[ok] Wrote {out_vocals} and {out_instr}")


def main():
    parser = argparse.ArgumentParser(description="Produce vocals and instrumental outputs using Demucs stems.")
    parser.add_argument("--input-dir", default="input_flacs", help="Folder with input .flac/.wav files (used when no explicit files are provided)")
    parser.add_argument("--out-dir", default="out_karaoke", help="Folder for two-file outputs")
    parser.add_argument("--model", default="htdemucs", help="Demucs model name")
    parser.add_argument("--stems-root", default="out_demucs", help="Root folder where Demucs writes stems")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("inputs", nargs="*", help="Optional explicit file paths to process (overrides --input-dir)")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    stems_root = Path(args.stems_root)

    if args.inputs:
        files = [Path(p) for p in args.inputs]
    else:
        files = [
            *in_dir.glob("*.flac"),
            *in_dir.glob("*.wav"),
            *in_dir.glob("*.mp3"),
            *in_dir.glob("*.m4a"),
        ]
        if not files:
            print(f"No inputs found in {in_dir}. Drop files there and re-run.")
            return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        try:
            make_outputs_for_track(
                track_path=f,
                out_dir=out_dir,
                model=args.model,
                stems_root=stems_root,
                overwrite=args.overwrite,
            )
        except subprocess.CalledProcessError as e:
            print(f"[error] Demucs failed for {f.name}: {e}")
        except Exception as e:
            print(f"[error] {f.name}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
