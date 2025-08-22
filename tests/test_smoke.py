import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def test_karaoke_smoke(tmp_path: Path):
    # Generate 0.5s mono sine wave at 16kHz to keep it tiny and fast
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    x = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    inp = tmp_path / "beep.wav"
    sf.write(inp, x, sr)

    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    env["SMOKE_TEST"] = "1"

    script = Path("scripts") / "make_karaoke_outputs.py"
    cmd = [sys.executable, str(script), str(inp), "--out-dir", str(out_dir), "--overwrite"]
    subprocess.check_call(cmd, env=env)

    base = inp.stem
    voc = out_dir / f"{base}_vocals.wav"
    ins = out_dir / f"{base}_instrumental.wav"
    assert voc.exists() and ins.exists()
