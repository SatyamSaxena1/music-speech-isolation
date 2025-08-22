"""Run pretrained Demucs on all FLACs in input_flacs/ and write stems to out_demucs/.

This script uses the demucs Python package (requires installation). It will use a pretrained model
('demucs' default) to separate into stems (vocals, bass, drums, other).
"""
import os
import sys
import shutil
import torch

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

try:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
except Exception:
    apply_model = None
    get_model = None

try:
    import torchaudio
except Exception:
    torchaudio = None


def run():
    if get_model is None:
        print('Demucs not installed. Install with `pip install demucs`')
        return
    model = get_model(name='htdemucs')
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    in_dir = 'input_flacs'
    out_dir = 'out_demucs'
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(in_dir) if f.lower().endswith('.flac')]
    if not files:
        print('No FLAC files in', in_dir)
        return

    for fn in files:
        path = os.path.join(in_dir, fn)
        print('Processing', path)
        # demucs apply_model expects paths and will write outputs to out_dir
        try:
            # demucs apply_model is low-level; use CLI-style separate for robust handling
            from demucs.separate import main as demucs_separate_main
            # demucs CLI expects args: ['-n', 'htdemucs', '--out', out_dir, path]
            args = ['-n', 'htdemucs', '--out', out_dir, path]
            demucs_separate_main(args)
            print('Wrote stems for', fn, 'to', out_dir)
        except Exception as e:
            print('Demucs failed for', fn, e)

if __name__ == '__main__':
    run()
