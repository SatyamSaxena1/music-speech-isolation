Speech Isolation — project scaffold

This workspace contains a starter scaffold for a speech isolation project (speech/music/noise separation and vocal removal).

Structure
- src/speech_isolation: library code (dataset, models, training, evaluation, inference, metrics)
- configs/config.yaml: default training/inference parameters
- scripts/demo.py: small demo to run inference on an input file
- requirements.txt: Python dependencies

Quick start (Windows PowerShell):

# Create a virtual env and install
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

# Run demo
python scripts\demo.py --input path\to\mixture.wav --out_dir out

See `configs/config.yaml` for recommended training parameters and metrics.

One-click karaoke (vocals + instrumental):
- Put .flac/.wav/.mp3/.m4a in `input_flacs/`
- Run: `cmd /c run_karaoke.bat`
- Outputs: `out_karaoke/<Track>_vocals.wav` and `out_karaoke/<Track>_instrumental.wav`
- To force re-generate: `cmd /c run_karaoke.bat overwrite`

GitHub push checklist:
- Confirm `.gitignore` excludes large media and outputs
- No personal media under `input_flacs/` (only `README.md`)
- Commit and push:
	- `git init` (if needed) → `git add .` → `git commit -m "Initial push: speech isolation + karaoke"`
	- `git remote add origin <your-repo-url>` → `git push -u origin main`
