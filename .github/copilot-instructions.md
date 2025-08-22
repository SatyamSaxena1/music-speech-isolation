# Copilot instructions for this repository

Purpose: make an AI coding agent productive fast in this speech/music/noise separation project.

## What this project is
- Single Python repo for speech/music/noise separation and vocal removal.
- Key dirs: `src/speech_isolation/` (library), `scripts/` (CLIs), `configs/config.yaml` (canonical runtime/training), `input_flacs/` (inputs), `out_*` (outputs).

## Big picture: architecture and flow
- Input audio (FLAC) → separation (mid/side, learned ConvTasNet-like, or pretrained Demucs) → optional post-processing → evaluation (SI-SDR/SDR/STOI/PESQ/WER) → listening artifacts in `out_versions/`.
- Main implementations:
	- `scripts/run_demucs.py` runs pretrained htdemucs and writes stems to `out_demucs/htdemucs/<track>/`.
	- `src/speech_isolation/models.py` small Conv-TasNet-like prototype used by training/inference.
	- `src/speech_isolation/transformer_models.py` prototype spectrogram Transformer (STFT/ISTFT with consistent params).
	- Evaluation uses Demucs vocals as pseudo-reference when true stems are absent.

## Developer workflows (Windows PowerShell)
- Setup venv and deps: `python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt`
- Run Demucs on `input_flacs/`: `python scripts\run_demucs.py`
- Generate listening versions: set `PYTHONPATH=src` then `python scripts\generate_versions.py`
- Evaluate (writes `out_demucs/eval_report.csv`): `python scripts\evaluate_separations.py`
- Quick model smoke test: `set PYTHONPATH=src` then `python scripts\test_transformer_models.py`

## Conventions and patterns
- Keep `src/speech_isolation/__init__.py` import-light; import submodules explicitly (e.g., `from speech_isolation.train import train`).
- Audio I/O prefers `soundfile` with `scipy`/`torchaudio`/`ffmpeg` fallbacks; preserve original sample rate where possible.
- Output naming:
	- Demucs: `out_demucs/htdemucs/<track>/<stem>.wav`
	- Learned prototype: `out_separated_learned/<base>_vocals_learned.wav`
	- Finetuned model: `out_finetuned/<base>_vocals_finetuned.wav`
	- Versions: `out_versions/<track>/vNN_description.wav`
- Device/dtype: always move model and tensors to the same device; use CPU chunking or AMP if memory limited.

## Integration points and external deps
- Demucs CLI downloads weights automatically; ensure `ffmpeg`/torchaudio backends are available.
- ASR for WER may download a wav2vec2 model on first run (~300+ MB).
- Optional libs (pesq, pystoi, mir_eval) may be absent; evaluator handles missing metrics by reporting `None`.

## Files to inspect for patterns
- `src/speech_isolation/train.py` — training loop with SI-SDR loss, optimizer/scheduler.
- `src/speech_isolation/infer.py` — robust audio read/write and device handling.
- `scripts/generate_versions.py` — post-processing pipeline, device/dtype, chunking patterns.
- `scripts/evaluate_separations.py` — discovery of references/outputs and metric computation.

## First actions for an AI agent
- Run the quick tests and Demucs once to warm caches and reveal environment issues:
	1) `set PYTHONPATH=src`; `python scripts\test_transformer_models.py`
	2) `python scripts\run_demucs.py`
	3) `python scripts\evaluate_separations.py`
- When editing inference/train code, validate with short synthetic audio or a 1–2s clip before processing full songs.

## Examples to follow
- Device movement: ensure `model.to(device)` and `tensor.to(device)` before forward (see `scripts/generate_versions.py`).
- STFT/ISTFT: keep window/hop identical between calls (see `transformer_models.SpectrogramTransformerSeparator`).

## Ask the developer when unsure
- Preferred CPU vs GPU for long files, dataset locations (e.g., MUSDB18), and consent to download large assets.

## Editing rules
- Keep `__init__` lightweight; avoid heavy import-time side effects.
- After changes, run the scripts above; if adding long tasks, write progress under `out_versions/progress.json` or `out_demucs/progress.json`.

