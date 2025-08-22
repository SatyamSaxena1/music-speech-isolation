Drop your FLAC files here for separation

- Put one or more .flac files into this folder.
- Filenames should be plain ASCII (no special characters) to avoid shell issues.

Recommended workflow:
1. Drop your .flac files into `input_flacs`.
2. If you want, I can add an automatic converter that converts each FLAC to WAV and runs the inference demo to produce `speech.wav` and `background.wav` for each input.

Notes:
- The current demo expects WAV input at 16 kHz; if FLACs are a different sample rate I'll add resampling in the converter.
- To proceed, tell me whether you want an automatic converter+batch inference script added and whether you prefer outputs next to each file or in a single `out/` folder.
