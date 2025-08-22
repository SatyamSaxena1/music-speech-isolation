"""Evaluate separation outputs using Demucs vocals as pseudo-reference.
Computes SI-SDR (numpy), STOI (pystoi), PESQ (if available), and WER via torchaudio wav2vec2 ASR + jiwer.
Saves report to out_demucs/eval_report.csv and prints results.
"""
import os
import sys
import csv
import numpy as np
import torch
import torchaudio

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from speech_isolation.evaluate_metrics import evaluate_refs
from speech_isolation.metrics import si_sdr

try:
    from jiwer import wer
except Exception:
    wer = None

# ASR bundle
try:
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    asr_model = bundle.get_model().to('cuda' if torch.cuda.is_available() else 'cpu')
    labels = bundle.get_labels()
    sample_rate_asr = bundle.sample_rate
except Exception:
    bundle = None
    asr_model = None
    labels = None
    sample_rate_asr = 16000


def load_wav(path, target_sr=None):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)  # mono
    if target_sr is not None and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0).numpy(), int(sr)


def ctc_greedy_decode(emissions, labels):
    # emissions: Tensor (time, n_labels)
    indices = torch.argmax(emissions, dim=-1).cpu().numpy()
    prev = None
    out = []
    for ix in indices:
        if ix != prev and ix < len(labels):
            token = labels[ix]
            out.append(token)
        prev = ix
    # join tokens (labels are characters)
    txt = ''.join(out).replace('|', ' ').strip()
    return txt


def transcribe(path):
    if asr_model is None:
        return None
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate_asr:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate_asr)
    waveform = waveform.to(next(asr_model.parameters()).device)
    with torch.inference_mode():
        emissions, _ = asr_model(waveform)
    emissions = emissions[0].cpu()
    txt = ctc_greedy_decode(emissions, labels)
    return txt


def main():
    demucs_dir = os.path.join('out_demucs', 'htdemucs')
    mid_dir = 'out_separated'
    learned_dir = 'out_separated_learned'
    finetuned_dir = 'out_finetuned'
    report = []

    for root, dirs, files in os.walk(demucs_dir):
        for fn in files:
            if fn.lower() == 'vocals.wav':
                demucs_v = os.path.join(root, fn)
                # Demucs stores stems under out_demucs/htdemucs/<TrackName>/vocals.wav
                base = os.path.basename(root)
                mid_v = os.path.join(mid_dir, base + '_vocals.wav')
                learned_v = os.path.join(learned_dir, base + '_vocals_learned.wav')

                entries = {'track': base}
                try:
                    ref, sr = load_wav(demucs_v)
                except Exception as e:
                    print('Failed to load demucs vocals', demucs_v, e)
                    continue

                # mid/side
                finetuned_v = os.path.join(finetuned_dir, base + '_vocals_finetuned.wav')
                for name, path in (('mid', mid_v), ('learned', learned_v), ('finetuned', finetuned_v)):
                    if os.path.exists(path):
                        est, sr2 = load_wav(path, target_sr=sr)
                        # align lengths
                        n = min(len(ref), len(est))
                        ref_a = ref[:n]
                        est_a = est[:n]
                        entries[f'{name}_si_sdr'] = float(si_sdr(est_a, ref_a))
                        try:
                            metrics = evaluate_refs(est_a, ref_a, sr)
                            entries[f'{name}_sdr'] = metrics.get('sdr')
                            entries[f'{name}_stoi'] = metrics.get('stoi')
                            entries[f'{name}_pesq'] = metrics.get('pesq')
                        except Exception:
                            entries[f'{name}_sdr'] = None
                            entries[f'{name}_stoi'] = None
                            entries[f'{name}_pesq'] = None
                        # ASR WER vs Demucs transcript
                        if asr_model is not None and wer is not None:
                            ref_txt = transcribe(demucs_v)
                            hyp_txt = transcribe(path)
                            try:
                                entries[f'{name}_wer_vs_demucs'] = wer(ref_txt, hyp_txt)
                            except Exception:
                                entries[f'{name}_wer_vs_demucs'] = None
                        else:
                            entries[f'{name}_wer_vs_demucs'] = None
                    else:
                        entries[f'{name}_si_sdr'] = None
                        entries[f'{name}_sdr'] = None
                        entries[f'{name}_stoi'] = None
                        entries[f'{name}_pesq'] = None
                        entries[f'{name}_wer_vs_demucs'] = None

                report.append(entries)

    out_csv = os.path.join('out_demucs', 'eval_report.csv')
    if report:
        keys = sorted(set(k for d in report for k in d.keys()))
        with open(out_csv, 'w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            for r in report:
                writer.writerow(r)
        print('Wrote evaluation report to', out_csv)
        for r in report:
            print(r)
    else:
        print('No demucs vocals found to evaluate')

if __name__ == '__main__':
    main()
