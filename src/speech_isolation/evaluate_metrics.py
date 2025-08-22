import numpy as np
from .metrics import si_sdr

try:
    import mir_eval
except Exception:
    mir_eval = None

try:
    from pystoi import stoi
except Exception:
    stoi = None

try:
    from pesq import pesq
except Exception:
    pesq = None

try:
    from jiwer import wer
except Exception:
    wer = None


def evaluate_refs(est, ref, sr):
    """est, ref: 1D numpy arrays"""
    results = {}
    results['si_sdr'] = si_sdr(est, ref)
    if mir_eval is not None:
        try:
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(ref[np.newaxis, :], est[np.newaxis, :])
            results['sdr'] = float(sdr[0])
        except Exception:
            results['sdr'] = None
    if stoi is not None:
        try:
            results['stoi'] = float(stoi(ref, est, sr))
        except Exception:
            results['stoi'] = None
    if pesq is not None:
        try:
            results['pesq'] = float(pesq(sr, ref, est, 'wb'))
        except Exception:
            results['pesq'] = None
    # WER requires a reference transcript and hypothesis; placeholder
    results['wer'] = None
    return results
