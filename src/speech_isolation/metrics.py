import numpy as np

# Simple SI-SDR placeholder

def si_sdr(est, ref, eps=1e-8):
    # est, ref: 1D numpy
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    s_target = np.dot(est, ref) * ref / (np.dot(ref, ref) + eps)
    e_noise = est - s_target
    si = 10 * np.log10((np.sum(s_target**2)+eps) / (np.sum(e_noise**2)+eps))
    return si
