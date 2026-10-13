import hashlib
import numpy as np


def phase_seed(label):
    return int.from_bytes(hashlib.sha256(('20261008:' + label).encode()).digest()[:8], 'little')


def phases(y, count, rng):
    f = np.fft.rfft(y)
    v = np.repeat(f[:, None], count, axis=1)
    end = len(f) - 1 if len(y) % 2 == 0 else len(f)
    v[1:end] *= np.exp(1j * rng.uniform(0, 2 * np.pi, (end - 1, count)))
    if len(y) % 2 == 0:
        v[-1] *= rng.choice([-1.0, 1.0], count)
    return np.fft.irfft(v, n=len(y), axis=0)
