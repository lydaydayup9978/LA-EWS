import hashlib
import numpy as np
from arch.bootstrap import StationaryBootstrap, optimal_block_length

SEED = 20260926


def rng_for(tag):
    s = int.from_bytes(hashlib.sha256(f'{SEED}:{tag}'.encode()).digest()[:8], 'little')
    return np.random.default_rng(s)


def norm_cols(a):
    a = np.asarray(a, float)
    b = a - a.mean(axis=0)
    norms = np.sqrt(np.sum(b * b, axis=0))
    if np.any(norms < 1e-14):
        raise ValueError('Degenerate correlation column')
    return b / norms


def correlations(X, y):
    return norm_cols(X).T @ norm_cols(y)


def phase_draws(y, n, rng):
    f = np.fft.rfft(y)
    z = np.broadcast_to(f, (n, len(f))).copy()
    end = len(f) - 1 if len(y) % 2 == 0 else len(f)
    z[:, 1:end] = np.abs(f[1:end]) * np.exp(1j * rng.uniform(0, 2 * np.pi, (n, end - 1)))
    if len(y) % 2 == 0:
        z[:, -1] = np.abs(f[-1]) * rng.choice([-1, 1], n)
    return np.fft.irfft(z, n=len(y), axis=1)


def best(r, lags):
    inds = np.flatnonzero(r >= np.max(r) - 1e-12)
    return int(lags[inds[0]]), [int(lags[i]) for i in inds]


def block_info(Z):
    raw = optimal_block_length(Z)['stationary'].to_numpy()
    if not np.all(np.isfinite(raw)):
        return len(Z), None, None
    return max(2, int(np.ceil(raw.max()))), float(raw.max()), raw.tolist()


def ci_discrete(vals):
    return np.quantile(vals, [0.025, 0.975], method='inverted_cdf').astype(int).tolist()
