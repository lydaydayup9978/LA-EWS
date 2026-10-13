"""STL residuals and rolling-window EWS (AR1, DFA)."""

import numpy as np
from statsmodels.tsa.seasonal import STL
from scipy.signal import detrend
import fathon
from fathon import fathonUtils as fu


def residual(values, stl_configs, method='stl'):
    x = np.asarray(values, float)
    r = x.copy()
    season = np.zeros(len(x))
    for config in stl_configs:
        f = STL(r, **config).fit()
        season += f.seasonal
        r = f.resid
    if method == 'stl':
        return r
    z = x - season
    if method == 'linear':
        return detrend(z, type='linear')
    if method != 'gaussian':
        raise ValueError(method)
    t = np.arange(len(x), dtype=float)
    bw = int(
        np.rint(
            0.9
            * min(t.std(ddof=1), np.subtract(*np.percentile(t, [75, 25])) / 1.34)
            * len(t) ** (-0.2)
        )
    )
    sigma = bw * 0.3706506
    dist = t[:, None] - t[None, :]
    kernel = np.exp(-0.5 * (dist / sigma) ** 2)
    kernel[np.abs(dist) > 4 * sigma] = 0
    return z - (kernel @ z) / kernel.sum(axis=1)


def rolling_ews(residuals, width=120, order=3):
    """DFA exponent (scales 10..width, residual window used directly) and lag-1 autocorrelation for N-W windows."""
    r = np.asarray(residuals, float)
    dfa = []
    ar1 = []
    for i in range(len(r) - width):
        w = r[i : i + width]
        d = fathon.DFA(w)
        d.computeFlucVec(fu.linRangeByStep(10, width), revSeg=True, polOrd=order)
        dfa.append(d.fitFlucVec()[0])
        ar1.append(np.corrcoef(w[:-1], w[1:])[0, 1])
    return np.array(dfa), np.array(ar1)
