"""CCM (simplex cross mapping as in pyEDM 2.5.0) with the neighbour geometry cached for each library,
so that the observed target and all surrogate targets are evaluated on the same random libraries."""

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix


def standardize(x):
    x = np.asarray(x, float).ravel()
    assert np.isfinite(x).all() and np.std(x) > 0
    return (x - x.mean()) / x.std()


def embedding(source, e, tau):
    idx = np.arange((e - 1) * tau, len(source))
    return np.ascontiguousarray(np.column_stack([source[idx - j * tau] for j in range(e)])), idx


def build_cache(source, e, tau, libsize, samples, radius, rng_seed):
    emb, idx = embedding(source, e, tau)
    m = len(idx)
    L = min(libsize, m)
    k = min(e + 1, L - 1)
    rng = np.random.default_rng(rng_seed)
    q = min(k + 1 + 2 * radius, L)
    r_m = np.arange(m)[:, None]
    rr = []
    cc = []
    vv = []
    mask = np.zeros((samples, m), bool)
    for s in range(samples):
        lib = rng.choice(m, L, replace=False)
        lib.sort()
        ds, local = KDTree(emb[lib]).query(emb, k=q)
        if ds.ndim == 1:
            ds, local = ds[:, None], local[:, None]
        glob = lib[local]
        invalid = (glob == r_m) | (np.abs(idx[:, None] - idx[glob]) <= radius)
        valid = ~invalid
        cs = np.cumsum(valid, axis=1)
        first = valid & (cs <= k)
        suff = cs[:, -1] >= k
        first[~suff] = False
        nncols = np.zeros((m, k), int)
        nncols[suff] = np.where(first)[1].reshape(int(suff.sum()), k)
        nd = ds[r_m, nncols]
        ng = glob[r_m, nncols]
        dmin = nd[:, :1]
        w = np.exp(-nd / np.where(dmin > 0, dmin, 1.0))
        w = np.where(dmin == 0, np.where(nd == 0, 1.0, 0.0), w)
        w /= np.where(w.sum(axis=1, keepdims=True) > 0, w.sum(axis=1, keepdims=True), 1.0)
        mask[s] = suff
        rr.extend(np.repeat(s * m + np.where(suff)[0], k))
        cc.extend(ng[suff].ravel())
        vv.extend(w[suff].ravel())
    mat = csr_matrix((vv, (rr, cc)), shape=(samples * m, m))
    return {
        'matrix': mat,
        'mask': mask,
        'idx': idx,
        'samples': samples,
        'min_valid_predictions': int(mask.sum(axis=1).min()),
    }


def evaluate(cache, targets, return_trials=False):
    """Each column is one target series, with identical geometry for all columns."""
    if targets.ndim == 1:
        targets = targets[:, None]
    y = targets[cache['idx']]
    s = cache['samples']
    m = len(y)
    b = y.shape[1]
    pred = (cache['matrix'] @ y).reshape(s, m, b)
    mask = cache['mask'][:, :, None]
    n = mask.sum(axis=1)
    ps = (pred * mask).sum(axis=1)
    ys = (y[None, :, :] * mask).sum(axis=1)
    cross = (pred * y[None, :, :] * mask).sum(axis=1) - ps * ys / n
    pv = (pred * pred * mask).sum(axis=1) - ps * ps / n
    yv = (y[None, :, :] ** 2 * mask).sum(axis=1) - ys * ys / n
    with np.errstate(invalid='ignore', divide='ignore'):
        rho = cross / np.sqrt(np.maximum(pv, 0) * np.maximum(yv, 0))
    return rho if return_trials else np.nanmean(rho, axis=0)


def phase_surrogates(y, count, rng):
    f = np.fft.rfft(y)
    end = len(f) - 1 if len(y) % 2 == 0 else len(f)
    out = np.repeat(f[:, None], count, axis=1)
    out[1:end] *= np.exp(1j * rng.uniform(0, 2 * np.pi, size=(end - 1, count)))
    return np.fft.irfft(out, n=len(y), axis=0)
