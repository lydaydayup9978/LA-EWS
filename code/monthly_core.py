from pathlib import Path
import hashlib, json, time
import numpy as np
import pandas as pd
import pyEDM
from scipy.spatial.distance import cdist
import ccm_core as core

OUT = Path(__file__).resolve().parents[1] / 'data/monthly_ccm'
LAGS = np.arange(-24, 25)
B = 1000
SAMPLES = 30
RADIUS = 12


def write(p, v):
    p.write_text(
        json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )


def seed(s):
    return int.from_bytes(hashlib.sha256(('monthly20261010:' + s).encode()).digest()[:4], 'little')


def inputs():
    with np.load(OUT / 'inputs.npz') as z:
        return {k: z[k].copy() for k in z.files}


def jobs():
    return [
        (r, v, d)
        for r in ['AM', 'NB', 'TP']
        for v in ['TCI', 'TET', 'PAI', 'LAmean']
        for d in ['LA_to_NDVI', 'NDVI_to_LA']
    ]


def identity(job):
    return '_'.join(job)


def pair(job, data):
    region, var, direction = job
    s, t = ('NDVI', var) if direction == 'LA_to_NDVI' else (var, 'NDVI')
    return data[region + '_' + s], data[region + '_' + t], region + '_' + s, region + '_' + t


def pvalue(obs, null):
    return float((1 + np.count_nonzero(null >= obs - 1e-12)) / (len(null) + 1))


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty(len(p))
    out[o] = np.minimum(q, 1)
    return out


def select_embedding(x):
    # Equal test times for every candidate; strictly forward self prediction.
    n = len(x)
    maxspan = 84
    rows = []
    for e in range(2, 9):
        for tau in range(1, 13):
            scores = []
            for split, end in [(0.50, 0.65), (0.65, 0.80), (0.80, 1.0)]:
                train = np.arange(maxspan, int(n * split) - 1)
                test = np.arange(int(n * split) + 12, min(int(n * end), n - 1))
                a = np.column_stack([x[train - j * tau] for j in range(e)])
                b = np.column_stack([x[test - j * tau] for j in range(e)])
                dist = cdist(b, a)
                ix = np.argsort(dist, axis=1)[:, : e + 1]
                ds = np.take_along_axis(dist, ix, axis=1)
                w = np.exp(-ds / np.maximum(ds[:, :1], 1e-12))
                w /= w.sum(axis=1, keepdims=True)
                pred = (w * x[train[ix] + 1]).sum(axis=1)
                scores.append(float(np.corrcoef(pred, x[test + 1])[0, 1]))
            rows.append(dict(E=e, tau=tau, mean_self_rho=float(np.mean(scores)), fold_rhos=scores))
    best = max(r['mean_self_rho'] for r in rows)
    chosen = min(
        [r for r in rows if r['mean_self_rho'] >= best - 0.01], key=lambda r: (r['E'], r['tau'])
    )
    return dict(
        **chosen,
        best_mean_self_rho=best,
        selection='Smallest E then tau within 0.01 of best mean chronological 1-step self-prediction rho.',
        all_candidates=rows,
    )


def settings(job, data, parameters):
    full, target, sk, tk = pair(job, data)
    par = parameters[sk]
    e, tau = par['E'], par['tau']
    source = full[24:-24]
    m = len(source) - (e - 1) * tau
    sizes = sorted(set([v for v in [50, 100, 150, 200, 250, 300, m] if v <= m]))
    return source, target, sk, tk, e, tau, m, sizes


def native(source, target, e, tau, L, radius, rseed, samples):
    df = pd.DataFrame({'Time': np.arange(len(source)), 'source': source, 'target': target})
    r = pyEDM.CCM(
        dataFrame=df,
        columns='source',
        target='target',
        E=e,
        tau=-tau,
        Tp=0,
        libSizes=[L],
        sample=samples,
        seed=rseed,
        exclusionRadius=radius,
        parallel=False,
        includeData=False,
    )
    return float(r['source:target'].iloc[0])


def run_one(job):
    started = time.perf_counter()
    data = inputs()
    parameters = json.loads((OUT / 'embeddings.json').read_text(encoding='utf-8'))
    s, y, sk, tk, e, tau, m, sizes = settings(job, data, parameters)
    surrogate = core.phase_surrogates(y, B, np.random.default_rng(seed(tk + ':phase')))
    specerr = float(np.max(abs(abs(np.fft.rfft(surrogate, axis=0)) - abs(np.fft.rfft(y))[:, None])))
    assert specerr < 1e-9
    observed = np.empty((len(sizes), len(LAGS)))
    r0 = np.empty(len(LAGS))
    null_min = np.empty((B, len(LAGS)))
    null_max = np.empty_like(null_min)
    minpred = m
    for i, L in enumerate(sizes):
        ns = 1 if L == m else SAMPLES
        rseed = seed(sk + ':library:' + str(L))
        cache = core.build_cache(s, e, tau, L, ns, RADIUS, rseed)
        minpred = min(minpred, cache['min_valid_predictions'])
        assert cache['min_valid_predictions'] == m
        aligned = np.column_stack([y[24 + ell : len(y) - 24 + ell] for ell in LAGS])
        observed[i] = core.evaluate(cache, aligned)
        if i in [0, len(sizes) - 1]:
            dest = null_min if i == 0 else null_max
            for j, ell in enumerate(LAGS):
                for b in range(0, B, 100):
                    dest[b : b + 100, j] = core.evaluate(
                        cache, surrogate[24 + ell : len(y) - 24 + ell, b : b + 100]
                    )
    cache0 = core.build_cache(s, e, tau, m, 1, 0, seed(sk + ':library:' + str(m)))
    r0 = core.evaluate(cache0, np.column_stack([y[24 + ell : len(y) - 24 + ell] for ell in LAGS]))
    gain = observed[-1] - observed[0]
    ngain = null_max - null_min
    nsmax = null_max.max(axis=1)
    ngmax = ngain.max(axis=1)
    neg = np.flatnonzero(LAGS < 0)
    best = int(np.argmax(observed[-1]))
    bestneg = int(neg[np.argmax(observed[-1, neg])])
    ps = pvalue(observed[-1, bestneg], nsmax)
    pg = pvalue(gain[bestneg], ngmax)
    pj = max(ps, pg)
    summary = dict(
        id=identity(job),
        region=job[0],
        variable=job[1],
        hypothesized_direction=job[2],
        source=sk,
        target=tk,
        E=e,
        tau=tau,
        embedding_span=(e - 1) * tau,
        n_anchor=m,
        source_anchor_first=str(data['months'][24 + (e - 1) * tau]),
        source_anchor_last=str(data['months'][443]),
        library_sizes=sizes,
        B=B,
        radius=RADIUS,
        global_peak_ell=int(LAGS[best]),
        global_peak_rho=float(observed[-1, best]),
        global_peak_p_skill=pvalue(observed[-1, best], nsmax),
        negative_peak_ell=int(LAGS[bestneg]),
        negative_peak_rho=float(observed[-1, bestneg]),
        negative_peak_gain=float(gain[bestneg]),
        p_skill_search_adjusted=ps,
        p_gain_search_adjusted=pg,
        p_joint=pj,
        negative_global_peak=bool(LAGS[best] < 0),
        boundary_peak=bool(abs(LAGS[best]) == 24),
        nominal_negative_feature=bool(
            ps < 0.05
            and pg < 0.05
            and observed[-1, bestneg] > 0
            and gain[bestneg] > 0
            and LAGS[best] < 0
        ),
        radius0_peak_ell=int(LAGS[np.argmax(r0)]),
        radius0_peak_rho=float(np.max(r0)),
        radius0_rho_at_primary_peak=float(r0[best]),
        mean_self_prediction_rho=parameters[sk]['mean_self_rho'],
        null_max_rho_q95=float(np.quantile(nsmax, 0.95)),
        null_max_gain_q95=float(np.quantile(ngmax, 0.95)),
        spectrum_error=specerr,
        min_valid_predictions=minpred,
        elapsed_seconds=time.perf_counter() - started,
    )
    all_p_skill = [pvalue(v, nsmax) for v in observed[-1]]
    all_p_gain = [pvalue(v, ngmax) for v in gain]
    assert (
        np.isfinite(observed).all() and np.isfinite(null_max).all() and np.isfinite(null_min).all()
    )
    np.savez_compressed(
        OUT / 'results' / f'{identity(job)}.npz',
        lags=LAGS,
        sizes=sizes,
        rho=observed,
        rho_radius0=r0,
        null_rho_min=null_min,
        null_rho_max=null_max,
        p_skill_adjusted=all_p_skill,
        p_gain_adjusted=all_p_gain,
    )
    write(OUT / 'results' / f'{identity(job)}.json', summary)
    return summary
