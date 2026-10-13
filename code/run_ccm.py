"""Convergent cross mapping for the ten Fig. 4 lags of each region and indicator (Fig. 5).
Null model: 1,000 phase-randomized surrogates of the target series, with the selected lag and embedding fixed."""

import os

for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from statsmodels.stats.multitest import multipletests
import pyEDM

H = Path(__file__).resolve().parents[1] / 'data'
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--rank', type=int, help='run a single rank only')
ap.add_argument('--group', help='run a single group only, e.g. AM_DFA')
ap.add_argument('--out', type=Path, default=H.parent / 'outputs/ccm_rerun')
args = ap.parse_args()
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
import ccm_core as core
import ccm_phase as phase
import embedding_selection as emb

B = 1000


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


write(
    'protocol.json',
    dict(
        n=240,
        B=B,
        lags='Fig. 4 top 10 by |r|',
        embedding='tau 1..12, then E 2..10, tolerance 0.02',
        null='phase-randomized target surrogates',
        samples=30,
        exclusion_radius=0,
    ),
)
z = np.load(H / 'figure4/inputs.npz')
top = pd.read_csv(H / 'figure4/source_data_top10.csv')
rows = []
params = []
checks = []
start = time.time()
if args.rank is not None:
    top = top[top['rank'].eq(args.rank)]
if args.group:
    top = top[top.id.eq(args.group)]
if top.empty:
    raise ValueError('No matching CCM group/rank')
for ix, r in top.iterrows():
    key = r['id']
    lags = z[key + '_lags_months']
    la = core.standardize(z[key + '_X'][:, np.flatnonzero(lags == int(r.lag_years * 12))[0]])
    ndvi = core.standardize(z[key + '_y'])
    for direction, source, target in [('main', ndvi, la), ('reverse', la, ndvi)]:
        case = f'{key}_{int(r.lag_years*12)}_{direction}'
        fp = OUT / (case + '.npz')
        jp = OUT / (case + '.json')
        if fp.exists() and jp.exists():
            rows.append(json.loads(jp.read_text()))
            continue
        tau, e, ps = emb.choose_tau_e(source, target)
        ps['case'] = case
        params.append(ps)
        sizes = np.arange(max(10, e + 2), min(200, 240 - (e - 1) * tau) + 1, 5)
        seed = 42 + int(r.lag_years * 12) + e + tau
        ys = phase.phases(target, B, np.random.default_rng(phase.phase_seed('240:' + case)))
        specerr = float(
            np.max(abs(abs(np.fft.rfft(ys, axis=0)) - abs(np.fft.rfft(target))[:, None]))
        )
        vals = []
        for L in sizes:
            cache = core.build_cache(source, e, tau, int(L), 30, 0, seed)
            vals.append(core.evaluate(cache, np.column_stack([target, ys])))
        vals = np.asarray(vals).T
        obs = vals[0]
        null = vals[1:]
        if int(r['rank']) == 1:
            native = pyEDM.CCM(
                dataFrame=pd.DataFrame(
                    {'Time': np.arange(240), 'source': source, 'target': target}
                ),
                columns='source',
                target='target',
                E=e,
                tau=-tau,
                Tp=0,
                knn=e + 1,
                libSizes=sizes.tolist(),
                sample=30,
                seed=seed,
                exclusionRadius=0,
                parallel=False,
            )['source:target'].to_numpy()
            checks.append(dict(case=case, difference_from_pyEDM=float(np.max(abs(native - obs)))))
        kt = np.array(
            [
                kendalltau(sizes, v, alternative='greater', method='asymptotic').statistic
                for v in vals
            ]
        )
        delta = vals[:, -1] - vals[:, 0]
        nominal = np.array(
            [kendalltau(sizes, v, alternative='greater', method='asymptotic').pvalue for v in vals]
        )
        pv = lambda a: float((1 + np.count_nonzero(a[1:] >= a[0])) / (B + 1))
        row = dict(
            case=case,
            id=key,
            region=key.split('_')[0],
            metric=key.split('_')[1],
            rank=int(r['rank']),
            lag_years=int(r.lag_years),
            direction=direction,
            n=240,
            E=e,
            tau=tau,
            Lmin=int(sizes[0]),
            Lmax=int(sizes[-1]),
            pearson_r=float(r.r),
            rho=float(obs[-1]),
            gain=float(delta[0]),
            kendall_tau=float(kt[0]),
            nominal_kendall_p=float(nominal[0]),
            p_endpoint=pv(vals[:, -1]),
            p_gain=pv(delta),
            p_tau=pv(kt),
            null_nominal_growth_fraction=float(np.mean(nominal[1:] < 0.05)),
            spectrum_error=specerr,
            B=B,
        )
        np.savez_compressed(fp, lib_sizes=sizes, observed=obs, rho=null, tau=kt[1:], gain=delta[1:])
        jp.write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(
            f'{len(rows)}/120 {case}: endpoint={row["p_endpoint"]:.4f} gain={row["p_gain"]:.4f}',
            flush=True,
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'partial.csv', index=False)
df = pd.DataFrame(rows)
full = len(df) == 120
if full:
    df['q_endpoint_BH120'] = multipletests(df.p_endpoint, method='fdr_bh')[1]
for direction in ['main', 'reverse'] if full else []:
    idx = df.direction.eq(direction)
    for v in ['gain', 'tau']:
        df.loc[idx, 'q_' + v + '_BH60'] = multipletests(df.loc[idx, 'p_' + v], method='fdr_bh')[1]
df.to_csv(OUT / 'results.csv', index=False)
if params:
    pd.concat(params).to_csv(OUT / 'embedding_selection.csv', index=False)
write('pyedm_comparison.json', checks)
print('Completed', len(df), 'mappings in %.0f s' % (time.time() - start))
