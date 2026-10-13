"""Fig. 4 lead-lag correlations and Tables S5-S7.
Null model for the maximum correlation: stationary block bootstrap with mean block length n/10, n/5 and n/3."""

from pathlib import Path
import argparse, hashlib
import numpy as np
import pandas as pd
from statistical_core import (
    correlations,
    block_info,
    best,
    ci_discrete,
    rng_for,
    StationaryBootstrap,
)

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [r + '_' + m for r in ['AM', 'NB', 'TP'] for m in ['DFA', 'AR1']]


def inference(key, X, y, lags):
    B = 1000
    n = len(y)
    r = correlations(X, y)
    draws = {}
    result = {}
    block, _, _ = block_info(np.column_stack([y, X]))
    bs = StationaryBootstrap(block, np.column_stack([y, X]), seed=rng_for(key + ':paired_boot'))
    vals = []
    for _ in range(B):
        ix = bs.update_indices()
        vals.append(best(correlations(X[ix], y[ix]), lags)[0])
    draws[key + '_bootstrap_lags'] = np.asarray(vals)
    lo, hi = ci_discrete(vals)
    result.update(block_length=block, range_low_years=lo / 12, range_high_years=hi / 12)
    for div in [10, 5, 3]:
        L = round(n / div)
        seed = int.from_bytes(
            hashlib.sha256(f'20261011:{key}:stationary:{L}'.encode()).digest()[:8], 'little'
        )
        rng = np.random.default_rng(seed)
        ids = np.empty((B, n), int)
        ids[:, 0] = rng.integers(0, n, B)
        for t in range(1, n):
            restart = rng.random(B) < 1 / L
            fresh = rng.integers(0, n, B)
            ids[:, t] = np.where(restart, fresh, (ids[:, t - 1] + 1) % n)
        null = np.array([correlations(X[ix], y).max() for ix in ids])
        result['p_block_n' + str(div)] = (1 + np.sum(null >= r.max())) / 1001
        draws[key + '_block_n' + str(div)] = null
    return result, draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resample', action='store_true')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/figure4')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = ROOT / 'data'
    z = np.load(data / 'figure4/inputs.npz')
    tables = data / 'tables'
    s5a = pd.read_csv(tables / 'S5A.csv').set_index('id')
    s5b = []
    s5c = []
    profiles = []
    tops = []
    draws = {}
    for key in GROUPS:
        X, y, lags = [z[key + '_' + suffix] for suffix in ['X', 'y', 'lags_months']]
        r = correlations(X, y)
        j = np.flatnonzero(r >= r.max() - 1e-12)[0]
        for l, v in zip(lags, r):
            profiles.append(dict(id=key, n=240, lag_years=l / 12, r=v))
        for rank, i in enumerate(np.lexsort((lags, -abs(r)))[:10], 1):
            tops.append(
                dict(id=key, n=240, rank=rank, lag_years=lags[i] / 12, r=r[i], abs_r=abs(r[i]))
            )
        gap, _, _ = block_info(np.column_stack([y, X])[:120])
        tr = correlations(X[:120], y[:120])
        tl = best(tr, lags)[0]
        ti = np.flatnonzero(lags == tl)[0]
        s5b.append(
            dict(
                id=key,
                train_n=120,
                gap_steps=gap,
                test_n=120 - gap,
                training_selected_lead_years=tl / 12,
                heldout_fixed_lag_r=correlations(X[120 + gap :, [ti]], y[120 + gap :])[0],
            )
        )
        trimmed = correlations(X[24:-24], y[24:-24])
        tj = np.flatnonzero(trimmed >= trimmed.max() - 1e-12)[0]
        s5c.append(
            dict(
                id=key,
                retained_n=192,
                original_peak_years=lags[j] / 12,
                trimmed_r_at_original_peak=trimmed[j],
                trimmed_peak_years=lags[tj] / 12,
                trimmed_max_r=trimmed.max(),
            )
        )
        if a.resample:
            result, d = inference(key, X, y, lags)
            draws.update(d)
            for col, v in result.items():
                s5a.loc[key, col] = v
    pd.DataFrame(profiles).to_csv(a.out / 'all_lags.csv', index=False)
    top = pd.DataFrame(tops)
    top.to_csv(a.out / 'top10.csv', index=False)
    pd.DataFrame(s5b).to_csv(a.out / 'S5B.csv', index=False)
    pd.DataFrame(s5c).to_csv(a.out / 'S5C.csv', index=False)
    if a.resample:
        s5a.to_csv(a.out / 'S5A.csv')
        np.savez_compressed(a.out / 'distributions.npz', **draws)
    sens = np.load(tables / 'S6_extended_inputs.npz')
    rows = []
    extended = []
    for key in GROUPS:
        lags = z[key + '_lags_months']
        for method in ['stl', 'linear', 'gaussian']:
            r = correlations(sens[key + '_' + method + '_X'], sens[key + '_' + method + '_y'])
            rows.append(
                dict(id=key, method=method, n=240, lead_years=best(r, lags)[0] / 12, max_r=r.max())
            )
        r = correlations(sens[key + '_extended_X'], z[key + '_y'])
        lags = sens[key + '_extended_lags']
        extended.append(
            dict(id=key, n=240, extended_peak=best(r, lags)[0] / 12, extended_r=r.max())
        )
    pd.DataFrame(rows).to_csv(a.out / 'S6A.csv', index=False)
    pd.DataFrame(extended).to_csv(a.out / 'S6B.csv', index=False)
    w = np.load(tables / 'inputs.npz')
    rows = []
    for key in GROUPS:
        for width in [36, 60, 120, 180, 240]:
            tag = key + '_w' + str(width)
            X, y, R, lags = [w[tag + '_' + s] for s in ['X', 'y', 'reference', 'lags']]
            r = correlations(X, y)
            ref = correlations(R, y)
            rows.append(
                dict(
                    id=key,
                    window_years=width // 12,
                    n=240,
                    lead_years=best(r, lags)[0] / 12,
                    max_r=r.max(),
                    reference_lead_same_segment=best(ref, lags)[0] / 12,
                    reference_r_same_segment=ref.max(),
                )
            )
    pd.DataFrame(rows).to_csv(a.out / 'S7.csv', index=False)
    if a.plot:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 8})
        fig, axs = plt.subplots(3, 2, figsize=(7.2, 8.6), layout='constrained')
        for ax, key in zip(axs.flat, GROUPS):
            d = top[top.id.eq(key)]
            ax.barh(
                np.arange(10),
                d.abs_r,
                color={'AM': '#708090', 'NB': '#D17A00', 'TP': '#D76B93'}[key[:2]],
            )
            ax.set_yticks(np.arange(10), d.lag_years.astype(int))
            ax.invert_yaxis()
            ax.set(title=key, xlabel='|Pearson r|', ylabel='LA lead (years)', xlim=(0, 1.1))
            for i, v in enumerate(d.r):
                ax.text(abs(v) + 0.01, i, f'{v:.2f}', va='center', fontsize=7)
        for ext in ['png', 'svg', 'pdf']:
            fig.savefig(a.out / ('Figure4.' + ext), dpi=300)
        plt.close(fig)
    print('Fig. 4 and Tables S5-S7 written to', a.out)


if __name__ == '__main__':
    main()
