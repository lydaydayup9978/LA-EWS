"""Compute component EWS and LA-EWS from the coupling residuals, NDVI EWS, and the six Fig. 4 pairs."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import fathon
from fathon import fathonUtils as fu
from ews import rolling_ews

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'outputs/ews'
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / 'parameters/components.json').read_text())
    groups = {}
    for c in cfg:
        residual = np.load(ROOT / c['input'])
        dfa, ar1 = rolling_ews(residual, c['window'], c['order'])
        for name, a in [('DFA', dfa), ('AR1', ar1)]:
            groups.setdefault(c['region'] + '_' + name, []).append(a[c['crop'] :])
        print(c['case'], flush=True)
    ensemble = {
        key: np.mean([a[: min(map(len, cols))] for a in cols], axis=0)
        for key, cols in groups.items()
    }
    np.savez_compressed(out / 'regional_ensembles.npz', **ensemble)
    # AM NDVI: 120-month windows of the 1982-2022 regional series; Fig. 4 uses positions 120:360.
    r = pd.read_csv(ROOT / 'data/ndvi/AM_monthly.csv').stl_residual.to_numpy()
    rows = []
    for i in range(len(r) - 119):
        w = r[i : i + 120]
        d = fathon.DFA(w)
        d.computeFlucVec(fu.linRangeByStep(10, 120), revSeg=True, polOrd=3)
        rows.append((np.corrcoef(w[:-1], w[1:])[0, 1], d.fitFlucVec()[0]))
    arr = np.array(rows)
    nd = pd.read_csv(ROOT / 'data/ndvi/AM_ews.csv')
    print(
        'AM NDVI EWS, max difference from data/ndvi/AM_ews.csv:',
        np.abs(arr - nd[['ar1', 'dfa']].to_numpy()).max(),
    )
    ndvi = {'AM_AR1': arr[120:360, 0], 'AM_DFA': arr[120:360, 1]}
    # NB/TP NDVI: final 240 EWS values.
    for region in ['NB', 'TP']:
        dfa, ar1 = rolling_ews(
            np.load(ROOT / f'data/ndvi/{region}_residual.npz')['residual'], 120, 3
        )
        ndvi[region + '_DFA'] = dfa[-240:]
        ndvi[region + '_AR1'] = ar1[-240:]
    fig = np.load(ROOT / 'data/figure4/inputs.npz')
    alignment = json.loads((ROOT / 'parameters/alignment.json').read_text())
    pairs = {}
    for key, A in alignment.items():
        lags = fig[key + '_lags_months']
        X = ensemble[key][A + np.arange(240)[:, None] - lags]
        ex, ey = np.max(abs(X - fig[key + '_X'])), np.max(abs(ndvi[key] - fig[key + '_y']))
        print(key, 'max difference from data/figure4/inputs.npz: X %.1e, y %.1e' % (ex, ey))
        pairs[key + '_X'] = X
        pairs[key + '_y'] = ndvi[key]
        pairs[key + '_lags_months'] = lags
    np.savez_compressed(out / 'figure4_pairs.npz', **pairs)
    print('Saved to', out)


if __name__ == '__main__':
    main()
