"""Spatial Kendall tau maps: conjunction P values and Benjamini-Yekutieli adjustment (Fig. 7, Figs. S7-S8, Table S4)."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
BOUNDS = [-1, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
COLORS = [
    '#438E35',
    '#6AAC45',
    '#9BCD64',
    '#CAE493',
    '#FFFFBE',
    '#FAD39A',
    '#EE8B5A',
    '#D64B34',
    '#B61125',
    '#800026',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/spatial')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tables = {
        p: pd.read_csv(
            ROOT / 'data/spatial' / f'{p}_pixel_results.csv.gz', float_precision='round_trip'
        )
        for p in ['historical', 'future']
    }
    joint = np.concatenate(
        [d[['pai_p', 'tet_p', 'tci_p']].max(axis=1).to_numpy() for d in tables.values()]
    )
    q = false_discovery_control(joint, method='by')
    offset = 0
    summary = []
    maps = []
    for period, d in tables.items():
        n = len(d)
        newq = q[offset : offset + n]
        offset += n
        tau = d[['pai_tau', 'tet_tau', 'tci_tau']].mean(axis=1).to_numpy()
        p = d[['pai_p', 'tet_p', 'tci_p']].max(axis=1).to_numpy()
        with rasterio.open(ROOT / 'data/spatial' / f'{period}_q_BY.tif') as src:
            profile = src.profile.copy()
            shape = src.shape
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        rr = d.row.to_numpy(int)
        cc = d.col.to_numpy(int)
        for kind, values in [
            ('q_BY', newq),
            ('mean_tau_BY_masked', np.where(newq < 0.05, tau, np.nan)),
        ]:
            grid = np.full(shape, -9999.0)
            grid[rr, cc] = np.where(np.isfinite(values), values, -9999.0)
            with rasterio.open(args.out / f'{period}_{kind}.tif', 'w', **profile) as out:
                out.write(grid, 1)
        for label, selected in [('original', p < 0.05), ('BY', newq < 0.05)]:
            summary.append(
                dict(
                    period=period,
                    criterion=label,
                    valid_pixels=n,
                    significant_pixels=int(selected.sum()),
                )
            )
            grid = np.full(shape, np.nan)
            grid[rr[selected], cc[selected]] = tau[selected]
            valid = np.full(shape, np.nan)
            valid[rr, cc] = 1.0
            maps.append((period, label, grid, extent, valid))
    pd.DataFrame(summary).to_csv(args.out / 'summary.csv', index=False)
    if args.plot:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm

        plt.rcParams.update(
            {'font.family': 'Arial', 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
        )
        fig, axes = plt.subplots(2, 2, figsize=(8, 7), layout='constrained')
        for ax, (period, label, grid, extent, valid) in zip(axes.flat, maps):
            ax.imshow(
                valid,
                extent=extent,
                origin='upper',
                interpolation='nearest',
                cmap=ListedColormap(['#d9d9d9']),
                vmin=0,
                vmax=1,
            )
            im = ax.imshow(
                grid,
                extent=extent,
                origin='upper',
                interpolation='nearest',
                cmap=ListedColormap(COLORS),
                norm=BoundaryNorm(BOUNDS, len(COLORS), clip=True),
            )
            ax.set_title(period.capitalize() + ' ' + label)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        fig.colorbar(
            im, ax=axes, boundaries=BOUNDS, ticks=BOUNDS, label='Mean Kendall tau', shrink=0.85
        )
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(args.out / ('spatial_EWS.' + ext), dpi=300)
        plt.close(fig)
    print('Spatial maps written to', args.out)


if __name__ == '__main__':
    main()
