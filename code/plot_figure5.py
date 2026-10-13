from pathlib import Path
import numpy as np, pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

H = Path(__file__).resolve().parents[1] / 'data/ccm'
OUT = H.parents[1] / 'outputs/figures'
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(H / 'results.csv')
plt.rcParams.update(
    {
        'font.family': 'Arial',
        'font.size': 7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.6,
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
        'legend.frameon': False,
    }
)
fig, axes = plt.subplots(3, 2, figsize=(180 / 25.4, 177 / 25.4), sharex=True, sharey=True)
for panel, ax, (region, metric) in zip(
    'abcdef', axes.flat, [(r, m) for r in ['AM', 'NB', 'TP'] for m in ['AR1', 'DFA']]
):
    r = df[
        (df.region == region) & (df.metric == metric) & (df['rank'] == 1) & (df.direction == 'main')
    ].iloc[0]
    z = np.load(H / (r['case'] + '.npz'))
    x = z['lib_sizes']
    v = z['rho']
    assert len(v) == 1000
    ax.fill_between(x, *np.quantile(v, [0.05, 0.95], axis=0), color='#dde6eb', zorder=0)
    for y in v[:10]:
        ax.plot(x, y, color='#8ca4b4', lw=0.4, alpha=0.55)
    ax.plot(x, np.median(v, axis=0), color='#55778d', lw=1.1)
    ax.plot(x, z['observed'], color='#b2593f', lw=1.6)
    ax.set_title(f'{region} {metric} | h = {int(r.lag_years*12)} months', loc='left', fontsize=8)
    ax.text(-0.13, 1.1, panel, transform=ax.transAxes, weight='bold', fontsize=10)
    ax.text(
        0.04,
        0.05,
        f'Endpoint p = {r.p_endpoint:.3f}\nGrowth p = {r.p_gain:.3f}',
        transform=ax.transAxes,
        fontsize=6.5,
    )
    ax.set_ylim(-0.1, 1.02)
    ax.set_xlim(0, 205)
    if ax in axes[:, 0]:
        ax.set_ylabel('Cross-map skill (ρ)')
    if ax in axes[-1, :]:
        ax.set_xlabel('Library size')
fig.legend(
    handles=[
        Line2D([], [], color='#b2593f', lw=1.6, label='Observed'),
        Line2D([], [], color='#55778d', lw=1.1, label='Null median'),
        Patch(color='#dde6eb', label='Null 5–95%'),
        Line2D([], [], color='#8ca4b4', lw=0.6, label='10 fixed null examples'),
    ],
    loc='lower center',
    ncol=4,
    bbox_to_anchor=(0.5, 0.008),
    fontsize=6.5,
)
fig.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.11, hspace=0.40, wspace=0.17)
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(OUT / f'Figure5_240.{ext}', dpi=600, facecolor='white')
fig.savefig(OUT / 'Figure5_preview.png', dpi=150)
