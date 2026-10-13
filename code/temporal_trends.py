import numpy as np
from scipy.stats import kendalltau, false_discovery_control
from arch.bootstrap import StationaryBootstrap, optimal_block_length
from statistical_core import rng_for, phase_draws

B_PHASE = 1000
B_BOOT = 1000


def bootstrap_ci(y, ident):
    n = len(y)
    t = np.linspace(-1, 1, n)
    X = np.column_stack([np.ones(n), t])
    fit = X @ np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - fit
    resid -= resid.mean()
    block_raw = float(optimal_block_length(resid)['stationary'].iloc[0])
    if not np.isfinite(block_raw):
        block_raw = np.nan
    block = max(2, int(np.ceil(block_raw))) if np.isfinite(block_raw) else n
    v = [np.var(q, ddof=1) for q in np.array_split(resid, 2)]
    quarter = np.array([np.mean(q) for q in np.array_split(resid, 4)])
    info = dict(
        block_length_raw=block_raw,
        block_length=block,
        n_over_block=n / block,
        residual_half_variance_ratio=max(v) / min(v) if min(v) > 0 else np.inf,
        residual_quarter_mean_range_sd=float(np.ptp(quarter) / np.std(resid, ddof=1)),
        residual_acf1=float(np.corrcoef(resid[:-1], resid[1:])[0, 1]),
        ci_status='model_conditional_exploratory',
        ci_low=np.nan,
        ci_high=np.nan,
    )
    if n / block < 5 or np.std(resid) < 1e-12 or not np.all(np.isfinite(resid)):
        info['ci_status'] = 'withheld_insufficient_blocks_or_degenerate_residual'
        return info, None
    bs = StationaryBootstrap(block, resid, seed=rng_for(ident + ':bootstrap'))
    vals = np.empty(B_BOOT)
    for i, (pos, kw) in enumerate(bs.bootstrap(B_BOOT)):
        vals[i] = kendalltau(t, fit + pos[0]).statistic
    info['ci_low'], info['ci_high'] = map(float, np.quantile(vals, [0.025, 0.975]))
    return info, vals


def analyze(series_by_identifier):
    rows = []
    for ident, values in series_by_identifier.items():
        y = np.asarray(values, float)
        t = np.arange(len(y))
        if not np.isfinite(y).all() or np.std(y) == 0:
            raise ValueError(ident)
        tau = kendalltau(t, y).statistic
        rng = rng_for(ident + ':phase')
        count = 0
        for pos in range(0, B_PHASE, 512):
            draws = phase_draws(y, min(512, B_PHASE - pos), rng)
            count += sum(abs(kendalltau(t, z).statistic) >= abs(tau) for z in draws)
        ci, _ = bootstrap_ci(y, ident)
        rows.append(
            dict(id=ident, n=len(y), tau=float(tau), p_phase=(count + 1) / (B_PHASE + 1), **ci)
        )
    q = false_discovery_control(np.array([r['p_phase'] for r in rows]), method='by')
    for row, value in zip(rows, q):
        row['q_BY'] = float(value)
    return rows
