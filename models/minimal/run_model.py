"""Minimal land-atmosphere/vegetation model used for Figs. S9 and S10.

All times are dimensionless (tau_L = 1).
"""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import brentq
from scipy.special import expit

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'parameters.json').read_text(encoding='utf-8'))
DATA = HERE / 'data'
FIG = HERE / 'figures'
for folder in [DATA, FIG]:
    folder.mkdir(exist_ok=True)
COL = {'L': '#3577A3', 'V': '#CB7E43', 'F': '#72777D', 'accent': '#657D55'}
plt.rcParams.update(
    {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'legend.fontsize': 7,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.7,
        'lines.linewidth': 1.6,
        'legend.frameon': False,
    }
)


def dump(name, value):
    (HERE / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8'
    )


def csv_out(name, rows):
    with (DATA / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save(fig, name):
    fig.savefig(FIG / (name + '.png'), dpi=240, facecolor='white')
    fig.savefig(FIG / (name + '.svg'), facecolor='white')
    fig.savefig(FIG / (name + '.pdf'), facecolor='white')
    plt.close(fig)


def panel(ax, label, title):
    ax.set_title(title, loc='left', pad=9)
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=10, fontweight='bold')


def lin_matrix(tau=8, a=0.25, b=0.8):
    return np.array([[-1.0, a], [b / tau, -1.0 / tau]])


def response(J, B, t):
    final = -np.linalg.solve(J, B)
    vals, vecs = np.linalg.eig(J)
    transient = (vecs @ (np.exp(np.outer(vals, t)) * np.linalg.solve(vecs, final)[:, None])).real
    return final[:, None] - transient, final


def tq(J, B, q, i):
    final = -np.linalg.solve(J, B)
    if abs(final[i]) < 1e-12:
        return None
    horizon = 40 / -max(np.linalg.eigvals(J).real)
    return float(
        brentq(lambda t: (final - expm(J * t) @ final)[i] / final[i] - q, 0, horizon, xtol=1e-11)
    )


def linear_experiments():
    cfg = P['linear_model']
    rows = []
    ts = np.linspace(0, 80, 1601)
    curves = []
    for tau in cfg['tau_V_grid']:
        for eta in cfg['eta_grid']:
            J = lin_matrix(tau, cfg['a'], cfg['b'])
            B = np.array([1, eta / tau])
            for q in cfg['response_threshold_fractions']:
                tL, tV = tq(J, B, q, 0), tq(J, B, q, 1)
                rows.append(dict(tau_V=tau, eta=eta, q=q, t_L=tL, t_V=tV, lead=tV - tL))
    for name, tau, a, b, eta in [
        ('cascade', 8, 0.25, 0.8, 0),
        ('fast_vegetation_direct', 0.25, 0.25, 0.8, 4),
        ('common_step', 8, 0, 0, 1),
        ('disconnected', 8, 0, 0, 0),
    ]:
        J = lin_matrix(tau, a, b)
        B = np.array([1, eta / tau])
        y, final = response(J, B, ts)
        for j, t in enumerate(ts):
            curves.append(
                dict(
                    case=name,
                    time=t,
                    L=y[0, j],
                    V=y[1, j],
                    L_fraction=y[0, j] / final[0] if final[0] else 0,
                    V_fraction=y[1, j] / final[1] if final[1] else 0,
                )
            )
    csv_out('linear_response_grid.csv', rows)
    csv_out('linear_step_curves.csv', curves)
    return rows, curves


def H(L, s=0.07, theta=0.8):
    return expit((L - theta) / s)


def fold(alpha=0.6, s=0.07, theta=0.8):
    if alpha <= 4 * s:
        return None
    V = (1 - np.sqrt(1 - 4 * s / alpha)) / 2
    L = theta + s * np.log(V / (1 - V))
    return dict(L=float(L), V=float(V), F=float(L - alpha * V))


def eq_low(F, alpha=0.6):
    f = fold(alpha)
    upper = f['L'] if f is not None and F <= f['F'] else F + alpha + 2
    return float(brentq(lambda L: L - alpha * H(L) - F, -2, upper, xtol=1e-13))


def first_cross(t, y, level):
    ids = np.flatnonzero(y >= level)
    if not len(ids):
        return None
    i = ids[0]
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (level - y[i - 1]) * (t[i] - t[i - 1]) / (y[i] - y[i - 1]))


def nonlinear_experiments():
    cfg = P['nonlinear_model']
    rows = []
    main = {}
    all_curves = {}
    F0, F1 = cfg['forcing_start'], cfg['forcing_end']
    for alpha in cfg['alphas']:
        L0 = eq_low(F0, alpha)
        V0 = H(L0)
        for tau in cfg['tau_V_grid']:
            for duration in cfg['ramp_durations']:
                t = np.arange(0, duration + 8 * tau + 0.025, 0.05)

                def rhs(time, y):
                    F = F0 + (F1 - F0) * min(time / duration, 1)
                    return [F - y[0] + alpha * y[1], (H(y[0]) - y[1]) / tau]

                sol = solve_ivp(
                    rhs,
                    [0, t[-1]],
                    [L0, V0],
                    t_eval=t,
                    method='DOP853',
                    rtol=1e-9,
                    atol=1e-11,
                    max_step=0.5,
                )
                assert sol.success and np.isfinite(sol.y).all()
                tL = first_cross(t, sol.y[0], L0 + 0.1)
                tV = first_cross(t, sol.y[1], V0 + 0.1)
                tmid = first_cross(t, sol.y[1], 0.5)
                f = fold(alpha)
                rows.append(
                    dict(
                        alpha=alpha,
                        tau_V=tau,
                        ramp_duration=duration,
                        t_L_increment_01=tL,
                        t_V_increment_01=tV,
                        state_threshold_lead=tV - tL if tV is not None and tL is not None else None,
                        t_V_half=tmid,
                        static_fold_F=f['F'] if f else None,
                        static_fold_time=(f['F'] - F0) / (F1 - F0) * duration if f else None,
                    )
                )
                key = f'a{alpha:g}_tv{tau:g}_ramp{duration:g}'
                all_curves[key] = np.column_stack(
                    [t, F0 + (F1 - F0) * np.minimum(t / duration, 1), sol.y.T]
                )
                if tau == 8 and duration == 200:
                    main[alpha] = all_curves[key]
    csv_out('nonlinear_sweep.csv', rows)
    np.savez_compressed(DATA / 'nonlinear_all_trajectories.npz', **all_curves)
    example = []
    for alpha, arr in main.items():
        for t, F, L, V in arr:
            example.append(dict(alpha=alpha, time=t, F=F, L=L, V=V))
    csv_out('nonlinear_example.csv', example)
    return rows, main


def equilibrium(q, alpha=0.6, tau=8):
    ref = fold(0.6)
    if alpha > 0.28:
        F = 0.2 + q * (fold(alpha)['F'] - 0.2)
        L = eq_low(F, alpha)
    else:
        L0 = eq_low(0.2, alpha)
        L = L0 + q * (ref['L'] - L0)
        F = L - alpha * H(L)
    V = H(L)
    slope = V * (1 - V) / 0.07
    J = np.array([[-1.0, alpha], [slope / tau, -1 / tau]])
    Q = np.diag([2e-4, 2e-4 / tau])
    S = solve_continuous_lyapunov(J, -Q)
    residual = np.max(abs(J @ S + S @ J.T + Q))
    assert max(np.linalg.eigvals(J).real) < 0 and np.linalg.eigvalsh(S).min() > 0
    return dict(
        L=L,
        V=float(V),
        F=float(F),
        slope=float(slope),
        loop_gain=float(alpha * slope),
        J=J,
        S=S,
        Q=Q,
        lyapunov_residual=float(residual),
    )


def root_cov(S):
    vals, vecs = np.linalg.eigh((S + S.T) / 2)
    assert vals.min() > -1e-12
    return vecs @ np.diag(np.sqrt(np.maximum(vals, 0)))


def population_ac(J, S, obs):
    return np.diag(expm(J) @ S) / (np.diag(S) + obs)


def ac_records(J, S, obs, n, reps, seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    A = expm(J)
    innovation = S - A @ S @ A.T
    R = root_cov(innovation)
    state = rng.standard_normal((reps, 2)) @ root_cov(S).T
    data = np.empty((reps, n, 2))
    for k in range(n):
        data[:, k, :] = state + rng.standard_normal((reps, 2)) * np.sqrt(obs)
        state = state @ A.T + rng.standard_normal((reps, 2)) @ R.T
    x, y = data[:, :-1], data[:, 1:]
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    return (x * y).sum(axis=1) / np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))


def wilson(k, n):
    p = k / n
    z = 1.959963984540054
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    width = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - width), float(centre + width)


def ews_experiments():
    cfg = P['ews_experiment']
    qs = np.linspace(0, 0.995, cfg['control_levels'])
    cases = [
        ('default', 0.6, 8, 0, 0),
        ('V_noisy', 0.6, 8, 0, 9),
        ('L_noisy', 0.6, 8, 9, 0),
        ('equal_timescale', 0.6, 1, 0, 0),
        ('fast_V', 0.6, 0.25, 0, 0),
        ('no_feedback', 0, 8, 0, 0),
    ]
    rows = []
    summaries = []
    cutoffrows = []
    records = {}
    for ci, (name, alpha, tau, rL, rV) in enumerate(cases):
        base = equilibrium(0, alpha, tau)
        obs = np.diag(base['S']) * [rL, rV]
        for n in cfg['sample_lengths']:
            baseline = ac_records(
                base['J'],
                base['S'],
                obs,
                n,
                cfg['baseline_calibration_replicates'],
                [cfg['seed'], ci, n, 0],
            )
            records[f'{name}_n{n}_baseline_calibration'] = baseline
            cutoff = np.quantile(baseline, 0.95, axis=0)
            cutoffrows.append(
                dict(
                    case=name,
                    n=n,
                    cutoff_L=cutoff[0],
                    cutoff_V=cutoff[1],
                    obs_var_L=obs[0],
                    obs_var_V=obs[1],
                )
            )
            first = [None, None]
            firstvalid = [None, None]
            for qi, q in enumerate(qs):
                e = equilibrium(float(q), alpha, tau)
                samples = ac_records(
                    e['J'],
                    e['S'],
                    obs,
                    n,
                    cfg['evaluation_replicates'],
                    [cfg['seed'], ci, n, 1, qi],
                )
                records[f'{name}_n{n}_q{qi}'] = samples
                successes = (samples > cutoff).sum(axis=0)
                power = successes / cfg['evaluation_replicates']
                pop = population_ac(e['J'], e['S'], obs)
                f = fold(alpha)
                warning = bool(f and 3 * np.sqrt(e['S'][0, 0]) >= f['L'] - e['L'])
                intervals = [wilson(int(k), cfg['evaluation_replicates']) for k in successes]
                row = dict(
                    case=name,
                    n=n,
                    q=float(q),
                    F=e['F'],
                    L_eq=e['L'],
                    V_eq=e['V'],
                    loop_gain=e['loop_gain'],
                    lambda_slow=float(max(np.linalg.eigvals(e['J']).real)),
                    rho_L=float(pop[0]),
                    rho_V=float(pop[1]),
                    var_L=float(e['S'][0, 0]),
                    var_V=float(e['S'][1, 1]),
                    power_L=float(power[0]),
                    power_V=float(power[1]),
                    lower_L=intervals[0][0],
                    upper_L=intervals[0][1],
                    lower_V=intervals[1][0],
                    upper_V=intervals[1][1],
                    local_3sd_reaches_fold=warning,
                )
                rows.append(row)
                for i in [0, 1]:
                    if first[i] is None and power[i] >= 0.8:
                        first[i] = float(q)
                    if firstvalid[i] is None and power[i] >= 0.8 and not warning:
                        firstvalid[i] = float(q)
            summaries.append(
                dict(
                    case=name,
                    n=n,
                    first_q80_L=first[0],
                    first_q80_V=first[1],
                    q80_gap=first[1] - first[0] if None not in first else None,
                    first_q80_local_L=firstvalid[0],
                    first_q80_local_V=firstvalid[1],
                )
            )
        print('EWS case complete:', name, flush=True)
    csv_out('ews_detectability.csv', rows)
    csv_out('ews_thresholds.csv', cutoffrows)
    csv_out('ews_detection_summary.csv', summaries)
    np.savez_compressed(DATA / 'ews_all_replicate_ar1.npz', **records)
    return rows, summaries


def common_driver():
    rows = []
    curves = []
    perturb = []
    lags = np.linspace(-40, 40, 1601)
    for tauF in P['common_driver_model']['tau_F_grid']:
        J = np.array([[-1 / tauF, 0, 0], [1.0, -1.0, 0], [1 / 8, 0, -1 / 8]])
        Q = np.diag([2e-4 / tauF, 0, 0])
        S = solve_continuous_lyapunov(J, -Q)
        rho = []
        for lag in lags:
            cov = (S @ expm(J.T * lag))[1, 2] if lag >= 0 else (expm(J * (-lag)) @ S)[1, 2]
            rho.append(float(cov / np.sqrt(S[1, 1] * S[2, 2])))
        idx = int(np.argmax(rho))
        ac = population_ac(J, S, np.zeros(3))
        rows.append(
            dict(
                tau_F=tauF,
                peak_lag=float(lags[idx]),
                peak_correlation=rho[idx],
                rho_L=float(ac[1]),
                rho_V=float(ac[2]),
                intrinsic_lambda_L=-1,
                intrinsic_lambda_V=-0.125,
            )
        )
        for lag, r in zip(lags, rho):
            curves.append(dict(tau_F=tauF, lag=lag, correlation=r))
        for t in np.linspace(0, 40, 401):
            dy = expm(J * t) @ np.array([0.0, 1.0, 0.0])
            perturb.append(dict(tau_F=tauF, time=t, delta_L=dy[1], delta_V=dy[2]))
    csv_out('common_driver_summary.csv', rows)
    csv_out('common_driver_lag_curves.csv', curves)
    csv_out('common_driver_local_intervention.csv', perturb)
    return rows, curves


def figures(linear, steps, nonlin, main, ews, esummary, common, ccurves):
    fig = plt.figure(figsize=(183 / 25.4, 6.05), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])
    ax = fig.add_subplot(gs[0, 0])
    arr = main[0.6]
    ax.plot(arr[:, 0], arr[:, 2], color=COL['L'], label='L: LA process stress')
    ax.plot(arr[:, 0], arr[:, 3], color=COL['V'], label='V: vegetation stress')
    ax.plot(arr[:, 0], arr[:, 1], color=COL['F'], ls=':', label='F: climate forcing')
    ff = fold()
    tf = (ff['F'] - 0.2) / 0.6 * 200
    ax.axvline(tf, color='.5', ls='--', lw=0.8, label='Static fold crossed')
    ax.set(xlabel='Time / tau_L', ylabel='Dimensionless state', xlim=(0, 260))
    ax.legend(loc='upper left')
    panel(ax, 'a', 'Nonlinear response to gradual forcing')
    ax = fig.add_subplot(gs[0, 1])
    taus = P['linear_model']['tau_V_grid']
    etas = P['linear_model']['eta_grid']
    grid = np.array(
        [
            [
                next(
                    r['lead'] for r in linear if r['tau_V'] == t and r['eta'] == e and r['q'] == 0.5
                )
                for e in etas
            ]
            for t in taus
        ]
    )
    norm = matplotlib.colors.TwoSlopeNorm(
        vmin=min(-0.1, float(grid.min())), vcenter=0, vmax=float(grid.max())
    )
    im = ax.imshow(grid, origin='lower', aspect='auto', cmap='RdBu', norm=norm)
    ax.set_xticks(range(len(etas)), [str(e) for e in etas], rotation=45)
    ax.set_yticks(range(len(taus)), [str(t) for t in taus])
    ax.set(xlabel='Direct climate forcing of V (eta)', ylabel='tau_V / tau_L')
    fig.colorbar(im, ax=ax, pad=0.025, label='t50(V) - t50(L) [time units]')
    panel(ax, 'b', 'Lead can shrink or reverse')
    ax = fig.add_subplot(gs[1, 0])
    for case, style, lab in [
        ('cascade', '-', 'Cascade'),
        ('fast_vegetation_direct', '--', 'Fast V + direct forcing'),
    ]:
        rr = [r for r in steps if r['case'] == case]
        for var in ['L', 'V']:
            ax.plot(
                [r['time'] for r in rr],
                [r[var + '_fraction'] for r in rr],
                color=COL[var],
                ls=style,
                label=lab + ': ' + var,
            )
    ax.set(
        xlim=(0, 25),
        xlabel='Time / tau_L',
        ylabel='Fraction of own final response',
        ylim=(-0.02, 1.04),
    )
    ax.axhline(0.5, color='.8', lw=0.7)
    ax.legend(loc='lower right', fontsize=6.6)
    panel(ax, 'c', 'A fixed threshold defines response timing')
    ax = fig.add_subplot(gs[1, 1])
    q = np.linspace(0, 0.995, 151)
    stability_rows = []
    for alpha, style, lab in [(0.6, '-', 'Positive feedback'), (0, '--', 'No feedback')]:
        vals = [equilibrium(float(x), alpha) for x in q]
        eig = [max(np.linalg.eigvals(e['J']).real) for e in vals]
        stability_rows.extend(
            dict(alpha=alpha, q=float(x), F=e['F'], L_eq=e['L'], lambda_slow=float(v))
            for x, e, v in zip(q, vals, eig)
        )
        ax.plot(q, eig, style, color=COL['accent'] if alpha else COL['F'], label=lab)
    ax.axhline(0, lw=0.7, color='.65')
    ax.set(xlabel='Approach coordinate q', ylabel='Dominant eigenvalue', ylim=(-0.145, 0.005))
    ax.legend()
    panel(ax, 'd', 'State delay alone is not slowing down')
    csv_out('stability_curves.csv', stability_rows)
    save(fig, 'mechanism_and_conditions')

    fig, axs = plt.subplots(2, 3, figsize=(183 / 25.4, 5.0), layout='constrained')
    for ax, (name, label), letter in zip(
        axs[0],
        [
            ('default', 'Observation error:\nnone'),
            ('V_noisy', 'Observation error:\nvegetation only'),
            ('L_noisy', 'Observation error:\nLA only'),
        ],
        ['a', 'b', 'c'],
    ):
        rr = [r for r in ews if r['case'] == name and r['n'] == 120]
        for var in ['L', 'V']:
            ax.plot(
                [r['q'] for r in rr], [r['power_' + var] for r in rr], color=COL[var], label=var
            )
            ax.fill_between(
                [r['q'] for r in rr],
                [r['lower_' + var] for r in rr],
                [r['upper_' + var] for r in rr],
                color=COL[var],
                alpha=0.14,
                lw=0,
            )
        warn = [r['q'] for r in rr if r['local_3sd_reaches_fold']]
        if warn:
            ax.axvspan(min(warn), 1, color='.8', alpha=0.35)
        ax.axhline(0.8, color='.5', ls=':', lw=0.8)
        ax.set(xlabel='Approach coordinate q', ylabel='Detection fraction', ylim=(-0.02, 1.02))
        ax.legend(loc='upper left')
        panel(ax, letter, label)
    ax = axs[1, 0]
    rr = [r for r in ews if r['case'] == 'default' and r['n'] == 120]
    for var in ['L', 'V']:
        ax.plot([r['q'] for r in rr], [r['rho_' + var] for r in rr], color=COL[var], label=var)
    ax.set(xlabel='Approach coordinate q', ylabel='Population AR1', ylim=(0, 1.02))
    ax.legend()
    panel(ax, 'd', 'Population AR1:\nshared slow mode')
    ax = axs[1, 1]
    for tauF in [1, 4, 16]:
        rr = [r for r in ccurves if r['tau_F'] == tauF]
        ax.plot([r['lag'] for r in rr], [r['correlation'] for r in rr], label=f'tau_F = {tauF}')
    ax.axvline(0, color='.7', ls=':', lw=0.8)
    ax.set(xlabel='Lag (positive: L earlier)', ylabel='Corr[L(t), V(t + lag)]', xlim=(-20, 30))
    ax.legend()
    panel(ax, 'e', 'Shared climate:\nleading correlation')
    ax = axs[1, 2]
    for var in ['L', 'V']:
        ax.plot(
            [r['tau_F'] for r in common],
            [r['rho_' + var] for r in common],
            marker='o',
            ms=3,
            color=COL[var],
            label=var,
        )
    ax.set(xscale='log', xlabel='Climate memory tau_F', ylabel='Population AR1', ylim=(0, 1.02))
    ax.legend()
    panel(ax, 'f', 'Shared climate:\nincreasing memory')
    save(fig, 'ews_and_common_driver')


def main():
    linear, steps = linear_experiments()
    print('Linear response grid complete', flush=True)
    nonlinear, curves = nonlinear_experiments()
    print('Nonlinear sweeps complete', flush=True)
    ews, esummary = ews_experiments()
    common, ccurves = common_driver()
    figures(linear, steps, nonlinear, curves, ews, esummary, common, ccurves)
    summary = {
        'linear_cases': len(linear) // 3,
        'linear_threshold_comparisons': len(linear),
        'linear_positive_t50': sum(r['lead'] > 0 for r in linear if r['q'] == 0.5),
        'linear_negative_t50': sum(r['lead'] < 0 for r in linear if r['q'] == 0.5),
        'cascade_t50': next(
            r for r in linear if r['tau_V'] == 8 and r['eta'] == 0 and r['q'] == 0.5
        ),
        'reverse_example_t50': next(
            r for r in linear if r['tau_V'] == 0.25 and r['eta'] == 4 and r['q'] == 0.5
        ),
        'nonlinear_cases': len(nonlinear),
        'fold': fold(),
        'nonlinear_example': next(
            r
            for r in nonlinear
            if r['alpha'] == 0.6 and r['tau_V'] == 8 and r['ramp_duration'] == 200
        ),
        'ews_detectability': esummary,
        'common_driver': common,
        'software': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'matplotlib': matplotlib.__version__,
        },
    }
    dump('summary.json', summary)
    print('Model runs complete.')


if __name__ == '__main__':
    main()
