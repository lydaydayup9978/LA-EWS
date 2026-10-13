"""Compare the recomputed tables (outputs/) and the saved resampling results (data/) with the reported values."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau, false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)


def close(a, b, tol=1e-11):
    if not np.allclose(a, b, rtol=0, atol=tol):
        raise AssertionError((np.asarray(a), np.asarray(b)))


def main():
    checks = {}
    z = np.load(D / 'figure4/inputs.npz')
    null = np.load(D / 'tables/distributions.npz')
    a = pd.read_csv(D / 'tables/S5A.csv')
    top = pd.read_csv(D / 'figure4/source_data_top10.csv')
    for r in a.itertuples():
        X, y, lags = [z[r.id + '_' + k] for k in ['X', 'y', 'lags_months']]
        assert X.shape == (240, len(lags))
        rr = np.array([pearsonr(x, y).statistic for x in X.T])
        close(rr.max(), r.max_r)
        rank = np.lexsort((lags, -abs(rr)))[:10]
        tt = top[top.id.eq(r.id)].sort_values('rank')
        close(tt.r, rr[rank])
        close(tt.lag_years, lags[rank] / 12)
        for method in ['block_n10', 'block_n5', 'block_n3']:
            vals = null[r.id + '_' + method]
            assert len(vals) == 1000
            close((1 + np.sum(vals >= rr.max())) / 1001, getattr(r, 'p_' + method))
        close(
            np.quantile(null[r.id + '_bootstrap_lags'], [0.025, 0.975], method='inverted_cdf') / 12,
            [r.range_low_years, r.range_high_years],
        )
    checks['figure4_bars'] = 60
    checks['S5A_groups'] = 6
    for table, keys, cols in [
        (
            'S5B',
            ['id'],
            ['training_selected_lead_years', 'gap_steps', 'test_n', 'heldout_fixed_lag_r'],
        ),
        ('S5C', ['id'], ['trimmed_peak_years', 'trimmed_max_r', 'trimmed_r_at_original_peak']),
        ('S6A', ['id', 'method'], ['lead_years', 'max_r']),
        ('S6B', ['id'], ['extended_peak', 'extended_r']),
        (
            'S7',
            ['id', 'window_years'],
            ['lead_years', 'max_r', 'reference_lead_same_segment', 'reference_r_same_segment'],
        ),
    ]:
        actual = pd.read_csv(OUT / 'figure4' / f'{table}.csv').set_index(keys).sort_index()
        expected = pd.read_csv(D / 'tables' / f'{table}.csv').set_index(keys).sort_index()
        assert actual.index.equals(expected.index)
        for col in cols:
            close(actual[col], expected[col])
        checks[table + '_rows'] = len(actual)
    cc = pd.read_csv(D / 'ccm/results.csv')
    assert len(cc) == 120 and cc.B.eq(1000).all()
    for r in cc.itertuples():
        zz = np.load(D / 'ccm' / (r.case + '.npz'))
        obs = zz['observed']
        v = zz['rho']
        assert v.shape[0] == 1000
        close((1 + np.sum(v[:, -1] >= obs[-1])) / 1001, r.p_endpoint)
        close((1 + np.sum(v[:, -1] - v[:, 0] >= obs[-1] - obs[0])) / 1001, r.p_gain)
        close((1 + np.sum(zz['tau'] >= kendalltau(zz['lib_sizes'], obs).statistic)) / 1001, r.p_tau)
    close(false_discovery_control(cc.p_endpoint), cc.q_endpoint_BH120)
    for direction in ['main', 'reverse']:
        rows = cc[cc.direction.eq(direction)]
        for metric in ['gain', 'tau']:
            close(false_discovery_control(rows['p_' + metric]), rows['q_' + metric + '_BH60'])
    counts = (
        cc.groupby('direction').q_endpoint_BH120.apply(lambda x: int((x < 0.05).sum())).to_dict()
    )
    assert counts == {'main': 14, 'reverse': 6}
    assert (cc.q_gain_BH60 < 0.05).sum() == 0 and (cc.q_tau_BH60 < 0.05).sum() == 0
    checks['CCM_mappings'] = 120
    checks['CCM_endpoint_BH_counts'] = counts
    tr = pd.read_csv(D / 'temporal/results.csv')
    td = np.load(D / 'temporal/segments.npz')
    tn = np.load(D / 'temporal/distributions.npz')
    for r in tr.itertuples():
        y = td[r.id]
        tau = kendalltau(np.arange(len(y)), y).statistic
        close(tau, r.tau)
        close((1 + np.sum(abs(tn[r.id + '_phase_tau']) >= abs(tau))) / 1001, r.p_phase)
    close(false_discovery_control(tr.p_phase, method='by'), tr.q_BY)
    checks['S3_groups'] = 6
    mr = pd.read_csv(D / 'monthly_ccm/results.csv')
    assert len(mr) == 24
    for r in mr.itertuples():
        v = np.load(D / 'monthly_ccm/results' / (r.id + '.npz'))
        lags = v['lags']
        obs = v['rho']
        null_hi = v['null_rho_max']
        null_lo = v['null_rho_min']
        assert len(null_hi) == 1000
        idx = np.flatnonzero(lags < 0)
        j = idx[np.argmax(obs[-1, idx])]
        ps = (1 + np.sum(null_hi.max(axis=1) >= obs[-1, j] - 1e-12)) / 1001
        pg = (1 + np.sum((null_hi - null_lo).max(axis=1) >= obs[-1, j] - obs[0, j] - 1e-12)) / 1001
        close(ps, r.p_skill_search_adjusted)
        close(pg, r.p_gain_search_adjusted)
    close(false_discovery_control(mr.p_joint), mr.q_BH24)
    checks['monthly_CCM_mappings'] = 24
    spatial = pd.read_csv(OUT / 'spatial/summary.csv')
    assert spatial.significant_pixels.tolist() == [39880, 32205, 34405, 25837]
    checks['spatial_significant_pixels'] = spatial.significant_pixels.tolist()
    rerun = OUT / 'figure4' / 'S5A.csv'
    if rerun.exists():
        new = pd.read_csv(rerun).set_index('id').loc[a.id]
        for col in [
            'block_length',
            'range_low_years',
            'range_high_years',
            'p_block_n10',
            'p_block_n5',
            'p_block_n3',
        ]:
            close(new[col], a.set_index('id')[col])
        checks['S5A_resampling_rerun'] = True
    checks['passed'] = True
    (OUT / 'validation.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
