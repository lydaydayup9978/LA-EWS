"""CCM between monthly LA and NDVI residuals (24 mappings, 1,000 surrogates)."""

from pathlib import Path
import argparse, shutil
import pandas as pd
import monthly_core as m

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/monthly_ccm_rerun')
    a = ap.parse_args()
    (a.out / 'results').mkdir(parents=True, exist_ok=True)
    for name in ['inputs.npz', 'embeddings.json']:
        shutil.copy2(m.OUT / name, a.out / name)
    m.OUT = a.out
    rows = []
    for job in m.jobs():
        rows.append(m.run_one(job))
        print('Completed', len(rows), 'of 24', flush=True)
    d = pd.DataFrame(rows)
    d['q_BH24'] = m.bh(d.p_joint.to_numpy())
    d.to_csv(a.out / 'results.csv', index=False)
