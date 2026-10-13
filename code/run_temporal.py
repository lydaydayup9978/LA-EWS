"""Table S3: Kendall trend tests for the six rising segments in Fig. 2 (1,000 phase-randomized surrogates)."""

from pathlib import Path
import numpy as np
import pandas as pd
from temporal_trends import analyze

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    z = np.load(ROOT / 'data/temporal/segments.npz')
    ref = pd.read_csv(ROOT / 'data/temporal/results.csv')
    rows = analyze({key: z[key] for key in ref.id})
    df = pd.DataFrame(rows).set_index('id')
    ref = ref.set_index('id').loc[df.index]
    out = ROOT / 'outputs/temporal'
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'S3.csv')
    print(df[['n', 'tau', 'p_phase', 'q_BY']])
