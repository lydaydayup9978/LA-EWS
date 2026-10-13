import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

TAU_CANDIDATES = list(range(1, 13))
E_CANDIDATES = list(range(2, 11))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 5 or np.nanstd(a[m]) == 0 or np.nanstd(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def embed_indices(n: int, e: int, tau: int) -> np.ndarray:
    return np.arange((e - 1) * tau, n)


def embed_series(x: np.ndarray, e: int, tau: int) -> tuple[np.ndarray, np.ndarray]:
    idx = embed_indices(len(x), e, tau)
    if len(idx) == 0:
        return np.empty((0, e)), np.asarray([], dtype=int)
    mat = np.column_stack([x[idx - j * tau] for j in range(e)])
    finite = np.isfinite(mat).all(axis=1)
    return mat[finite], idx[finite]


def simplex_cv_score(source: np.ndarray, target: np.ndarray, e: int, tau: int) -> float:
    manifold, idx = embed_series(source, e, tau)
    if len(idx) < max(30, e + 5):
        return np.nan
    target_values = target[idx]
    finite = np.isfinite(target_values)
    manifold = manifold[finite]
    target_values = target_values[finite]
    if len(target_values) < max(30, e + 5):
        return np.nan
    split = max(int(math.floor(len(target_values) * 0.6)), e + 3)
    if len(target_values) - split < 10:
        return np.nan
    lib_x = manifold[:split]
    lib_y = target_values[:split]
    pred_x = manifold[split:]
    obs = target_values[split:]
    k = min(e + 1, len(lib_y))
    if k < 2:
        return np.nan
    dists = cdist(pred_x, lib_x)
    preds = []
    for row in dists:
        nn = np.argpartition(row, k - 1)[:k]
        d = row[nn]
        d0 = max(float(np.nanmin(d)), 1e-12)
        w = np.exp(-d / d0)
        preds.append(float(np.sum(w * lib_y[nn]) / np.sum(w)))
    return corr(np.asarray(preds), obs)


def choose_tau_e(source: np.ndarray, target: np.ndarray) -> tuple[int, int, pd.DataFrame]:
    tau_rows = []
    for tau in TAU_CANDIDATES:
        score = simplex_cv_score(source, target, e=2, tau=tau)
        tau_rows.append(
            {"stage": "tau", "candidate_tau": tau, "candidate_E": 2, "simplex_rho": score}
        )
    tau_df = pd.DataFrame(tau_rows)
    max_tau_score = tau_df["simplex_rho"].max(skipna=True)
    if not np.isfinite(max_tau_score):
        selected_tau = 1
    else:
        close = tau_df[tau_df["simplex_rho"] >= max_tau_score - 0.02]
        selected_tau = int(close.sort_values("candidate_tau").iloc[0]["candidate_tau"])

    e_rows = []
    for e in E_CANDIDATES:
        score = simplex_cv_score(source, target, e=e, tau=selected_tau)
        e_rows.append(
            {"stage": "E", "candidate_tau": selected_tau, "candidate_E": e, "simplex_rho": score}
        )
    e_df = pd.DataFrame(e_rows)
    max_e_score = e_df["simplex_rho"].max(skipna=True)
    if not np.isfinite(max_e_score):
        selected_e = 2
    else:
        close = e_df[e_df["simplex_rho"] >= max_e_score - 0.02]
        selected_e = int(close.sort_values("candidate_E").iloc[0]["candidate_E"])

    rows = pd.concat([tau_df, e_df], ignore_index=True)
    rows["selected_tau"] = selected_tau
    rows["selected_E"] = selected_e
    return selected_tau, selected_e, rows
