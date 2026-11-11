"""Level 2 engine: player archetypes with standardization, K-means, PCA, stability and a
leakage-safe downstream test. Pure functions (no Streamlit), tested.

Required columns: player_id, season, position, + 3 or more numeric feature columns.
Optional: position_group, snaps, production, next_production (needed for the Production Candidate test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, mean_absolute_error, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import validation as V

RESERVED = {"player_id", "season", "position", "position_group", "snaps", "production",
            "next_production", "true_archetype", "team"}
MIN_GAIN = 0.02            # archetypes must cut next-season MAE by >= 2% vs the baseline
MIN_STABILITY = 0.80       # mean ARI across random restarts
MIN_TEST_SEASONS = 3
MIN_TEST_ROWS = 200
PRIMARY_METRIC = ("Mean absolute error of next-season production, rolling origin by season. "
                  "Registered in code before any data is evaluated.")


def feature_candidates(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in RESERVED and pd.api.types.is_numeric_dtype(df[c])]


def prepare(df, features, group=None, min_snaps=0):
    d = df.copy()
    if group and "position_group" in d.columns:
        d = d[d["position_group"] == group]
    if min_snaps and "snaps" in d.columns:
        d = d[d["snaps"] >= min_snaps]
    return d.dropna(subset=features).reset_index(drop=True)


def standardize(d, features):
    scaler = StandardScaler().fit(d[features])
    return scaler, scaler.transform(d[features])


def silhouette_by_k(Z, ks=range(2, 9), seed=0, sample=2000):
    out = {}
    for k in ks:
        labels = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
        out[k] = float(silhouette_score(Z, labels, sample_size=min(sample, len(Z)), random_state=seed))
    return out


def fit_kmeans(Z, k, seed=0):
    return KMeans(k, n_init=10, random_state=seed).fit(Z)


def stability(Z, k, seeds=(0, 1, 2, 3, 4)):
    """Mean adjusted Rand index between clusterings from different random starts (1 = identical)."""
    labs = [KMeans(k, n_init=1, random_state=s).fit_predict(Z) for s in seeds]
    pairs = [adjusted_rand_score(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]
    return float(np.mean(pairs))


def pca_2d(Z):
    p = PCA(2).fit(Z)
    return p.transform(Z), p.explained_variance_ratio_


def profiles(d, labels, features):
    """Cluster profile in z-scores (how far each cluster sits from the average player, in SDs)."""
    z = (d[features] - d[features].mean()) / d[features].std()
    t = z.groupby(labels).mean()
    t.insert(0, "players", pd.Series(labels).value_counts().sort_index().to_numpy())
    return t


def _design(d, base_cols, clusters=None, k=None, positions=None):
    X = [d[base_cols].to_numpy(float)]
    X.append(pd.get_dummies(pd.Categorical(d["position"], categories=positions)).to_numpy(float))
    if clusters is not None:
        X.append(np.eye(k)[clusters])
    return np.hstack(X)


def evaluate_downstream(df, features, k, group=None, min_snaps=0):
    """Do archetypes help predict NEXT season's production beyond 'this season + position'?
    Each fold: scaler, K-means and regressions are fit on seasons < T only, tested on season T."""
    need = {"production", "next_production"}
    if not need <= set(df.columns):
        return {"ok": False, "reason": "Needs `production` and `next_production` columns for the downstream test."}
    d = prepare(df, features, group, min_snaps).dropna(subset=["next_production"])
    positions = sorted(d["position"].unique())
    rows, leak_ok, stab = [], True, []
    for T, train, test in V.rolling_origin(d, "season", min_train_periods=2):
        if len(train) < 50 or test.empty:
            continue
        leak_ok &= V.no_leakage(train, test)
        scaler = StandardScaler().fit(train[features])
        Ztr, Zte = scaler.transform(train[features]), scaler.transform(test[features])
        km = KMeans(k, n_init=10, random_state=0).fit(Ztr)
        stab.append(stability(Ztr, k))
        base = Ridge(1.0).fit(_design(train, ["production"], positions=positions), train["next_production"])
        arch = Ridge(1.0).fit(_design(train, ["production"], km.labels_, k, positions), train["next_production"])
        rows.append(pd.DataFrame({
            "season": T, "actual": test["next_production"].to_numpy(),
            "same_as_last": test["production"].to_numpy(),
            "baseline": base.predict(_design(test, ["production"], positions=positions)),
            "archetypes": arch.predict(_design(test, ["production"], km.predict(Zte), k, positions))}))
    if not rows:
        return {"ok": False, "reason": "Not enough seasons with next-season outcomes."}
    r = pd.concat(rows, ignore_index=True)
    mae = {c: float(mean_absolute_error(r["actual"], r[c])) for c in ("same_as_last", "baseline", "archetypes")}
    per = r.groupby("season").apply(lambda g: pd.Series(
        {"test_rows": len(g), **{c: mean_absolute_error(g["actual"], g[c]) for c in ("same_as_last", "baseline", "archetypes")}}),
        include_groups=False).reset_index()
    return {"ok": True, "rows": r, "per_season": per, "mae": mae,
            "gain": 1 - mae["archetypes"] / mae["baseline"],
            "wins": int((per["archetypes"] < per["baseline"]).sum()), "n_seasons": len(per),
            "leak_ok": leak_ok, "stability": float(np.mean(stab))}


def gates(result, data_info):
    if not result.get("ok"):
        return [("Evaluation ran", False, result.get("reason", ""))]
    is_real = data_info["source"] != "simulated" and data_info.get("real")
    return [
        ("Real data you have rights to use", bool(is_real),
         "Simulated data can never qualify." if data_info["source"] == "simulated"
         else ("Marked as real on upload." if is_real else "Upload is not marked as real data.")),
        ("Baseline defined", True, "Ridge regression: this season's production + listed position (no archetypes)."),
        ("Metric predefined", True, PRIMARY_METRIC),
        ("Time-based validation (concept 26)",
         result["n_seasons"] >= MIN_TEST_SEASONS and len(result["rows"]) >= MIN_TEST_ROWS,
         f"{result['n_seasons']} test seasons, {len(result['rows'])} player-seasons "
         f"(need >= {MIN_TEST_SEASONS} and >= {MIN_TEST_ROWS})."),
        ("Archetypes beat the baseline",
         result["gain"] >= MIN_GAIN and result["wins"] > result["n_seasons"] / 2,
         f"{result['gain']:+.1%} MAE vs baseline; wins {result['wins']} of {result['n_seasons']} seasons "
         f"(need >= {MIN_GAIN:.0%} and a majority)."),
        ("No temporal leakage (concept 26)", bool(result["leak_ok"]),
         "Scaler, K-means and regressions are refit inside every fold on earlier seasons only."),
        ("Stable clusters", result["stability"] >= MIN_STABILITY,
         f"Mean agreement (ARI) across random restarts {result['stability']:.2f} (need >= {MIN_STABILITY})."),
        ("Calibration (concept 25)", True, "Not applicable: this output is a number, not a probability."),
    ]


def verdict(gate_list):
    if all(p for _, p, _ in gate_list):
        return "Meets Production Candidate criteria (review before any use)"
    return "Learning Lab only"
