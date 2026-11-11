"""Beta-Binomial shrinkage engine for the Level 1 project. Pure functions (no Streamlit), tested.

Schema expected: player_id, season, successes, opportunities  (+ optional position, team).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from . import validation as V

PRIMARY_METRIC = ("Opportunity-weighted mean squared error of the predicted next-season rate "
                  "(lower is better). Registered in code before any data is evaluated.")
INTERVAL = 0.80          # nominal coverage for the calibration gate
CAL_TOLERANCE = 0.05     # coverage must be within +/- 5 points of nominal
MIN_GAIN = 0.02          # must beat the best baseline by at least 2% of its error
MIN_TEST_SEASONS = 3
MIN_TEST_ROWS = 200
MAX_ECE = 0.05           # calibration gate for the "top-third next season" probabilities
EVENT = ("Top-third season: next-season rate at or above the top-third cutoff of the most recent "
         "training season (qualifiers only). Cutoff and qualifying volume are fixed before the test season.")


def fit_beta_prior(k, n):
    """Empirical Bayes: choose Beta(a, b) that maximizes the beta-binomial MARGINAL likelihood
    (the 'evidence' from lesson L1.5). Returns (a, b)."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    m0 = np.clip(k.sum() / n.sum(), 1e-3, 1 - 1e-3)

    def nll(theta):
        mu = 1 / (1 + np.exp(-theta[0]))
        kappa = np.exp(theta[1])
        return -stats.betabinom.logpmf(k, n, mu * kappa, (1 - mu) * kappa).sum()

    res = optimize.minimize(nll, x0=[np.log(m0 / (1 - m0)), np.log(50.0)], method="Nelder-Mead",
                            options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 400})
    mu = 1 / (1 + np.exp(-res.x[0]))
    kappa = float(np.clip(np.exp(res.x[1]), 1.0, 1e5))
    return mu * kappa, (1 - mu) * kappa


def fit_priors(df: pd.DataFrame, group: str | None = None) -> dict:
    """Return {group_value: (a, b)}; key '__all__' when not grouping."""
    if not group:
        return {"__all__": fit_beta_prior(df["successes"], df["opportunities"])}
    return {g: fit_beta_prior(d["successes"], d["opportunities"]) for g, d in df.groupby(group)}


def prior_for(priors: dict, key):
    return priors.get(key, priors.get("__all__", next(iter(priors.values()))))


def posterior_table(df: pd.DataFrame, priors: dict, group: str | None = None, level=0.95) -> pd.DataFrame:
    """One row per player (all seasons pooled): raw rate, posterior mean and credible interval."""
    keys = ["player_id"] + ([group] if group else [])
    agg = df.groupby(keys, as_index=False)[["successes", "opportunities"]].sum()
    ab = np.array([prior_for(priors, r[group] if group else "__all__") for _, r in agg.iterrows()])
    a = ab[:, 0] + agg["successes"].to_numpy()
    b = ab[:, 1] + (agg["opportunities"] - agg["successes"]).to_numpy()
    lo, hi = stats.beta.ppf([(1 - level) / 2, (1 + level) / 2], a[:, None], b[:, None]).T
    agg["raw_rate"] = agg["successes"] / agg["opportunities"]
    agg["prior_mean"] = ab[:, 0] / ab.sum(1)
    agg["post_mean"] = a / (a + b)
    agg["ci_low"], agg["ci_high"] = lo, hi
    agg["weight_on_data"] = agg["opportunities"] / (agg["opportunities"] + ab.sum(1))
    return agg


def evaluate_next_season(df: pd.DataFrame, group: str | None = None) -> dict:
    """Rolling-origin validation. For each season T: fit the prior on seasons < T only, use each
    player's seasons < T as his history, and predict his season-T rate. Never touches T's data
    until scoring."""
    rows, leak_ok = [], True
    for T, train, test in V.rolling_origin(df, "season"):
        if train.empty or test.empty:
            continue
        leak_ok &= V.no_leakage(train, test, "season")
        priors = fit_priors(train, group)
        keys = ["player_id"] + ([group] if group else [])
        hist = train.groupby(keys, as_index=False)[["successes", "opportunities"]].sum()
        last_season = train["season"].max()
        recent = train[train["season"] == last_season]
        last = (train.sort_values("season").groupby(keys, as_index=False).last()
                .assign(last_rate=lambda d: d["successes"] / d["opportunities"])[keys + ["last_rate"]])
        m = test.merge(hist, on=keys, suffixes=("", "_hist")).merge(last, on=keys, how="left")
        if m.empty:
            continue
        if group:
            gmean = train.groupby(group)["successes"].sum() / train.groupby(group)["opportunities"].sum()
            pop = m[group].map(gmean).fillna(train["successes"].sum() / train["opportunities"].sum())
        else:
            pop = pd.Series(train["successes"].sum() / train["opportunities"].sum(), index=m.index)
        ab = np.array([prior_for(priors, r[group] if group else "__all__") for _, r in m.iterrows()])
        a = ab[:, 0] + m["successes_hist"].to_numpy()
        b = ab[:, 1] + (m["opportunities_hist"] - m["successes_hist"]).to_numpy()
        n_t = m["opportunities"].to_numpy()
        k_t = m["successes"].to_numpy()
        lo, hi = stats.betabinom.interval(INTERVAL, n_t, a, b)
        # Binary event (concept 25): cutoff and qualifying volume come from the last TRAINING season
        q_min = float(recent["opportunities"].median())
        qual = recent[recent["opportunities"] >= q_min]
        cut = float((qual["successes"] / qual["opportunities"]).quantile(2 / 3))
        need = np.ceil(cut * n_t).astype(int)
        rows.append(pd.DataFrame({
            "season": T, "n": n_t, "actual": k_t / n_t,
            "raw": (m["successes_hist"] / m["opportunities_hist"]).to_numpy(),
            "population": pop.to_numpy(), "model": a / (a + b),
            "covered": (k_t >= lo) & (k_t <= hi),
            "qualifies": n_t >= q_min, "event": (k_t >= need).astype(int),
            "p_event": stats.betabinom.sf(need - 1, n_t, a, b),
            "last_rate": m["last_rate"].to_numpy(),
            "base_rate": float(((qual["successes"] / qual["opportunities"]) >= cut).mean()),
        }))
    if not rows:
        return {"ok": False, "reason": "Need players who appear in at least two different seasons."}
    r = pd.concat(rows, ignore_index=True)

    def wmse(d, col):
        return float(np.average((d[col] - d["actual"]) ** 2, weights=d["n"]))

    per = r.groupby("season").apply(lambda d: pd.Series({
        "test_rows": len(d), "raw": wmse(d, "raw"), "population": wmse(d, "population"),
        "model": wmse(d, "model"), "coverage": d["covered"].mean()}), include_groups=False).reset_index()
    overall = {c: wmse(r, c) for c in ("raw", "population", "model")}
    best_base = min(overall["raw"], overall["population"])
    gain = 1 - overall["model"] / best_base
    wins = int((per["model"] < per[["raw", "population"]].min(axis=1)).sum())
    coverage = float(r["covered"].mean())
    e = r[r["qualifies"]]
    prob = {"n": len(e), "event_rate": float(e["event"].mean()),
            "brier": V.brier(e["p_event"], e["event"]),
            "brier_base": V.brier(e["base_rate"], e["event"]),
            "skill": V.brier_skill(e["p_event"], e["event"], e["base_rate"]),
            "ece": V.ece(e["p_event"], e["event"]),
            "reliability": V.reliability_table(e["p_event"], e["event"]),
            "auc_model": V.auc(e["p_event"], e["event"]),
            "auc_last_season": V.auc(e["last_rate"], e["event"])}
    return {"ok": True, "rows": r, "per_season": per, "overall": overall, "gain": gain,
            "wins": wins, "n_seasons": len(per), "coverage": coverage, "leak_ok": leak_ok, "prob": prob}


def gates(result: dict, data_info: dict) -> list:
    """Production Candidate gates: list of (name, passed, detail). All must pass."""
    if not result.get("ok"):
        return [("Evaluation ran", False, result.get("reason", ""))]
    is_real = data_info["source"] != "simulated" and data_info.get("real")
    return [
        ("Real data you have rights to use", bool(is_real),
         "Simulated data can never qualify." if data_info["source"] == "simulated"
         else ("Marked as real on upload." if is_real else "Upload is not marked as real data.")),
        ("Baseline defined", True, "Two baselines: player's raw past rate, and the population/position mean."),
        ("Metric predefined", True, PRIMARY_METRIC),
        ("Time-based validation (concept 26)",
         result["n_seasons"] >= MIN_TEST_SEASONS and len(result["rows"]) >= MIN_TEST_ROWS,
         f"Rolling origin: train on all seasons before T, test on T. {result['n_seasons']} test seasons, "
         f"{len(result['rows'])} player-seasons (need >= {MIN_TEST_SEASONS} and >= {MIN_TEST_ROWS})."),
        ("Beats the best baseline",
         result["gain"] >= MIN_GAIN and result["wins"] > result["n_seasons"] / 2,
         f"{result['gain']:+.1%} vs best baseline overall; wins {result['wins']} of {result['n_seasons']} seasons "
         f"(need >= {MIN_GAIN:.0%} and a majority)."),
        ("No temporal leakage (concept 26)", bool(result["leak_ok"]),
         "Checked on every fold: prior, player history, top-third cutoff and qualifying volume use only earlier seasons."),
        ("Calibrated probabilities (concept 25)",
         result["prob"]["ece"] <= MAX_ECE and result["prob"]["skill"] > 0,
         f"P(top-third next season): calibration error {result['prob']['ece']:.3f} (need <= {MAX_ECE}), "
         f"Brier skill vs base rate {result['prob']['skill']:+.1%} (need > 0)."),
        ("Calibrated intervals", abs(result["coverage"] - INTERVAL) <= CAL_TOLERANCE,
         f"{INTERVAL:.0%} predictive intervals covered {result['coverage']:.1%} of outcomes "
         f"(need {INTERVAL - CAL_TOLERANCE:.0%}–{INTERVAL + CAL_TOLERANCE:.0%})."),
    ]


def verdict(gate_list: list) -> str:
    if all(p for _, p, _ in gate_list):
        return "Meets Production Candidate criteria (review before any use)"
    return "Learning Lab only"
