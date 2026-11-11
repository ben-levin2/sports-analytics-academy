"""Level 3 survival engine: Kaplan-Meier, a small Cox proportional-hazards model (NumPy only),
and a leakage-safe 'contribute by year 2' test. Pure functions, tested. No extra dependencies.

Roster schema: player_id, class_year, seasons_observed, contributed (1 = event in last observed season)
Optional: recruit_rating, position, team, left_before_contributing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import validation as V

HORIZON = 2               # "contribute by season 2"
MIN_GAIN = 0.02           # Cox must cut the Brier score by >= 2% vs the rating-tier baseline
MAX_ECE = 0.05
MIN_TEST_CLASSES = 3
MIN_TEST_ROWS = 200
PRIMARY_METRIC = (f"Brier score of P(contributes by season {HORIZON}) for each new recruiting class, "
                  "predicted at enrollment. Registered in code before any data is evaluated.")


# ---------------------------------------------------------------- Kaplan-Meier
def kaplan_meier(t, e) -> pd.DataFrame:
    """Survival = share who have NOT had the event yet, counting censored players correctly.
    Includes a 95% Greenwood confidence band."""
    t, e = np.asarray(t, float), np.asarray(e, int)
    rows, s, gw = [], 1.0, 0.0
    for time in np.unique(t):
        at_risk = int((t >= time).sum())
        d = int(((t == time) & (e == 1)).sum())
        c = int(((t == time) & (e == 0)).sum())
        if at_risk:
            s *= 1 - d / at_risk
            if at_risk > d:
                gw += d / (at_risk * (at_risk - d))
        se = s * np.sqrt(gw)
        rows.append((time, at_risk, d, c, s, max(0.0, s - 1.96 * se), min(1.0, s + 1.96 * se)))
    return pd.DataFrame(rows, columns=["time", "at_risk", "events", "censored", "survival", "lo", "hi"])


def km_at(km: pd.DataFrame, time: float) -> float:
    s = km.loc[km["time"] <= time, "survival"]
    return float(s.iloc[-1]) if len(s) else 1.0


# ---------------------------------------------------------------- Cox proportional hazards
def cox_fit(X, t, e, l2: float = 1e-4, iters: int = 50):
    """Breslow partial likelihood, Newton's method. Returns dict with beta, se, baseline hazard."""
    X, t, e = np.asarray(X, float), np.asarray(t, float), np.asarray(e, int)
    p = X.shape[1]
    beta = np.zeros(p)
    times = np.unique(t[e == 1])
    for _ in range(iters):
        eta = X @ beta
        w = np.exp(eta - eta.max())
        g, H = -l2 * beta, -l2 * np.eye(p)
        for s in times:
            risk = t >= s
            ev = (t == s) & (e == 1)
            d = ev.sum()
            wr = w[risk]
            S0 = wr.sum()
            S1 = wr @ X[risk]
            S2 = (X[risk] * wr[:, None]).T @ X[risk]
            g += X[ev].sum(0) - d * S1 / S0
            H -= d * (S2 / S0 - np.outer(S1, S1) / S0 ** 2)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    se = np.sqrt(np.diag(np.linalg.inv(-H)))
    eta = X @ beta
    H0 = []
    cum = 0.0
    for s in times:
        risk = t >= s
        d = ((t == s) & (e == 1)).sum()
        cum += d / np.exp(eta[risk]).sum()
        H0.append((s, cum))
    return {"beta": beta, "se": se, "H0": pd.DataFrame(H0, columns=["time", "cum_hazard"])}


def cox_prob_by(model, X, time) -> np.ndarray:
    """P(event by `time` | x) = 1 - exp(-H0(time) * exp(x * beta))."""
    h0 = model["H0"].loc[model["H0"]["time"] <= time, "cum_hazard"]
    h0 = float(h0.iloc[-1]) if len(h0) else 0.0
    return 1 - np.exp(-h0 * np.exp(np.asarray(X, float) @ model["beta"]))


def design(df: pd.DataFrame, rating_mean=None, rating_sd=None, positions=None):
    """Covariates: recruit rating (in SDs) + position dummies (first position = reference)."""
    cols, names = [], []
    if "recruit_rating" in df.columns:
        m = df["recruit_rating"].mean() if rating_mean is None else rating_mean
        s = df["recruit_rating"].std() if rating_sd is None else rating_sd
        cols.append(((df["recruit_rating"] - m) / s).to_numpy())
        names.append("rating (+1 SD)")
    if "position" in df.columns and positions is not None and len(positions) > 1:
        for p in positions[1:]:
            cols.append((df["position"] == p).to_numpy(float))
            names.append(f"position {p}")
    X = np.column_stack(cols) if cols else np.zeros((len(df), 0))
    return X, names


# ---------------------------------------------------------------- leakage-safe evaluation
def truncate(df: pd.DataFrame, last_known_season: int) -> pd.DataFrame:
    """What we would have known at the end of `last_known_season`: later seasons are cut off."""
    d = df[df["class_year"] <= last_known_season].copy()
    avail = last_known_season - d["class_year"] + 1
    d["t"] = np.minimum(d["seasons_observed"], avail)
    d["e"] = ((d["contributed"] == 1) & (d["seasons_observed"] <= avail)).astype(int)
    d["known_through"] = d["class_year"] + d["t"] - 1
    return d


def evaluate_contribution(df: pd.DataFrame) -> dict:
    """For each recruiting class C: train on classes < C as known at the end of season C-1,
    predict P(contribute by season 2) at enrollment, score once the outcome is known."""
    data_end = int((df["class_year"] + df["seasons_observed"] - 1).max())
    classes = sorted(df["class_year"].unique())
    rows, leak_ok = [], True
    for C in classes:
        if C + HORIZON - 1 > data_end:
            continue
        train = truncate(df[df["class_year"] < C], C - 1)
        if train["class_year"].nunique() < 2 or train["e"].sum() < 20:
            continue
        leak_ok &= bool((train["known_through"] <= C - 1).all())
        test = df[df["class_year"] == C].copy()
        test["y"] = ((test["contributed"] == 1) & (test["seasons_observed"] <= HORIZON)).astype(int)

        # Baseline: Kaplan-Meier within rating tiers (thirds of the training ratings)
        if "recruit_rating" in df.columns:
            cuts = train["recruit_rating"].quantile([1 / 3, 2 / 3]).to_numpy()
            tier_tr = np.digitize(train["recruit_rating"], cuts)
            tier_te = np.digitize(test["recruit_rating"], cuts)
            p_tier = {k: 1 - km_at(kaplan_meier(train["t"][tier_tr == k], train["e"][tier_tr == k]), HORIZON)
                      for k in np.unique(tier_tr)}
            base = np.array([p_tier.get(k, np.nan) for k in tier_te])
        else:
            base = np.full(len(test), 1 - km_at(kaplan_meier(train["t"], train["e"]), HORIZON))
        overall = 1 - km_at(kaplan_meier(train["t"], train["e"]), HORIZON)

        positions = sorted(train["position"].unique()) if "position" in df.columns else None
        m, s = train["recruit_rating"].mean() if "recruit_rating" in df.columns else None, \
            train["recruit_rating"].std() if "recruit_rating" in df.columns else None
        Xtr, _ = design(train, m, s, positions)
        Xte, _ = design(test, m, s, positions)
        if Xtr.shape[1]:
            model = cox_fit(Xtr, train["t"], train["e"])
            p_cox = cox_prob_by(model, Xte, HORIZON)
        else:
            p_cox = np.full(len(test), overall)
        rows.append(pd.DataFrame({"class_year": C, "y": test["y"].to_numpy(), "baseline": base,
                                  "cox": p_cox, "overall": overall}))
    if not rows:
        return {"ok": False, "reason": "Need at least 3 recruiting classes with two seasons of follow-up."}
    r = pd.concat(rows, ignore_index=True).dropna()
    per = r.groupby("class_year").apply(lambda d: pd.Series({
        "recruits": len(d), "contributed_by_2": d["y"].mean(),
        "baseline": V.brier(d["baseline"], d["y"]), "cox": V.brier(d["cox"], d["y"])}),
        include_groups=False).reset_index()
    b = {c: V.brier(r[c], r["y"]) for c in ("overall", "baseline", "cox")}
    return {"ok": True, "rows": r, "per_class": per, "brier": b, "gain": 1 - b["cox"] / b["baseline"],
            "wins": int((per["cox"] < per["baseline"]).sum()), "n_classes": len(per),
            "ece": V.ece(r["cox"], r["y"]), "reliability": V.reliability_table(r["cox"], r["y"]),
            "auc_cox": V.auc(r["cox"], r["y"]), "auc_baseline": V.auc(r["baseline"], r["y"]),
            "leak_ok": leak_ok, "has_rating": "recruit_rating" in df.columns}


def gates(res: dict, data_info: dict) -> list:
    if not res.get("ok"):
        return [("Evaluation ran", False, res.get("reason", ""))]
    is_real = data_info["source"] != "simulated" and data_info.get("real")
    return [
        ("Real data you have rights to use", bool(is_real),
         "Simulated data can never qualify." if data_info["source"] == "simulated"
         else ("Marked as real on upload." if is_real else "Upload is not marked as real data.")),
        ("Baseline defined", res["has_rating"],
         "Kaplan-Meier within rating tiers (thirds)." if res["has_rating"]
         else "Needs a recruit_rating column for the rating-tier baseline and the Cox model."),
        ("Metric predefined", True, PRIMARY_METRIC),
        ("Time-based validation (concept 26)",
         res["n_classes"] >= MIN_TEST_CLASSES and len(res["rows"]) >= MIN_TEST_ROWS,
         f"Each class predicted using only earlier classes as known before it enrolled. "
         f"{res['n_classes']} test classes, {len(res['rows'])} recruits (need >= {MIN_TEST_CLASSES} and >= {MIN_TEST_ROWS})."),
        ("Cox beats the baseline",
         res["gain"] >= MIN_GAIN and res["wins"] > res["n_classes"] / 2,
         f"{res['gain']:+.1%} Brier vs rating-tier baseline; wins {res['wins']} of {res['n_classes']} classes "
         f"(need >= {MIN_GAIN:.0%} and a majority)."),
        ("No temporal leakage (concept 26)", bool(res["leak_ok"]),
         "Training outcomes are cut off at the season before each test class enrolls."),
        ("Calibrated probabilities (concept 25)", res["ece"] <= MAX_ECE,
         f"Calibration error {res['ece']:.3f} (need <= {MAX_ECE})."),
    ]


def verdict(gate_list):
    return ("Meets Production Candidate criteria (review before any use)"
            if all(p for _, p, _ in gate_list) else "Learning Lab only")
