"""Level 3 spatial engine: Gaussian-process (Kriging) maps of recruit production, a density-aware
regional summary, and a leakage-safe test of whether location adds predictive value.
Uses scikit-learn (already installed). Pure functions, tested.

Hometown schema: player_id, lat, lon, production. Optional: local_density, class_year, recruit_rating.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
from sklearn.linear_model import Ridge

from . import validation as V

MIN_GAIN = 0.02
INTERVAL = 0.80
CAL_TOL = 0.05
MIN_TEST_CLASSES = 3
MIN_TEST_ROWS = 200
TRAIN_CAP = 600            # GPs cost grows with n^3; each fold uses at most this many training players
Z80 = 1.2816
PRIMARY_METRIC = ("Mean absolute error of production for each new recruiting class, predicted from earlier "
                  "classes only. Registered in code before any data is evaluated.")


def kernel(name: str, length: float = 8.0, nu: float = 1.5):
    """Similarity rule. Length scale is in degrees of latitude/longitude (1 degree ~ 69 miles north-south)."""
    base = RBF(length, (1.0, 60.0)) if name == "RBF" else Matern(length, (1.0, 60.0), nu=nu)
    # y is normalized to variance 1 inside the GP, so signal and noise are on that scale
    return C(0.1, (1e-3, 10.0)) * base + WhiteKernel(0.8, (1e-3, 10.0))


def fit_gp(X, y, name="Matern", nu=1.5, seed=0, cap=TRAIN_CAP):
    X, y = np.asarray(X, float), np.asarray(y, float)
    if len(X) > cap:
        idx = np.random.default_rng(seed).choice(len(X), cap, replace=False)
        X, y = X[idx], y[idx]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GaussianProcessRegressor(kernel(name, nu=nu), normalize_y=True, random_state=seed).fit(X, y)


def predict_latent(gp, X):
    """Map prediction and uncertainty about the REGIONAL AVERAGE (excludes individual player noise)."""
    mu, sd = gp.predict(np.asarray(X, float), return_std=True)
    noise = gp.kernel_.k2.noise_level * float(np.ravel(gp._y_train_std)[0]) ** 2
    return mu, np.sqrt(np.clip(sd ** 2 - noise, 0, None))


def grid(df, nx=60, ny=35):
    gx, gy = np.meshgrid(np.linspace(df["lon"].min(), df["lon"].max(), nx),
                         np.linspace(df["lat"].min(), df["lat"].max(), ny))
    return gx, gy, np.c_[gx.ravel(), gy.ravel()]


def regional_table(df, size=5.0):
    """Coarse lat/lon boxes: how many players, their average production (with standard error),
    and players per unit of population density. Separates 'more players' from 'better players'."""
    d = df.copy()
    d["region"] = (np.floor(d["lat"] / size) * size).astype(int).astype(str) + "N, " + \
                  (np.floor(-d["lon"] / size) * size).astype(int).astype(str) + "W"
    agg = d.groupby("region").agg(players=("production", "size"), mean_production=("production", "mean"),
                                  sd=("production", "std"))
    agg["std_error"] = agg["sd"] / np.sqrt(agg["players"])
    if "local_density" in d.columns:
        agg["players_per_density"] = agg["players"] / d.groupby("region")["local_density"].sum()
    return agg.drop(columns="sd").sort_values("players", ascending=False)


def empirical_variogram(df, value="production", bins=12, max_dist=20.0, n=1500, seed=0):
    """Average half squared difference between pairs of players at each distance (Kriging's view)."""
    d = df.sample(min(n, len(df)), random_state=seed)
    P = d[["lon", "lat"]].to_numpy()
    v = d[value].to_numpy()
    i, j = np.triu_indices(len(d), 1)
    dist = np.hypot(*(P[i] - P[j]).T)
    gamma = 0.5 * (v[i] - v[j]) ** 2
    edges = np.linspace(0, max_dist, bins + 1)
    k = np.digitize(dist, edges) - 1
    ok = (k >= 0) & (k < bins)
    out = pd.DataFrame({"bin": k[ok], "gamma": gamma[ok]}).groupby("bin")["gamma"].agg(["mean", "size"])
    out["distance"] = (edges[:-1] + edges[1:])[out.index] / 2
    return out.rename(columns={"mean": "semivariance", "size": "pairs"}).reset_index(drop=True)


def _baseline_cols(df):
    return [c for c in ("recruit_rating", "local_density") if c in df.columns]


def evaluate_spatial(df, name="Matern", nu=1.5):
    """For each class C: baseline = ridge on rating + density (no location), fitted on classes < C;
    model = baseline + a GP on the baseline's leftover error, by hometown. Scored on class C."""
    if "class_year" not in df.columns:
        return {"ok": False, "reason": "Needs a class_year column for time-based validation."}
    cols = _baseline_cols(df)
    rows, leak_ok = [], True
    for Cy, train, test in V.rolling_origin(df, "class_year", min_train_periods=3):
        leak_ok &= V.no_leakage(train, test, "class_year")
        if cols:
            ridge = Ridge(1.0).fit(train[cols], train["production"])
            base_tr, base_te = ridge.predict(train[cols]), ridge.predict(test[cols])
        else:
            base_tr = np.full(len(train), train["production"].mean())
            base_te = np.full(len(test), train["production"].mean())
        gp = fit_gp(train[["lon", "lat"]], train["production"] - base_tr, name, nu, seed=int(Cy))
        mu, sd = gp.predict(test[["lon", "lat"]].to_numpy(), return_std=True)
        resid_sd = float(np.std(train["production"] - base_tr))
        y = test["production"].to_numpy()
        rows.append(pd.DataFrame({
            "class_year": Cy, "y": y, "baseline": base_te, "model": base_te + mu,
            "covered": np.abs(y - (base_te + mu)) <= Z80 * sd,
            "covered_base": np.abs(y - base_te) <= Z80 * resid_sd}))
    if not rows:
        return {"ok": False, "reason": "Need at least 4 recruiting classes."}
    r = pd.concat(rows, ignore_index=True)

    def mae(d, c):
        return float(np.mean(np.abs(d["y"] - d[c])))

    per = r.groupby("class_year").apply(lambda d: pd.Series({
        "players": len(d), "baseline": mae(d, "baseline"), "model": mae(d, "model"),
        "coverage": d["covered"].mean()}), include_groups=False).reset_index()
    m = {"baseline": mae(r, "baseline"), "model": mae(r, "model")}
    return {"ok": True, "rows": r, "per_class": per, "mae": m, "gain": 1 - m["model"] / m["baseline"],
            "wins": int((per["model"] < per["baseline"]).sum()), "n_classes": len(per),
            "coverage": float(r["covered"].mean()), "coverage_base": float(r["covered_base"].mean()),
            "leak_ok": leak_ok, "controls": cols}


def gates(res, data_info):
    if not res.get("ok"):
        return [("Evaluation ran", False, res.get("reason", ""))]
    is_real = data_info["source"] != "simulated" and data_info.get("real")
    return [
        ("Real data you have rights to use", bool(is_real),
         "Simulated data can never qualify." if data_info["source"] == "simulated"
         else ("Marked as real on upload." if is_real else "Upload is not marked as real data.")),
        ("Baseline defined", True, f"Ridge regression on {', '.join(res['controls']) or 'the overall mean'}, no location."),
        ("Metric predefined", True, PRIMARY_METRIC),
        ("Time-based validation (concept 26)",
         res["n_classes"] >= MIN_TEST_CLASSES and len(res["rows"]) >= MIN_TEST_ROWS,
         f"{res['n_classes']} test classes, {len(res['rows'])} players (need >= {MIN_TEST_CLASSES} and >= {MIN_TEST_ROWS})."),
        ("Location beats the baseline",
         res["gain"] >= MIN_GAIN and res["wins"] > res["n_classes"] / 2,
         f"{res['gain']:+.1%} MAE vs baseline; wins {res['wins']} of {res['n_classes']} classes "
         f"(need >= {MIN_GAIN:.0%} and a majority)."),
        ("No temporal leakage (concept 26)", bool(res["leak_ok"]), "Every fold trains on earlier recruiting classes only."),
        ("Controls for population density", "local_density" in res["controls"],
         "local_density is in the baseline, so a crowded region can't pass as a talented one."
         if "local_density" in res["controls"] else "Add a local_density (population) column."),
        ("Calibrated intervals (concept 25)", abs(res["coverage"] - INTERVAL) <= CAL_TOL,
         f"{INTERVAL:.0%} intervals covered {res['coverage']:.1%} of players (need {INTERVAL - CAL_TOL:.0%}–{INTERVAL + CAL_TOL:.0%})."),
    ]


def verdict(gate_list):
    return ("Meets Production Candidate criteria (review before any use)"
            if all(p for _, p, _ in gate_list) else "Learning Lab only")
