"""Level 5 engine: a hierarchical Bayesian model for player rates, fitted with a NumPy Gibbs sampler
(MCMC), plus R-hat diagnostics and a season-by-season contest against 'same as last season' and the
Level 1 shrinkage model. No PyMC needed (PyMC is optional; see requirements-optional.txt).

Model, on the log-odds ('logit') scale, for player i in season s:
    y_is = position mean + experience effect + team effect + player effect + season wobble + sampling noise
Team and player effects are partially pooled: they share a common spread that is learned from data.
Rate-events schema: player_id, season, successes, opportunities (+ position, team).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import shrinkage as S
from . import validation as V

MIN_GAIN = 0.02
INTERVAL = 0.80
CAL_TOL = 0.05
MAX_RHAT = 1.05           # practical threshold for this teaching sampler (1.01 is the strict standard)
MIN_TEST_SEASONS = 3
MIN_TEST_ROWS = 200
PRIMARY_METRIC = ("Opportunity-weighted mean squared error of the predicted next-season rate "
                  "(same metric as the Level 1 project). Registered in code before any data is evaluated.")


def expit(z):
    return 1 / (1 + np.exp(-z))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Adds experience (earlier seasons in the data, capped at 3) and the empirical logit + its variance."""
    d = df.sort_values(["player_id", "season"]).copy()
    d["experience"] = d.groupby("player_id").cumcount().clip(upper=3)
    k, n = d["successes"].to_numpy(float), d["opportunities"].to_numpy(float)
    d["y"] = np.log((k + 0.5) / (n - k + 0.5))
    d["v"] = 1 / (k + 0.5) + 1 / (n - k + 0.5)
    if "position" not in d.columns:
        d["position"] = "all"
    if "team" not in d.columns:
        d["team"] = "all"
    return d.reset_index(drop=True)


def _design(d, positions):
    X = [(d["position"] == p).to_numpy(float) for p in positions]
    X += [(d["experience"] == e).to_numpy(float) for e in (1, 2, 3)]
    names = [f"position {p}" for p in positions] + ["experience 1", "experience 2", "experience 3+"]
    return np.column_stack(X), names


def gibbs(d: pd.DataFrame, iters: int = 700, burn: int = 250, chains: int = 3, seed: int = 0) -> dict:
    """MCMC for the hierarchical model. Gibbs steps for effects and spreads, a Metropolis step for the
    season-wobble variance. Returns posterior draws (after burn-in) for every chain."""
    positions = sorted(d["position"].unique())
    teams, t_idx = np.unique(d["team"], return_inverse=True)
    players, p_idx = np.unique(d["player_id"], return_inverse=True)
    X, names = _design(d, positions)
    y, v = d["y"].to_numpy(), d["v"].to_numpy()
    J, M, P = len(players), len(teams), X.shape[1]
    a0, b0 = 1.0, 0.05                                  # weak inverse-gamma prior on every spread
    out = {"beta": [], "sd_player": [], "sd_team": [], "sd_season": [], "u": [], "tau": []}
    for c in range(chains):
        rng = np.random.default_rng(seed + 101 * c)
        beta = rng.normal(np.mean(y), 0.5, P) * np.r_[np.ones(len(positions)), np.zeros(3)]
        u, tau = np.zeros(J), np.zeros(M)
        s2u, s2t, s2s = rng.uniform(0.02, 0.3, 3)
        draws = {k: [] for k in out}
        for it in range(iters):
            w = 1 / (v + s2s)
            # fixed effects (position means, experience steps)
            r = y - tau[t_idx] - u[p_idx]
            prec = (X * w[:, None]).T @ X + np.eye(P) / 100.0
            cov = np.linalg.inv(prec)
            beta = rng.multivariate_normal(cov @ (X.T @ (w * r)), cov)
            eta = X @ beta
            # player effects: partial pooling toward 0 with spread sd_player
            r = y - eta - tau[t_idx]
            pp = np.bincount(p_idx, w, J) + 1 / s2u
            u = rng.normal(np.bincount(p_idx, w * r, J) / pp, 1 / np.sqrt(pp))
            # team effects
            r = y - eta - u[p_idx]
            pt = np.bincount(t_idx, w, M) + 1 / s2t
            tau = rng.normal(np.bincount(t_idx, w * r, M) / pt, 1 / np.sqrt(pt)) if M > 1 else np.zeros(M)
            # re-centre: move the average player/team effect into the fixed effects. This leaves every
            # prediction unchanged but stops the position means and the effects drifting together,
            # which otherwise makes the chains mix slowly (a standard trick for hierarchical samplers).
            npos = len(positions)
            pos_of_player = np.bincount(p_idx, X[:, :npos].argmax(1), J) // np.maximum(np.bincount(p_idx, None, J), 1)
            for j in range(npos):
                m = pos_of_player == j
                if m.any():
                    shift = u[m].mean()
                    u[m] -= shift
                    beta[j] += shift
            if M > 1:
                shift = tau.mean()
                tau -= shift
                beta[:npos] += shift
            # spreads
            s2u = 1 / rng.gamma(a0 + J / 2, 1 / (b0 + (u ** 2).sum() / 2))
            s2t = 1 / rng.gamma(a0 + M / 2, 1 / (b0 + (tau ** 2).sum() / 2)) if M > 1 else 1e-6
            # season wobble: random-walk Metropolis on log variance
            res = y - eta - tau[t_idx] - u[p_idx]

            def logpost(s2):
                return (stats.norm.logpdf(res, 0, np.sqrt(v + s2)).sum()
                        + stats.invgamma.logpdf(s2, a0, scale=b0) + np.log(s2))

            prop = s2s * np.exp(rng.normal(0, 0.30))
            if np.log(rng.random()) < logpost(prop) - logpost(s2s):
                s2s = prop
            if it >= burn:
                for k, val in (("beta", beta), ("sd_player", np.sqrt(s2u)), ("sd_team", np.sqrt(s2t)),
                               ("sd_season", np.sqrt(s2s)), ("u", u.copy()), ("tau", tau.copy())):
                    draws[k].append(val)
        for k in out:
            out[k].append(np.array(draws[k]))
    res = {k: np.array(v_) for k, v_ in out.items()}          # shape: chains x draws x ...
    res.update({"names": names, "positions": positions, "teams": teams, "players": players})
    return res


def rhat(chains: np.ndarray) -> float:
    """Split R-hat for one scalar: ~1.00 when chains agree; above ~1.05 means 'not converged yet'."""
    c = np.asarray(chains)
    half = c.shape[1] // 2
    s = np.concatenate([c[:, :half], c[:, half:2 * half]])
    n = s.shape[1]
    B = n * s.mean(1).var(ddof=1)
    W = s.var(1, ddof=1).mean()
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def ess(chains: np.ndarray) -> float:
    """Effective sample size: how many independent draws the correlated chains are worth.
    Adds up autocorrelations until they first turn negative (the standard 'initial positive' rule),
    so random noise in the autocorrelations isn't counted as real stickiness."""
    x = np.asarray(chains).reshape(np.shape(chains)[0], -1)
    total = 0.0
    for ch in x:
        ch = ch - ch.mean()
        n = len(ch)
        ac = np.correlate(ch, ch, "full")[n - 1:] / (ch.var() * n)
        rho_sum = 0.0
        for t in range(1, n // 2):
            if ac[t] < 0:
                break
            rho_sum += ac[t]
        total += n / (1 + 2 * rho_sum)
    return float(total)


def diagnostics(fit: dict) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(fit["names"]):
        rows.append((name, fit["beta"][:, :, j]))
    for k in ("sd_player", "sd_team", "sd_season"):
        rows.append((k.replace("_", " "), fit[k]))
    return pd.DataFrame([{"parameter": n, "mean": float(c.mean()), "sd": float(c.std()),
                          "R-hat": rhat(c), "ESS": ess(c)} for n, c in rows])


def predict(fit: dict, rows: pd.DataFrame, rng=None):
    """Posterior-mean rate and 80% predictive interval of successes, for new player-seasons."""
    rng = rng or np.random.default_rng(0)
    X, _ = _design(rows, fit["positions"])
    beta = fit["beta"].reshape(-1, X.shape[1])
    tau = fit["tau"].reshape(beta.shape[0], -1)
    u = fit["u"].reshape(beta.shape[0], -1)
    s_p, s_s = fit["sd_player"].ravel(), fit["sd_season"].ravel()
    t_map = {t: i for i, t in enumerate(fit["teams"])}
    p_map = {p: i for i, p in enumerate(fit["players"])}
    ti = rows["team"].map(t_map).to_numpy()
    pi = rows["player_id"].map(p_map).to_numpy()
    eta = beta @ X.T                                                    # draws x rows
    eta += np.where(np.isnan(ti.astype(float)), 0, tau[:, np.nan_to_num(ti.astype(float)).astype(int)])
    new_player = np.isnan(pi.astype(float))
    eta += np.where(new_player, rng.normal(0, 1, eta.shape) * s_p[:, None],
                    u[:, np.nan_to_num(pi.astype(float)).astype(int)])
    eta_next = eta + rng.normal(0, 1, eta.shape) * s_s[:, None]
    p = expit(eta_next)
    k = rng.binomial(rows["opportunities"].to_numpy()[None, :], p)
    lo, hi = np.quantile(k, [(1 - INTERVAL) / 2, (1 + INTERVAL) / 2], axis=0)
    return p.mean(0), lo, hi


def evaluate(df: pd.DataFrame, iters: int = 900, burn: int = 300, chains: int = 3, last_folds: int = 6) -> dict:
    """Season by season: fit everything on seasons < T only, predict players' season-T rates.
    Compares 'same as last season', Level 1 position shrinkage, and the hierarchical model."""
    d = prepare(df)
    folds = list(V.rolling_origin(d, "season", min_train_periods=2))[-last_folds:]
    rows, leak_ok, worst_rhat = [], True, 1.0
    for T, train, test in folds:
        leak_ok &= V.no_leakage(train, test)
        hist = train.groupby("player_id")
        last = hist.apply(lambda g: g.sort_values("season").iloc[-1], include_groups=False)
        test = test[test["player_id"].isin(last.index)].copy()
        if test.empty:
            continue
        test["same_as_last"] = test["player_id"].map(last["successes"] / last["opportunities"]).to_numpy()
        priors = S.fit_priors(train, "position")
        sums = hist[["successes", "opportunities"]].sum()
        ab = np.array([S.prior_for(priors, p) for p in test["position"]])
        k_h = test["player_id"].map(sums["successes"]).to_numpy()
        n_h = test["player_id"].map(sums["opportunities"]).to_numpy()
        test["level1"] = (ab[:, 0] + k_h) / (ab.sum(1) + n_h)
        fit = gibbs(train, iters, burn, chains, seed=int(T))
        diag = diagnostics(fit)
        worst_rhat = max(worst_rhat, float(diag["R-hat"].max()))
        mean, lo, hi = predict(fit, test, np.random.default_rng(int(T)))
        test["hierarchical"] = mean
        test["covered"] = (test["successes"] >= lo) & (test["successes"] <= hi)
        test["actual"] = test["successes"] / test["opportunities"]
        rows.append(test[["season", "player_id", "position", "opportunities", "actual",
                          "same_as_last", "level1", "hierarchical", "covered"]])
    if not rows:
        return {"ok": False, "reason": "Need players who appear in at least two seasons, and 3+ seasons overall."}
    r = pd.concat(rows, ignore_index=True)

    def wmse(g, c):
        return float(np.average((g[c] - g["actual"]) ** 2, weights=g["opportunities"]))

    methods = ("same_as_last", "level1", "hierarchical")
    per = r.groupby("season").apply(lambda g: pd.Series({"test_rows": len(g), **{m: wmse(g, m) for m in methods}}),
                                    include_groups=False).reset_index()
    mse = {m: wmse(r, m) for m in methods}
    best_base = min(mse["same_as_last"], mse["level1"])
    return {"ok": True, "rows": r, "per_season": per, "mse": mse, "gain": 1 - mse["hierarchical"] / best_base,
            "gain_vs_level1": 1 - mse["hierarchical"] / mse["level1"],
            "wins": int((per["hierarchical"] < per[["same_as_last", "level1"]].min(axis=1)).sum()),
            "n_seasons": len(per), "coverage": float(r["covered"].mean()), "leak_ok": leak_ok,
            "worst_rhat": worst_rhat}


def gates(res: dict, data_info: dict) -> list:
    if not res.get("ok"):
        return [("Evaluation ran", False, res.get("reason", ""))]
    is_real = data_info["source"] != "simulated" and data_info.get("real")
    return [
        ("Real data you have rights to use", bool(is_real),
         "Simulated data can never qualify." if data_info["source"] == "simulated"
         else ("Marked as real on upload." if is_real else "Upload is not marked as real data.")),
        ("Baselines defined", True, "'Same as last season' and the Level 1 position-shrinkage model."),
        ("Metric predefined", True, PRIMARY_METRIC),
        ("Time-based validation (concept 26)",
         res["n_seasons"] >= MIN_TEST_SEASONS and len(res["rows"]) >= MIN_TEST_ROWS,
         f"{res['n_seasons']} test seasons, {len(res['rows'])} player-seasons "
         f"(need >= {MIN_TEST_SEASONS} and >= {MIN_TEST_ROWS})."),
        ("Beats BOTH baselines",
         res["gain"] >= MIN_GAIN and res["wins"] > res["n_seasons"] / 2,
         f"{res['gain']:+.1%} vs the better baseline ({res['gain_vs_level1']:+.1%} vs Level 1); wins "
         f"{res['wins']} of {res['n_seasons']} seasons (need >= {MIN_GAIN:.0%} and a majority)."),
        ("No temporal leakage (concept 26)", bool(res["leak_ok"]),
         "Every model, prior and MCMC fit uses seasons before the test season only."),
        ("MCMC converged (concept 22)", res["worst_rhat"] <= MAX_RHAT,
         f"Worst R-hat across folds and parameters {res['worst_rhat']:.3f} (need <= {MAX_RHAT})."),
        ("Calibrated intervals (concept 25)", abs(res["coverage"] - INTERVAL) <= CAL_TOL,
         f"{INTERVAL:.0%} predictive intervals covered {res['coverage']:.1%} "
         f"(need {INTERVAL - CAL_TOL:.0%}–{INTERVAL + CAL_TOL:.0%})."),
    ]


def verdict(gate_list):
    return ("Meets Production Candidate criteria (review before any use)"
            if all(p for _, p, _ in gate_list) else "Learning Lab only")
