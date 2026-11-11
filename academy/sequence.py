"""Level 4 engine: Gaussian hidden Markov models (forward filter, Viterbi, Baum-Welch), simple
sequence forecasts, split-conformal intervals, and a leakage-safe game-by-game evaluation.
NumPy/SciPy only. PyTorch is optional and only used by `torch_lstm_available()` callers.

Game-log schema: player_id, season, week, grade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import validation as V

MIN_GAIN = 0.02
INTERVAL = 0.80
CAL_TOL = 0.05
MIN_TEST_SEASONS = 3
MIN_TEST_ROWS = 200
MIN_GAMES_PER_SEASON = 8           # median observed games per player-season needed for a fair test
MIN_HISTORY = 2                    # forecast only once a player has 2+ games this season
PRIMARY_METRIC = ("Mean absolute error of the next game's grade, walk-forward within each test season "
                  "(only earlier games used). Registered in code before any data is evaluated.")


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


TORCH_NOTE = ("PyTorch isn't installed, so the real-LSTM lab is skipped. Everything else runs without it. "
              "To enable it locally: `pip install -r requirements-optional.txt` "
              "(CPU-only, smaller: `pip install torch --index-url https://download.pytorch.org/whl/cpu`).")


# ---------------------------------------------------------------- sequences
def sequences(df, value="grade"):
    """List of arrays, one per player-season, in week order (missing weeks simply skipped)."""
    return [g.sort_values("week")[value].to_numpy(float)
            for _, g in df.groupby(["player_id", "season"]) if len(g)]


def running_mean_residuals(df):
    """grade minus the player's season-to-date mean of EARLIER games (needs MIN_HISTORY games)."""
    out = []
    for (pid, s), g in df.sort_values("week").groupby(["player_id", "season"]):
        y = g["grade"].to_numpy(float)
        prior = np.r_[np.nan, np.cumsum(y)[:-1] / np.arange(1, len(y))]
        r = y - prior
        r[:MIN_HISTORY] = np.nan
        out.append(pd.DataFrame({"player_id": pid, "season": s, "week": g["week"].to_numpy(),
                                 "grade": y, "s2d_mean": prior, "resid": r}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------- HMM
def _emis(x, means, sds):
    return stats.norm.pdf(np.asarray(x)[:, None], means[None, :], sds[None, :]) + 1e-300


def forward(x, pi, A, means, sds):
    """Filtered P(state_t | games 1..t) for each game, plus the log-likelihood."""
    B = _emis(x, means, sds)
    alpha = np.zeros((len(x), len(pi)))
    c = np.zeros(len(x))
    a = pi * B[0]
    for t in range(len(x)):
        if t:
            a = (alpha[t - 1] @ A) * B[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, float(np.log(c).sum())


def viterbi(x, pi, A, means, sds):
    """Single most likely hidden-state path."""
    lB = np.log(_emis(x, means, sds))
    lA = np.log(A)
    d = np.log(pi) + lB[0]
    back = np.zeros((len(x), len(pi)), int)
    for t in range(1, len(x)):
        m = d[:, None] + lA
        back[t] = m.argmax(0)
        d = m.max(0) + lB[t]
    path = [int(d.argmax())]
    for t in range(len(x) - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return np.array(path[::-1])


def fit_hmm(seqs, K=2, iters=60, seed=0, tol=1e-5):
    """Baum-Welch (an EM algorithm, Level 5) for Gaussian emissions shared across sequences.
    States are sorted so state 0 has the HIGHEST mean ('hot')."""
    seqs = [np.asarray(s, float) for s in seqs if len(s) >= 2]
    allx = np.concatenate(seqs)
    means = np.quantile(allx, np.linspace(0.75, 0.25, K)) if K > 1 else np.array([allx.mean()])
    sds = np.full(K, allx.std())
    pi = np.full(K, 1 / K)
    A = np.full((K, K), 0.2 / max(K - 1, 1)) + np.eye(K) * (0.8 - 0.2 / max(K - 1, 1)) if K > 1 else np.ones((1, 1))
    prev = -np.inf
    for _ in range(iters):
        g0, xi_sum, gam_sum, gx, gxx, ll = np.zeros(K), np.zeros((K, K)), np.zeros(K), np.zeros(K), np.zeros(K), 0.0
        for x in seqs:
            B = _emis(x, means, sds)
            T = len(x)
            alpha, c = np.zeros((T, K)), np.zeros(T)
            a = pi * B[0]
            for t in range(T):
                if t:
                    a = (alpha[t - 1] @ A) * B[t]
                c[t] = a.sum()
                alpha[t] = a / c[t]
            beta = np.ones((T, K))
            for t in range(T - 2, -1, -1):
                beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(1, keepdims=True)
            for t in range(1, T):
                xi = alpha[t - 1][:, None] * A * (B[t] * beta[t])[None, :] / c[t]
                xi_sum += xi
            g0 += gamma[0]
            gam_sum += gamma.sum(0)
            gx += gamma.T @ x
            gxx += gamma.T @ x ** 2
            ll += np.log(c).sum()
        pi = g0 / g0.sum()
        if K > 1:
            A = xi_sum / xi_sum.sum(1, keepdims=True)
        means = gx / gam_sum
        sds = np.sqrt(np.maximum(gxx / gam_sum - means ** 2, 1.0))
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    order = np.argsort(-means)
    return {"pi": pi[order], "A": A[np.ix_(order, order)], "means": means[order], "sds": sds[order],
            "loglik": float(ll), "n_obs": int(sum(len(s) for s in seqs)), "K": K}


def hmm_loglik(seqs, m):
    return float(sum(forward(np.asarray(s, float), m["pi"], m["A"], m["means"], m["sds"])[1]
                     for s in seqs if len(s)))


def bic(m):
    K = m["K"]
    n_params = (K - 1) + K * (K - 1) + 2 * K
    return -2 * m["loglik"] + n_params * np.log(m["n_obs"])


def hmm_next_offset(resid_so_far, m):
    """Expected next residual: P(state next game | games so far) x each state's mean."""
    x = np.asarray(resid_so_far, float)
    if len(x) == 0:
        p = m["pi"]
    else:
        alpha, _ = forward(x, m["pi"], m["A"], m["means"], m["sds"])
        p = alpha[-1] @ m["A"]
    return float(p @ m["means"])


# ---------------------------------------------------------------- forecasting + conformal
def ewma_forecast(y_prior, alpha):
    w = (1 - alpha) ** np.arange(len(y_prior))[::-1]
    return float((w * y_prior).sum() / w.sum())


def conformal_halfwidth(abs_resid, level=INTERVAL):
    """Split conformal: the ceil((n+1)*level)/n quantile of calibration |errors|."""
    r = np.sort(np.asarray(abs_resid, float))
    n = len(r)
    if n == 0:
        return np.nan
    k = int(np.ceil((n + 1) * level)) - 1
    return float(r[min(k, n - 1)])


def _predict_season(test, alpha, hmm):
    rows = []
    for (pid, s), g in test.sort_values("week").groupby(["player_id", "season"]):
        y = g["grade"].to_numpy(float)
        prior_mean = np.r_[np.nan, np.cumsum(y)[:-1] / np.arange(1, len(y))]
        resid = y - prior_mean
        for t in range(MIN_HISTORY, len(y)):
            past = y[:t]
            r_hist = resid[MIN_HISTORY:t]
            rows.append({"player_id": pid, "season": s, "week": int(g["week"].iloc[t]), "y": y[t],
                         "season_mean": past.mean(), "last_game": past[-1],
                         "ewma": ewma_forecast(past, alpha),
                         "hmm": past.mean() + hmm_next_offset(r_hist, hmm)})
    return pd.DataFrame(rows)


def evaluate_games(df, alphas=(0.1, 0.2, 0.3, 0.5)):
    """For each test season S: choose the EWMA weight and fit the HMM on seasons < S only,
    then forecast every game of season S from earlier games. Conformal intervals are
    calibrated on the previous season's errors and checked on season S."""
    if not {"player_id", "season", "week", "grade"} <= set(df.columns):
        return {"ok": False, "reason": "Needs player_id, season, week and grade columns."}
    per_ps = df.groupby(["player_id", "season"]).size()
    median_games = float(per_ps.median())
    out, leak_ok = [], True
    for S, train, test in V.rolling_origin(df, "season", min_train_periods=2):
        leak_ok &= V.no_leakage(train, test)
        res_tr = running_mean_residuals(train).dropna(subset=["resid"])
        if res_tr.empty:
            continue
        hmm = fit_hmm([g["resid"].to_numpy() for _, g in res_tr.groupby(["player_id", "season"]) if len(g) >= 2])
        last = train[train["season"] == train["season"].max()]
        best_alpha = min(alphas, key=lambda a: np.mean(np.abs(
            (p := _predict_season(last, a, hmm))["y"] - p["ewma"])) if len(last) else 0)
        calib = _predict_season(last, best_alpha, hmm)
        pred = _predict_season(test, best_alpha, hmm)
        if pred.empty:
            continue
        for m in ("season_mean", "ewma", "hmm"):
            pred[f"hw_{m}"] = conformal_halfwidth(np.abs(calib["y"] - calib[m])) if len(calib) else np.nan
        pred["alpha"] = best_alpha
        out.append(pred)
    if not out:
        return {"ok": False, "reason": "Need at least 3 seasons of game logs."}
    r = pd.concat(out, ignore_index=True)
    methods = ("season_mean", "last_game", "ewma", "hmm")
    mae = {m: float(np.mean(np.abs(r["y"] - r[m]))) for m in methods}
    best = min(("ewma", "hmm"), key=mae.get)
    per = r.groupby("season").apply(lambda d: pd.Series(
        {"forecasts": len(d), **{m: np.mean(np.abs(d["y"] - d[m])) for m in methods}}),
        include_groups=False).reset_index()
    cov = {m: float(np.mean(np.abs(r["y"] - r[m]) <= r[f"hw_{m}"])) for m in ("season_mean", "ewma", "hmm")}
    return {"ok": True, "rows": r, "per_season": per, "mae": mae, "best": best,
            "gain": 1 - mae[best] / mae["season_mean"],
            "wins": int((per[best] < per["season_mean"]).sum()), "n_seasons": len(per),
            "coverage": cov, "halfwidth": float(r[f"hw_{best}"].mean()),
            "leak_ok": leak_ok, "median_games": median_games, "seasons_total": int(df["season"].nunique())}


def gates(res, data_info):
    if not res.get("ok"):
        return [("Evaluation ran", False, res.get("reason", ""))]
    is_real = data_info["source"] != "simulated" and data_info.get("real")
    b = res["best"]
    return [
        ("Real data you have rights to use", bool(is_real),
         "Simulated data can never qualify." if data_info["source"] == "simulated"
         else ("Marked as real on upload." if is_real else "Upload is not marked as real data.")),
        ("Baseline defined", True, "Season-to-date average of the player's earlier games."),
        ("Metric predefined", True, PRIMARY_METRIC),
        ("Sample sufficiency", res["median_games"] >= MIN_GAMES_PER_SEASON and res["seasons_total"] >= 4,
         f"Median {res['median_games']:.0f} games per player-season, {res['seasons_total']} seasons "
         f"(need >= {MIN_GAMES_PER_SEASON} and >= 4). Below this, results are learning exercises only."),
        ("Time-based validation (concept 26)",
         res["n_seasons"] >= MIN_TEST_SEASONS and len(res["rows"]) >= MIN_TEST_ROWS,
         f"Walk-forward within season, models fitted on earlier seasons. {res['n_seasons']} test seasons, "
         f"{len(res['rows'])} forecasts (need >= {MIN_TEST_SEASONS} and >= {MIN_TEST_ROWS})."),
        ("Beats the baseline",
         res["gain"] >= MIN_GAIN and res["wins"] > res["n_seasons"] / 2,
         f"Best method ({b}) {res['gain']:+.1%} MAE vs season-to-date average; wins {res['wins']} of "
         f"{res['n_seasons']} seasons (need >= {MIN_GAIN:.0%} and a majority)."),
        ("No temporal leakage (concept 26)", bool(res["leak_ok"]),
         "EWMA weight, HMM and conformal calibration use earlier seasons only; each forecast uses earlier games only."),
        ("Calibrated intervals (concept 28)", abs(res["coverage"][b] - INTERVAL) <= CAL_TOL,
         f"Conformal {INTERVAL:.0%} intervals (±{res['halfwidth']:.1f}) covered {res['coverage'][b]:.1%} "
         f"(need {INTERVAL - CAL_TOL:.0%}–{INTERVAL + CAL_TOL:.0%})."),
    ]


def verdict(gate_list):
    return ("Meets Production Candidate criteria (review before any use)"
            if all(p for _, p, _ in gate_list) else "Learning Lab only")


# ---------------------------------------------------------------- windows + neural forecasters
def windows(df, past=6, value="grade"):
    """(X, y) pairs: the previous `past` games -> the next game, within each player-season."""
    X, y = [], []
    for _, g in df.groupby(["player_id", "season"]):
        v = g.sort_values("week")[value].to_numpy(float)
        for t in range(past, len(v)):
            X.append(v[t - past:t])
            y.append(v[t])
    return np.array(X).reshape(-1, past), np.array(y)


def windowed_mlp(Xtr, ytr, Xte, seed=0):
    """NumPy/scikit-learn fallback: a small neural net on a fixed 6-game window. NOT an LSTM."""
    from sklearn.neural_network import MLPRegressor
    mu, sd = Xtr.mean(), Xtr.std()
    m = MLPRegressor(hidden_layer_sizes=(16,), alpha=1e-2, max_iter=2000, early_stopping=True, random_state=seed)
    m.fit((Xtr - mu) / sd, (ytr - mu) / sd)
    return m.predict((Xte - mu) / sd) * sd + mu


def torch_lstm(Xtr, ytr, Xte, hidden=16, epochs=60, seed=0):
    """A real LSTM (only when PyTorch is installed). Inputs are standardized game windows."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    mu, sd = float(Xtr.mean()), float(Xtr.std())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
    X = torch.tensor((Xtr - mu) / sd, dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor((ytr - mu) / sd, dtype=torch.float32)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for k in range(0, len(X), 64):
            idx = perm[k:k + 64]
            opt.zero_grad()
            loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        Xt = torch.tensor((Xte - mu) / sd, dtype=torch.float32).unsqueeze(-1)
        return net(Xt).numpy() * sd + mu
