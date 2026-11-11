"""Level 6 engine: Bayesian optimization (Gaussian-process surrogate + expected improvement),
random search, and a tuning objective built from the Level 1 shrinkage model.
NumPy / SciPy / scikit-learn only.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel

from . import data, shrinkage as S


# ---------------------------------------------------------------- the 'expensive' function to tune
def tuning_objective(df=None):
    """Returns f(x) = next-season error (weighted MSE x 1e4) of Level 1 shrinkage, for two settings:
       x[0] = log2 of a multiplier on the fitted prior strength (0 = the empirical-Bayes fit),
       x[1] = weight on position-specific priors vs one shared prior (0..1).
    Folds are precomputed so each call is fast; in real use each call might take minutes."""
    df = data.make_pressure_by_position() if df is None else df
    folds = []
    for T in sorted(df["season"].unique())[1:]:
        train, test = df[df["season"] < T], df[df["season"] == T]
        hist = train.groupby("player_id")[["successes", "opportunities"]].sum()
        t = test[test["player_id"].isin(hist.index)]
        if t.empty:
            continue
        pos = S.fit_priors(train, "position")
        glob = S.fit_priors(train)["__all__"]
        ab_pos = np.array([pos[p] for p in t["position"]])
        folds.append({"k": t["player_id"].map(hist["successes"]).to_numpy(float),
                      "n": t["player_id"].map(hist["opportunities"]).to_numpy(float),
                      "m_pos": ab_pos[:, 0] / ab_pos.sum(1), "s_pos": ab_pos.sum(1),
                      "m_glob": glob[0] / sum(glob), "s_glob": sum(glob),
                      "actual": (t["successes"] / t["opportunities"]).to_numpy(), "w": t["opportunities"].to_numpy(float)})

    def f(x):
        mult, w = 2.0 ** float(x[0]), float(np.clip(x[1], 0, 1))
        err, tot = 0.0, 0.0
        for fd in folds:
            m = w * fd["m_pos"] + (1 - w) * fd["m_glob"]
            s = mult * (w * fd["s_pos"] + (1 - w) * fd["s_glob"])
            pred = (m * s + fd["k"]) / (s + fd["n"])
            err += np.sum(fd["w"] * (pred - fd["actual"]) ** 2)
            tot += fd["w"].sum()
        return float(err / tot * 1e4)
    return f


TUNING_BOUNDS = np.array([[-4.0, 4.0], [0.0, 1.0]])


# ---------------------------------------------------------------- Bayesian optimization
def surrogate(X, y, seed=0, restarts=2):
    k = C(1.0, (1e-3, 1e3)) * Matern(np.ones(np.shape(X)[1]), (1e-2, 1e2), nu=2.5) + WhiteKernel(1e-3, (1e-8, 1e-1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GaussianProcessRegressor(k, normalize_y=True, random_state=seed,
                                        n_restarts_optimizer=restarts).fit(X, y)


def expected_improvement(mu, sd, best, xi=0.01):
    """For MINIMIZING: how much we expect a point to beat the best so far, counting uncertainty."""
    sd = np.maximum(sd, 1e-9)
    z = (best - mu - xi) / sd
    return (best - mu - xi) * stats.norm.cdf(z) + sd * stats.norm.pdf(z)


def random_points(bounds, n, rng):
    b = np.asarray(bounds, float)
    return b[:, 0] + rng.random((n, len(b))) * (b[:, 1] - b[:, 0])


def bayes_opt(f, bounds, n_init=4, n_iter=11, seed=0, n_cand=2000):
    """Minimize f with a GP surrogate + expected improvement. Returns X, y in evaluation order."""
    rng = np.random.default_rng(seed)
    X = random_points(bounds, n_init, rng)
    y = np.array([f(x) for x in X])
    for _ in range(n_iter):
        gp = surrogate(X, y, seed)
        cand = random_points(bounds, n_cand, rng)
        mu, sd = gp.predict(cand, return_std=True)
        x_new = cand[np.argmax(expected_improvement(mu, sd, y.min()))]
        X = np.vstack([X, x_new])
        y = np.append(y, f(x_new))
    return X, y


def random_search(f, bounds, n=15, seed=0):
    X = random_points(bounds, n, np.random.default_rng(seed))
    return X, np.array([f(x) for x in X])


def grid_search(f, bounds, n=15):
    side = max(2, int(np.floor(np.sqrt(n))))
    g = np.array(np.meshgrid(*[np.linspace(lo, hi, side) for lo, hi in bounds])).reshape(len(bounds), -1).T
    return g, np.array([f(x) for x in g])


def contest(f, bounds, budget=15, seeds=20):
    """Best value found within the same budget, averaged over many random starts."""
    bo, rs = [], []
    for s in range(seeds):
        bo.append(np.minimum.accumulate(bayes_opt(f, bounds, 4, budget - 4, seed=s)[1]))
        rs.append(np.minimum.accumulate(random_search(f, bounds, budget, seed=s)[1]))
    g = grid_search(f, bounds, budget)[1].min()
    return np.array(bo), np.array(rs), float(g)
