"""Level 6 lineup engine: the $40M defensive-line build.
Choose 6 players (at least 2 EDGE and 2 DT) with total cap <= $40M to maximize
score = sum(projected_value - risk * value_sd). Compares an exact solver (MILP), a greedy pick,
random search and Bayesian optimization under a budget, plus a season-by-season back-test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .optimize import expected_improvement, surrogate

CAP = 40.0
ROSTER = 6
MIN_POS = {"EDGE": 2, "DT": 2}
MIN_GAIN = 0.02
MIN_SEASONS = 3
MAX_CURSE = 0.10          # chosen players must realize >= 90% of their projected value on average
PRIMARY_METRIC = ("Total REALIZED value of the chosen 6-man group, per season, choosing only from "
                  "pre-season projections. Registered in code before any data is evaluated.")


def scores(pool, risk=0.0):
    return (pool["projected_value"] - risk * pool.get("value_sd", 0)).to_numpy(float)


def feasible(pool, x):
    x = np.asarray(x, bool)
    if x.sum() != ROSTER or pool["cap_hit_m"].to_numpy()[x].sum() > CAP + 1e-9:
        return False
    pos = pool["position"].to_numpy()[x]
    return all((pos == p).sum() >= k for p, k in MIN_POS.items())


def total(pool, x, risk=0.0):
    return float(scores(pool, risk)[np.asarray(x, bool)].sum())


def exact(pool, risk=0.0):
    """Mixed-integer linear program: the true best lineup (fast, because the score is simple)."""
    s = scores(pool, risk)
    n = len(pool)
    A = [pool["cap_hit_m"].to_numpy(float), np.ones(n)]
    lo, hi = [-np.inf, ROSTER], [CAP, ROSTER]
    for p, k in MIN_POS.items():
        A.append((pool["position"] == p).to_numpy(float))
        lo.append(k)
        hi.append(np.inf)
    res = milp(-s, integrality=np.ones(n), bounds=Bounds(0, 1), constraints=LinearConstraint(np.array(A), lo, hi))
    if not res.success:
        return None
    return np.round(res.x).astype(bool)


def greedy(pool, risk=0.0):
    """Best value per dollar first: fill the required EDGE/DT spots, then the rest, skipping anything over the cap."""
    s = scores(pool, risk)
    ratio = s / pool["cap_hit_m"].to_numpy()
    order = np.argsort(-ratio)
    x = np.zeros(len(pool), bool)
    pos = pool["position"].to_numpy()
    caps = pool["cap_hit_m"].to_numpy()
    # cheapest possible way to finish the roster, so greedy never paints itself into a corner
    def can_finish(x):
        need = {p: max(0, k - (pos[x] == p).sum()) for p, k in MIN_POS.items()}
        left = ROSTER - x.sum()
        if sum(need.values()) > left:
            return False
        cost, used = 0.0, x.copy()
        for p, k in need.items():
            idx = [i for i in np.argsort(caps) if pos[i] == p and not used[i]][:k]
            if len(idx) < k:
                return False
            cost += caps[idx].sum()
            used[idx] = True
        rest = [i for i in np.argsort(caps) if not used[i]][:left - sum(need.values())]
        return caps[x].sum() + cost + caps[rest].sum() <= CAP + 1e-9
    for i in order:
        if x.sum() == ROSTER:
            break
        x[i] = True
        if not can_finish(x):
            x[i] = False
    return x if feasible(pool, x) else None


def random_lineup(pool, rng, tries=500):
    pos = pool["position"].to_numpy()
    n = len(pool)
    for _ in range(tries):
        pick = list(rng.choice(np.flatnonzero(pos == "EDGE"), 2, replace=False)) + \
               list(rng.choice(np.flatnonzero(pos == "DT"), 2, replace=False))
        rest = [i for i in rng.permutation(n) if i not in pick][:ROSTER - 4]
        x = np.zeros(n, bool)
        x[pick + rest] = True
        if feasible(pool, x):
            return x
    return None


def random_search(pool, budget, seed=0, risk=0.0):
    rng = np.random.default_rng(seed)
    vals = [total(pool, x, risk) for x in (random_lineup(pool, rng) for _ in range(budget)) if x is not None]
    return np.maximum.accumulate(vals)


def _neighbors(pool, x, rng, k=40):
    out = []
    on, off = np.flatnonzero(x), np.flatnonzero(~x)
    for _ in range(k * 3):
        y = x.copy()
        y[rng.choice(on)] = False
        y[rng.choice(off)] = True
        if feasible(pool, y):
            out.append(y)
        if len(out) >= k:
            break
    return out


def bayes_opt(pool, budget, seed=0, risk=0.0, n_init=10):
    """BO over lineups: a GP learns score from the 0/1 'who is in' vector; expected improvement picks
    the next lineup to try from random lineups and one-swap neighbours of the best ones so far."""
    rng = np.random.default_rng(seed)
    X = [x for x in (random_lineup(pool, rng) for _ in range(n_init)) if x is not None]
    y = [total(pool, x, risk) for x in X]
    while len(y) < budget:
        gp = surrogate(np.array(X, float), -np.array(y), seed, restarts=0)  # minimize negative score
        top = np.argsort(y)[-3:]
        cand = [random_lineup(pool, rng) for _ in range(60)]
        for t in top:
            cand += _neighbors(pool, X[t], rng, k=25)
        seen = {tuple(x) for x in X}
        cand = [c for c in cand if c is not None and tuple(c) not in seen]
        if not cand:
            break
        mu, sd = gp.predict(np.array(cand, float), return_std=True)
        x_new = cand[int(np.argmax(expected_improvement(mu, sd, -max(y))))]
        X.append(x_new)
        y.append(total(pool, x_new, risk))
    return np.maximum.accumulate(y)


def contest(pool, budget=40, seeds=5, risk=0.0):
    best = total(pool, exact(pool, risk), risk)
    g = greedy(pool, risk)
    bo = np.array([bayes_opt(pool, budget, s, risk) for s in range(seeds)])
    rs = np.array([random_search(pool, budget, s, risk) for s in range(seeds)])
    return {"optimum": best, "greedy": total(pool, g, risk) if g is not None else np.nan,
            "bo": bo, "random": rs}


# ---------------------------------------------------------------- back-test (Production Candidate)
def backtest(df, risk=0.0, n_random=200):
    """Each season: choose using PROJECTIONS only, score with REALIZED value."""
    if not {"season", "realized_value"} <= set(df.columns):
        return {"ok": False, "reason": "Needs season and realized_value columns for a back-test."}
    rows, leak_ok, chosen = [], True, []
    for s, pool in df.groupby("season"):
        pool = pool.reset_index(drop=True)
        x_opt, x_gr = exact(pool, risk), greedy(pool, risk)
        if x_opt is None or x_gr is None:
            continue
        shuffled = pool.assign(realized_value=np.random.default_rng(int(s)).permutation(pool["realized_value"].to_numpy()))
        leak_ok &= bool((exact(shuffled, risk) == x_opt).all())          # choice must not depend on outcomes
        rng = np.random.default_rng(int(s))
        rnd = [pool["realized_value"].to_numpy()[x].sum() for x in (random_lineup(pool, rng) for _ in range(n_random)) if x is not None]
        real = pool["realized_value"].to_numpy()
        rows.append({"season": s, "optimizer": real[x_opt].sum(), "greedy": real[x_gr].sum(),
                     "random_median": float(np.median(rnd)),
                     "optimizer_projected": pool["projected_value"].to_numpy()[x_opt].sum()})
        chosen.append(pool[x_opt])
    if not rows:
        return {"ok": False, "reason": "No season had a feasible lineup."}
    per = pd.DataFrame(rows)
    ch = pd.concat(chosen)
    return {"ok": True, "per_season": per, "n_seasons": len(per),
            "gain": float(per["optimizer"].sum() / per["greedy"].sum() - 1),
            "wins": int((per["optimizer"] > per["greedy"]).sum()),
            "realized_ratio": float(ch["realized_value"].sum() / ch["projected_value"].sum()),
            "leak_ok": leak_ok}


def gates(res, data_info):
    if not res.get("ok"):
        return [("Evaluation ran", False, res.get("reason", ""))]
    is_real = data_info["source"] != "simulated" and data_info.get("real")
    return [
        ("Real data you have rights to use", bool(is_real),
         "Simulated data can never qualify." if data_info["source"] == "simulated"
         else ("Marked as real on upload." if is_real else "Upload is not marked as real data.")),
        ("Baseline defined", True, "Greedy value-per-dollar pick (and the median random legal lineup)."),
        ("Metric predefined", True, PRIMARY_METRIC),
        ("Time-based back-test (concept 26)", res["n_seasons"] >= MIN_SEASONS,
         f"{res['n_seasons']} seasons, each chosen from pre-season projections (need >= {MIN_SEASONS})."),
        ("Optimizer beats greedy on REALIZED value", res["gain"] >= MIN_GAIN and res["wins"] > res["n_seasons"] / 2,
         f"{res['gain']:+.1%} realized value vs greedy; better in {res['wins']} of {res['n_seasons']} seasons "
         f"(need >= {MIN_GAIN:.0%} and a majority)."),
        ("No leakage", bool(res["leak_ok"]), "Shuffling realized values never changes the chosen lineup."),
        ("Projections honest for chosen players (optimizer's curse)", res["realized_ratio"] >= 1 - MAX_CURSE,
         f"Chosen players realized {res['realized_ratio']:.0%} of their projected value (need >= {1 - MAX_CURSE:.0%})."),
        ("Input value model passed its own gates", False,
         "Not automated: the projections come from another model, which must pass its own Production Candidate checks."),
    ]


def verdict(gate_list):
    return ("Meets Production Candidate criteria (review before any use)"
            if all(p for _, p, _ in gate_list) else "Learning Lab only")
