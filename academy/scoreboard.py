"""Integrated scoreboard: run every level's automated test on the current data (your uploads where
present, simulated otherwise) and say plainly which methods beat their simple baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import (archetypes as A, data, datasets, hierarchical as H, lineup as L, optimize as O,
               sequence as Q, shrinkage as S, spatial as SP, survival as SV)


def _beats(gl, key):
    return next((ok for n, ok, _ in gl if n.startswith(key)), False)


def _others_ok(gl):
    """All gates except 'real data' and gates that can't be automated."""
    return all(ok for n, ok, _ in gl if not n.startswith("Real data") and not n.startswith("Input value model"))


STUDIES = [
    # (id, level, project, method, baseline, runner)
    ("p1", 1, "P1 Bayesian Player Evaluation", "Position shrinkage (empirical Bayes)", "Raw past rate / position mean"),
    ("p2", 2, "P2 Archetype Discovery", "K-means archetypes as a feature", "This season's production + position"),
    ("p3a", 3, "P3A Time to Contribution", "Cox proportional hazards", "Kaplan-Meier by rating tier"),
    ("p3b", 3, "P3B Geographic Recruiting", "Kriging (GP) on hometown", "Rating + population density"),
    ("p4", 4, "P4 Game-by-Game", "Best of EWMA / HMM", "Season-to-date average"),
    ("p5", 5, "P5 Hierarchical (with team/exp effects)", "Hierarchical Bayes (MCMC)", "Best of Level 1 / same as last season"),
    ("p5b", 5, "P5 Hierarchical (no team/exp effects)", "Hierarchical Bayes (MCMC)", "Best of Level 1 / same as last season"),
    ("p6a", 6, "P6A $40M D-line build", "Exact lineup optimizer", "Greedy value-per-dollar"),
    ("p6t", 6, "L6 tuning", "Bayesian optimization (10 tries)", "Random search (10 tries)"),
]


def run_one(sid):
    """Returns dict: gain (vs baseline), beats (bool), others_ok (bool), source, gates."""
    if sid == "p1":
        df, info = datasets.load("rate_events", "Pressure rate by position")
        res = S.evaluate_next_season(df, "position" if "position" in df.columns else None)
        gl = S.gates(res, info)
        return {"gain": res.get("gain", np.nan), "beats": _beats(gl, "Beats the best baseline"), "gates": gl, "info": info}
    if sid == "p2":
        df, info = datasets.load("player_profiles")
        feats = A.feature_candidates(df)
        grp = "DL" if "position_group" in df.columns else None
        res = A.evaluate_downstream(df, feats, 3, grp, 100)
        gl = A.gates(res, info)
        return {"gain": res.get("gain", np.nan), "beats": _beats(gl, "Archetypes beat"), "gates": gl, "info": info}
    if sid == "p3a":
        df, info = datasets.load("roster_careers")
        res = SV.evaluate_contribution(df)
        gl = SV.gates(res, info)
        return {"gain": res.get("gain", np.nan), "beats": _beats(gl, "Cox beats"), "gates": gl, "info": info}
    if sid == "p3b":
        df, info = datasets.load("hometowns")
        res = SP.evaluate_spatial(df)
        gl = SP.gates(res, info)
        return {"gain": res.get("gain", np.nan), "beats": _beats(gl, "Location beats"), "gates": gl, "info": info}
    if sid == "p4":
        df, info = datasets.load("game_logs")
        res = Q.evaluate_games(df)
        gl = Q.gates(res, info)
        return {"gain": res.get("gain", np.nan), "beats": _beats(gl, "Beats the baseline"), "gates": gl, "info": info}
    if sid in ("p5", "p5b"):
        variant = "Pressure rate with team & experience effects" if sid == "p5" else "Pressure rate by position"
        df, info = datasets.load("rate_events", variant)
        if sid == "p5b" and info["source"] != "simulated":
            return None                                   # with real data there is only one P5 row
        res = H.evaluate(df)
        gl = H.gates(res, info)
        return {"gain": res.get("gain", np.nan), "beats": _beats(gl, "Beats BOTH"), "gates": gl, "info": info}
    if sid == "p6a":
        df, info = datasets.load("lineup_candidates")
        res = L.backtest(df)
        gl = L.gates(res, info)
        return {"gain": res.get("gain", np.nan), "beats": _beats(gl, "Optimizer beats"), "gates": gl, "info": info}
    if sid == "p6t":
        bo, rs, _ = O.contest(O.tuning_objective(), O.TUNING_BOUNDS, budget=10, seeds=5)
        gain = 1 - bo[:, -1].mean() / rs[:, -1].mean()
        info = {"source": "simulated", "label": "Level 1 tuning objective", "real": False}
        gl = [("Beats random search with the same budget", gain >= 0.02, f"{gain:+.1%} lower error after 10 tries.")]
        return {"gain": gain, "beats": gain >= 0.02, "gates": gl, "info": info}
    raise KeyError(sid)


def run_all(ids=None):
    rows = []
    for sid, lvl, proj, method, base in STUDIES:
        if ids and sid not in ids:
            continue
        r = run_one(sid)
        if r is None:
            continue
        rows.append({"level": lvl, "project": proj, "method": method, "simple baseline": base,
                     "gain vs baseline": r["gain"], "beats baseline?": "YES" if r["beats"] else "no",
                     "other gates pass?": "yes" if _others_ok(r["gates"]) else "no",
                     "data": "simulated" if r["info"]["source"] == "simulated" else f"your data ({r['info']['label']})",
                     "_gates": r["gates"]})
    return pd.DataFrame(rows)


def plain_summary(table: pd.DataFrame) -> list:
    out = []
    for _, r in table.iterrows():
        g = r["gain vs baseline"]
        gtxt = "n/a" if pd.isna(g) else f"{g:+.1%}"
        if r["beats baseline?"] == "YES":
            out.append(f"**{r['method']}** ({r['project']}) beats {r['simple baseline']} by {gtxt}.")
        else:
            out.append(f"**{r['method']}** ({r['project']}) does NOT clearly beat {r['simple baseline']} ({gtxt}).")
    return out
