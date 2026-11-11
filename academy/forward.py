"""Concept 36: frozen forward tests.

1. Before outcomes exist, write predictions AND the scoring plan into one canonical text, and hash it (SHA-256).
2. Later, recompute the hash (proves nothing was edited), then score exactly as the plan says.
3. If the outcome window hasn't closed when you score, the plan scores ranking (AUC) only: a model of
   "ever happens" can't be calibration-checked against a window that has seen only part of the outcomes.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import validation as V


def make_plan(class_year: int, window_years: int = 5, written: str = "") -> dict:
    return {"target": f"drafted within {window_years} years of enrolling",
            "class_year": int(class_year), "window_end": int(class_year + window_years),
            "metrics_if_complete": ["auc", "calibration"], "metrics_if_partial": ["auc"],
            "rule": "If scored before window_end, score ranking (AUC) only; calibration waits for the full window.",
            "written": written}


def canonical(preds: pd.DataFrame, plan: dict) -> str:
    """One exact text for predictions + plan: sorted, fixed decimals, no spaces. Same input -> same text."""
    rows = sorted([str(i), round(float(p), 6)] for i, p in zip(preds["player_id"], preds["p"]))
    return json.dumps({"plan": plan, "predictions": rows}, sort_keys=True, separators=(",", ":"))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze(preds: pd.DataFrame, plan: dict) -> dict:
    payload = canonical(preds, plan)
    return {"payload": payload, "hash": sha256(payload)}


def verify(frozen: dict) -> bool:
    return sha256(frozen["payload"]) == frozen["hash"]


def unpack(frozen: dict):
    d = json.loads(frozen["payload"])
    preds = pd.DataFrame(d["predictions"], columns=["player_id", "p"])
    return preds, d["plan"]


def score(frozen: dict, outcomes: pd.DataFrame, observed_through: int) -> dict:
    """outcomes: player_id + y (1 = happened by observed_through). Scores only what the frozen plan allows."""
    if not verify(frozen):
        return {"ok": False, "reason": "Hash does not match: the frozen predictions or plan were changed."}
    preds, plan = unpack(frozen)
    m = preds.merge(outcomes[["player_id", "y"]], on="player_id", how="inner")
    complete = observed_through >= plan["window_end"]
    metrics = plan["metrics_if_complete"] if complete else plan["metrics_if_partial"]
    out = {"ok": True, "complete": complete, "metrics": metrics, "n": len(m), "events": int(m["y"].sum()),
           "auc": V.auc(m["p"], m["y"]), "mean_forecast": float(m["p"].mean()), "observed_rate": float(m["y"].mean())}
    if "calibration" in metrics:
        out["ece"] = V.ece(m["p"], m["y"])
        out["brier"] = V.brier(m["p"], m["y"])
    else:
        out["ece_if_misused"] = V.ece(m["p"], m["y"])        # shown only to explain the trap; NOT a score
    return out


def simulate_forecast(per_class: int = 120, seed: int = 36, class_year: int = 2022, as_of: int = 2023):
    """Fit a Bayesian logistic on classes finished by `as_of`, predict P(drafted within 5 years) for `class_year`."""
    from .data import make_draft_classes
    from .smalldata import Design, fit_logistic, predict_logistic
    now = make_draft_classes(per_class=per_class, seed=seed, data_end=as_of)
    train = now[now["class_year"] + 5 <= as_of]
    test = now[now["class_year"] == class_year]
    d = Design(train)
    model = fit_logistic(d.linear(train), train["drafted"], *d.priors(True))
    p = predict_logistic(model, d.linear(test), moderated=True)
    return pd.DataFrame({"player_id": test["player_id"].to_numpy(), "p": np.round(p, 6)})


def simulate_outcomes(per_class: int = 120, seed: int = 36, class_year: int = 2022, observed_through: int = 2027):
    """The same simulated players, with drafts revealed only up to `observed_through`."""
    from .data import make_draft_classes
    later = make_draft_classes(per_class=per_class, seed=seed, data_end=observed_through)
    c = later[later["class_year"] == class_year]
    return pd.DataFrame({"player_id": c["player_id"].to_numpy(), "y": c["drafted"].to_numpy(int)})
