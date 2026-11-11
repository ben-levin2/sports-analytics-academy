"""Concepts 29-33 and the Small-Data Draft Model project (P3C).

Four models predict P(drafted within 5 years) for a recruiting class, judged at the end of its first season:
  plain     plain logistic regression (almost no penalty)
  rating_only  logistic on rating alone (a simple reference)
The Learning Lab judges gains against the plain logistic. The Production Candidate test judges them against the
STRONGER of plain logistic and rating-only (higher pooled test AUC), so a weak baseline can't create a fake win.
  bayes     Bayesian logistic regression with informative Normal priors (concepts 29-30)
  gam       monotonic generalized additive model: smooth curves that can only rise (concept 31)
  survival  discrete-time survival on person-period rows, so unfinished classes still count (concept 33)

Everything is NumPy/SciPy. The rules of the Production Candidate test are fixed here, before any data is seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata

from . import validation as V

PERIODS = (3, 4, 5)                 # drafted 3, 4 or 5 years after the class enrolled
WINDOW = PERIODS[-1]
LAG = 1                             # predictions are made one year after enrolling (year-1 stats known)
MODELS = ("plain", "bayes", "gam", "survival")
CHALLENGERS = MODELS[1:]
NAMES = {"plain": "Plain logistic", "bayes": "Bayesian logistic", "gam": "Monotonic GAM",
         "survival": "Discrete-time survival", "rating_only": "Rating only (reference)"}

MIN_AUC_GAIN = 0.02
BOOT_B = 1000
BOOT_ALPHA = 0.05 / len(CHALLENGERS)          # one-sided, split across the 3 challengers (Bonferroni)
MAX_ECE = 0.05
MIN_TEST_CLASSES = 4
MIN_TEST_DRAFTED = 15
MIN_TRAIN_ROWS = 60
MIN_TRAIN_DRAFTED = 5
PRIMARY_METRIC = ("Pooled AUC of P(drafted within 5 years) over every later class whose draft window has closed, "
                  "each class predicted one year after it enrolled from earlier classes only. A challenger must "
                  f"beat the stronger of the plain logistic and rating-only baselines by at least +{MIN_AUC_GAIN:.2f} AUC, with the class-level "
                  f"(cluster) bootstrap lower bound above 0 at the {BOOT_ALPHA:.2%} level, and calibration error "
                  f"<= {MAX_ECE}.")

# Priors for the Bayesian models, written down before seeing any data. Units: log-odds per 1 SD of input.
PRIOR = {"intercept": (float(logit(0.10)), 1.5),     # about 1 in 10 recruits drafted, very unsure
         "rating": (0.8, 0.4),                       # higher rating -> more likely drafted; moderate size
         "year1": (0.8, 0.4),                        # better year-1 production -> more likely drafted
         "position": (0.0, 0.5)}                     # positions differ a little, direction unknown


# ---------------------------------------------------------------- concept 29-30: penalized / Bayesian logistic
def fit_logistic(X, y, prior_mean=None, prior_sd=None, weights=None, max_iter=100):
    """MAP logistic regression with independent Normal priors (= L2 / ridge penalty).
    prior_sd = large -> plain logistic. Returns beta and the Laplace covariance (posterior uncertainty)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, p = X.shape
    m = np.zeros(p) if prior_mean is None else np.asarray(prior_mean, float)
    sd = np.full(p, 100.0) if prior_sd is None else np.asarray(prior_sd, float)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    prec = 1.0 / sd ** 2
    beta = m.copy()
    for _ in range(max_iter):
        mu = expit(X @ beta)
        grad = X.T @ (w * (y - mu)) - prec * (beta - m)
        H = X.T @ (X * (w * mu * (1 - mu))[:, None]) + np.diag(prec)
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    mu = expit(X @ beta)
    H = X.T @ (X * (w * mu * (1 - mu))[:, None]) + np.diag(prec)
    return {"beta": beta, "cov": np.linalg.inv(H)}


def predict_logistic(model, X, moderated=False):
    """Probabilities. moderated=True averages over the posterior uncertainty (MacKay's probit approximation)."""
    X = np.asarray(X, float)
    eta = X @ model["beta"]
    if moderated:
        var = np.einsum("ij,jk,ik->i", X, model["cov"], X)
        eta = eta / np.sqrt(1 + np.pi * var / 8)
    return expit(eta)


# ---------------------------------------------------------------- design matrices
class Design:
    """Standardize rating and year-1 percentile with TRAINING stats; position dummies vs the most common one."""

    def __init__(self, train: pd.DataFrame):
        self.mr, self.sr = train["rating"].mean(), train["rating"].std() or 1.0
        self.my, self.sy = train["year1_percentile"].mean(), train["year1_percentile"].std() or 1.0
        if "position" in train.columns:
            counts = train["position"].value_counts()
            self.ref = counts.index[0]
            self.positions = sorted(p for p in counts.index if p != self.ref)
        else:
            self.ref, self.positions = None, []

    def z(self, df):
        return ((df["rating"].to_numpy(float) - self.mr) / self.sr,
                (df["year1_percentile"].to_numpy(float) - self.my) / self.sy)

    def pos(self, df):
        if not self.positions:
            return np.zeros((len(df), 0))
        return np.column_stack([(df["position"] == p).to_numpy(float) for p in self.positions])

    def linear(self, df):
        zr, zy = self.z(df)
        return np.column_stack([np.ones(len(df)), zr, zy, self.pos(df)])

    def priors(self, informative=True):
        k = len(self.positions)
        if not informative:
            return np.zeros(3 + k), np.full(3 + k, 100.0)
        m = np.r_[PRIOR["intercept"][0], PRIOR["rating"][0], PRIOR["year1"][0], np.zeros(k)]
        s = np.r_[PRIOR["intercept"][1], PRIOR["rating"][1], PRIOR["year1"][1], np.full(k, PRIOR["position"][1])]
        return m, s


# ---------------------------------------------------------------- concept 31: monotonic GAM
def ramp_basis(z, knots, scale=0.5):
    """Smooth ramps: softplus((z - knot)/scale) * scale. Each one is flat, then bends upward at its knot.
    With NON-NEGATIVE weights, any sum of them (plus a non-negative slope) can only go up."""
    z = np.asarray(z, float)[:, None]
    return scale * np.logaddexp(0, (z - np.asarray(knots)[None, :]) / scale)


KNOTS = np.array([-1.0, 0.0, 1.0])


def fit_monotone_gam(design: Design, df, y, lam=2.0, pos_sd=0.5, monotone=True):
    """Logistic GAM: intercept + slope*z + ramps for each input (weights >= 0) + position effects.
    lam penalizes the ramp weights (wiggliness); positions get a Normal(0, pos_sd) prior. Fit by L-BFGS-B.
    monotone=False drops the >= 0 bounds (an ordinary penalized spline, free to wiggle)."""
    zr, zy = design.z(df)
    B = np.column_stack([zr, ramp_basis(zr, KNOTS), zy, ramp_basis(zy, KNOTS)])
    P = design.pos(df)
    X = np.column_stack([np.ones(len(df)), B, P])
    y = np.asarray(y, float)
    nb, k = B.shape[1], P.shape[1]
    ramp_idx = np.r_[2:2 + len(KNOTS), 3 + len(KNOTS):3 + 2 * len(KNOTS)]
    pos_idx = np.arange(1 + nb, 1 + nb + k)

    def nll(b):
        eta = X @ b
        mu = expit(eta)
        f = -np.sum(y * eta - np.logaddexp(0, eta)) + 0.5 * lam * np.sum(b[ramp_idx] ** 2) \
            + 0.5 * np.sum(b[pos_idx] ** 2) / pos_sd ** 2
        g = -X.T @ (y - mu)
        g[ramp_idx] += lam * b[ramp_idx]
        g[pos_idx] += b[pos_idx] / pos_sd ** 2
        return f, g

    b0 = np.zeros(X.shape[1])
    b0[0] = logit(np.clip(y.mean(), 0.02, 0.98))
    bounds = [(None, None)] + [(0, None) if monotone else (None, None)] * nb + [(None, None)] * k
    r = minimize(nll, b0, jac=True, method="L-BFGS-B", bounds=bounds)
    return {"beta": r.x, "design": design, "nb": nb}


def predict_gam(model, df):
    d = model["design"]
    zr, zy = d.z(df)
    X = np.column_stack([np.ones(len(df)), zr, ramp_basis(zr, KNOTS), zy, ramp_basis(zy, KNOTS), d.pos(df)])
    return expit(X @ model["beta"])


def gam_curve(model, which, grid_z):
    """The fitted shape for one input (log-odds, centered at z = 0), other inputs held at average."""
    b = model["beta"]
    nk = len(KNOTS)
    if which == "rating":
        slope, ramps, = b[1], b[2:2 + nk]
    else:
        slope, ramps = b[2 + nk], b[3 + nk:3 + 2 * nk]
    f = slope * grid_z + ramp_basis(grid_z, KNOTS) @ ramps
    f0 = slope * 0 + ramp_basis(np.array([0.0]), KNOTS) @ ramps
    return f - f0


# ---------------------------------------------------------------- concept 33: discrete-time survival
def person_period(df: pd.DataFrame, as_of: int) -> pd.DataFrame:
    """One row per player per draft year he was eligible AND that had happened by `as_of`.
    event = 1 in the year he was drafted; he leaves the risk set after that. Unfinished players simply
    stop early (censored): their 'not drafted yet' years still count as information."""
    out = []
    dy = df["draft_year"] if "draft_year" in df.columns else pd.Series(np.nan, index=df.index)
    for k in PERIODS:
        cal = df["class_year"] + k
        at_risk = (cal <= as_of) & ~(dy < cal)                    # observed, and not drafted earlier
        rows = df[at_risk].copy()
        rows["period"] = k
        rows["event"] = (dy[at_risk] == cal[at_risk]).astype(int)
        out.append(rows)
    return pd.concat(out, ignore_index=True)


def survival_X(design: Design, df, periods_col):
    zr, zy = design.z(df)
    per = np.column_stack([(periods_col == k).to_numpy(float) for k in PERIODS])
    return np.column_stack([per, zr, zy, design.pos(df)])


def fit_survival(design: Design, pp: pd.DataFrame):
    k = len(design.positions)
    m = np.r_[np.full(len(PERIODS), float(logit(0.04))), PRIOR["rating"][0], PRIOR["year1"][0], np.zeros(k)]
    s = np.r_[np.full(len(PERIODS), 1.5), PRIOR["rating"][1], PRIOR["year1"][1], np.full(k, PRIOR["position"][1])]
    return fit_logistic(survival_X(design, pp, pp["period"]), pp["event"], m, s)


def predict_survival(model, design: Design, df):
    """P(drafted within the window) = 1 - product over periods of (1 - hazard)."""
    surv = np.ones(len(df))
    for k in PERIODS:
        X = survival_X(design, df, pd.Series(k, index=df.index))
        surv *= 1 - predict_logistic(model, X)
    return 1 - surv


# ---------------------------------------------------------------- fitting all models for one cutoff
def fit_predict(train_full: pd.DataFrame, pp: pd.DataFrame | None, test: pd.DataFrame) -> dict:
    """train_full: classes whose 5-year window had closed; pp: person-period rows (may include unfinished
    classes); test: the class being predicted. Returns {model: probabilities}."""
    d = Design(train_full)
    X, Xt, y = d.linear(train_full), d.linear(test), train_full["drafted"].to_numpy(float)
    out = {"plain": predict_logistic(fit_logistic(X, y, *d.priors(False)), Xt)}
    out["bayes"] = predict_logistic(fit_logistic(X, y, *d.priors(True)), Xt, moderated=True)
    out["gam"] = predict_gam(fit_monotone_gam(d, train_full, y), test)
    out["rating_only"] = predict_logistic(fit_logistic(X[:, :2], y, np.zeros(2), np.full(2, 100.0)), Xt[:, :2])
    if pp is not None and len(pp) and pp["event"].sum() >= MIN_TRAIN_DRAFTED:
        ds = Design(pp.drop_duplicates("player_id"))
        out["survival"] = predict_survival(fit_survival(ds, pp), ds, test)
    return out


def _has_draft_years(df):
    return "draft_year" in df.columns and df["draft_year"].notna().any()


def evaluate(df: pd.DataFrame, boot: int = BOOT_B, seed: int = 0, baseline: str = "best") -> dict:
    """Class-by-class, time-respecting test. For each class C whose 5-year window has closed, predict at
    as_of = C + 1 using only what was known then:
      - logistic-type models: classes c <= C + 1 - 5 (window closed by as_of)
      - survival: every earlier class, with draft years only up to as_of (unfinished classes included).
    baseline: "best" = the stronger (higher pooled AUC) of plain logistic and rating-only (Production Candidate);
              "plain" = plain logistic (Learning Lab)."""
    df = df.copy()
    has_dy = _has_draft_years(df)
    data_end = int(df["draft_year"].max()) if has_dy else int(df["class_year"].max()) + WINDOW
    rows, leak_ok, n_full, n_pp = [], True, [], []
    for C in sorted(df["class_year"].unique()):
        if C + WINDOW > data_end:
            continue                                         # test outcome not final yet
        as_of = C + LAG
        full = df[df["class_year"] + WINDOW <= as_of]
        if len(full) < MIN_TRAIN_ROWS or full["drafted"].sum() < MIN_TRAIN_DRAFTED:
            continue
        pp = person_period(df[df["class_year"] < C], as_of) if has_dy else None
        test = df[df["class_year"] == C]
        leak_ok &= bool((full["class_year"] + WINDOW <= as_of).all())
        if has_dy:
            leak_ok &= bool((pp["class_year"] + pp["period"] <= as_of).all())
            drafted_late = full["draft_year"].notna() & (full["draft_year"] > as_of)
            leak_ok &= not bool(drafted_late.any())
        preds = fit_predict(full, pp, test)
        n_full.append(len(full))
        n_pp.append(0 if pp is None else pp["player_id"].nunique())
        rows.append(pd.DataFrame({"class_year": C, "y": test["drafted"].to_numpy(int), **preds}))
    if not rows:
        return {"ok": False, "reason": "Need classes whose 5-year draft window has closed, with at least "
                                       f"{MIN_TRAIN_ROWS} finished recruits (and {MIN_TRAIN_DRAFTED}+ drafted) before them."}
    r = pd.concat(rows, ignore_index=True)
    models = [m for m in ("plain", "bayes", "gam", "survival", "rating_only") if m in r.columns]
    summ = pd.DataFrame({m: {"auc": V.auc(r[m], r["y"]), "brier": V.brier(r[m], r["y"]),
                             "ece": V.ece(r[m], r["y"])} for m in models}).T
    base = "plain"
    if baseline == "best" and summ.loc["rating_only", "auc"] > summ.loc["plain", "auc"]:
        base = "rating_only"
    summ["gain"] = summ["auc"] - summ.loc[base, "auc"]
    boot_res = cluster_bootstrap(r, [m for m in CHALLENGERS if m in r.columns], B=boot, seed=seed, base=base) if boot else {}
    return {"ok": True, "rows": r, "summary": summ, "boot": boot_res, "baseline": base, "leak_ok": leak_ok, "has_draft_years": has_dy,
            "n_classes": int(r["class_year"].nunique()), "n_drafted": int(r["y"].sum()), "n_test": len(r),
            "train_rows_full": n_full, "train_players_survival": n_pp, "data_end": data_end}


# ---------------------------------------------------------------- cluster bootstrap
def fast_auc(score, y):
    y = np.asarray(y)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(score)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cluster_bootstrap(r: pd.DataFrame, models, B=BOOT_B, seed=0, base="plain") -> dict:
    """Resample whole CLASSES with replacement (players in one class share a year, a staff, a draft),
    recompute the pooled AUC gain over the baseline each time."""
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in r.groupby("class_year")]
    y = r["y"].to_numpy()
    cols = {m: r[m].to_numpy() for m in [base, *models]}
    draws = {m: [] for m in models}
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        b_auc = fast_auc(cols[base][idx], y[idx])
        for m in models:
            draws[m].append(fast_auc(cols[m][idx], y[idx]) - b_auc)
    out = {}
    for m, d in draws.items():
        d = np.array(d)
        d = d[~np.isnan(d)]
        out[m] = {"lower": float(np.quantile(d, BOOT_ALPHA)), "lo90": float(np.quantile(d, 0.05)),
                  "hi90": float(np.quantile(d, 0.95)), "p_positive": float(np.mean(d > 0))}
    return out


# ---------------------------------------------------------------- gates
def gates(res: dict, data_info: dict, model: str) -> list:
    if not res.get("ok"):
        return [("Evaluation ran", False, res.get("reason", ""))]
    is_real = data_info["source"] != "simulated" and data_info.get("real")
    g = [("Real data you have rights to use", bool(is_real),
          "Simulated data can never qualify." if data_info["source"] == "simulated"
          else ("Marked as real on upload." if is_real else "Upload is not marked as real data.")),
         ("Baseline defined", True,
          f"Stronger of plain logistic and rating-only; used here: {NAMES[res.get('baseline', 'plain')]} "
          f"(AUC {res['summary'].loc[res.get('baseline', 'plain'), 'auc']:.3f})."),
         ("Metric predefined", True, PRIMARY_METRIC),
         ("Time-based validation (concept 26)",
          res["n_classes"] >= MIN_TEST_CLASSES and res["n_drafted"] >= MIN_TEST_DRAFTED,
          f"Train on earlier classes, test on later ones. {res['n_classes']} test classes, {res['n_drafted']} drafted "
          f"players in them (need >= {MIN_TEST_CLASSES} and >= {MIN_TEST_DRAFTED})."),
         ("No leakage (concept 26)", bool(res["leak_ok"]),
          "Every training outcome was known one year after the test class enrolled.")]
    if model not in res["summary"].index:
        g.append(("Model could run", False, "Needs a draft_year column (and enough drafted players) for person-period rows."
                  if model == "survival" else "Model did not run."))
        return g
    s = res["summary"].loc[model]
    b = res["boot"].get(model)
    g += [("AUC gain over the baseline", s["gain"] >= MIN_AUC_GAIN,
           f"{s['auc']:.3f} vs {res['summary'].loc[res.get('baseline', 'plain'), 'auc']:.3f} = {s['gain']:+.3f} "
           f"(need >= +{MIN_AUC_GAIN:.2f})."),
          ("Gain holds up in the cluster bootstrap", bool(b) and b["lower"] > 0,
           (f"Resampling whole classes: lower bound {b['lower']:+.3f} at the {BOOT_ALPHA:.2%} level "
            f"(90% range {b['lo90']:+.3f} to {b['hi90']:+.3f}); need above 0.") if b else "Bootstrap not run."),
          ("Calibrated probabilities (concept 25)", s["ece"] <= MAX_ECE,
           f"Calibration error {s['ece']:.3f} (need <= {MAX_ECE}).")]
    return g


def verdict(gate_list):
    return ("Meets Production Candidate criteria (review before any use)"
            if all(p for _, p, _ in gate_list) else "Learning Lab only")


# ---------------------------------------------------------------- Learning Lab: when does it fail?
def failure_study(sizes=(10, 22, 45, 90), noises=(0.0, 0.5, 1.0), seeds=(1, 2, 3, 4)) -> pd.DataFrame:
    """Average AUC gain over the plain baseline as data gets scarcer or noisier (no bootstrap)."""
    from .data import make_draft_classes
    out = []
    for kind, values in (("Recruits per class", sizes), ("Input noise (SD)", noises)):
        for v in values:
            for s in seeds:
                df = make_draft_classes(per_class=v if kind.startswith("Recruits") else 22,
                                        noise=v if kind.startswith("Input") else 0.0, seed=100 + s)
                res = evaluate(df, boot=0, baseline="plain")
                if not res["ok"]:
                    continue
                for m in CHALLENGERS:
                    if m in res["summary"].index:
                        out.append({"study": kind, "value": v, "seed": s, "model": m,
                                    "gain": res["summary"].loc[m, "gain"], "auc": res["summary"].loc[m, "auc"], "ece": res["summary"].loc[m, "ece"],
                                    "plain_auc": res["summary"].loc["plain", "auc"]})
    return pd.DataFrame(out)
