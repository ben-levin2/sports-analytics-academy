"""Level 5 project: Hierarchical Player Development (NumPy MCMC, mixed model, three-way contest)."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .. import hierarchical as H, theme as T, ui
from .p3_level3 import _gates, _load, _log

PYMC_SKETCH = '''import pymc as pm
# d = academy.hierarchical.prepare(df); integer codes for position, team, player, experience
with pm.Model() as model:
    beta_pos = pm.Normal("beta_pos", -2.5, 1.0, shape=n_positions)
    gamma = pm.Normal("gamma", 0, 0.5, shape=3)                    # experience steps 1, 2, 3+
    sd_team = pm.HalfNormal("sd_team", 0.5)
    sd_player = pm.HalfNormal("sd_player", 0.5)
    sd_season = pm.HalfNormal("sd_season", 0.3)
    tau = pm.Normal("tau", 0, sd_team, shape=n_teams)               # partially pooled team effects
    u = pm.Normal("u", 0, sd_player, shape=n_players)               # partially pooled player effects
    exp_effect = pm.math.concatenate([[0.0], gamma])[exp_idx]
    eta = beta_pos[pos_idx] + exp_effect + tau[team_idx] + u[player_idx]
    pm.Normal("y", eta, pm.math.sqrt(d["v"].values + sd_season ** 2), observed=d["y"].values)
    idata = pm.sample(1000, tune=1000, chains=4)                    # NUTS: a smarter MCMC sampler
# then: arviz.summary(idata) for R-hat and ESS'''


@st.cache_data(show_spinner="Running the hierarchical MCMC sampler…")
def _fit(df):
    d = H.prepare(df)
    f = H.gibbs(d)
    u = pd.Series(f["u"].reshape(-1, len(f["players"])).mean(0), index=f["players"])
    tau = pd.Series(f["tau"].reshape(-1, len(f["teams"])).mean(0), index=f["teams"])
    return H.diagnostics(f), u, tau, d


@st.cache_data(show_spinner="Fitting the mixed-effects model (statsmodels)…")
def _mixed(df):
    import statsmodels.formula.api as smf
    d = H.prepare(df)
    d = d[d["opportunities"] >= 100].copy()
    d["exp"] = d["experience"].astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("y ~ 0 + C(position) + C(exp)", d, groups=d["player_id"]).fit()
    return m.params, float(np.sqrt(m.cov_re.iloc[0, 0])), float(np.sqrt(m.scale)), len(d)


@st.cache_data(show_spinner="Running the season-by-season contest (about 15 seconds)…")
def _evaluate(df):
    return H.evaluate(df)


def _pick(key):
    """Like the shared loader, but defaults to the dataset WITH team and experience effects."""
    from .. import datasets
    if datasets.status("rate_events")["source"] == "simulated":
        names = list(datasets.SPECS["rate_events"].simulated)
        default = "Pressure rate with team & experience effects"
        if st.session_state.get(f"{key}_variant") not in names:
            st.session_state[f"{key}_variant"] = default
    return _load("rate_events", key)


def learning_lab(project):
    df, info = _pick("p5lab")
    if not {"position", "team"} <= set(df.columns):
        st.warning("This dataset has no position/team columns, so the hierarchy has only players. Results will be close to Level 1.")
    diag, u, tau, d = _fit(df)

    st.markdown("#### 1 · Convergence first (L5.3)")
    worst = float(diag["R-hat"].max())
    (st.success if worst <= H.MAX_RHAT else st.error)(
        f"Worst R-hat {worst:.3f} ({'converged' if worst <= H.MAX_RHAT else 'NOT converged: do not trust the numbers below'}).")
    st.dataframe(diag.round(3), hide_index=True)

    st.markdown("#### 2 · What the model learned (L5.5)")
    beta = diag.set_index("parameter")["mean"]
    pos_rows = [p for p in beta.index if p.startswith("position")]
    rates = pd.DataFrame({"rookie": H.expit(beta[pos_rows].to_numpy()),
                          "3rd+ season": H.expit(beta[pos_rows].to_numpy() + beta.get("experience 3+", 0))},
                         index=[p.replace("position ", "") for p in pos_rows])
    c1, c2 = st.columns(2)
    c1.markdown("**Average player's rate by position and experience**")
    c1.dataframe(rates.round(3))
    c2.markdown("**Where the variation comes from (log-odds spreads)**")
    c2.dataframe(diag[diag["parameter"].str.startswith("sd")][["parameter", "mean"]].round(3), hide_index=True)

    st.markdown("#### 3 · Partial pooling at work (L5.4)")
    snaps = d.groupby("player_id")["opportunities"].sum().reindex(u.index)
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.scatter(snaps, u, s=6, color=T.BLUE, alpha=0.5)
    ax.set_xscale("log"), ax.axhline(0, color=T.MUTED, lw=1)
    ax.set_xlabel("Player's total snaps (log scale)"), ax.set_ylabel("Estimated talent effect")
    ui.show_fig(fig)
    st.caption("The funnel: players with few snaps are pulled toward 0; only players with lots of data can sit far from it.")

    st.markdown("#### 4 · Same structure, fitted as a mixed model (L5.6)")
    params, sd_p, sd_r, n = _mixed(df)
    st.dataframe(pd.DataFrame({"mixed model (statsmodels)": params.round(3)}))
    st.caption(f"Player spread {sd_p:.3f}, residual {sd_r:.3f}, on {n} player-seasons with 100+ snaps. "
               "Compare the experience steps with the MCMC table above. The mixed model has no team effect "
               "or per-row luck weights, so expect small differences.")

    with st.expander("The same hierarchical model in PyMC (optional library)"):
        try:
            import pymc  # noqa: F401
            st.success("PyMC is installed. You can run this sketch locally.")
        except ImportError:
            st.info("PyMC isn't installed (optional): `pip install -r requirements-optional.txt`.")
        st.code(PYMC_SKETCH, language="python")


def production_candidate(project):
    df, info = _pick("p5prod")
    if st.button("▶ Run the three-way contest", type="primary", key="p5_eval"):
        st.session_state["p5_eval_args"] = info["label"]
    if st.session_state.get("p5_eval_args") != info["label"]:
        st.caption("Metric, baselines and gates are fixed in code before any data is evaluated.")
        return
    res = _evaluate(df)
    gl = H.gates(res, info)
    verdict = H.verdict(gl)
    _log(project, f"{verdict} [{info['label']}]")
    _gates(gl, verdict)
    if not res.get("ok"):
        return
    mse = res["mse"]
    m = st.columns(3)
    m[0].metric("'Same as last season'", f"{mse['same_as_last'] * 1e4:.2f}", help="Weighted MSE × 10⁻⁴")
    m[1].metric("Level 1 shrinkage", f"{mse['level1'] * 1e4:.2f}")
    m[2].metric("Hierarchical (MCMC)", f"{mse['hierarchical'] * 1e4:.2f}", f"{res['gain_vs_level1']:+.1%} vs Level 1")
    per = res["per_season"]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for col, c, lab in [("same_as_last", T.MUTED, "Same as last season"), ("level1", T.AMBER, "Level 1 shrinkage"),
                        ("hierarchical", T.BLUE, "Hierarchical")]:
        ax.plot(per["season"], per[col] * 1e4, marker="o", color=c, lw=2, label=lab)
    ax.set_xlabel("Test season"), ax.set_ylabel("Weighted MSE ×10⁻⁴"), ax.legend(fontsize=8)
    ui.show_fig(fig)
    st.dataframe(per.round(6), hide_index=True)
    st.caption("Try both simulated datasets. With team and experience effects, the hierarchy should help. "
               "Without them, it shouldn't, and the gates will say so.")
