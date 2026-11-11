"""Level 6 projects: $40M D-line optimizer and the Integrated Research Project (scoreboard)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .. import lineup as L, progress, scoreboard as SB, theme as T, ui
from .p3_level3 import _gates, _load, _log


# =================================================================== 6A: lineup
@st.cache_data(show_spinner="Racing the optimizers (about 25 seconds)…")
def _contest(pool, risk):
    return L.contest(pool, budget=40, seeds=5, risk=risk)


@st.cache_data(show_spinner="Back-testing every season…")
def _backtest(df, risk):
    return L.backtest(df, risk)


def lineup_lab(project):
    df, info = _load("lineup_candidates", "p6alab")
    seasons = sorted(df["season"].unique()) if "season" in df.columns else [None]
    c1, c2 = st.columns(2)
    season = c1.selectbox("Season's market", seasons, index=len(seasons) - 1, key="p6a_season")
    risk = c2.slider("Risk penalty λ (value − λ × uncertainty)", 0.0, 2.0, 0.0, 0.25, key="p6a_risk")
    pool = (df[df["season"] == season] if season is not None else df).reset_index(drop=True)
    if "value_sd" not in pool.columns:
        pool = pool.assign(value_sd=0.0)

    st.markdown("#### 1 · Objective, constraints, decision (L6.2)")
    st.markdown(f"- **Decision:** which {L.ROSTER} of the {len(pool)} candidates to sign\n"
                f"- **Constraints:** total cap ≤ ${L.CAP:.0f}M, exactly {L.ROSTER} players, at least 2 EDGE and 2 DT\n"
                f"- **Objective:** maximize Σ (projected value − {risk:g} × uncertainty)")

    st.markdown("#### 2 · The exact best lineup vs greedy")
    x_opt, x_gr = L.exact(pool, risk), L.greedy(pool, risk)
    if x_opt is None:
        st.error("No legal lineup exists under these rules.")
        return
    cols = ["player_id", "position", "cap_hit_m", "projected_value", "value_sd"]
    a, b = st.columns(2)
    a.markdown(f"**Exact solver** · score {L.total(pool, x_opt, risk):.2f} · cap ${pool['cap_hit_m'][x_opt].sum():.1f}M")
    with a:
        ui.table(pool.loc[x_opt, cols])
    if x_gr is not None:
        b.markdown(f"**Greedy** · score {L.total(pool, x_gr, risk):.2f} · cap ${pool['cap_hit_m'][x_gr].sum():.1f}M")
        with b:
            ui.table(pool.loc[x_gr, cols])

    st.markdown("#### 3 · Race: Bayesian optimization vs random search (same 40-try budget)")
    if st.button("Run the race", key="p6a_race"):
        st.session_state["p6a_race_args"] = (season, risk)
    if st.session_state.get("p6a_race_args") == (season, risk):
        c = _contest(pool, risk)
        n = np.arange(1, c["bo"].shape[1] + 1)
        fig, ax = plt.subplots(figsize=(7.5, 3.0))
        ax.plot(n, c["bo"].mean(0) / c["optimum"], color=T.BLUE, lw=2.4, label="Bayesian optimization")
        ax.plot(np.arange(1, c["random"].shape[1] + 1), c["random"].mean(0) / c["optimum"], color=T.AMBER, lw=2.4, label="Random search")
        ax.axhline(c["greedy"] / c["optimum"], color=T.ROSE, ls="--", lw=1.4, label="Greedy (1 try)")
        ax.axhline(1, color=T.GREEN, lw=1.2, label="Exact solver")
        ax.set_xlabel("Tries"), ax.set_ylabel("Share of best possible score"), ax.legend(fontsize=7)
        ui.show_fig(fig)
        st.caption(f"After 40 tries: BO {c['bo'][:, -1].mean() / c['optimum']:.0%}, random "
                   f"{c['random'][:, -1].mean() / c['optimum']:.0%}, greedy {c['greedy'] / c['optimum']:.0%}. "
                   "The exact solver gets 100% instantly, because this score is a simple sum.")

    st.markdown("#### 4 · Trade-off: how much does more cap buy?")
    caps = np.arange(20, 61, 5)
    vals = []
    for cap in caps:
        old, L.CAP = L.CAP, float(cap)
        try:
            x = L.exact(pool, risk)
            vals.append(L.total(pool, x, risk) if x is not None else np.nan)
        finally:
            L.CAP = old
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.plot(caps, vals, marker="o", color=T.BLUE, lw=2)
    ax.axvline(40, color=T.MUTED, ls="--", lw=1)
    ax.set_xlabel("Cap ($M)"), ax.set_ylabel("Best score")
    ui.show_fig(fig)
    st.caption("Diminishing returns: each extra $5M buys less once the bargains are gone.")


def lineup_prod(project):
    df, info = _load("lineup_candidates", "p6aprod")
    risk = st.slider("Risk penalty λ (fix it before running)", 0.0, 2.0, 0.0, 0.25, key="p6a_prod_risk")
    if st.button("Run the season-by-season back-test", type="primary", key="p6a_eval"):
        st.session_state["p6a_eval_args"] = (info["label"], risk)
    if st.session_state.get("p6a_eval_args") != (info["label"], risk):
        st.caption("Metric, baseline and gates are fixed in code before any data is evaluated.")
        return
    res = _backtest(df, risk)
    gl = L.gates(res, info)
    verdict = L.verdict(gl)
    _log(project, f"{verdict} [{info['label']}, risk {risk:g}]")
    _gates(gl, verdict)
    if not res.get("ok"):
        return
    per = res["per_season"]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for col, c, lab in [("optimizer", T.BLUE, "Exact optimizer (realized)"), ("greedy", T.ROSE, "Greedy (realized)"),
                        ("random_median", T.MUTED, "Median random lineup"), ("optimizer_projected", T.GREEN, "Optimizer (projected)")]:
        ax.plot(per["season"], per[col], marker="o", color=c, lw=2 if col != "optimizer_projected" else 1.2,
                ls="-" if col != "optimizer_projected" else "--", label=lab)
    ax.set_xlabel("Season"), ax.set_ylabel("Total value of the 6"), ax.legend(fontsize=7)
    ui.show_fig(fig)
    ui.table(per)
    st.caption("The dashed green line is what the optimizer expected; the gap down to the blue line is the optimizer's curse.")


# =================================================================== 6B: integrated
@st.cache_data(show_spinner="Running every level's test (about a minute)…")
def _scoreboard(_token):
    return SB.run_all()


def _show_scoreboard(tb):
    show = tb.drop(columns="_gates").copy()
    show["gain vs baseline"] = show["gain vs baseline"].map(lambda g: "n/a" if pd.isna(g) else f"{g:+.1%}")
    ui.table(show.rename(columns=lambda c: c[:1].upper() + c[1:]))
    wins = tb[tb["beats baseline?"] == "YES"]
    st.markdown(f"**{len(wins)} of {len(tb)} methods beat their simple baseline.**")
    for line in SB.plain_summary(tb):
        st.markdown(f"- {line}")
    with st.expander("Every gate, study by study"):
        for _, r in tb.iterrows():
            st.markdown(f"**{r['project']}**")
            ui.gates(r["_gates"])


def integrated_lab(project):
    st.markdown("#### 1 · The scoreboard: which methods earn their complexity?")
    st.caption("Runs each level's automated test on your uploaded data where present, simulated data otherwise.")
    if st.button("Run the scoreboard", type="primary", key="p6b_run"):
        st.session_state["p6b_token"] = st.session_state.get("p6b_token", 0) + 1
    if "p6b_token" in st.session_state:
        _show_scoreboard(_scoreboard(st.session_state["p6b_token"]))

    st.markdown("#### 2 · The research template, worked example")
    st.markdown("""
| Step | Worked example (Level 5 study) |
|---|---|
| **Research question** | Does a hierarchical model predict next-season pressure rate better than Level 1 shrinkage and 'same as last season'? |
| **Data preparation** | Player-seasons with successes, opportunities, position, team; experience computed from earlier seasons only. |
| **Baseline model** | 'Same as last season' and Level 1 position shrinkage. |
| **Advanced model** | Hierarchical Bayes (position + experience + team + player), fitted by MCMC. |
| **Validation** | Rolling origin by season; MCMC convergence checked in every fold. |
| **Uncertainty analysis** | 80% predictive intervals; coverage checked. |
| **Written conclusions** | Wins only when team/experience effects exist (+13%); loses without them (−8%). Simulated, so it says nothing about real football yet. |
""")


def integrated_prod(project):
    state = ui.get_state()
    rec = state["projects"].get(project.id, {})
    reg = rec.get("registration")
    st.markdown("#### 1 · Register the study (before running anything)")
    studies = {sid: name for sid, _, name, _, _ in SB.STUDIES if sid not in ("p5b", "p6t")}
    with st.form("p6b_reg"):
        q = st.text_area("Research question", value=(reg or {}).get("question", ""), height=70)
        primary = st.selectbox("Primary study", list(studies), format_func=studies.get,
                               index=list(studies).index((reg or {}).get("primary", "p5")) if reg else 5)
        metric = st.text_input("Metric (use the study's predefined metric)", value=(reg or {}).get("metric", ""))
        baseline = st.text_input("Simple baseline it must beat", value=(reg or {}).get("baseline", ""))
        saved = st.form_submit_button("Save registration")
    if saved:
        if not q.strip() or not metric.strip() or not baseline.strip():
            st.error("Fill in the question, metric and baseline.")
        else:
            progress.set_registration(state, project.id, {"question": q, "primary": primary, "metric": metric, "baseline": baseline})
            st.rerun()
    if reg:
        st.success(f"Registered {ui.fmt_ts(reg['ts'])}: {reg['question']}")

    st.markdown("#### 2 · Run the primary study")
    if st.button("Run the registered study", type="primary", key="p6b_eval", disabled=not reg):
        st.session_state["p6b_eval"] = reg["primary"]
    if not reg or st.session_state.get("p6b_eval") != reg["primary"]:
        st.caption("Save a registration first. The verdict uses the primary study's own gates.")
        return
    r = SB.run_one(reg["primary"])
    gl = [("Question, metric and baseline registered before analysis", True, f"Registered {ui.fmt_ts(reg['ts'])}.")] + r["gates"]
    verdict = "Meets Production Candidate criteria (review before any use)" if all(ok for _, ok, _ in gl) else "Learning Lab only"
    _log(project, f"{verdict} [{studies[reg['primary']]}]")
    _gates(gl, verdict)
