"""Level 4 project: Game-by-Game Player Performance (HMM, forecasts, conformal, optional LSTM)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .. import sequence as Q, theme as T, ui
from .p3_level3 import _gates, _load, _log


@st.cache_data(show_spinner="Fitting hidden Markov models…")
def _hmms(df):
    rs = Q.running_mean_residuals(df).dropna(subset=["resid"])
    seqs = [g["resid"].to_numpy() for _, g in rs.groupby(["player_id", "season"])]
    return {k: Q.fit_hmm(seqs, k) for k in (1, 2, 3)}


@st.cache_data(show_spinner="Running the season-by-season forecast test…")
def _evaluate(df):
    return Q.evaluate_games(df)


@st.cache_data(show_spinner="Training the LSTM (PyTorch)…")
def _lstm(df):
    seasons = sorted(df["season"].unique())
    cut = seasons[-3] if len(seasons) >= 4 else seasons[-1]
    Xtr, ytr = Q.windows(df[df["season"] < cut])
    Xte, yte = Q.windows(df[df["season"] >= cut])
    out = {"6-game average": Xte.mean(1), "EWMA (0.3)": np.array([Q.ewma_forecast(x, 0.3) for x in Xte]),
           "Real LSTM": Q.torch_lstm(Xtr, ytr, Xte)}
    return {k: float(np.mean(np.abs(yte - v))) for k, v in out.items()}, len(Xtr), int(cut)


def _sufficiency(df):
    per = df.groupby(["player_id", "season"]).size()
    med = float(per.median())
    seasons = int(df["season"].nunique())
    weeks = int(df["week"].max())
    miss = 1 - len(df) / (len(per) * weeks) if weeks else 0.0
    ok = med >= Q.MIN_GAMES_PER_SEASON and seasons >= 4
    ui.table(pd.DataFrame({"measure": ["seasons", "player-seasons", "median games per player-season",
                                           "share of scheduled games missing (approx.)"],
                               "value": [str(seasons), str(len(per)), f"{med:.0f}", f"{miss:.0%}"]}), hide_index=True)
    if ok:
        st.info("Enough games for a fair walk-forward test, but still far too few per player for an LSTM.")
    else:
        st.warning("Sample is too thin: treat everything below as a LEARNING EXERCISE only. No predictive claims.")


def learning_lab(project):
    df, info = _load("game_logs", "p4lab")
    st.markdown("#### 1 · Is there enough data?")
    _sufficiency(df)

    st.markdown("#### 2 · Hot and cold states (L4.1–L4.2)")
    fits = _hmms(df)
    ui.table(pd.DataFrame([{"states": k, "log-likelihood": round(m["loglik"], 1), "BIC (lower = better)": round(Q.bic(m), 1),
                                "state means (vs season-to-date avg)": ", ".join(f"{v:+.1f}" for v in m["means"])}
                               for k, m in fits.items()]), hide_index=True)
    m2 = fits[2]
    st.caption(f"2-state model: stay hot {m2['A'][0, 0]:.2f}, stay cold {m2['A'][1, 1]:.2f}. "
               "Fitted on each game minus the player's average of EARLIER games this season.")

    rs = Q.running_mean_residuals(df).dropna(subset=["resid"])
    keys = [k for k, g in rs.groupby(["player_id", "season"]) if len(g) >= 6][:200]
    if keys:
        k = st.selectbox("Player-season", keys, format_func=lambda x: f"{x[0]} · {x[1]}", key="p4_ps")
        g = rs.set_index(["player_id", "season"]).loc[k].sort_values("week")
        alpha, _ = Q.forward(g["resid"].to_numpy(), m2["pi"], m2["A"], m2["means"], m2["sds"])
        path = Q.viterbi(g["resid"].to_numpy(), m2["pi"], m2["A"], m2["means"], m2["sds"])
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 3.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
        a1.plot(g["week"], g["grade"], marker="o", color=T.MUTED, label="grade")
        a1.plot(g["week"], g["s2d_mean"], color=T.AMBER, ls="--", label="season-to-date avg (earlier games)")
        a1.legend(fontsize=7), a1.set_ylabel("Grade")
        a2.fill_between(g["week"], alpha[:, 0], step="mid", color=T.ROSE, alpha=0.35, label="live P(hot)")
        a2.step(g["week"], (path == 0).astype(float), where="mid", color=T.TEXT, lw=1.2, label="Viterbi: hot?")
        if "true_state" in df.columns:
            truth = df.set_index(["player_id", "season", "week"]).loc[[(*k, w) for w in g["week"]], "true_state"]
            a2.scatter(g["week"], (truth.to_numpy() == "hot") * 1.0, s=18, color=T.GREEN, zorder=5, label="truth (simulated)")
        a2.set_ylim(-0.05, 1.05), a2.set_xlabel("Week"), a2.legend(fontsize=6, ncol=3)
        ui.show_fig(fig)

    st.markdown("#### 3 · Forecasts and conformal ranges (L4.6)")
    res = _evaluate(df)
    if res.get("ok"):
        mae = res["mae"]
        names = {"season_mean": "season-to-date average (baseline)", "last_game": "last game",
                 "ewma": "EWMA (weight chosen on earlier seasons)", "hmm": "HMM (hot/cold)"}
        ui.table(pd.DataFrame({"forecaster": [names[m] for m in mae], "average error": [round(v, 3) for v in mae.values()],
                                   "80% conformal coverage": [f"{res['coverage'][m]:.1%}" if m in res["coverage"] else "–"
                                                              for m in mae]}), hide_index=True)
        st.caption("All forecasts use only earlier games; settings come from earlier seasons. "
                   "'Last game' is the worst: one game is mostly noise (L1.6).")

    st.markdown("#### 4 · Optional: a real LSTM (L4.4)")
    if not Q.torch_available():
        st.info(Q.TORCH_NOTE)
    elif st.button("Train a real LSTM and compare", key="p4_lstm"):
        mae, ntr, cut = _lstm(df)
        ui.table(pd.DataFrame({"forecaster": list(mae), "average error": [round(v, 3) for v in mae.values()]}),
                     hide_index=True)
        st.caption(f"Trained on {ntr} six-game windows before {cut}; tested on {cut} onward. "
                   "With this little data, don't expect it to beat the EWMA.")


def production_candidate(project):
    df, info = _load("game_logs", "p4prod")
    if st.button("Run season-by-season forecast test", type="primary", key="p4_eval"):
        st.session_state["p4_eval_args"] = info["label"]
    if st.session_state.get("p4_eval_args") != info["label"]:
        st.caption("Metric, baseline and gates are fixed in code before any data is evaluated.")
        return
    res = _evaluate(df)
    gl = Q.gates(res, info)
    verdict = Q.verdict(gl)
    _log(project, f"{verdict} [{info['label']}]")
    _gates(gl, verdict)
    if not res.get("ok"):
        return
    mae = res["mae"]
    m = st.columns(4)
    m[0].metric("Season-to-date avg (baseline)", f"{mae['season_mean']:.2f}")
    m[1].metric("EWMA", f"{mae['ewma']:.2f}")
    m[2].metric("HMM", f"{mae['hmm']:.2f}")
    m[3].metric(f"Best ({res['best']}) vs baseline", f"{res['gain']:+.1%}")
    ui.table(res["per_season"])
    st.caption("A gain under the bar is reported as it is. The honest conclusion may be "
               "'use the season-to-date average'.")
