"""Level 3C project: Small-Data Draft Model (concepts 29, 30, 31, 33, plus 25 and 26)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import json

from .. import data, forward as F, smalldata as SD, theme as T, ui
from .p3_level3 import _load, _log, _reliability

COLORS = {"plain": T.FAINT, "bayes": T.PRIMARY, "gam": T.ACCENT, "survival": T.NEUTRAL_LIGHT,
          "rating_only": T.PRIMARY_DARK}


@st.cache_data(show_spinner="Running the class-by-class competition…")
def _eval(df, boot, baseline="best"):
    return SD.evaluate(df, boot=boot, baseline=baseline)


@st.cache_data(show_spinner="Running the failure study (about 10–20 seconds)…")
def _failure():
    return SD.failure_study()


@st.cache_data(show_spinner=False)
def _sim(per_class, noise, seed):
    return data.make_draft_classes(per_class=per_class, noise=noise, seed=seed)


def _summary_table(res):
    s = res["summary"]
    rows = []
    for m in s.index:
        b = res["boot"].get(m)
        base = res.get("baseline", "plain")
        name = SD.NAMES[m] + (" (baseline)" if m == base else "")
        rows.append({"Model": name, "Test AUC": f"{s.loc[m, 'auc']:.3f}",
                     "AUC gain vs baseline": "baseline" if m == base else
                     ("—" if m in ("plain", "rating_only") else f"{s.loc[m, 'gain']:+.3f}"),
                     "Brier": f"{s.loc[m, 'brier']:.4f}", "Calibration error": f"{s.loc[m, 'ece']:.3f}",
                     "Bootstrap 90% range of gain": f"{b['lo90']:+.3f} to {b['hi90']:+.3f}" if b else "—"})
    t = pd.DataFrame(rows)
    ui.table(t)


# =================================================================== Learning Lab
def learning_lab(project):
    ui.sim_banner("This lab always uses simulated draft classes, so you can control their size and noise. "
                  "Your own CSV is used in the Production Candidate tab.")
    c1, c2, c3 = st.columns(3)
    per = c1.select_slider("Recruits per class", [10, 15, 22, 30, 45, 60], value=22, key="p3c_per")
    noise = c2.select_slider("Measurement noise on inputs (SD)", [0.0, 0.5, 1.0], value=0.0, key="p3c_noise")
    seed = c3.number_input("Program (random seed)", 1, 999, 29, key="p3c_seed")
    df = _sim(per, noise, int(seed))
    res = _eval(df, 300, "plain")

    ui.section(1, "The competition", "p3c-1")
    if not res.get("ok"):
        st.warning(res.get("reason"))
        return
    st.caption(f"{res['n_classes']} test classes ({res['n_test']} recruits, {res['n_drafted']} drafted). "
               f"Each class predicted one year after enrolling from earlier classes only. Bootstrap uses 300 "
               f"class resamples here (1,000 in the Production Candidate tab).")
    _summary_table(res)
    s = res["summary"]
    best = max(SD.CHALLENGERS, key=lambda m: s.loc[m, "gain"] if m in s.index else -9)
    if s.loc["rating_only", "auc"] >= s.loc[best, "auc"] - 0.005:
        st.info(f"Honest note: the rating-only reference ({s.loc['rating_only', 'auc']:.3f}) does about as well as the "
                f"best challenger ({s.loc[best, 'auc']:.3f}). Much of the 'gain' here is the plain baseline "
                "overfitting its position dummies, not the challengers being clever.")

    ui.section(2, "Calibration", "p3c-2")
    pick = st.radio("Model", [m for m in SD.MODELS if m in s.index], format_func=lambda m: SD.NAMES[m],
                    horizontal=True, index=list(SD.MODELS).index(best), key="p3c_cal")
    from .. import validation as V
    _reliability(V.reliability_table(res["rows"][pick], res["rows"]["y"]))
    st.caption(f"Calibration error {s.loc[pick, 'ece']:.3f} (the Production Candidate gate needs <= {SD.MAX_ECE}). "
               "With about 2 drafted players a class, each dot rests on very few picks.")

    ui.section(3, "What each model learned", "p3c-3")
    last = res["data_end"] - SD.WINDOW
    train = df[df["class_year"] <= last]
    if len(train) >= 40 and train["drafted"].sum() >= 3:
        d = SD.Design(train)
        y = train["drafted"].to_numpy()
        grid = np.linspace(-2, 2, 81)
        y1 = d.my + d.sy * grid
        truth = 2.4 / (1 + np.exp(-(y1 - 75) / 7)) - 2.4 / (1 + np.exp(-(d.my - 75) / 7))
        plain = SD.fit_logistic(d.linear(train), y, *d.priors(False))
        bayes = SD.fit_logistic(d.linear(train), y, *d.priors(True))
        gam = SD.fit_monotone_gam(d, train, y)
        fig, ax = plt.subplots(figsize=(7.5, 3.3))
        ax.plot(grid, truth, color=T.NEUTRAL_LIGHT, ls="--", lw=2, label="Truth (simulated)")
        ax.plot(grid, plain["beta"][2] * grid, color=T.FAINT, lw=1.8, label="Plain logistic")
        ax.plot(grid, bayes["beta"][2] * grid, color=T.PRIMARY, lw=2, label="Bayesian logistic")
        ax.plot(grid, SD.gam_curve(gam, "year1", grid), color=T.ACCENT, lw=2.4, label="Monotonic GAM")
        ax.set_xlabel("Year-1 production (SDs from average)"), ax.set_ylabel("Effect on draft log-odds")
        ax.set_title(f"Fitted on every finished class, through {last} ({len(train)} recruits, {int(y.sum())} drafted)",
                     fontsize=9)
        ui.show_fig(fig)
        m, sd = bayes["beta"][1], np.sqrt(bayes["cov"][1, 1])
        st.markdown(f"**Bayesian rating slope:** prior {SD.PRIOR['rating'][0]:.2f} ± {SD.PRIOR['rating'][1]:.2f} → "
                    f"posterior {m:.2f} ± {sd:.2f} (plain logistic: {plain['beta'][1]:.2f}).")
    if res["has_draft_years"]:
        k = pd.DataFrame({"Test class": [str(c) for c in sorted(res["rows"]["class_year"].unique())],
                          "Recruits, logistic models": res["train_rows_full"],
                          "Recruits, survival model": res["train_players_survival"]})
        st.markdown("**Training recruits per test class.** The survival model also learns from classes whose draft "
                    "window hadn't closed yet (concept 33).")
        ui.table(k)

    ui.section(4, "When it fails", "p3c-4")
    st.markdown("Rerun the whole competition on 4 simulated programs at each setting: first shrinking the classes, "
                "then adding noise to the inputs. No bootstrap here, just the average gain.")
    if st.button("Run the failure study", key="p3c_fail"):
        st.session_state["p3c_fail_run"] = True
    if st.session_state.get("p3c_fail_run"):
        f = _failure()
        agg = f.groupby(["study", "value", "model"]).agg(gain=("gain", "mean"), lo=("gain", "min"), hi=("gain", "max"),
                                                         auc=("auc", "mean"), plain=("plain_auc", "mean"),
                                                         ece=("ece", "mean")).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
        for ax, study in zip(axes, ["Recruits per class", "Input noise (SD)"]):
            g = agg[agg["study"] == study]
            for m in SD.CHALLENGERS:
                h = g[g["model"] == m]
                ax.plot(h["value"], h["gain"], marker="o", color=COLORS[m], lw=2, label=SD.NAMES[m])
                ax.fill_between(h["value"], h["lo"], h["hi"], color=COLORS[m], alpha=0.06)
            ax.axhline(SD.MIN_AUC_GAIN, color=T.MUTED, ls="--", lw=1)
            ax.axhline(0, color=T.BORDER, lw=1)
            ax.set_xlabel(study), ax.set_ylabel("AUC gain vs plain logistic")
        axes[0].legend(fontsize=7)
        ui.show_fig(fig)
        lv = agg[agg["model"] == "bayes"].set_index(["study", "value"])
        small = lv.loc[("Recruits per class", 10)]
        big = lv.loc[("Recruits per class", 90)]
        noisy = lv.loc[("Input noise (SD)", 1.0)]
        clean = lv.loc[("Input noise (SD)", 0.0)]
        st.markdown(f"""
- **Too few rows (10 a class):** every model collapses. The plain baseline's AUC is {small['plain']:.2f} and the Bayesian model's {small['auc']:.2f}; gains range from {small['lo']:+.3f} to {small['hi']:+.3f} across programs, and calibration error is {small['ece']:.2f}. Nothing can be trusted.
- **Plenty of rows (90 a class):** the plain model stops overfitting, so the gain shrinks to {big['gain']:+.3f}. Priors matter most in the middle.
- **Too noisy (1 SD of error):** every model gets worse (Bayesian AUC {clean['auc']:.2f} → {noisy['auc']:.2f}). The *gain* can look bigger only because the baseline gets worse faster. A bigger gain over a weaker baseline is not a better model.
- Dashed line: the +{SD.MIN_AUC_GAIN:.2f} bar the Production Candidate test requires.
""")


# =================================================================== Production Candidate
def production_candidate(project):
    back, fwd = st.tabs(["Backtest", "Forward test"])
    with back:
        _backtest(project)
    with fwd:
        forward_test(project)


def _backtest(project):
    df, info = _load("draft_outcomes", "p3cprod")
    if "draft_year" not in df.columns or not df["draft_year"].notna().any():
        st.info("No draft_year column: the survival model can't run, and every class is treated as finished. "
                "Only upload classes whose 5-year draft window has closed, or add draft_year.")
    if st.button("Run class-by-class evaluation", type="primary", key="p3c_eval"):
        st.session_state["p3c_eval_args"] = info["label"]
    if st.session_state.get("p3c_eval_args") != info["label"]:
        st.caption("Baseline, metric, priors and gates are fixed in code before any data is evaluated. "
                   "Takes about 5–10 seconds (1,000 bootstrap resamples).")
        return
    res = _eval(df, SD.BOOT_B)
    lists = {m: SD.gates(res, info, m) for m in SD.CHALLENGERS}
    passed = [m for m, gl in lists.items() if SD.verdict(gl).startswith("Meets")]
    overall = ("Meets Production Candidate criteria for: " + ", ".join(SD.NAMES[m] for m in passed)
               + " (review before any use)") if passed else "Learning Lab only"
    _log(project, f"{overall} [{info['label']}]")
    ui.html(f'<div class="aa-verdict {"ok" if passed else "no"}">Verdict: {ui.esc(overall)}</div>')
    if not res.get("ok"):
        ui.gates(lists["bayes"])
        return
    st.markdown(f"**Baseline used:** {SD.NAMES[res['baseline']]}, the stronger of plain logistic "
                f"(AUC {res['summary'].loc['plain', 'auc']:.3f}) and rating only "
                f"(AUC {res['summary'].loc['rating_only', 'auc']:.3f}). Gains below are measured against it.")
    _summary_table(res)
    st.caption(f"{res['n_classes']} test classes, {res['n_test']} recruits, {res['n_drafted']} drafted. "
               f"Three challengers are tested, so the bootstrap bar is split three ways ({SD.BOOT_ALPHA:.2%} each).")
    tabs = st.tabs([SD.NAMES[m] for m in SD.CHALLENGERS])
    for tab, m in zip(tabs, SD.CHALLENGERS):
        with tab:
            ui.gates(lists[m], SD.verdict(lists[m]))
    s = res["summary"]
    if s.loc["rating_only", "auc"] > s.loc["plain", "auc"] + SD.MIN_AUC_GAIN:
        st.warning(f"The rating-only reference (AUC {s.loc['rating_only', 'auc']:.3f}) beats the plain baseline "
                   f"({s.loc['plain', 'auc']:.3f}) by more than the bar. Beating this baseline is easier than it "
                   "looks; check each challenger against rating only too before believing a pass. "
                   "(The gates above already use the stronger baseline.)")
    per = res["rows"].groupby("class_year").agg(recruits=("y", "size"), drafted=("y", "sum")).reset_index()
    per["class_year"] = per["class_year"].astype(str)
    ui.table(per, rename={"class_year": "Class", "recruits": "Recruits", "drafted": "Drafted"})


# =================================================================== Forward test (concept 36)
def freeze_forecast(per_class: int, seed: int) -> dict:
    """Predict the simulated 2022 class as of spring 2023 and freeze predictions + plan (simulation settings inside)."""
    preds = F.simulate_forecast(per_class=per_class, seed=seed)
    plan = F.make_plan(2022, written="spring 2023 (simulated)")
    plan["simulation"] = {"per_class": int(per_class), "seed": int(seed)}
    return F.freeze(preds, plan)


def score_forecast(frozen: dict, observed_through: int) -> dict:
    if not F.verify(frozen):
        return {"ok": False, "reason": "Hash does not match: the frozen predictions or plan were changed."}
    _, plan = F.unpack(frozen)
    sim = plan["simulation"]
    out = F.simulate_outcomes(per_class=sim["per_class"], seed=sim["seed"], class_year=plan["class_year"],
                              observed_through=observed_through)
    return F.score(frozen, out, observed_through)


def forward_test(project):
    ui.sim_banner("Forward tests here use simulated outcomes: the 2022 class is predicted as of spring 2023, "
                  "and drafts are revealed only when you score.")
    st.markdown("**1 · Freeze.** Predictions and the scoring plan are hashed together before any outcome exists.")
    c1, c2 = st.columns(2)
    per = c1.select_slider("Recruits in the class", [22, 60, 120], value=120, key="p3c_fwd_per")
    seed = c2.number_input("Simulated program (seed)", 1, 999, 36, key="p3c_fwd_seed")
    if st.button("Freeze predictions", type="primary", key="p3c_fwd_freeze"):
        st.session_state["p3c_frozen"] = freeze_forecast(int(per), int(seed))
        st.session_state.pop("p3c_fwd_score", None)
    up = st.file_uploader("…or load a forecast you froze earlier (JSON)", type="json", key="p3c_fwd_up")
    if up is not None:
        try:
            st.session_state["p3c_frozen"] = json.loads(up.getvalue().decode("utf-8"))
        except Exception:
            st.error("That file isn't a frozen forecast.")
    fz = st.session_state.get("p3c_frozen")
    if not fz:
        return
    preds, plan = F.unpack(fz)
    st.code(f"SHA-256: {fz['hash']}\nClass {plan['class_year']}: {len(preds)} predictions, mean {preds['p'].mean():.3f}\n"
            f"Target: {plan['target']} (window ends {plan['window_end']})\nPlan: {plan['rule']}", language="text")
    st.download_button("Download the frozen forecast", json.dumps(fz), file_name=f"forecast_{fz['hash'][:10]}.json",
                       mime="application/json", key="p3c_fwd_dl")
    st.markdown("**2 · Score later**, exactly as the frozen plan says.")
    when = st.select_slider("Score after the draft of", [2025, 2026, 2027], value=2027, key="p3c_fwd_when")
    if st.button("Reveal outcomes and score", key="p3c_fwd_go"):
        st.session_state["p3c_fwd_score"] = (when, score_forecast(fz, when))
    got = st.session_state.get("p3c_fwd_score")
    if not got:
        return
    when, res = got
    if not res["ok"]:
        st.error(res["reason"])
        return
    _log(project, f"Forward test scored after {when} [hash {fz['hash'][:10]}]")
    m = st.columns(3)
    m[0].metric("Window complete?", "Yes" if res["complete"] else "No", f"ends {plan['window_end']}", delta_color="off")
    m[1].metric("AUC (scored)", f"{res['auc']:.3f}")
    if "ece" in res:
        m[2].metric("Calibration error (scored)", f"{res['ece']:.3f}")
    else:
        m[2].metric("Calibration", "not scored", f"window incomplete", delta_color="off")
        st.info(f"Only {res['observed_rate']:.3f} drafted so far vs a forecast of {res['mean_forecast']:.3f} for the full "
                f"window. Calibration would read {res['ece_if_misused']:.3f}, but that's the window mismatch, not the "
                "model: the plan scores ranking only until the window closes.")
    st.caption("Simulated forward test: it practices the procedure and can never count toward a Production Candidate verdict.")
