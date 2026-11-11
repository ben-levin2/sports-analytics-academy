"""Shared Streamlit UI helpers."""
from __future__ import annotations

import html as _html
import os
from datetime import datetime

import streamlit as st

from . import config, progress, runner


# ------------------------------------------------------------------ state
def get_state() -> dict:
    """Progress lives on disk; session_state is only a per-session cache of it."""
    if "progress" not in st.session_state:
        st.session_state["progress"], st.session_state["load_info"] = progress.load()
    return st.session_state["progress"]


def go(page: str, lesson_id: str | None = None, project_id: str | None = None) -> None:
    """on_click callback: jump to a page (optionally a specific lesson or project)."""
    st.session_state["nav"] = page
    if lesson_id:
        st.session_state["lesson_pick"] = lesson_id
    if project_id:
        st.session_state["proj_pick"] = project_id


def can_edit_code() -> bool:
    """Editable code = arbitrary code execution. On by default locally, off on Community Cloud
    unless you set ACADEMY_ALLOW_CODE_EDIT=1 or the secret allow_code_edit = true."""
    if os.environ.get("ACADEMY_ALLOW_CODE_EDIT") == "1":
        return True
    try:
        if bool(st.secrets.get("allow_code_edit", False)):
            return True
    except Exception:
        pass
    return not config.IS_CLOUD


# ------------------------------------------------------------------ html bits
def esc(s) -> str:
    return _html.escape(str(s))


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%b %d, %H:%M")
    except (TypeError, ValueError):
        return str(ts)


def label(text: str) -> None:
    html(f'<div class="aa-label">{esc(text)}</div>')


def hero(kicker: str, title: str, sub: str = "") -> None:
    html(f'<div class="aa-hero"><div class="aa-label">{esc(kicker)}</div>'
         f'<h1>{esc(title)}</h1>' + (f'<p>{esc(sub)}</p>' if sub else "") + '</div>')


def bar(frac: float, big: bool = False) -> str:
    frac = max(0.0, min(1.0, frac))
    return f'<div class="aa-bar{" big" if big else ""}"><div style="width:{frac * 100:.0f}%"></div></div>'


def tile(lbl: str, value: str, sub: str = "") -> str:
    return (f'<div class="aa-tile"><div class="aa-label">{esc(lbl)}</div>'
            f'<div class="v">{esc(value)}</div><div class="s">{esc(sub)}</div></div>')


def pill(text: str, kind: str = "blue") -> str:
    return f'<span class="aa-pill {kind}">{esc(text)}</span>'


CONCEPT_PILL = {
    "planned": ("Planned", "gray"), "new": ("Not started", "gray"), "learning": ("Learning", "amber"),
    "proficient": ("Proficient", "blue"), "mastered": ("Mastered", "green"),
}


def concept_pill(status: str) -> str:
    text, kind = CONCEPT_PILL[status]
    return pill(text, kind)


STATUS_PILL = {
    "new": ("Not started", "gray"), "opened": ("Opened", "gray"),
    "review": ("Needs review", "amber"), "completed": ("Completed", "blue"), "mastered": ("Mastered", "green"),
}


def status_pill(status: str) -> str:
    text, kind = STATUS_PILL[status]
    return pill(text, kind)


def section(num: str, title: str, anchor: str | None = None) -> None:
    aid = f' id="{esc(anchor)}"' if anchor else ""
    html(f'<div class="aa-sec"{aid}><span class="n">{esc(num)}</span><span class="t">{esc(title)}</span></div>')


def step_nav(steps) -> None:
    """Row of jump links: steps = [(anchor, label), ...]."""
    html('<div class="aa-steps">' + "".join(f'<a href="#{esc(a)}" target="_self">{esc(l)}</a>' for a, l in steps) + "</div>")


# ------------------------------------------------------------------ tables
FRIENDLY = {
    "player_id": "Player", "season": "Season", "team": "Team", "position": "Position", "week": "Week",
    "successes": "Successes", "opportunities": "Opportunities", "raw_rate": "Raw rate", "prior_mean": "Prior mean",
    "post_mean": "Posterior mean", "ci_low": "95% low", "ci_high": "95% high", "weight_on_data": "Weight on own data",
    "test_rows": "Player-seasons tested", "raw": "Raw past rate", "population": "Population mean", "model": "Model",
    "coverage": "Interval coverage", "same_as_last": "Same as last season", "baseline": "Baseline",
    "archetypes": "With archetypes", "class_year": "Class", "recruits": "Recruits",
    "contributed_by_2": "Contributed by year 2", "cox": "Cox model", "players": "Players", "forecasts": "Forecasts",
    "season_mean": "Season-to-date avg", "last_game": "Last game", "ewma": "EWMA", "hmm": "HMM",
    "level1": "Level 1 shrinkage", "hierarchical": "Hierarchical", "optimizer": "Optimizer (realized)",
    "greedy": "Greedy (realized)", "random_median": "Median random lineup", "optimizer_projected": "Optimizer (projected)",
    "cap_hit_m": "Cap hit ($M)", "projected_value": "Projected value", "value_sd": "Uncertainty",
    "realized_value": "Realized value", "mean_production": "Avg production", "std_error": "Std. error",
    "players_per_density": "Players per density unit", "parameter": "Parameter", "mean": "Mean", "sd": "SD",
    "ESS": "Effective draws", "R-hat": "R-hat", "recruit_rating": "Recruit rating", "snaps": "Snaps",
}
PERCENT_COLS = {"raw_rate", "post_mean", "ci_low", "ci_high", "prior_mean", "weight_on_data", "coverage",
                "contributed_by_2", "Interval coverage"}


def _smart(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    if x != x:
        return "–"
    a = abs(x)
    if a >= 100:
        return f"{x:,.0f}"
    if a >= 10:
        return f"{x:.1f}"
    if a >= 1:
        return f"{x:.2f}"
    if a == 0:
        return "0"
    return f"{x:.3g}"


def table(df, rename: dict | None = None, percent=(), hide_index: bool = True, height=None, keep_index_label=None):
    """Display a table with friendly column names and sensible decimals."""
    import pandas as pd
    d = df.copy()
    if not hide_index and keep_index_label:
        d.index.name = keep_index_label
    pct = set(percent) | {c for c in d.columns if c in PERCENT_COLS}
    names = {c: (rename or {}).get(c, FRIENDLY.get(c, str(c).replace("_", " ").capitalize() if "_" in str(c) else c))
             for c in d.columns}
    fmt = {}
    for c in d.columns:
        if c in pct:
            fmt[names[c]] = lambda v: "–" if v != v else f"{float(v):.1%}"
        elif pd.api.types.is_integer_dtype(d[c]):
            fmt[names[c]] = "{:,}"
        elif pd.api.types.is_float_dtype(d[c]):
            fmt[names[c]] = _smart
    d = d.rename(columns=names)
    kw = {"hide_index": hide_index}
    if height:
        kw["height"] = height
    st.dataframe(d.style.format(fmt), **kw)


# ------------------------------------------------------------------ verdicts & gates
def gates(gate_list, verdict: str | None = None) -> None:
    if verdict:
        ok = verdict.startswith("Meets")
        html(f'<div class="aa-verdict {"ok" if ok else "no"}">Verdict: {esc(verdict)}</div>')
    html("".join(f'<div class="aa-gate"><span class="st {"pass" if ok else "fail"}">{"PASS" if ok else "FAIL"}</span>'
                 f'<span><b>{esc(name)}</b><br><span class="d">{esc(detail)}</span></span></div>'
                 for name, ok, detail in gate_list))


def sim_banner(detail: str = "Every player and number here is fictional, generated by code. "
                             "Nothing below is a real football statistic.") -> None:
    html(f'<div class="aa-sim"><b>Simulated data.</b> {esc(detail)}</div>')


def _direct_labels(ax) -> None:
    """Replace the legend with labels at the end of each line, when that is unambiguous:
    2-5 labelled lines, each with 5+ points, and nothing else in the legend."""
    import numpy as np
    leg = ax.get_legend()
    if leg is None:
        return
    handles, labels = ax.get_legend_handles_labels()
    lines = [l for l in ax.get_lines() if not l.get_label().startswith("_")]
    if not 2 <= len(lines) <= 5 or len(handles) != len(lines) or any(len(l.get_xdata()) < 5 for l in lines):
        return
    ends = []
    for l in lines:
        x, y = np.asarray(l.get_xdata(), float), np.asarray(l.get_ydata(), float)
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            return
        ends.append([x[ok][-1], y[ok][-1], l])
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.07
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):                      # nudge labels apart so they don't overlap
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
    for x, y, l in ends:
        ax.annotate(l.get_label(), (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=l.get_color(), annotation_clip=False)
    leg.remove()


def show_fig(fig) -> None:
    """Every chart goes through here: one style, direct labels where clear, tight layout."""
    import matplotlib.pyplot as plt
    for ax in fig.get_axes():
        try:
            _direct_labels(ax)
        except Exception:
            pass
    try:
        fig.tight_layout()
    except Exception:
        pass
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------ code execution UI
def show_output(res: dict) -> None:
    if res.get("error"):
        st.error("The code raised an error:")
        st.code(res["error"], language="text")
    if res.get("text"):
        st.code(res["text"].rstrip(), language="text")
    for img in res.get("images", []):
        st.image(img)
    if not res.get("error") and not res.get("text") and not res.get("images"):
        st.caption("The code ran but printed nothing.")


def code_runner(code: str, key: str, height: int = 360) -> None:
    """Show code, a Run button, and (when allowed) an editable copy."""
    st.code(code, language="python")
    out_key = f"{key}_out"
    if st.button("Run this code", key=f"{key}_run"):
        st.session_state[out_key] = runner.run_code(code)
    if can_edit_code():
        with st.expander("Edit and run your own version"):
            edited = st.text_area("Your code", value=code, height=height, key=f"{key}_edit",
                                  label_visibility="collapsed")
            if st.button("Run edited code", key=f"{key}_run_edit"):
                st.session_state[out_key] = runner.run_code(edited)
    if out_key in st.session_state:
        show_output(st.session_state[out_key])
