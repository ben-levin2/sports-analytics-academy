import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from .. import config, curriculum, progress, theme as T, ui
from ..projects_data import PROJECTS


def _score_chart(state):
    attempts = sorted((a["ts"], a["score"]) for rec in state["lessons"].values() for a in rec["attempts"])
    if not attempts:
        st.caption("No quizzes taken yet. Your score history will appear here.")
        return
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ys = [s * 100 for _, s in attempts]
    ax.plot(range(1, len(ys) + 1), ys, marker="o", color=T.BLUE, lw=2)
    ax.axhline(config.COMPLETE_THRESHOLD * 100, color=T.AMBER, ls="--", lw=1, label="Pass (70%)")
    ax.axhline(config.MASTERY_THRESHOLD * 100, color=T.PRIMARY_LIGHT, ls=":", lw=1, label="Mastery (90%)")
    ax.set_ylim(0, 105), ax.set_xlabel("Quiz attempt #"), ax.set_ylabel("Score %")
    ax.legend(loc="lower right")
    ui.show_fig(fig)


def render():
    state = ui.get_state()
    info = st.session_state.get("load_info", {})
    ui.hero("Dashboard", "Sports Analytics Academy",
            "Statistics and machine learning, learned by applying them to football research questions.")

    if info.get("source") == "error":
        st.warning(info["message"])
    elif info.get("source") == "new" and not state["lessons"]:
        st.info("No saved progress found. If you studied here before and the app restarted, "
                "restore your backup on the **Progress & Backup** page.")

    s = curriculum.summary(state)
    n_concepts = len(curriculum.CONCEPTS)
    cols = st.columns(4)
    tiles = [
        ("Concepts mastered", f"{len(s['concepts_mastered'])} / {n_concepts}", f"{len(s['concepts_proficient'])} proficient or better"),
        ("Lessons completed", f"{len(s['completed'])} / {s['total']}", f"{len(s['mastered_lessons'])} at 90%+"),
        ("Current level", f"Level {s['level'].number}", s["level"].title),
        ("Learning streak", f"{s['streak']} day{'s' if s['streak'] != 1 else ''}", "quizzes, labs, project steps"),
    ]
    for c, (lbl, v, sub) in zip(cols, tiles):
        with c:
            ui.html(ui.tile(lbl, v, sub))

    left, right = st.columns([3, 2], gap="large")
    with right:
        ui.label("Recommended next")
        lesson, reason = curriculum.recommended(state)
        with st.container(border=True):
            if lesson:
                ui.html(ui.pill(f"Lesson {lesson.code_label}") + ui.status_pill(progress.lesson_status(state, lesson.id)))
                st.markdown(f"#### {lesson.title}")
                st.caption(reason)
                st.button("Open lesson", key="dash_open", type="primary",
                          on_click=ui.go, args=("Lessons", lesson.id))
            else:
                st.markdown(reason)
                st.button("Open the Level 1 project", key="dash_proj", on_click=ui.go,
                          args=("Projects", None, PROJECTS[0].id))
        ui.label("Recent activity")
        acts = list(reversed(state["activity"]))[:5]
        if not acts:
            st.caption("Nothing yet. Your quizzes, labs and project steps will show up here.")
        for a in acts:
            ui.html(f'<div class="aa-row"><span>{ui.esc(a["text"])}</span><span class="t">{ui.fmt_ts(a["ts"])}</span></div>')

    with left:
        ui.label("Your six levels")
        stats = curriculum.level_stats(state)
        for row in (stats[:3], stats[3:]):
            cs = st.columns(3)
            for c, ls in zip(cs, row):
                lv = ls["level"]
                n_c = len(ls["concepts"])
                cur = " cur" if lv.number == s["level"].number else ""
                c.markdown(f'<div class="aa-lv{cur}" style="min-height:8.4rem"><h4>LEVEL {lv.number}</h4>'
                       f'<div class="lt">{ui.esc(lv.title)}</div>'
                       + ui.bar(ls["mastered"] / n_c if n_c else 0)
                       + f'<div style="font-size:.74rem;color:#93A3BA">{ls["mastered"]}/{n_c} concepts mastered'
                         f'<br>{ls["done"]}/{ls["total"]} lessons completed</div></div>', unsafe_allow_html=True)
        st.button("Open the full roadmap", key="dash_roadmap", on_click=ui.go, args=("Learning Roadmap",))

    ui.label("Projects")
    rows = []
    for p in PROJECTS:
        rec = state["projects"].get(p.id, {})
        lab_steps = [k for k in rec.get("steps", []) if str(k).startswith("lab-")]
        runs = rec.get("runs", [])
        pc = ("Candidate" if runs[-1]["verdict"].startswith("Meets") else "Learning Lab only") if runs else "Not run"
        rows.append({"Project": f"{p.label} · {p.title}", "Level": p.level,
                     "Lab steps done": f"{len(lab_steps)} of {len(p.learning.steps)}", "Latest verdict": pc})
    ui.table(pd.DataFrame(rows))

    with st.expander("Quiz score history"):
        _score_chart(state)
