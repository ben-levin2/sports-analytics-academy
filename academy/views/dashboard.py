import matplotlib.pyplot as plt
import streamlit as st

from .. import config, curriculum, progress, theme as T, ui
from ..projects_data import PROJECTS
from .curriculum_map import render_map


def _score_chart(state):
    attempts = sorted((a["ts"], a["score"]) for rec in state["lessons"].values() for a in rec["attempts"])
    if not attempts:
        st.caption("No quizzes taken yet. Your score history will appear here.")
        return
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ys = [s * 100 for _, s in attempts]
    ax.plot(range(1, len(ys) + 1), ys, marker="o", color=T.BLUE, lw=2)
    ax.axhline(config.COMPLETE_THRESHOLD * 100, color=T.AMBER, ls="--", lw=1, label="Pass (70%)")
    ax.axhline(config.MASTERY_THRESHOLD * 100, color=T.GREEN, ls=":", lw=1, label="Mastery (90%)")
    ax.set_ylim(0, 105), ax.set_xlabel("Quiz attempt #"), ax.set_ylabel("Score %")
    ax.legend(loc="lower right")
    ui.show_fig(fig)


def render():
    state = ui.get_state()
    info = st.session_state.get("load_info", {})
    ui.hero("Research Dashboard", "Sports Analytics Academy",
            "Statistics, machine learning and modeling, learned by applying them to football research questions.")

    if info.get("source") == "error":
        st.warning(info["message"])
    elif info.get("source") == "new" and not state["lessons"]:
        st.info("No saved progress found. If you studied here before and the app restarted, "
                "restore your backup on the **Progress & Backup** page.")

    s = curriculum.summary(state)
    n_concepts = len(curriculum.CONCEPTS)
    cols = st.columns(5)
    tiles = [
        ("Lessons completed", f"{len(s['completed'])}/{s['total']}", f"{len(s['mastered_lessons'])} at 90%+ · best score counts"),
        ("Concepts mastered", f"{len(s['concepts_mastered'])}/{n_concepts}", f"{len(s['concepts_proficient'])} proficient · latest answers count"),
        ("Current level", f"Level {s['level'].number}", s["level"].title),
        ("Curriculum built", f"{len(s['concepts_built'])}/{n_concepts}", "concepts with lessons so far"),
        ("Learning streak", f"{s['streak']} day{'s' if s['streak'] != 1 else ''}", "quizzes, labs and project steps"),
    ]
    for c, (lbl, v, sub) in zip(cols, tiles):
        with c:
            ui.html(ui.tile(lbl, v, sub))
    ui.html(ui.bar(len(s["concepts_mastered"]) / n_concepts, big=True))

    ui.label("Curriculum map")
    render_map(state)

    left, right = st.columns([3, 2], gap="large")
    with right:
        ui.label("Recommended next")
        lesson, reason = curriculum.recommended(state)
        with st.container(border=True):
            if lesson:
                ui.html(ui.pill(lesson.code_label) + ui.status_pill(progress.lesson_status(state, lesson.id)))
                st.markdown(f"#### {lesson.title}")
                st.caption(reason)
                st.button("Open lesson →", key="dash_open", type="primary",
                          on_click=ui.go, args=("Lessons", lesson.id))
            else:
                st.markdown(reason)
                st.button("Open the Level 1 project →", key="dash_proj", on_click=ui.go,
                          args=("Projects", None, PROJECTS[0].id))

        ui.label("Recent activity")
        acts = list(reversed(state["activity"]))[:8]
        if not acts:
            st.caption("Nothing yet.")
        for a in acts:
            ui.html(f'<div class="aa-row"><span>{ui.esc(a["text"])}</span><span class="t">{ui.fmt_ts(a["ts"])}</span></div>')

    with left:
        ui.label("Projects")
        for p in PROJECTS:
            rec = state["projects"].get(p.id, {})
            lab_steps = [k for k in rec.get("steps", []) if str(k).startswith("lab-")]
            runs = rec.get("runs", [])
            pc = runs[-1]["verdict"] if runs else ("not run yet" if p.status == "built" else "planned")
            ui.html(f'<div class="aa-row"><span><b>{p.label}</b> · {ui.esc(p.title)}</span>'
                    f'<span class="t">LAB {len(lab_steps)}/{len(p.learning.steps)} · PC: {ui.esc(pc)}</span></div>')
        st.write("")
        ui.label("Quiz score history")
        _score_chart(state)
