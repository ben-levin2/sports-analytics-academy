import streamlit as st

from .. import curriculum, progress, ui
from ..lessons import LOAD_ERRORS
from ..projects_data import PROJECTS
from .curriculum_map import render_map


def render():
    state = ui.get_state()
    ui.hero("Curriculum", "Learning Roadmap",
            "28 tracked concepts in six levels. Each level ends in a project with a Learning Lab and a Production Candidate track.")
    render_map(state)
    current = curriculum.current_level(state).number

    for ls in curriculum.level_stats(state):
        lv = ls["level"]
        tag = " · CURRENT" if lv.number == current else ""
        head = (f"Level {lv.number}: {lv.title}  ·  {ls['mastered']}/{len(ls['concepts'])} concepts mastered"
                f"  ·  {ls['done']}/{ls['total']} lessons{tag}")
        with st.expander(head, expanded=lv.number == current):
            st.caption(lv.tagline)
            ui.label("Concepts")
            for c in sorted(ls["concepts"], key=lambda c: c.num):
                status = curriculum.concept_status(state, c.id)
                score, n, _ = curriculum.concept_score(state, c.id)
                where = ", ".join(l.code_label for l in curriculum.lessons_for_concept(c.id)) or "lesson planned"
                extra = f" · latest {score:.0%} on {n} questions" if score is not None else ""
                st.markdown(f"**{c.num}. {c.name}** {ui.concept_pill(status)}  \n"
                            f"<span style='color:#8FA3BF;font-size:.85rem'>{ui.esc(c.summary)} · {where}{extra}</span>",
                            unsafe_allow_html=True)

            if ls["lessons"]:
                ui.label("Lessons")
            for l in ls["lessons"]:
                status = progress.lesson_status(state, l.id)
                c1, c2, c3 = st.columns([6, 2.2, 1.2])
                c1.markdown(f"**{l.code_label} · {l.title}**  \n"
                            f"<span style='color:#8FA3BF;font-size:.85rem'>{l.subtitle} · {l.minutes} min</span>",
                            unsafe_allow_html=True)
                best = (f"<span style='color:#8FA3BF;font-size:.8rem'>best {progress.best_score(state, l.id):.0%}</span>"
                        if state["lessons"].get(l.id, {}).get("attempts") else "")
                c2.markdown(ui.status_pill(status) + best, unsafe_allow_html=True)
                c3.button("Open", key=f"rm_{l.id}", on_click=ui.go, args=("Lessons", l.id))

            ui.label("Projects")
            for p in [p for p in PROJECTS if p.level == lv.number]:
                c1, c3 = st.columns([8.2, 1.2])
                built = "fully built" if p.status == "built" else "starter: requirements, gates, runnable Learning Lab"
                c1.markdown(f"**{p.label} · {p.title}**  \n<span style='color:#8FA3BF;font-size:.85rem'>{built}</span>",
                            unsafe_allow_html=True)
                c3.button("Open", key=f"rmp_{p.id}", on_click=ui.go, args=("Projects", None, p.id))

            comps = [c for c in curriculum.COMPARISONS if c[1] == lv.number]
            if comps:
                ui.label("Method comparisons taught here")
                st.markdown("\n".join(f"- {name} ({where})" for name, _, where in comps))

    if LOAD_ERRORS:
        st.error("Some lesson files failed to load:\n\n" + "\n".join(f"- {e}" for e in LOAD_ERRORS))

    with st.expander("How progress is measured"):
        st.markdown("""
**Lessons** (best attempt counts)
- *Opened*: you viewed it. Never counts as completion.
- *Completed*: best quiz score of 70% or more. *Mastered*: 90% or more.

**Concepts** (latest attempt counts, tracked separately)
- Every quiz question is tagged with one concept. A concept's score is your average on its questions in the **most recent** attempt of each lesson that teaches it.
- *Proficient* at 70%+. *Mastered* at 90%+, and only once every lesson teaching the concept has been attempted.
- A later, worse attempt can drop a concept from mastered back to proficient, even though the lesson stays completed.
""")
