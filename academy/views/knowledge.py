import pandas as pd
import streamlit as st

from .. import curriculum, progress, ui
from ..lessons import all_lessons


def render():
    state = ui.get_state()
    ui.hero("Assessment", "Knowledge Checks",
            "Quiz performance across lessons. Quizzes live at the end of each lesson.")
    lessons = all_lessons()

    st.markdown("### Concept mastery (latest attempts)")
    crow = []
    for c in curriculum.CONCEPTS:
        score, n, _ = curriculum.concept_score(state, c.id)
        crow.append({"#": c.num, "Concept": c.name, "Level": c.level,
                     "Status": curriculum.concept_status(state, c.id).title(),
                     "Latest score": f"{score:.0%}" if score is not None else "–", "Questions": n,
                     "Taught in": ", ".join(l.code_label for l in curriculum.lessons_for_concept(c.id)) or "planned"})
    ui.table(pd.DataFrame(crow))

    st.markdown("### Lessons (best attempt)")
    rows = []
    for l in lessons:
        attempts = state["lessons"].get(l.id, {}).get("attempts", [])
        rows.append({
            "Lesson": f"{l.code_label} · {l.title}",
            "Status": progress.lesson_status(state, l.id).title(),
            "Attempts": len(attempts),
            "Best": f"{progress.best_score(state, l.id):.0%}" if attempts else "–",
            "Last": f"{attempts[-1]['score']:.0%}" if attempts else "–",
            "Last taken": ui.fmt_ts(attempts[-1]["ts"]) if attempts else "–",
        })
    ui.table(pd.DataFrame(rows))

    st.markdown("### Review list")
    st.caption("Questions you didn't fully get right on your most recent attempt.")
    any_missed = False
    for l in lessons:
        attempts = state["lessons"].get(l.id, {}).get("attempts", [])
        if not attempts:
            continue
        missed = [q for q in l.quiz if attempts[-1]["per_q"].get(q.id, 0) < 1]
        if not missed:
            continue
        any_missed = True
        with st.container(border=True):
            st.markdown(f"**{l.code_label} · {l.title}**")
            for q in missed:
                st.markdown(f"- {q.prompt}  \n  <span style='color:#93A3BA;font-size:.85rem'>{q.category} · {q.kind}</span>",
                            unsafe_allow_html=True)
            st.button("Retake quiz", key=f"kc_{l.id}", on_click=ui.go, args=("Lessons", l.id))
    if not any_missed:
        st.caption("Nothing to review. Either you haven't taken a quiz yet, or your last attempts were perfect.")
