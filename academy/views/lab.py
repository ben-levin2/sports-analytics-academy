import streamlit as st

from .. import progress, runner, ui
from ..labs import EXERCISES


def _toggle(ex):
    state = ui.get_state()
    progress.set_lab_done(state, ex.id, ex.title, st.session_state[f"lab_done_{ex.id}"])


def _reset(key, code):
    st.session_state[key] = code


def render():
    state = ui.get_state()
    ui.hero("Python Lab", "Learn by writing code",
            "NumPy, pandas, Matplotlib and scikit-learn on simulated college-football data. Beginner to advanced.")
    ui.sim_banner("Every dataset in the Lab comes from academy/data.py: fictional players generated with NumPy. "
                  "Some include hidden 'true' values, which real data never gives you.")

    done = sum(progress.lab_done(state, e.id) for e in EXERCISES)
    ui.html(f'<div class="aa-label">{done} of {len(EXERCISES)} exercises completed</div>' + ui.bar(done / len(EXERCISES)))

    by_id = {e.id: e for e in EXERCISES}
    pick = st.selectbox("Exercise", list(by_id), key="lab_pick",
                        format_func=lambda i: f"{'Done · ' if progress.lab_done(state, i) else ''}"
                                              f"[{by_id[i].level}] {by_id[i].title}")
    ex = by_id[pick]

    st.markdown(f"### {ex.title}")
    ui.html(ui.pill(ex.level, {"Beginner": "gray", "Intermediate": "blue", "Advanced": "amber"}[ex.level])
            + "".join(ui.pill(c, "gray") for c in ex.concepts))
    st.markdown(f"**Goal:** {ex.goal}")

    out_key = f"lab_out_{ex.id}"
    code_key = f"lab_code_{ex.id}"
    if ui.can_edit_code():
        st.caption("Fill in the TODOs, then run. The self-check lines at the bottom tell you when you're right.")
        if code_key not in st.session_state:
            st.session_state[code_key] = ex.starter
        code = st.text_area("Code editor", key=code_key, height=420, label_visibility="collapsed")
        c1, c2, _ = st.columns([1, 1, 3])
        if c1.button("Run", key=f"lab_run_{ex.id}", type="primary"):
            st.session_state[out_key] = runner.run_code(code)
        c2.button("Reset to starter", key=f"lab_reset_{ex.id}", on_click=_reset, args=(code_key, ex.starter))
    else:
        st.info("Code editing is turned off on this deployment (it would let anyone run code on the server). "
                "Run the app locally, or set the secret `allow_code_edit = true` on a private app.")
        st.code(ex.starter, language="python")
        if st.button("Run starter", key=f"lab_run_{ex.id}"):
            st.session_state[out_key] = runner.run_code(ex.starter)

    if out_key in st.session_state:
        ui.show_output(st.session_state[out_key])

    with st.expander("Hint"):
        st.markdown(ex.hint)
    with st.expander("Solution"):
        st.code(ex.solution, language="python")
        if st.button("Run solution", key=f"lab_sol_{ex.id}"):
            st.session_state[out_key] = runner.run_code(ex.solution)
            st.rerun()

    st.checkbox("I completed this exercise (all self-checks pass)", value=progress.lab_done(state, ex.id),
                key=f"lab_done_{ex.id}", on_change=_toggle, args=(ex,))
