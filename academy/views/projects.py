import streamlit as st

from .. import curriculum, datasets, progress, ui
from ..projects_data import PROJECTS
from . import p1_bayes

INTERACTIVE = {"p1-bayesian-player-evaluation": (p1_bayes.learning_lab, p1_bayes.production_candidate)}


def _toggle(p, key, label):
    state = ui.get_state()
    progress.set_project_step(state, p.id, p.title, key, st.session_state[f"proj_{p.id}_{key}"], label)


def _steps(p, prefix, steps, rec):
    for i, step in enumerate(steps):
        key = f"{prefix}-{i}"
        st.checkbox(f"{i + 1}. {step}", value=key in rec["steps"], key=f"proj_{p.id}_{key}",
                    on_change=_toggle, args=(p, key, step))


def _dataset(p):
    spec = datasets.SPECS.get(p.dataset_id)
    if not spec:
        return
    src = datasets.status(spec.id)
    where = {"upload": "your uploaded CSV", "path": "a configured file path"}.get(src["source"], "SIMULATED data")
    st.markdown(f"**{spec.name}**: {spec.description}  \nCurrently using: **{where}**.")
    st.markdown("| Column | Meaning | |\n|---|---|---|\n"
                + "\n".join(f"| `{c}` | {m} | required |" for c, m in spec.required.items())
                + "".join(f"\n| `{c}` | {m} | optional |" for c, m in spec.optional.items()))
    st.button("Upload or replace this dataset →", key=f"ds_{p.id}", on_click=ui.go, args=("Data Sources",))


def render():
    state = ui.get_state()
    ui.hero("Research Projects", "Projects",
            "Each project has a Learning Lab (learn the method) and a Production Candidate track (prove it works).")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.markdown("**🧪 Learning Lab**  \nExploratory. Teaches a method. Simulated data allowed. "
                    "Doesn't need to beat anything. Limitations stated.")
        c2.markdown("**🏭 Production Candidate**  \nReal data, a defined baseline, a predefined metric, time-respecting "
                    "validation, beats the baseline, checked for leakage and calibration. "
                    "Never labelled production-ready just because it runs.")

    by_id = {p.id: p for p in PROJECTS}
    if st.session_state.get("proj_pick") not in by_id:
        st.session_state["proj_pick"] = PROJECTS[0].id
    p = by_id[st.selectbox("Project", list(by_id), key="proj_pick",
                           format_func=lambda i: f"{by_id[i].label} · Level {by_id[i].level} · {by_id[i].title}"
                                                 + ("" if by_id[i].status == "built" else "  (starter)"))]
    rec = progress.project_rec(state, p.id) if p.id in state["projects"] else {"steps": [], "notes": ""}

    st.markdown(f"## {p.label} · {p.title}")
    concept_pills = "".join(ui.pill(f"#{curriculum.CONCEPT_BY_ID[c].num} {curriculum.CONCEPT_BY_ID[c].name}", "gray")
                            for c in p.concept_ids if c in curriculum.CONCEPT_BY_ID)
    ui.html(ui.pill(f"LEVEL {p.level}") + ui.pill("FULLY BUILT" if p.status == "built" else "STARTER",
                                                  "green" if p.status == "built" else "gray") + concept_pills)
    st.markdown(f"**Objective:** {p.objective}")
    with st.expander("PSFAL connection", expanded=p.status == "built"):
        st.markdown(p.psfal)
    with st.expander("Dataset requirements"):
        _dataset(p)

    lab_tab, prod_tab = st.tabs(["🧪 Learning Lab", "🏭 Production Candidate"])
    with lab_tab:
        ui.html(f'<div class="aa-track lab"><span class="aa-label">Learning Lab</span><br>{ui.esc(p.learning.summary)}</div>')
        _steps(p, "lab", p.learning.steps, rec)
        if p.learning.limitations:
            with st.container(border=True):
                ui.label("Limitations")
                st.markdown("\n".join(f"- {x}" for x in p.learning.limitations))
        if p.id in INTERACTIVE:
            INTERACTIVE[p.id][0](p)
        elif p.learning.code:
            ui.sim_banner("Starter code uses fictional data from academy/data.py.")
            ui.code_runner(p.learning.code, key=f"projcode_{p.id}")

    with prod_tab:
        pr = p.production
        ui.html(f'<div class="aa-track prod"><span class="aa-label">Production Candidate</span><br>{ui.esc(pr.summary)}</div>')
        st.markdown(f"- **Baseline:** {pr.baseline}\n- **Predefined metric:** {pr.metric}\n- **Validation:** {pr.validation}")
        if p.id in INTERACTIVE:
            INTERACTIVE[p.id][1](p)
        else:
            st.info("Automated gate checks for this project are not built yet, so it stays a **Learning Lab** "
                    "regardless of the checklist below. The checklist is your plan, not a verdict.")
            ui.label("Gates this project will have to pass")
            st.markdown("\n".join(f"- {g}" for g in pr.gates))
            ui.label("Plan")
            _steps(p, "prod", pr.steps, rec)

    ui.section("Q", "Final research questions")
    st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(p.questions, start=1)))
    st.markdown("#### Research notes")
    notes = st.text_area("Notes", value=rec.get("notes", ""), key=f"notes_{p.id}", height=130,
                         label_visibility="collapsed", placeholder="Findings, decisions, open questions…")
    if st.button("Save notes", key=f"save_notes_{p.id}"):
        progress.set_project_notes(state, p.id, notes)
        st.toast("Notes saved")
