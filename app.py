"""Sports Analytics Academy: entry point.  Run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Sports Analytics Academy", page_icon=":material/sports_football:", layout="wide",
                   initial_sidebar_state="expanded")

from academy import config, curriculum, progress, theme, ui  # noqa: E402
from academy.views import backup, dashboard, data_sources, knowledge, lab, lesson, projects, roadmap  # noqa: E402

PAGES = {
    "Dashboard": dashboard,
    "Learning Roadmap": roadmap,
    "Lessons": lesson,
    "Python Lab": lab,
    "Knowledge Checks": knowledge,
    "Projects": projects,
    "Data Sources": data_sources,
    "Progress & Backup": backup,
}

theme.apply_mpl_style()
ui.html(theme.css_block())
st.session_state.setdefault("nav", "Dashboard")
state = ui.get_state()

with st.sidebar:
    ui.html('<div class="aa-brand">Sports Analytics <span>Academy</span></div>'
            '<div style="font-size:.78rem;color:#93A3BA;margin-top:2px">Personal research academy</div>')
    st.write("")
    st.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed")
    st.divider()
    s = curriculum.summary(state)
    ui.html(f'<div style="font-size:.8rem;color:#E7ECF4;font-weight:500">Level {s["level"].number} · {ui.esc(s["level"].title)}</div>'
            + ui.bar(len(s["concepts_mastered"]) / len(curriculum.CONCEPTS))
            + f'<div style="font-size:.76rem;color:#93A3BA">{len(s["concepts_mastered"])} of {len(curriculum.CONCEPTS)} concepts mastered<br>'
              f'{len(s["completed"])} of {s["total"]} lessons · {s["streak"]}-day streak</div>')
    pending = int(state["meta"].get("changes_since_export") or 0)
    if pending >= config.BACKUP_NUDGE_AFTER:
        st.warning(f"{pending} changes since your last backup.")
        st.button("Back up now", on_click=ui.go, args=("Progress & Backup",), key="nudge_backup")

PAGES[st.session_state["nav"]].render()
