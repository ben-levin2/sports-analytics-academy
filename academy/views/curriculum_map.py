"""Visual curriculum map: six levels, every concept coloured by mastery, projects per level."""
from .. import curriculum, ui
from ..projects_data import PROJECTS

LEGEND = (
    '<div class="aa-legend">'
    '<span style="--c:#22304A;--b:transparent">Not started</span>'
    '<span style="--c:#6A5124;--b:#2A2214">Learning (&lt;70%)</span>'
    '<span style="--c:#2D4E80;--b:#14243F">Proficient (70%+)</span>'
    '<span style="--c:#5B9BF5;--b:#1B3A66">Mastered (90%+)</span>'
    '</div>')


def project_tag(state, p) -> str:
    runs = state["projects"].get(p.id, {}).get("runs", [])
    pc = "not run"
    if runs:
        pc = "candidate" if runs[-1]["verdict"].startswith("Meets") else "lab only"
    return f"{p.label} · {pc}"


def render_map(state, show_projects: bool = True) -> None:
    statuses = {c.id: curriculum.concept_status(state, c.id) for c in curriculum.CONCEPTS}
    current = curriculum.current_level(state).number
    cols = []
    for lv in curriculum.LEVELS:
        nodes = "".join(
            f'<div class="aa-node {statuses[c.id]}" title="{c.num}. {ui.esc(c.name)}: {ui.esc(c.summary)}"><b>{c.num}</b><span>{ui.esc(c.name)}</span></div>'
            for c in sorted(curriculum.concepts_in_level(lv.number), key=lambda c: c.num))
        projs = ""
        if show_projects:
            projs = '<div class="aa-proj">' + "<br>".join(
                project_tag(state, p) for p in PROJECTS if p.level == lv.number) + "</div>"
        cur = " cur" if lv.number == current else ""
        cols.append(f'<div class="aa-lv{cur}"><h4>LEVEL {lv.number}</h4>'
                    f'<div class="lt">{ui.esc(lv.title)}</div>{nodes}{projs}</div>')
    ui.html('<div class="aa-map">' + "".join(cols) + "</div>" + LEGEND)
