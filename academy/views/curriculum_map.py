"""Visual curriculum map: six levels, 28 concepts coloured by mastery, projects per level."""
from .. import curriculum, ui
from ..projects_data import PROJECTS

LEGEND = (
    '<div class="aa-legend">'
    '<span style="--c:#34507A;--b:transparent">Not started</span>'
    '<span style="--c:#7A5A1E;--b:#241D10">Learning (&lt;70%)</span>'
    '<span style="--c:#2D5C9E;--b:#10233F">Proficient (70%+)</span>'
    '<span style="--c:#1F6B45;--b:#0E2A21">Mastered (90%+)</span>'
    '<span style="--c:#263754;--b:transparent;border-style:dashed">Planned (no lesson yet)</span>'
    '</div>')


def project_tag(state, p) -> str:
    runs = state["projects"].get(p.id, {}).get("runs", [])
    pc = "PC: not run" if p.status == "built" else "PC: planned"
    if runs:
        pc = "PC: candidate" if runs[-1]["verdict"].startswith("Meets") else "PC: lab only"
    return f"{p.label} {ui.esc(p.title)} · {pc}"


def render_map(state, show_projects: bool = True) -> None:
    statuses = {c.id: curriculum.concept_status(state, c.id) for c in curriculum.CONCEPTS}
    current = curriculum.current_level(state).number
    cols = []
    for lv in curriculum.LEVELS:
        nodes = "".join(
            f'<div class="aa-node {statuses[c.id]}" title="{ui.esc(c.summary)}"><b>{c.num}</b>{ui.esc(c.name)}</div>'
            for c in sorted(curriculum.concepts_in_level(lv.number), key=lambda c: c.num))
        projs = ""
        if show_projects:
            projs = '<div class="aa-proj">' + "<br>".join(
                project_tag(state, p) for p in PROJECTS if p.level == lv.number) + "</div>"
        cur = " cur" if lv.number == current else ""
        cols.append(f'<div class="aa-lv{cur}"><h4>LEVEL {lv.number}</h4>'
                    f'<div class="lt">{ui.esc(lv.title)}</div>{nodes}{projs}</div>')
    ui.html('<div class="aa-map">' + "".join(cols) + "</div>" + LEGEND)
