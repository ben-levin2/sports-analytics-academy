import streamlit as st

from .. import config, curriculum, progress, quiz, ui
from ..lessons import all_lessons, get_lesson


STEPS = [("step-1", "1 Plain English"), ("step-2", "2 The problem"), ("step-3", "3 Football example"),
         ("step-4", "4 Visual"), ("step-5", "5–6 Math"), ("step-7", "7 Code"), ("step-8", "8 Line by line"),
         ("step-9", "9 PSFAL"), ("step-10", "10 Quiz")]


def _problem(l):
    with st.container(border=True):
        ui.label("The problem")
        st.markdown(l.problem)
        ui.label("Why someone would use it")
        st.markdown(l.why)
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            ui.label("When it is useful")
            st.markdown(l.when_useful)
    with c2:
        with st.container(border=True):
            ui.label("When it is NOT useful")
            st.markdown(l.when_not)
    with st.container(border=True):
        ui.label("Compared with related methods")
        st.markdown(l.compare)


def _walkthrough(l):
    cells = "".join(f"<code>{ui.esc(code)}</code><div>{ui.esc(text)}</div>" for code, text in l.walkthrough)
    ui.html(f'<div class="aa-walk">{cells}</div>')


def _quiz(l, state):
    rkey = f"quiz_result_{l.id}"
    st.caption(f"{len(l.quiz)} questions. Lesson passes at {config.COMPLETE_THRESHOLD:.0%} and is mastered at "
               f"{config.MASTERY_THRESHOLD:.0%} (best attempt). Concept mastery uses your latest attempt.")
    with st.form(f"quiz_{l.id}", border=True):
        responses = {}
        for i, q in enumerate(l.quiz, start=1):
            if i > 1:
                st.divider()
            kind = {"mc": "Multiple choice", "short": "Short answer", "numeric": "Calculate"}[q.kind]
            c = curriculum.CONCEPT_BY_ID.get(l.question_concept(q))
            ui.html(f'<div class="aa-label" style="margin-top:0">Question {i} · {ui.esc(kind)} · {ui.esc(q.category)}'
                    + (f' · concept {c.num}' if c else "") + "</div>")
            st.markdown(f"**{q.prompt}**")
            k = f"{l.id}_{q.id}"
            if q.kind == "mc":
                responses[q.id] = st.radio(q.prompt, q.options, index=None, key=k, label_visibility="collapsed")
            elif q.kind == "numeric":
                responses[q.id] = st.text_input(q.prompt, key=k, placeholder="e.g. 0.42 or 42%",
                                                label_visibility="collapsed")
            else:
                responses[q.id] = st.text_area(q.prompt, key=k, height=90, label_visibility="collapsed",
                                               placeholder="Explain in your own words…")
        submitted = st.form_submit_button("Submit answers", type="primary")

    if submitted:
        score, points, per_q, feedback = quiz.score_quiz(l.quiz, responses)
        progress.record_attempt(state, l.id, l.title, score, points, len(l.quiz), per_q)
        st.session_state[rkey] = {"score": score, "per_q": per_q, "feedback": feedback}
        st.rerun()

    res = st.session_state.get(rkey)
    if res:
        score = res["score"]
        if score >= config.MASTERY_THRESHOLD:
            st.success(f"Score {score:.0%}. Lesson mastered.")
        elif score >= config.COMPLETE_THRESHOLD:
            st.info(f"Score {score:.0%}. Lesson completed. Reach {config.MASTERY_THRESHOLD:.0%} to master it.")
        else:
            st.warning(f"Score {score:.0%}. Not yet passed. Review the feedback, revisit the sections, and retake.")
        for cid in l.concept_ids:
            c = curriculum.CONCEPT_BY_ID[cid]
            ui.html(f"Concept #{c.num} {ui.esc(c.name)}: " + ui.concept_pill(curriculum.concept_status(state, cid)))
        for i, q in enumerate(l.quiz, start=1):
            frac = res["per_q"].get(q.id, 0)
            tag = "Correct" if frac == 1 else ("Partly right" if frac > 0 else "Not yet")
            with st.expander(f"Q{i} · {tag} — {q.prompt[:80]}{'…' if len(q.prompt) > 80 else ''}", expanded=frac < 1):
                st.markdown(res["feedback"][q.id])


def render():
    state = ui.get_state()
    lessons = all_lessons()
    if not lessons:
        st.warning("No lessons found in academy/lessons/.")
        return
    ids = [l.id for l in lessons]
    if st.session_state.get("lesson_pick") not in ids:
        rec, _ = curriculum.recommended(state)
        st.session_state["lesson_pick"] = rec.id if rec else ids[0]

    by_id = {l.id: l for l in lessons}
    st.selectbox("Lesson", ids, key="lesson_pick",
                 format_func=lambda i: f"{by_id[i].code_label} · {by_id[i].title}")
    l = get_lesson(st.session_state["lesson_pick"])
    progress.mark_opened(state, l.id, l.title)
    lv = curriculum.level(l.level)

    ui.hero(f"Lesson {l.code_label} · Level {lv.number}: {lv.title}", l.title, l.subtitle)
    concepts = [curriculum.CONCEPT_BY_ID[c] for c in l.concept_ids]
    ui.html(ui.status_pill(progress.lesson_status(state, l.id)) + ui.pill(f"{l.minutes} min", "gray")
            + "".join(ui.pill(f"Concept {c.num}: {c.name}", "gray") for c in concepts))
    ui.step_nav(STEPS)

    ui.section("Step 1", "In plain English", "step-1")
    st.markdown(l.simple)

    ui.section("Step 2", "What problem it solves", "step-2")
    _problem(l)

    ui.section("Step 3", "College football example", "step-3")
    ui.sim_banner("The players and numbers in this example are fictional, chosen to be realistic.")
    st.markdown(l.sports)
    ui.html(f'<div class="aa-takeaway"><div class="aa-label">Key idea</div>{ui.esc(l.takeaway)}</div>')

    ui.section("Step 4", "See it", "step-4")
    if l.visual:
        try:
            l.visual()
        except Exception as exc:  # a broken visual should not hide the rest of the lesson
            st.error(f"This visual failed to render: {exc!r}")
    st.caption(l.visual_caption)

    ui.section("Steps 5–6", "The math, step by step, every symbol explained", "step-5")
    for m in l.math:
        with st.container(border=True):
            ui.label(m.title)
            st.latex(m.latex)
            st.markdown("**Every symbol:**\n" + m.where)
            if m.note:
                st.caption(m.note)

    ui.section("Step 7", "Python example", "step-7")
    st.markdown(l.code_notes)
    ui.code_runner(l.code, key=f"code_{l.id}")

    ui.section("Step 8", "The code, line by line", "step-8")
    _walkthrough(l)

    ui.section("Step 9", "How this applies to PSFAL", "step-9")
    st.markdown(l.psfal)

    ui.section("Step 10", "Knowledge check", "step-10")
    _quiz(l, state)

    st.divider()
    i = ids.index(l.id)
    c1, _, c3 = st.columns([1, 2, 1])
    if i > 0:
        c1.button(f"← {lessons[i - 1].code_label}", key="prev_lesson", on_click=ui.go, args=("Lessons", ids[i - 1]))
    if i < len(ids) - 1:
        c3.button(f"{lessons[i + 1].code_label} →", key="next_lesson", on_click=ui.go, args=("Lessons", ids[i + 1]))
