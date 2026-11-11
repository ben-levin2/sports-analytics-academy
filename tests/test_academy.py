"""Run:  pytest -q   (from the project root)"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["ACADEMY_DATA_DIR"] = tempfile.mkdtemp()   # never touch real progress or uploads
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from academy import config, curriculum, data, datasets, progress, quiz, shrinkage  # noqa: E402
from academy.labs import EXERCISES  # noqa: E402
from academy.lessons import LOAD_ERRORS, all_lessons  # noqa: E402
from academy.projects_data import PROJECTS  # noqa: E402
from academy.runner import run_code  # noqa: E402


# ---------------------------------------------------------------- curriculum integrity
def test_28_concepts_six_levels():
    nums = sorted(c.num for c in curriculum.CONCEPTS)
    assert nums == list(range(1, 29))
    assert len(curriculum.CONCEPT_BY_ID) == 28
    assert {c.level for c in curriculum.CONCEPTS} == {1, 2, 3, 4, 5, 6}
    assert len(curriculum.LEVELS) == 6


def test_lessons_well_formed_and_level1_complete():
    lessons = all_lessons()
    assert not LOAD_ERRORS
    assert len({l.id for l in lessons}) == len(lessons)
    for l in lessons:
        assert l.concept_ids and set(l.concept_ids) <= set(curriculum.CONCEPT_BY_ID), l.id
        assert len(l.quiz) >= 5 and l.math and l.code and len(l.walkthrough) >= 5, l.id
        for q in l.quiz:
            assert l.question_concept(q) in curriculum.CONCEPT_BY_ID
            if q.kind == "mc":
                assert len(q.notes) == len(q.options) and 0 <= q.answer < len(q.options)
            elif q.kind == "short":
                assert q.keywords and q.model_answer
            else:
                assert q.kind == "numeric" and q.answer is not None
    level1 = {c.id for c in curriculum.concepts_in_level(1)}
    assert level1 <= {cid for l in lessons for cid in l.concept_ids}, "every Level 1 concept has a lesson"


def test_projects_cover_every_level_with_both_tracks():
    assert {p.level for p in PROJECTS} == {1, 2, 3, 4, 5, 6}
    for p in PROJECTS:
        assert p.dataset_id in datasets.SPECS or p.dataset_id == "rate_events"
        assert p.learning.steps and p.learning.limitations and p.production.gates and p.production.metric
        assert set(p.concept_ids) <= set(curriculum.CONCEPT_BY_ID)


@pytest.mark.parametrize("lesson", all_lessons(), ids=lambda l: l.id)
def test_lesson_code_runs(lesson):
    assert run_code(lesson.code)["error"] is None


@pytest.mark.parametrize("ex", EXERCISES, ids=lambda e: e.id)
def test_lab_solution_passes_and_starter_runs(ex):
    sol = run_code(ex.solution)
    assert sol["error"] is None and "❌" not in sol["text"], sol["text"]
    assert run_code(ex.starter)["error"] is None


@pytest.mark.parametrize("p", [p for p in PROJECTS if p.learning.code], ids=lambda p: p.id)
def test_project_learning_code_runs(p):
    assert run_code(p.learning.code)["error"] is None


# ---------------------------------------------------------------- shrinkage engine + gates
def test_shrinkage_evaluation_and_gates():
    df = data.make_qb_completions()
    res = shrinkage.evaluate_next_season(df)
    assert res["ok"] and res["leak_ok"] and res["overall"]["model"] < res["overall"]["raw"]
    assert res["rows"]["season"].min() > df["season"].min()          # first season never a test season
    sim_gates = shrinkage.gates(res, {"source": "simulated", "real": False})
    assert shrinkage.verdict(sim_gates) == "Learning Lab only"       # simulated can never pass
    assert "production-ready" not in shrinkage.verdict(shrinkage.gates(res, {"source": "upload", "real": True})).lower()


def test_posterior_table_shrinks_low_volume_more():
    df = data.make_lb_tackle_share()
    tab = shrinkage.posterior_table(df, shrinkage.fit_priors(df))
    lo = tab.nsmallest(50, "opportunities")
    hi = tab.nlargest(50, "opportunities")
    assert lo["weight_on_data"].mean() < hi["weight_on_data"].mean()
    assert (tab["ci_low"] <= tab["post_mean"]).all() and (tab["post_mean"] <= tab["ci_high"]).all()


# ---------------------------------------------------------------- datasets
def test_dataset_upload_roundtrip_and_validation():
    bad = pd.DataFrame({"player_id": ["a"], "season": [2020], "successes": [5]})
    assert datasets.save_upload("rate_events", bad, "bad", True)
    worse = pd.DataFrame({"player_id": ["a"], "season": [2020], "successes": [9], "opportunities": [5]})
    assert datasets.save_upload("rate_events", worse, "bad", True)
    good = data.make_qb_completions().drop(columns="true_rate").head(300)
    assert datasets.save_upload("rate_events", good, "my qbs", True) == []
    df, info = datasets.load("rate_events")
    assert info["source"] == "upload" and info["real"] and len(df) == 300
    datasets.remove_upload("rate_events")
    df, info = datasets.load("rate_events")
    assert info["source"] == "simulated" and df.attrs.get("simulated")


# ---------------------------------------------------------------- grading and progress
def test_grading():
    l = all_lessons()[0]
    q_mc = next(q for q in l.quiz if q.kind == "mc")
    assert quiz.grade(q_mc, q_mc.options[q_mc.answer])[0] == 1 and quiz.grade(q_mc, None)[0] == 0
    q_num = next(q for q in l.quiz if q.kind == "numeric")
    assert quiz.grade(q_num, "0.661")[0] == 1 and quiz.grade(q_num, "66.1%")[0] == 1
    assert quiz.grade(q_num, "0.9")[0] == 0 and quiz.grade(q_num, "abc")[0] == 0
    q_short = next(q for q in l.quiz if q.kind == "short")
    assert quiz.grade(q_short, "small sample so shrink toward the league average")[0] == 1


def test_lesson_completion_and_concept_mastery_are_separate():
    l = all_lessons()[0]
    cid = l.concept_ids[0]
    s = progress.default_state()
    progress.mark_opened(s, l.id, l.title)
    assert progress.lesson_status(s, l.id) == "opened" and curriculum.concept_status(s, cid) == "new"
    perfect = {q.id: 1.0 for q in l.quiz}
    progress.record_attempt(s, l.id, l.title, 1.0, len(l.quiz), len(l.quiz), perfect)
    assert progress.is_mastered(s, l.id) and curriculum.concept_status(s, cid) == "mastered"
    weak = {q.id: 0.0 for q in l.quiz}
    progress.record_attempt(s, l.id, l.title, 0.0, 0, len(l.quiz), weak)
    assert progress.is_mastered(s, l.id)                       # best attempt still counts for the lesson
    assert curriculum.concept_status(s, cid) == "learning"     # latest attempt drives the concept
    assert curriculum.concept_status(s, "lstms") == "planned"


def test_persistence_roundtrip_merge_and_corrupt_file():
    s = progress.default_state()
    progress.record_attempt(s, "a", "A", 0.9, 4.5, 5, {"q1": 1})
    progress.set_project_step(s, "p", "P", "lab-1", True, "x")
    progress.log_project_run(s, "p", "P", "Learning Lab only")
    s2, info = progress.load()
    assert info["source"] == "file" and s2["lessons"]["a"]["best"] == 0.9 and s2["projects"]["p"]["runs"]
    other = progress.default_state()
    progress.set_project_step(other, "p", "P", "prod-0", True, "y")
    merged = progress.merge(s, progress.from_json(progress.to_json(other)))
    assert set(merged["projects"]["p"]["steps"]) == {"lab-1", "prod-0"} and merged["projects"]["p"]["runs"]
    with pytest.raises(ValueError):
        progress.from_json('{"hello": 1}')
    config.PROGRESS_FILE.write_text("{not json")
    s3, info = progress.load()
    assert info["source"] == "error" and s3["lessons"] == {}


def test_streak():
    from datetime import timedelta
    s = progress.default_state()
    t = progress.today()
    s["active_days"] = [(t - timedelta(days=d)).isoformat() for d in (0, 1, 2, 4)]
    assert progress.streak(s) == 3


# ---------------------------------------------------------------- full app smoke test
def test_app_pages_projects_and_quiz_flow():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
    assert not at.exception
    for page in ["Learning Roadmap", "Lessons", "Python Lab", "Knowledge Checks", "Projects",
                 "Data Sources", "Progress & Backup", "Dashboard"]:
        at.sidebar.radio(key="nav").set_value(page).run()
        assert not at.exception, (page, at.exception)

    at.sidebar.radio(key="nav").set_value("Projects").run()
    for p in PROJECTS:
        at.selectbox(key="proj_pick").set_value(p.id).run()
        assert not at.exception, (p.id, at.exception)
    at.selectbox(key="proj_pick").set_value(PROJECTS[0].id).run()
    at.button(key="p1_high").click().run()
    assert not at.exception
    at.button(key="p1_eval").click().run()
    assert not at.exception
    assert at.session_state["progress"]["projects"][PROJECTS[0].id]["runs"][-1]["verdict"].startswith("Learning Lab only")

    at.sidebar.radio(key="nav").set_value("Lessons").run()
    for lesson in all_lessons():
        at.selectbox(key="lesson_pick").set_value(lesson.id).run()
        assert not at.exception, lesson.id
    l = all_lessons()[0]
    at.selectbox(key="lesson_pick").set_value(l.id).run()
    for q in l.quiz:
        k = f"{l.id}_{q.id}"
        if q.kind == "mc":
            at.radio(key=k).set_value(q.options[q.answer])
        elif q.kind == "numeric":
            at.text_input(key=k).input(str(q.answer))
        else:
            at.text_area(key=k).input(q.model_answer)
    at.button[[b.label for b in at.button].index("Submit answers")].click().run()
    assert not at.exception
    st_ = at.session_state["progress"]
    assert progress.is_mastered(st_, l.id) and curriculum.concept_status(st_, l.concept_ids[0]) == "mastered"
