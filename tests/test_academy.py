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

from academy import (archetypes, config, curriculum, data, datasets, progress, quiz, shrinkage, spatial,  # noqa: E402
                     hierarchical, lineup, optimize, scoreboard, sequence, survival, validation)
from academy.labs import EXERCISES  # noqa: E402
from academy.lessons import LOAD_ERRORS, all_lessons  # noqa: E402
from academy.projects_data import PROJECTS  # noqa: E402
from academy.runner import run_code  # noqa: E402


# ---------------------------------------------------------------- curriculum integrity
def test_28_concepts_six_levels():
    nums = sorted(c.num for c in curriculum.CONCEPTS)
    assert nums == list(range(1, 34))                     # the original 28 plus small-data concepts 29-33
    assert len(curriculum.CONCEPT_BY_ID) == 33
    assert [c.num for c in curriculum.CONCEPTS[:28]] == list(range(1, 29))   # originals unchanged, in order
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


def test_every_project_has_an_interactive_page():
    from academy.views.projects import INTERACTIVE
    assert {p.id for p in PROJECTS} <= set(INTERACTIVE)


# ---------------------------------------------------------------- shrinkage engine + gates
def test_shrinkage_evaluation_and_gates():
    df = data.make_qb_completions()
    res = shrinkage.evaluate_next_season(df)
    assert res["ok"] and res["leak_ok"] and res["overall"]["model"] < res["overall"]["raw"]
    assert res["rows"]["season"].min() > df["season"].min()          # first season never a test season
    sim_gates = shrinkage.gates(res, {"source": "simulated", "real": False})
    assert shrinkage.verdict(sim_gates) == "Learning Lab only"       # simulated can never pass
    assert "production-ready" not in shrinkage.verdict(shrinkage.gates(res, {"source": "upload", "real": True})).lower()


def test_validation_helpers():
    import numpy as np
    y = np.array([1, 0, 0, 1])
    assert abs(validation.brier([0.9, 0.2, 0.6, 1.0], y) - (0.01 + 0.04 + 0.36 + 0) / 4) < 1e-12
    assert validation.brier_skill(y.astype(float), y, 0.5) == 1.0
    assert validation.ece(np.full(200, 0.3), np.r_[np.ones(60), np.zeros(140)]) < 1e-9
    assert validation.auc([0.1, 0.9, 0.8, 0.2], [0, 1, 1, 0]) == 1.0
    df = data.make_qb_completions()
    folds = list(validation.rolling_origin(df))
    assert len(folds) == df["season"].nunique() - 1
    assert all(validation.no_leakage(tr, te) for _, tr, te in folds)
    assert not validation.no_leakage(df, df[df["season"] == 2020])


def test_calibration_and_time_gates_are_wired_in():
    res = shrinkage.evaluate_next_season(data.make_pressure_by_position(), "position")
    names = [n for n, _, _ in shrinkage.gates(res, {"source": "upload", "real": True})]
    assert "Calibrated probabilities (concept 25)" in names
    assert "Time-based validation (concept 26)" in names and "No temporal leakage (concept 26)" in names
    p = res["prob"]
    assert 0 <= p["ece"] <= 1 and p["skill"] > 0 and len(p["reliability"]) == 5
    assert 0.25 < p["event_rate"] < 0.42


def test_level2_complete_and_archetype_engine():
    level2 = {c.id for c in curriculum.concepts_in_level(2)}
    assert level2 <= {cid for l in all_lessons() for cid in l.concept_ids}
    df = data.make_defensive_profiles()
    assert len(df) > 3000 and df.attrs.get("simulated")
    feats = archetypes.feature_candidates(df)
    assert set(data.PROFILE_FEATURES) == set(feats)
    d = archetypes.prepare(df, feats, "DL", 150)
    _, Z = archetypes.standardize(d, feats)
    assert archetypes.stability(Z, 3) > 0.8
    res = archetypes.evaluate_downstream(df, feats, 3, "DL", 100)
    assert res["ok"] and res["leak_ok"] and res["n_seasons"] >= 3
    gl = archetypes.gates(res, {"source": "simulated", "real": False})
    assert archetypes.verdict(gl) == "Learning Lab only"
    assert archetypes.gates({"ok": False, "reason": "x"}, {"source": "upload", "real": True})[0][1] is False


def test_level3_complete_and_survival_engine():
    import numpy as np
    level3 = {c.id for c in curriculum.concepts_in_level(3)}
    assert level3 <= {cid for l in all_lessons() for cid in l.concept_ids}
    # Kaplan-Meier on a tiny hand-checked example: S(2) = (1 - 20/100) * (1 - 14/70) = 0.64
    t = np.r_[np.ones(20), np.ones(10), np.full(14, 2), np.full(56, 3)]
    e = np.r_[np.ones(20), np.zeros(10), np.ones(14), np.zeros(56)]
    assert abs(survival.km_at(survival.kaplan_meier(t, e), 2) - 0.64) < 1e-9
    df = data.make_roster_careers()
    X, names = survival.design(df)
    m = survival.cox_fit(X, df["seasons_observed"], df["contributed"])
    assert 1.5 < np.exp(m["beta"][0]) < 2.0                       # simulated rating effect is recovered
    res = survival.evaluate_contribution(df)
    assert res["ok"] and res["leak_ok"] and res["n_classes"] >= 3
    assert survival.verdict(survival.gates(res, {"source": "simulated", "real": False})) == "Learning Lab only"
    tr = survival.truncate(df[df["class_year"] < 2022], 2021)
    assert (tr["known_through"] <= 2021).all()


def test_spatial_engine():
    df = data.make_hometown_production()
    t = spatial.regional_table(df)
    assert t["players"].sum() == len(df)
    vg = spatial.empirical_variogram(df)
    assert (vg["semivariance"] > 0).all()
    gp = spatial.fit_gp(df[["lon", "lat"]], df["production"] - df["production"].mean(), "RBF", cap=300)
    mu, sd = spatial.predict_latent(gp, [[-95.4, 29.8]])
    assert sd[0] >= 0
    res = spatial.evaluate_spatial(df)
    assert res["ok"] and res["leak_ok"] and 0.7 < res["coverage"] < 0.9
    gl = spatial.gates(res, {"source": "simulated", "real": False})
    assert spatial.verdict(gl) == "Learning Lab only"
    assert any(n == "Controls for population density" and ok for n, ok, _ in gl)
    assert spatial.gates(spatial.evaluate_spatial(df.drop(columns="class_year")), {"source": "upload", "real": True})[0][1] is False


def test_level4_complete_and_sequence_engine():
    import numpy as np
    level4 = {c.id for c in curriculum.concepts_in_level(4)}
    assert level4 <= {cid for l in all_lessons() for cid in l.concept_ids}
    df = data.make_game_logs()
    rs = sequence.running_mean_residuals(df).dropna(subset=["resid"])
    seqs = [g["resid"].to_numpy() for _, g in rs.groupby(["player_id", "season"])]
    m2 = sequence.fit_hmm(seqs, 2)
    assert m2["means"][0] > 0 > m2["means"][1]                       # hot above, cold below
    assert 0.6 < m2["A"][0, 0] < 0.95                                # recovers sticky (not near-1) form
    alpha, ll = sequence.forward(seqs[0], m2["pi"], m2["A"], m2["means"], m2["sds"])
    assert np.allclose(alpha.sum(1), 1) and np.isfinite(ll)
    assert len(sequence.viterbi(seqs[0], m2["pi"], m2["A"], m2["means"], m2["sds"])) == len(seqs[0])
    assert sequence.conformal_halfwidth(np.arange(1, 100), 0.8) == 80
    res = sequence.evaluate_games(df)
    assert res["ok"] and res["leak_ok"] and res["n_seasons"] >= 3
    assert all(0.7 < c < 0.9 for c in res["coverage"].values())
    assert sequence.verdict(sequence.gates(res, {"source": "simulated", "real": False})) == "Learning Lab only"
    # thin data must fail the sample-sufficiency gate
    thin = df[df["week"] <= 5]
    gl = sequence.gates(sequence.evaluate_games(thin), {"source": "upload", "real": True})
    assert any(n == "Sample sufficiency" and not ok for n, ok, _ in gl)


def test_level5_complete_and_hierarchical_engine():
    import numpy as np
    level5 = {c.id for c in curriculum.concepts_in_level(5)}
    assert level5 <= {cid for l in all_lessons() for cid in l.concept_ids}
    d = hierarchical.prepare(data.make_pressure_hierarchical())
    fit = hierarchical.gibbs(d, iters=500, burn=200)
    diag = hierarchical.diagnostics(fit).set_index("parameter")
    assert diag["R-hat"].max() < 1.1
    assert abs(diag.loc["experience 3+", "mean"] - 0.36) < 0.08         # recovers the simulated effect
    assert abs(diag.loc["sd player", "mean"] - 0.30) < 0.06
    chains = np.random.default_rng(0).normal(size=(4, 500))
    assert abs(hierarchical.rhat(chains) - 1) < 0.02 and hierarchical.ess(chains) > 1000
    stuck = chains + np.arange(4)[:, None] * 3
    assert hierarchical.rhat(stuck) > 1.5


def test_level5_contest_both_ways():
    gated = {"source": "upload", "real": True}
    with_fx = hierarchical.evaluate(data.make_pressure_hierarchical())
    assert with_fx["ok"] and with_fx["leak_ok"] and with_fx["gain_vs_level1"] > 0.05
    assert hierarchical.verdict(hierarchical.gates(with_fx, {"source": "simulated", "real": False})) == "Learning Lab only"
    without = hierarchical.evaluate(data.make_pressure_by_position())
    assert without["gain_vs_level1"] < 0                               # extra structure that isn't there hurts
    assert any(n == "Beats BOTH baselines" and not ok for n, ok, _ in hierarchical.gates(without, gated))


def test_statsmodels_is_required_and_pymc_optional():
    req = (ROOT / "requirements.txt").read_text()
    opt = (ROOT / "requirements-optional.txt").read_text()
    assert "statsmodels" in req and "pymc" not in req and "pymc" in opt
    import statsmodels  # noqa: F401


def test_all_28_concepts_have_lessons_and_every_project_is_built():
    taught = {cid for l in all_lessons() for cid in l.concept_ids}
    assert taught == set(curriculum.CONCEPT_BY_ID)
    assert all(p.status == "built" for p in PROJECTS)


def test_bayesian_optimization_engine():
    import numpy as np
    ei = optimize.expected_improvement(np.array([1.0, 2.0, 2.0]), np.array([0.1, 0.1, 1.0]), best=1.5)
    assert ei[0] > ei[2] > ei[1]                                       # promising > uncertain > bad-and-certain
    f = lambda x: (x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2
    X, y = optimize.bayes_opt(f, [[0, 1], [0, 1]], n_init=4, n_iter=10, seed=0, n_cand=500)
    assert y.min() < 0.01
    tune = optimize.tuning_objective()
    assert tune([0.0, 1.0]) < tune([4.0, 1.0]) and tune([0.0, 1.0]) < tune([0.0, 0.0])


def test_lineup_engine_and_backtest():
    import numpy as np
    m = data.make_dl_market()
    pool = m[m["season"] == 2025].reset_index(drop=True)
    x = lineup.exact(pool)
    g = lineup.greedy(pool)
    assert lineup.feasible(pool, x) and lineup.feasible(pool, g)
    assert lineup.total(pool, x) >= lineup.total(pool, g) - 1e-9
    assert lineup.random_search(pool, 15, seed=0)[-1] <= lineup.total(pool, x) + 1e-9
    assert lineup.bayes_opt(pool, 15, seed=0)[-1] <= lineup.total(pool, x) + 1e-9
    res = lineup.backtest(m)
    assert res["ok"] and res["leak_ok"] and res["n_seasons"] >= 3 and 0.5 < res["realized_ratio"] < 1.2
    gl = lineup.gates(res, {"source": "upload", "real": True})
    assert lineup.verdict(gl) == "Learning Lab only"                   # input-model gate can't be automated


def test_scoreboard_and_registration():
    row = scoreboard.run_one("p6a")
    assert row["beats"] in (True, False) and row["gates"]
    s = progress.default_state()
    progress.set_registration(s, "p6b", {"question": "q", "primary": "p5", "metric": "m", "baseline": "b"})
    assert s["projects"]["p6b"]["registration"]["primary"] == "p5" and "ts" in s["projects"]["p6b"]["registration"]
    merged = progress.merge(progress.default_state(), progress.from_json(progress.to_json(s)))
    assert merged["projects"]["p6b"]["registration"]["question"] == "q"


def test_torch_is_optional():
    req = (ROOT / "requirements.txt").read_text()
    assert "torch" not in req
    assert "torch" in (ROOT / "requirements-optional.txt").read_text()
    if not sequence.torch_available():
        assert "pip install" in sequence.TORCH_NOTE


def test_profile_upload_validation():
    df = data.make_defensive_profiles().drop(columns="true_archetype")
    assert datasets.save_upload("player_profiles", df[["player_id", "season", "position", "snaps"]], "x", True)
    assert datasets.save_upload("player_profiles", df, "mine", True) == []
    loaded, info = datasets.load("player_profiles")
    assert info["source"] == "upload" and len(loaded) == len(df)
    datasets.remove_upload("player_profiles")


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
    assert all(curriculum.concept_status(s, c.id) != "planned" for c in curriculum.CONCEPTS)


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
    at.selectbox(key="proj_pick").set_value("p2-archetypes").run()
    assert not at.exception
    at.button(key="p2_eval").click().run()
    assert not at.exception
    assert at.session_state["progress"]["projects"]["p2-archetypes"]["runs"][-1]["verdict"].startswith("Learning Lab only")
    for pid, btn in [("p3a-time-to-contribution", "p3a_eval"), ("p3b-geographic-recruiting", "p3b_eval"),
                     ("p4-game-by-game", "p4_eval"), ("p5-hierarchical-development", "p5_eval"),
                     ("p6a-lineup-optimization", "p6a_eval")]:
        at.selectbox(key="proj_pick").set_value(pid).run()
        assert not at.exception, (pid, at.exception)
        at.button(key=btn).click().run()
        assert not at.exception, (pid, at.exception)
        assert at.session_state["progress"]["projects"][pid]["runs"][-1]["verdict"].startswith("Learning Lab only")

    at.selectbox(key="proj_pick").set_value("p6b-integrated-research").run()
    assert not at.exception

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


# ---------------------------------------------------------------- small-data modeling (concepts 29-33, P3C)
def test_ridge_equals_normal_prior():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from academy import smalldata as SD
    df = data.make_draft_classes()
    tr = df[df["class_year"] <= 2013]
    d = SD.Design(tr)
    X = d.linear(tr)
    sk = LogisticRegression(C=0.5, max_iter=5000).fit(X[:, 1:], tr["drafted"])
    mine = SD.fit_logistic(X, tr["drafted"], prior_sd=np.r_[100.0, np.full(X.shape[1] - 1, np.sqrt(0.5))])
    assert np.allclose(mine["beta"][1:], sk.coef_[0], atol=2e-3)


def test_monotone_gam_never_decreases():
    import numpy as np
    from academy import smalldata as SD
    df = data.make_draft_classes()
    tr = df[df["class_year"] <= 2013]
    d = SD.Design(tr)
    grid = np.linspace(-3, 3, 121)
    m = SD.fit_monotone_gam(d, tr, tr["drafted"].to_numpy(), lam=0.05)
    for which in ("rating", "year1"):
        assert (np.diff(SD.gam_curve(m, which, grid)) >= -1e-9).all()


def test_person_period_censoring():
    import numpy as np
    from academy import smalldata as SD
    df = pd.DataFrame({"player_id": ["a", "b", "c", "d"], "class_year": [2015, 2015, 2019, 2021],
                       "position": "QB", "rating": 0.9, "year1_percentile": 50.0,
                       "drafted": [1, 0, 0, 0], "draft_year": [2019.0, np.nan, np.nan, np.nan]})
    pp = SD.person_period(df, as_of=2022)
    rows = pp.groupby("player_id")["period"].apply(list).to_dict()
    assert rows == {"a": [3, 4], "b": [3, 4, 5], "c": [3]}           # a drafted in year 4; c censored; d not yet eligible
    assert pp.loc[pp["player_id"] == "a", "event"].tolist() == [0, 1]
    assert pp.loc[pp["player_id"] != "a", "event"].sum() == 0


def test_survival_uses_unfinished_classes_without_bias():
    import numpy as np
    from academy import smalldata as SD
    later = data.make_draft_classes(per_class=60, data_end=2035)
    now = data.make_draft_classes(per_class=60, data_end=2022)
    truth = later[later["class_year"] <= 2021]["drafted"].mean()
    upto = now[now["class_year"] <= 2021]
    h = SD.person_period(upto, 2022).groupby("period")["event"].mean()
    est = 1 - np.prod(1 - h)
    assert abs(est - truth) < abs(upto["drafted"].mean() - truth)      # beats 'not yet = never'


def test_draft_model_evaluation_and_gates():
    from academy import smalldata as SD
    df = data.make_draft_classes()
    res = SD.evaluate(df, boot=200)
    assert res["ok"] and res["leak_ok"] and res["n_classes"] >= SD.MIN_TEST_CLASSES
    assert set(SD.MODELS) <= set(res["summary"].index)
    assert all(a < b for a, b in zip(res["train_rows_full"], res["train_players_survival"]))
    for m in SD.CHALLENGERS:
        gl = SD.gates(res, {"source": "simulated", "real": False}, m)
        assert SD.verdict(gl) == "Learning Lab only"                       # simulated data can never pass
    no_dy = df.drop(columns="draft_year")
    res2 = SD.evaluate(no_dy, boot=50)
    assert res2["ok"] and "survival" not in res2["summary"].index
    assert SD.verdict(SD.gates(res2, {"source": "upload", "real": True}, "survival")) == "Learning Lab only"


def test_draft_upload_validation():
    df = data.make_draft_classes().drop(columns=["true_z_rating", "true_year1"])
    bad = df.copy()
    bad.loc[0, "drafted"] = 2
    assert datasets.save_upload("draft_outcomes", bad, "bad", True)
    assert datasets.save_upload("draft_outcomes", df.drop(columns="year1_percentile"), "bad", True)
    assert datasets.save_upload("draft_outcomes", df, "mine", True) == []
    loaded, info = datasets.load("draft_outcomes")
    assert info["source"] == "upload" and info["real"] and len(loaded) == len(df)
    datasets.remove_upload("draft_outcomes")
