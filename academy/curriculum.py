"""The six levels and 28 tracked concepts, plus helpers that combine content with progress.

Two separate measures:
- Lesson completion  = your BEST quiz score on a lesson (>= 70% completed, >= 90% mastered).
- Concept mastery    = your LATEST answers on every question tagged with that concept, across all
                       lessons that teach it. A lucky old attempt can't keep a concept "mastered".
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config, progress
from .lessons import all_lessons
from .models import Concept


@dataclass(frozen=True)
class Level:
    number: int
    title: str
    tagline: str


LEVELS = (
    Level(1, "Bayesian Statistics & Shrinkage", "Reason honestly about small samples and uncertainty."),
    Level(2, "Player Archetypes & Latent Models", "Discover player types hidden in statistical profiles."),
    Level(3, "Player Development & Spatial Modeling", "Model time-to-contribution, geography, and honest forecasts."),
    Level(4, "Sequential Machine Learning", "Learn from game-by-game sequences."),
    Level(5, "Advanced Statistical Modeling", "Fit models that share information across groups."),
    Level(6, "Optimization & Research", "Make constrained decisions and run complete studies."),
)

_C = Concept
CONCEPTS = (
    _C(1, "bayesian-inference", "Bayesian inference", 1, "Update beliefs by combining prior knowledge with new evidence."),
    _C(2, "prior-distributions", "Prior distributions", 1, "Write down what you believe before seeing the data, and how strongly."),
    _C(3, "posterior-distributions", "Posterior distributions", 1, "Your full updated belief: best guess, uncertainty, decision probabilities."),
    _C(4, "likelihood", "Likelihood", 1, "How well each possible explanation predicts the data you saw."),
    _C(5, "marginal-likelihood", "Marginal likelihood / evidence", 1, "How well a whole model predicted the data: the basis of Bayesian model comparison."),
    _C(6, "stochastic-processes", "Stochastic processes", 1, "Randomness unfolding over time: how luck accumulates across a season."),
    _C(7, "latent-spaces", "Latent spaces", 2, "Compress many stats into a few hidden dimensions that describe a player."),
    _C(8, "k-means", "K-means clustering", 2, "Group players with similar profiles into archetypes."),
    _C(9, "autoencoders", "Autoencoders", 2, "A neural network that learns a compressed player profile."),
    _C(10, "vaes", "Variational autoencoders (VAEs)", 2, "Autoencoders with a smooth, probabilistic latent space."),
    _C(11, "survival-analysis", "Survival analysis", 3, "Model time until an event, including players we never saw reach it."),
    _C(12, "cox-ph", "Cox proportional hazards", 3, "How player traits speed up or slow down time-to-event."),
    _C(13, "gp-regression", "Gaussian process regression", 3, "Flexible curves with built-in uncertainty."),
    _C(14, "kernel-functions", "Kernel functions", 3, "The rule that decides which observations are 'similar'."),
    _C(15, "rbf-kernels", "RBF kernels", 3, "A very smooth notion of similarity."),
    _C(16, "matern-kernels", "Matérn kernels", 3, "A rougher, often more realistic notion of similarity."),
    _C(17, "kriging", "Kriging", 3, "Gaussian processes for maps: spatial prediction with uncertainty."),
    _C(18, "hmms", "Hidden Markov models (HMMs)", 4, "Hidden states (hot/cold) that generate observed performance."),
    _C(19, "lstms", "Long short-term memory networks (LSTMs)", 4, "Neural networks that remember across long sequences."),
    _C(20, "seq2seq", "Sequence-to-sequence models", 4, "Map one sequence (past games) to another (future games)."),
    _C(21, "em", "Expectation-maximization (EM)", 5, "Fit models with hidden variables by alternating guess and refit."),
    _C(22, "mcmc", "Markov chain Monte Carlo (MCMC)", 5, "Sample from posteriors too complex for formulas."),
    _C(23, "hierarchical-bayes", "Hierarchical Bayesian modeling", 5, "Partial pooling across positions, teams and players."),
    _C(24, "bayesian-optimization", "Bayesian optimization", 6, "Tune expensive settings with a surrogate model."),
    _C(25, "calibration-brier", "Calibration and Brier score", 3, "Do your 80% predictions happen about 80% of the time?"),
    _C(26, "time-based-cv", "Time-based cross-validation", 3, "Train on earlier seasons, test on later ones, never leak the future."),
    _C(27, "mixed-effects", "Mixed-effects models", 5, "Fixed + random effects: the frequentist cousin of hierarchical Bayes."),
    _C(28, "conformal-prediction", "Conformal prediction", 4, "Prediction intervals with a coverage guarantee."),
)
CONCEPT_BY_ID = {c.id: c for c in CONCEPTS}

# Method comparisons the curriculum must teach, and where they land.
COMPARISONS = (
    ("Bayesian regression vs ordinary regression", 1, "L1.4"),
    ("Ridge regression vs hierarchical modeling", 5, "planned"),
    ("Gaussian processes vs ordinary regression", 3, "planned"),
    ("HMMs vs LSTMs", 4, "planned"),
    ("Calibration vs discrimination", 3, "L3.1"),
    ("Prediction intervals vs confidence vs credible intervals", 1, "L1.4 (intro); full in Level 4"),
    ("Marginal likelihood vs cross-validation", 1, "L1.5"),
    ("Random splits vs time-based validation", 3, "L3.2"),
)


def level(n: int) -> Level:
    return next(l for l in LEVELS if l.number == n)


def concepts_in_level(n: int) -> list:
    return [c for c in CONCEPTS if c.level == n]


def lessons_in_level(n: int) -> list:
    return [l for l in all_lessons() if l.level == n]


def lessons_for_concept(cid: str) -> list:
    return [l for l in all_lessons() if cid in l.concept_ids]


# ---------------------------------------------------------------- concept mastery
def concept_score(state: dict, cid: str):
    """Mean credit on the concept's questions in each teaching lesson's LATEST attempt.
    Returns (score or None, n_questions_answered, all_teaching_lessons_attempted)."""
    vals, all_tried = [], True
    for l in lessons_for_concept(cid):
        attempts = state["lessons"].get(l.id, {}).get("attempts", [])
        if not attempts:
            all_tried = False
            continue
        latest = attempts[-1]["per_q"]
        vals += [float(latest.get(q.id, 0)) for q in l.quiz if l.question_concept(q) == cid]
    return (sum(vals) / len(vals) if vals else None), len(vals), all_tried


def concept_status(state: dict, cid: str) -> str:
    """planned | new | learning | proficient | mastered."""
    if not lessons_for_concept(cid):
        return "planned"
    score, n, all_tried = concept_score(state, cid)
    if score is None:
        return "new"
    if score >= config.MASTERY_THRESHOLD and all_tried:
        return "mastered"
    if score >= config.COMPLETE_THRESHOLD:
        return "proficient"
    return "learning"


# ---------------------------------------------------------------- level / summary
def level_stats(state: dict) -> list:
    out = []
    for lv in LEVELS:
        ls = lessons_in_level(lv.number)
        cs = concepts_in_level(lv.number)
        done = sum(progress.is_completed(state, l.id) for l in ls)
        mastered = sum(concept_status(state, c.id) == "mastered" for c in cs)
        out.append({"level": lv, "lessons": ls, "concepts": cs, "done": done, "total": len(ls),
                    "frac": done / len(ls) if ls else 0.0, "mastered": mastered,
                    "built": sum(bool(lessons_for_concept(c.id)) for c in cs)})
    return out


def current_level(state: dict) -> Level:
    """First level that still has an incomplete lesson; else the last level that has lessons."""
    stats = level_stats(state)
    for s in stats:
        if s["total"] and s["done"] < s["total"]:
            return s["level"]
    with_lessons = [s["level"] for s in stats if s["total"]]
    return with_lessons[-1] if with_lessons else LEVELS[0]


def summary(state: dict) -> dict:
    lessons = all_lessons()
    completed = [l for l in lessons if progress.is_completed(state, l.id)]
    statuses = {c.id: concept_status(state, c.id) for c in CONCEPTS}
    return {
        "lessons": lessons, "total": len(lessons), "completed": completed,
        "mastered_lessons": [l for l in lessons if progress.is_mastered(state, l.id)],
        "concept_status": statuses,
        "concepts_mastered": [c for c in CONCEPTS if statuses[c.id] == "mastered"],
        "concepts_proficient": [c for c in CONCEPTS if statuses[c.id] in ("proficient", "mastered")],
        "concepts_built": [c for c in CONCEPTS if statuses[c.id] != "planned"],
        "frac": len(completed) / len(lessons) if lessons else 0.0,
        "level": current_level(state), "streak": progress.streak(state),
    }


def recommended(state: dict):
    """Return (lesson | None, reason)."""
    lessons = all_lessons()
    for l in lessons:
        st = progress.lesson_status(state, l.id)
        if st == "review":
            return l, (f"Retake the quiz: your best score is {progress.best_score(state, l.id):.0%}, "
                       f"below the {config.COMPLETE_THRESHOLD:.0%} pass mark.")
        if st in ("new", "opened"):
            return l, "Next lesson in the sequence." if st == "new" else "You opened this but haven't passed its quiz yet."
    for c in CONCEPTS:
        if concept_status(state, c.id) in ("learning", "proficient"):
            l = lessons_for_concept(c.id)[0]
            return l, f"Concept {c.num} ({c.name}) isn't mastered on your latest attempt. Retake to lock it in."
    for l in lessons:
        if progress.lesson_status(state, l.id) == "completed":
            return l, f"Passed ({progress.best_score(state, l.id):.0%}) but not yet mastered. Retake for 90%+."
    return None, "Every available lesson is mastered. Work on the Level 1 project next."
