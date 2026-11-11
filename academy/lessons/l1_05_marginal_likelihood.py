import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import stats

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig

MODELS = {"Average QB  Beta(62, 38)": (62, 38), "Accurate QB  Beta(72, 28)": (72, 28), "Know nothing  Beta(1, 1)": (1, 1)}


def visual():
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Pass attempts", 10, 400, 200, key="l15_n")
        pct = st.slider("Completion % observed", 40, 90, 70, key="l15_pct")
    k = round(n * pct / 100)
    ks = np.arange(n + 1)
    fig, ax = plt.subplots()
    ev = {}
    for (name, (a, b)), c in zip(MODELS.items(), [T.BLUE, T.AMBER, T.MUTED]):
        pmf = stats.betabinom.pmf(ks, n, a, b)
        ev[name] = pmf[k]
        ax.plot(ks / n, pmf, color=c, lw=2.2, label=name.split("  ")[0])
    ax.axvline(k / n, color=T.ROSE, lw=1.5, ls="--", label=f"Observed: {k}/{n}")
    ax.set_xlabel("Completion % the model predicted BEFORE seeing the data")
    ax.set_ylabel("Predicted probability"), ax.legend(loc="upper left")
    with c2:
        show_fig(fig)
    cols = st.columns(3)
    for col, (name, v) in zip(cols, ev.items()):
        col.metric(name.split("  ")[0], f"{v:.4f}", help="Evidence = height of that model's curve at the observed result")
    names = list(ev)
    st.caption(f"Bayes factor, Accurate vs Average = {ev[names[1]] / ev[names[0]]:.2f}. "
               "Above 1 favours the Accurate model, below 1 the Average model.")


CODE = '''import numpy as np
from scipy import stats

# --- SIMULATED example: a fictional QB completes 140 of 200 passes (70%) ---
k, n = 140, 200

# Three competing MODELS of this quarterback, each written as a prior
models = {
    "Average QB  (about 62%)": (62, 38),
    "Accurate QB (about 72%)": (72, 28),
    "Know nothing (flat)":     (1, 1),
}

# Evidence = P(data | model): the probability each model gave to "140 of 200"
# BEFORE seeing it, averaged over every completion rate the model considered plausible.
evidence = {}
for name, (a, b) in models.items():
    evidence[name] = stats.betabinom.pmf(k, n, a, b)
    print(f"{name}: evidence = {evidence[name]:.5f}")

names = list(models)
bf = evidence[names[1]] / evidence[names[0]]
print(f"\\nBayes factor (Accurate vs Average): {bf:.2f}")
print(f"Bayes factor (Average vs Know-nothing): {evidence[names[0]] / evidence[names[2]]:.2f}")

# Warning: with ONE player, evidence always prefers a prior squeezed onto his own rate.
for strength in (10, 100, 1000):
    a, b = 0.70 * strength, 0.30 * strength
    print(f"one QB, prior 70% with strength {strength:>4}: evidence {stats.betabinom.pmf(k, n, a, b):.4f}")

# Empirical Bayes needs MANY players. 60 SIMULATED QBs whose true rates vary around 62%.
rng = np.random.default_rng(3)
true_rates = rng.beta(0.62 * 80, 0.38 * 80, size=60)      # real spread: strength 80
att = rng.integers(50, 400, size=60)
comp = rng.binomial(att, true_rates)

print("\\nTotal log-evidence across 60 QBs (prior mean 62%):")
for strength in (10, 30, 80, 200, 1000):
    a, b = 0.62 * strength, 0.38 * strength
    print(f"  strength {strength:>4}: {stats.betabinom.logpmf(comp, att, a, b).sum():.1f}")
print("Best strength is near 80, the true spread. Too weak or too strong both predict worse.")
'''

LESSON = Lesson(
    id="l1-05-marginal-likelihood", level=1, order=5,
    title="Marginal Likelihood: How Well Did the Whole Model Predict?",
    subtitle="Comparing models, and why it powers empirical Bayes",
    minutes=20,
    concept_ids=["marginal-likelihood"],
    simple="""
In L1.3 the **likelihood** scored one specific value: "if his true rate were exactly 70%, how likely is 140 of 200?"

The **marginal likelihood** (also called the **evidence**) scores a whole *model*, a whole set of beliefs:

> *"Before seeing any games, how much probability did this model put on what actually happened?"*

A model that predicted the result well gets high evidence. A model that bet on something else, or spread its bets so thin that it predicted nothing in particular, gets low evidence.

The ratio of two models' evidence is the **Bayes factor**: how many times better one model predicted the data than the other.
""",
    problem="You often have competing explanations: is this QB an average passer or a genuinely accurate one? Does a position-specific prior fit better than one prior for everyone? Evidence compares models on equal terms.",
    why="It rewards models that predicted the data, and automatically penalizes models that are too vague (a built-in Occam's razor). It is also how the Level 1 project picks its prior from the data (empirical Bayes).",
    when_useful="""
- Comparing a small number of clearly defined models
- Choosing a prior's mean and strength from many players (empirical Bayes)
- Simple models where it has a formula (Beta-Binomial, Normal-Normal)
""",
    when_not="""
- Very vague priors: evidence punishes them in ways that depend on arbitrary choices of width
- Complex models: it's very hard to compute, and naive estimates are unreliable
- When you care about prediction on new data: cross-validation answers that more directly
""",
    compare="""
**Likelihood vs marginal likelihood:** likelihood scores one parameter value. Marginal likelihood averages the likelihood over the prior, so it scores the whole model.

**Marginal likelihood vs cross-validation:** evidence asks "how well did the model predict this data, starting from the prior?". Cross-validation asks "how well does the fitted model predict held-out data?". They often agree. When the prior is vague or arbitrary, trust cross-validation more. The Level 1 project uses evidence to *fit* the prior, and time-based validation to *judge* the result.
""",
    sports="""
**Fictional scenario (simulated).** A QB completes **140 of 200** (70%). Three scouts propose three models:

| Model | Prior | Evidence P(140 of 200) |
|---|---|---|
| Average QB | Beta(62, 38), about 62% | 0.0141 |
| Accurate QB | Beta(72, 28), about 72% | **0.0327** |
| Know nothing | Beta(1, 1), flat | 0.0050 |

The Accurate model predicted the data best. But the Bayes factor against the Average model is only **2.3**: real but modest evidence, even after 200 attempts.

Notice the flat model loses to *both*. It spread its predictions evenly over every completion %, from 0% to 100%, so it couldn't put much probability anywhere. That is the built-in penalty for vagueness.
""",
    takeaway="Evidence scores a whole model by how much probability it gave to what actually happened, before it happened.",
    visual=visual,
    visual_caption="Each curve is a model's prediction made before any games. The evidence is the curve's height at the dashed line (what actually happened). Move the sliders: with 30 attempts, the models are hard to tell apart.",
    math=[
        MathBlock(
            "Step 1: the evidence averages the likelihood over the prior",
            r"P(D \mid M) = \int_0^1 P(D \mid \theta, M)\; P(\theta \mid M)\; d\theta",
            """
- $M$: a model (here, a choice of prior)
- $D$: the observed data (140 completions in 200 attempts)
- $P(D \\mid \\theta, M)$: the likelihood from L1.3
- $P(\\theta \\mid M)$: the model's prior from L1.2
- $\\int \\ldots d\\theta$: "add up over every possible rate θ", weighting each by how plausible the model thought it was
""",
            "This is the same P(D) that sat in the denominator of Bayes' theorem in L1.1.",
        ),
        MathBlock(
            "Step 2: for the Beta-Binomial it has a formula",
            r"P(k \mid n, \alpha, \beta) = \binom{n}{k}\,\frac{B(k+\alpha,\; n-k+\beta)}{B(\alpha,\beta)}",
            """
- $k, n$: completions and attempts
- $\\alpha, \\beta$: the prior's parameters
- $B(\\cdot,\\cdot)$: the Beta function, a normalizing constant
- In code this is `stats.betabinom.pmf(k, n, a, b)`
""",
        ),
        MathBlock(
            "Step 3: compare models with the Bayes factor",
            r"\text{BF}_{12} = \frac{P(D \mid M_1)}{P(D \mid M_2)}",
            """
- $\\text{BF}_{12} > 1$: the data favour model 1. Below 1: model 2
- Rough guide: 1–3 weak, 3–10 moderate, above 10 strong evidence
""",
            "Empirical Bayes (the Level 1 project) picks the prior's mean and strength that maximize the evidence across all players at once.",
        ),
    ],
    code=CODE,
    code_notes="Change `k, n` to `14, 20` (still 70%). The Bayes factor shrinks toward 1: less data, less ability to tell models apart. Then change the hidden 80 in `rng.beta(...)` to 300 and see the best strength follow it.",
    walkthrough=[
        ("k, n = 140, 200", "The data: 140 completions in 200 attempts."),
        ("models = {...}", "Three competing models. Each one is just a prior: a different belief about the QB."),
        ("stats.betabinom.pmf(k, n, a, b)", "The beta-binomial gives the probability of k successes in n tries when the rate itself is uncertain (Beta). That is exactly the evidence."),
        ("bf = evidence[...] / evidence[...]", "The Bayes factor: how many times better one model predicted the data than another."),
        ("one QB, prior 70% with strength ...", "With a single player, evidence keeps rising as the prior squeezes onto his own 70%. That's overfitting: one player can't tell you how much players differ."),
        ("true_rates = rng.beta(0.62 * 80, 0.38 * 80, size=60)", "Simulate 60 QBs whose true rates really do vary, with a hidden prior strength of 80."),
        ("comp = rng.binomial(att, true_rates)", "Each QB's completions: luck on top of his true rate."),
        ("stats.betabinom.logpmf(comp, att, a, b).sum()", "Total log-evidence over all 60 QBs for one candidate prior. Logs let us add instead of multiply."),
        ("for strength in (10, 30, 80, 200, 1000):", "Try several prior strengths. The best one sits near 80: empirical Bayes recovers how much players truly differ."),
    ],
    psfal="""
- **Fitting priors from data:** the Level 1 project chooses each position's prior by maximizing the evidence across hundreds of player-seasons (empirical Bayes). It's the same idea Model 8 relies on: lean on the rating-based prior when production is thin.
- **Comparing structures:** does a separate prior per position (EDGE / DT / LB) fit better than one shared prior? Compare their evidence, then confirm with time-based validation.
- **Guardrail:** evidence chooses between models you specified. It cannot tell you that *none* of them is right. Always check out-of-sample predictions too.
""",
    quiz=[
        Question(
            "q1", "mc", "The marginal likelihood (evidence) of a model is:",
            "Evidence is the likelihood averaged over the model's prior: the probability the model assigned to the observed data before seeing it.",
            options=["The probability the model gave to the observed data, averaged over its prior",
                     "The highest point of the likelihood curve", "The probability that the model is true",
                     "The posterior mean"],
            notes=["Correct.", "That's the maximum likelihood, for one best θ only.",
                   "Evidence is P(data | model), not P(model | data).", "The posterior mean is an estimate of θ, not a model score."],
            answer=0),
        Question(
            "q2", "numeric", "Model A's evidence is 0.032 and model B's is 0.014. What is the Bayes factor of A vs B?",
            "0.032 / 0.014 ≈ 2.29: the data are about 2.3 times more probable under A.",
            category="Application", answer=2.29, tolerance=0.05),
        Question(
            "q3", "mc", "Why does a flat 'know nothing' prior often have LOW evidence?",
            "It spreads its predictions over every possible outcome, so it can't put much probability on the one that happened.",
            options=["It spreads its bets over every outcome, so no single outcome gets much probability",
                     "Flat priors are mathematically invalid", "It always predicts 50%",
                     "It ignores the data entirely"],
            notes=["Correct: that's the built-in Occam's razor.", "They're valid, just vague.",
                   "Its prior mean is 50%, but it considers every rate equally plausible.", "Its posterior still uses the data."],
            answer=0),
        Question(
            "q4", "short", "How does the Level 1 project use the marginal likelihood?",
            "It fits the prior (mean and strength) by maximizing the evidence across many players: empirical Bayes.",
            category="Application",
            keywords=[["prior", "mean", "strength"], ["maxim", "fit", "choose", "pick", "estimate", "best"]],
            model_answer="It chooses the prior's mean and strength (per position) that maximize the evidence over all player-seasons. That's empirical Bayes."),
        Question(
            "q5", "mc", "You want to know which model will predict NEXT season best. Evidence and time-based validation disagree. Which do you trust for that question?",
            "Validation on later seasons directly measures what you care about. Evidence depends on prior choices and only scores the data you already have.",
            category="Application",
            options=["Time-based validation, because it directly tests predictions on future data",
                     "Evidence, because it is Bayesian", "Neither; pick the simpler model always",
                     "Whichever gives the larger number"],
            notes=["Correct.", "Being Bayesian doesn't make it the right tool for this question.",
                   "Simplicity matters, but you should check prediction directly.", "The two numbers are on different scales."],
            answer=0),
    ],
)
