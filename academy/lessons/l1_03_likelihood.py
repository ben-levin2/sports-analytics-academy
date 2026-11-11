import streamlit as st

from ..models import Lesson, MathBlock, Question
from ..ui import show_fig
from ._viz import likelihood_fig


def visual():
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Field-goal attempts (40-49 yds)", 1, 60, 10, key="l13_n")
        pct = st.slider("Make % in the sample", 0, 100, 80, key="l13_pct")
    k = round(n * pct / 100)
    with c2:
        show_fig(likelihood_fig(k, n))
    st.caption("The curve is not a probability distribution over θ. Its area does not equal 1. "
               "It only compares how well different values of θ explain this one dataset.")


CODE = '''import numpy as np
from scipy import stats

# --- SIMULATED example: a fictional kicker, 8 makes in 10 attempts (40-49 yds) ---
makes, attempts = 8, 10

# Likelihood of the data under three hypotheses about his true make rate
for theta in (0.6, 0.8, 0.9):
    like = stats.binom.pmf(makes, attempts, theta)
    print(f"If true rate = {theta:.1f}: P(8 of 10) = {like:.4f}")

# Evaluate the likelihood over a fine grid of possible rates
grid = np.linspace(0.001, 0.999, 999)
lik = stats.binom.pmf(makes, attempts, grid)
mle = grid[lik.argmax()]
print(f"\\nBest-fitting rate (MLE): {mle:.3f}  (= makes/attempts = {makes / attempts:.3f})")

# The likelihood is NOT a probability distribution over theta: its area is not 1
area = lik.sum() * (grid[1] - grid[0])
print(f"Area under likelihood curve: {area:.4f}  (about 1/(n+1) = {1 / (attempts + 1):.4f})")

# Log-likelihood: sums instead of products, which is what real software optimizes
loglik = stats.binom.logpmf(makes, attempts, grid)
print(f"Log-likelihood peaks at the same rate: {grid[loglik.argmax()]:.3f}")

# Different data types need different likelihoods. Sacks per game are counts -> Poisson
sacks = np.array([0, 1, 0, 2, 1, 0, 1, 3, 0, 1])      # fictional edge rusher, 10 games
rate_hat = sacks.mean()                                # MLE of a Poisson rate
print(f"\\nPoisson MLE for sacks/game: {rate_hat:.2f}")
'''

LESSON = Lesson(
    id="l1-03-likelihood", level=1, order=3,
    title="Likelihood: How Well Does Each Possibility Explain the Data?",
    subtitle="The engine inside every statistical model",
    minutes=15,
    concept_ids=["likelihood"],
    simple="""
Suppose you're unsure of a kicker's true ability. For each possible ability, ask:

> *"If his true make rate really were this, how surprising would the results I saw be?"*

The answer, computed for every candidate ability, is the **likelihood**. Abilities that make the observed results *unsurprising* have high likelihood. Abilities that make them bizarre have low likelihood.

Careful: likelihood is **not** "the probability that his ability is X". It ranks explanations. It does not (yet) give probabilities of them.
""",
    problem="You need a precise way to say how well a candidate explanation fits the data. Every model, from simple averages to neural networks, is fit by some version of this idea.",
    why="Likelihood turns a modeling assumption (e.g., makes are Binomial) into a number you can compare, optimize and combine with a prior to get a posterior.",
    when_useful="""
- Fitting any parametric model (rates, counts, regressions)
- Comparing explanations of the same data
- It's the data half of Bayes' theorem
""",
    when_not="""
- On its own it **cannot** give "probability the kicker is elite" (you need a prior for that)
- With tiny samples the best-fit value is extreme (5-for-5 means 100%)
- When the data model is wrong (e.g., plays assumed independent when they aren't)
""",
    compare="""
**Likelihood vs posterior:** likelihood only looks at data, and the posterior also includes the prior. **MLE vs Bayesian mean:** the MLE is the single best-fit value, and the posterior mean is pulled toward the prior. **Least squares:** ordinary regression is maximum likelihood with a Normal (bell-curve) error model. You've been using likelihood already.
""",
    sports="""
**Fictional scenario (simulated).** A kicker makes **8 of 10** field goals from 40-49 yards. Three scouts disagree on his true make rate:

| Hypothesis | P(8 of 10 \\| θ) |
|---|---|
| θ = 0.6 | 0.121 |
| θ = 0.8 | **0.302** |
| θ = 0.9 | 0.194 |

The data fits θ = 0.8 best, but 0.9 isn't far behind, since 8 of 10 is quite plausible for a 90% kicker too. Ten kicks simply can't separate them, and that is exactly what the flat top of the curve shows.
""",
    takeaway="Likelihood ranks explanations by how well they predict what you saw. It is not a probability of the explanation.",
    visual=visual,
    visual_caption="Raise the number of attempts and keep the make % fixed. The curve narrows, which means the data is separating the hypotheses.",
    math=[
        MathBlock(
            "Binomial likelihood for a success rate",
            r"L(\theta \mid k, n) = \binom{n}{k}\,\theta^{k}\,(1-\theta)^{n-k}",
            """
- $\\theta$: the unknown true make rate
- $n$: attempts, $k$: makes
- $\\binom{n}{k}$: the number of ways to arrange $k$ makes among $n$ kicks
- $\\theta^{k}$: probability of the makes. $(1-\\theta)^{n-k}$: probability of the misses
""",
            "Read it as a function of θ with the data (k, n) held fixed.",
        ),
        MathBlock(
            "Log-likelihood and the maximum",
            r"\ell(\theta) = \ln L(\theta) = k\ln\theta + (n-k)\ln(1-\theta) + \text{const},\qquad \hat\theta_{\text{MLE}} = \frac{k}{n}",
            """
- $\\ell(\\theta)$: log-likelihood (turns products into sums, which is numerically stable)
- $\\hat\\theta_{\\text{MLE}}$: the value of θ that maximizes the likelihood, the raw rate
""",
        ),
        MathBlock(
            "A different data model: counts",
            r"P(Y = y \mid \lambda) = \frac{\lambda^{y}e^{-\lambda}}{y!}",
            """
- $Y$: a count (sacks in a game), $y$: the observed count
- $\\lambda$: the average rate of events per game (the thing to estimate)
""",
            "Choosing the likelihood family (Binomial for yes/no, Poisson for counts, Normal for continuous values) is a modeling decision.",
        ),
    ],
    code=CODE,
    code_notes="Change `makes, attempts` to `5, 5`. The MLE becomes 100%. That's the small-sample problem the prior in the previous lessons fixes.",
    walkthrough=[
        ("makes, attempts = 8, 10", "The data: 8 made field goals in 10 tries."),
        ("stats.binom.pmf(makes, attempts, theta)", "The likelihood: if his true rate were theta, how probable is exactly 8 of 10?"),
        ("grid = np.linspace(0.001, 0.999, 999)", "999 candidate true rates between 0 and 1, so we can draw the whole likelihood curve."),
        ("mle = grid[lik.argmax()]", "argmax finds the candidate with the highest likelihood: the maximum likelihood estimate."),
        ("area = lik.sum() * (grid[1] - grid[0])", "Approximate the area under the curve. It is not 1, which proves likelihood is not a probability distribution over theta."),
        ("stats.binom.logpmf(...)", "Log-likelihood. Logs turn tiny products into manageable sums; the peak stays in the same place."),
        ("sacks = np.array([...])", "A different kind of data: counts per game. Counts need a different likelihood (Poisson)."),
        ("rate_hat = sacks.mean()", "For a Poisson model, the best-fitting rate is simply the average count."),
    ],
    psfal="""
- Pick the likelihood to match the data: **Binomial** for rates (completion, success), **Poisson** for counts (sacks, tackles), **Normal** for continuous scores (EPA-like measures).
- Plays are not perfectly independent (same game, same opponent). If counts vary more than a Poisson allows (**overdispersion**), say so, and consider a richer model.
- Every model you'll build later, from regressions to LSTMs, is just a likelihood plus a way to fit it.
""",
    quiz=[
        Question(
            "q1", "mc", "The likelihood function is best described as:",
            "Likelihood is P(data | θ) viewed as a function of θ.",
            options=["P(data | θ) viewed as a function of θ", "P(θ | data), the posterior",
                     "The prior belief about θ", "The probability the model is correct"],
            notes=["Correct.", "That's the posterior, which also needs a prior.", "The prior is chosen before seeing data.", "Likelihood says nothing directly about model correctness."],
            answer=0),
        Question(
            "q2", "numeric", "A kicker makes 8 of 10. If his true rate is 0.8, what is the likelihood P(8 of 10 | θ=0.8)? (Hint: C(10,8)=45)",
            "45 × 0.8⁸ × 0.2² = 45 × 0.16777 × 0.04 ≈ 0.302.",
            category="Application", answer=0.302, tolerance=0.005),
        Question(
            "q3", "mc", "A kicker is 5-for-5. What does the MLE say, and what is the lesson?",
            "The MLE is k/n = 100%, which shows how overconfident raw best-fit values are with tiny samples.",
            category="Application",
            options=["MLE = 100%, which shows MLE can be overconfident with small samples",
                     "MLE = 50%, because small samples are uncertain",
                     "MLE can't be computed", "MLE = the league average"],
            notes=["Correct. A prior would rein this in.", "The MLE ignores uncertainty entirely.", "It's easily computed: 5/5.", "The MLE uses only the data, not the league average."],
            answer=0),
        Question(
            "q4", "short", "What is the difference between a likelihood and a posterior?",
            "The posterior combines the likelihood with the prior.",
            keywords=[["prior", "belief", "before", "background"],
                      ["probab", "distribution", "combine", "multipl", "update", "data", "evidence"]],
            model_answer="The likelihood only measures how well each θ explains the data. The posterior multiplies it by the prior and normalizes, giving an actual probability distribution over θ."),
        Question(
            "q5", "mc", "You want to model sacks per game for an edge rusher. Which data model fits best?",
            "Sacks per game are counts of events, which is the natural domain of the Poisson.",
            category="Application",
            options=["Normal, since all numbers are continuous", "Binomial with n = 1", "Poisson", "Uniform"],
            notes=["Counts are discrete and can't go below zero.", "That models a single yes/no event.", "Correct.", "Uniform treats every value as equally likely."],
            answer=2),
    ],
)
