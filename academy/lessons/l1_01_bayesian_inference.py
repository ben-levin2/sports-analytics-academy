import streamlit as st

from ..models import Lesson, MathBlock, Question
from ..ui import show_fig
from ._viz import beta_update_fig


def visual():
    c1, c2 = st.columns([1, 2])
    with c1:
        mean = st.slider("Prior: typical completion rate", 0.40, 0.80, 0.62, 0.01, key="l11_mean")
        strength = st.slider("Prior strength (pretend past attempts)", 2, 400, 100, key="l11_strength")
        n = st.slider("New attempts observed", 1, 200, 15, key="l11_n")
        pct = st.slider("Completion % in the new sample", 0, 100, 93, key="l11_pct")
    k = round(n * pct / 100)
    a0, b0 = mean * strength, (1 - mean) * strength
    post_mean = (a0 + k) / (a0 + b0 + n)
    with c2:
        show_fig(beta_update_fig(a0, b0, k, n))
    m1, m2, m3 = st.columns(3)
    m1.metric("Raw rate", f"{k / n:.1%}")
    m2.metric("Prior mean", f"{mean:.1%}")
    m3.metric("Posterior mean", f"{post_mean:.1%}")


CODE = '''import numpy as np
from scipy import stats

# --- SIMULATED example: a fictional freshman QB in a spring scrimmage ---
completions, attempts = 14, 15

# Prior: "FBS QBs complete about 62% of passes", worth about 100 attempts of evidence
prior_mean, prior_strength = 0.62, 100
alpha0 = prior_mean * prior_strength          # 62 "prior completions"
beta0 = (1 - prior_mean) * prior_strength     # 38 "prior incompletions"

# Bayesian update (Beta-Binomial): add successes to alpha, failures to beta
alpha1 = alpha0 + completions
beta1 = beta0 + (attempts - completions)

posterior = stats.beta(alpha1, beta1)
lo, hi = posterior.ppf([0.025, 0.975])

print(f"Raw completion rate:   {completions / attempts:.3f}")
print(f"Prior mean:            {prior_mean:.3f}")
print(f"Posterior mean:        {posterior.mean():.3f}")
print(f"95% credible interval: ({lo:.3f}, {hi:.3f})")

# What if the same ~93% rate held up over more attempts?
for n in (15, 60, 200, 800):
    k = round(0.933 * n)
    post_mean = (alpha0 + k) / (alpha0 + beta0 + n)
    print(f"{n:>4} attempts -> posterior mean {post_mean:.3f}")
'''

LESSON = Lesson(
    id="l1-01-bayesian-inference", level=1, order=1,
    title="Bayesian Inference: Updating Beliefs with Evidence",
    subtitle="How to treat a small sample honestly",
    minutes=15,
    concept_ids=["bayesian-inference"],
    simple="""
Bayesian inference is a disciplined way to **update what you believe when new evidence arrives**.

1. Start with what you already believe (the **prior**).
2. Look at the new evidence (the **data**).
3. Blend the two into an updated belief (the **posterior**).

That is how a good coach already thinks. A true freshman goes 14-for-15 in a scrimmage. You don't conclude he's a 93% passer, and you don't ignore it either. You nudge your opinion upward, and you nudge it further if he keeps doing it for a whole season.
""",
    problem="Small samples are noisy. A raw average from 15 attempts can sit far from a player's true ability. Bayesian updating tells you how much to trust it.",
    why="You get an estimate **and** an honest measure of uncertainty, and you can use outside knowledge (position averages, past seasons) instead of throwing it away.",
    when_useful="""
- Few observations (freshmen, transfers, early season)
- Real background knowledge exists (league averages)
- You need probabilities for decisions, such as "chance he is above average"
""",
    when_not="""
- Huge samples: the prior washes out and a plain average is just as good
- No defensible prior, and no appetite to explain one
- You only need a quick descriptive stat for a report
""",
    compare="""
**Raw average:** simplest, but overreacts to small samples.
**Frequentist confidence interval:** also quantifies uncertainty, but cannot say "the probability the true rate is above X".
**Ridge / shrinkage regression:** closely related, since many penalties are priors in disguise.

A Bayesian model is **not** automatically better. With 2,000 attempts it gives essentially the same answer as the average, with more machinery.
""",
    sports="""
**Fictional scenario (simulated numbers).** A true freshman QB, call him *QB-17*, completes **14 of 15** passes in a spring scrimmage (93.3%).

- **What we already know:** FBS quarterbacks complete roughly **62%** of their passes. We treat that as worth about **100 attempts** of evidence.
- **What we saw:** 14 of 15.

A raw-average thinker says "93%!". A Bayesian blends 100 attempts of experience at 62% with 15 attempts at 93%, and lands near **66%**. That is better than average, but not a superstar yet. If he keeps hitting 93% over 200 attempts, the estimate climbs to about 83%, because the data is earning more weight.
""",
    takeaway="Posterior = prior knowledge + new evidence, weighted by how much of each you have.",
    visual=visual,
    visual_caption="Drag the sliders. Watch the blue posterior sit between the grey prior and the amber data. With few attempts it hugs the prior. With many attempts it moves to the data.",
    math=[
        MathBlock(
            "Bayes' theorem",
            r"P(\theta \mid D) = \frac{P(D \mid \theta)\,P(\theta)}{P(D)}",
            """
- $\\theta$: the unknown true completion rate (what we want to learn)
- $D$: the data we observed (14 completions in 15 attempts)
- $P(\\theta)$: the **prior**, our belief about $\\theta$ before the data
- $P(D \\mid \\theta)$: the **likelihood**, how likely this data is if the true rate were $\\theta$
- $P(D)$: the **evidence**, a normalizing constant that makes the result sum to 1
- $P(\\theta \\mid D)$: the **posterior**, our updated belief
""",
        ),
        MathBlock(
            "The Beta-Binomial shortcut",
            r"\theta \sim \text{Beta}(\alpha,\beta),\quad k \mid \theta \sim \text{Binomial}(n,\theta)\;\;\Rightarrow\;\;\theta \mid k \sim \text{Beta}(\alpha + k,\; \beta + n - k)",
            """
- $\\alpha,\\beta$: prior "completions" and "incompletions" (their sum is the prior strength)
- $n$: number of new attempts, $k$: number of new completions
- The posterior is a Beta again, so updating is just **addition**
""",
            "This neat closed form works because the Beta prior is *conjugate* to the Binomial likelihood. Most models don't have one, which is why Level 5 covers MCMC.",
        ),
        MathBlock(
            "The posterior mean is a weighted average",
            r"E[\theta \mid D] = \frac{\alpha+\beta}{\alpha+\beta+n}\cdot\frac{\alpha}{\alpha+\beta} \;+\; \frac{n}{\alpha+\beta+n}\cdot\frac{k}{n}",
            """
- First term: the **prior mean** $\\frac{\\alpha}{\\alpha+\\beta}$ with weight $\\frac{\\alpha+\\beta}{\\alpha+\\beta+n}$
- Second term: the **raw rate** $\\frac{k}{n}$ with weight $\\frac{n}{\\alpha+\\beta+n}$
- As $n$ grows, the weight on the raw rate approaches 1
""",
        ),
    ],
    code=CODE,
    code_notes="Try changing `prior_strength` to 10, then 1000, and rerun. Watch how far the posterior mean moves toward 93%.",
    walkthrough=[
        ("from scipy import stats", "Load SciPy's statistics module. It knows the Beta distribution, so we don't have to code it."),
        ("completions, attempts = 14, 15", "The new evidence: 14 completions in 15 attempts."),
        ("prior_mean, prior_strength = 0.62, 100", "Our belief before the scrimmage: about 62%, and we trust that as much as 100 attempts of evidence."),
        ("alpha0 = prior_mean * prior_strength", "Turn the belief into Beta parameters. alpha0 = 62 'pretend completions'."),
        ("beta0 = (1 - prior_mean) * prior_strength", "beta0 = 38 'pretend incompletions'. Together they total the prior strength, 100."),
        ("alpha1 = alpha0 + completions", "The Bayesian update: add real completions to the pretend ones."),
        ("beta1 = beta0 + (attempts - completions)", "...and add real incompletions to the pretend incompletions."),
        ("posterior = stats.beta(alpha1, beta1)", "Build the updated belief as a full distribution, not just one number."),
        ("lo, hi = posterior.ppf([0.025, 0.975])", "ppf finds the values with 2.5% and 97.5% of the belief below them: a 95% credible interval."),
        ("for n in (15, 60, 200, 800):", "Replay the update with more and more attempts at the same ~93% rate, to watch the data take over."),
    ],
    psfal="""
Use this whenever a player has **few observations**: early-season evaluation, transfer-portal targets with limited snaps, true freshmen.

- Replace raw rates (completion %, pressure rate, tackle-win rate) with **shrunk estimates** pulled toward a position or conference baseline.
- Report estimates with **uncertainty**, so a 12-snap star doesn't outrank a 600-snap starter on luck alone.
- Start with this simple Beta-Binomial model. Reach for MCMC only if it truly cannot represent your problem.
""",
    quiz=[
        Question(
            "q1", "mc", "What does Bayesian inference combine to produce a posterior?",
            "The posterior is built from the prior and the likelihood of the observed data.",
            options=["A prior belief and observed data", "Two independent samples of data",
                     "A p-value and a confidence interval", "A raw average and a median"],
            notes=["Right: this is the core recipe.", "Two datasets could be combined, but that's not what makes it Bayesian.",
                   "Those are frequentist tools.", "Those are summaries, not an updating method."],
            answer=0),
        Question(
            "q2", "numeric",
            "Prior: Beta(62, 38). The QB then completes 14 of 15. What is the posterior mean? (decimal or %)",
            "(62 + 14) / (62 + 38 + 15) = 76 / 115 ≈ 0.661.",
            category="Application", answer=0.661, tolerance=0.005),
        Question(
            "q3", "mc",
            "QB-A completes 14 of 15 (93.3%). QB-B completes 280 of 300 (93.3%). Same prior for both. Which has the higher posterior mean?",
            "More evidence pulls the estimate further from the prior toward the data: QB-B ≈ 0.855 vs QB-A ≈ 0.661.",
            category="Application",
            options=["QB-A, because small samples are more impressive",
                     "QB-B, because 300 attempts move the estimate further from the prior",
                     "They are equal because the raw rates are equal",
                     "It can't be determined without knowing the opponents"],
            notes=["Small samples are less reliable, not more impressive.", "Correct.",
                   "Equal raw rates do not mean equal evidence.", "Opponent strength matters in real life, but this question is about sample size."],
            answer=1),
        Question(
            "q4", "short", "In your own words, why not just trust a 14-for-15 sample?",
            "Small samples are noisy, so we pull the estimate toward a baseline.",
            keywords=[["small", "few", "sample", "noise", "noisy", "luck", "variance", "limited"],
                      ["prior", "baseline", "average", "typical", "regress", "shrink", "league"]],
            model_answer="Fifteen attempts is a small sample, so luck can dominate. The prior (typical completion rate) keeps the estimate from overreacting."),
        Question(
            "q5", "mc", "A QB has 2,000 career attempts. How will his Bayesian estimate compare to his raw average?",
            "With 2,000 attempts the data overwhelms a prior worth ~100 attempts.",
            options=["Dramatically different from the raw average", "Nearly identical to the raw average",
                     "The raw average becomes meaningless", "Bayesian inference can't be used with that much data"],
            notes=["Only true when the prior is very strong.", "Correct: the data outweighs the prior.",
                   "A big sample makes the raw average more meaningful.", "It works at any sample size; it just matters less."],
            answer=1),
    ],
)
