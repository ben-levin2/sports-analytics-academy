import streamlit as st

from ..models import Lesson, MathBlock, Question
from ..ui import show_fig
from ._viz import multi_posterior_fig


def visual():
    c1, c2 = st.columns([1, 2])
    with c1:
        targets = st.slider("Targets thrown at the CB", 5, 150, 12, key="l12_n")
        pct = st.slider("PBU % in the sample", 0, 60, 25, key="l12_pct")
        strength = st.slider("Informative prior strength", 5, 200, 40, key="l12_strength")
    k = round(targets * pct / 100)
    mean = 0.15
    priors = {
        "Flat prior": (1.0, 1.0),
        f"Informative (strength {strength})": (mean * strength, (1 - mean) * strength),
        f"Very strong (strength {strength * 10})": (mean * strength * 10, (1 - mean) * strength * 10),
    }
    with c2:
        show_fig(multi_posterior_fig(priors, k, targets))
    st.caption(f"Data: {k} PBUs on {targets} targets. League-typical PBU rate used for the informative priors: {mean:.0%}.")


CODE = '''import pandas as pd
from scipy import stats

def beta_prior(mean, strength):
    """Turn 'typical rate' + 'how many attempts is that worth' into Beta(alpha, beta)."""
    return mean * strength, (1 - mean) * strength

# --- SIMULATED example: a fictional cornerback, 3 pass breakups on 12 targets ---
k, n = 3, 12
typical_rate = 0.15

priors = {
    "Flat (know nothing)":  (1, 1),
    "Informative (n=40)":   beta_prior(typical_rate, 40),
    "Too strong (n=400)":   beta_prior(typical_rate, 400),
}

rows = []
for name, (a, b) in priors.items():
    post = stats.beta(a + k, b + n - k)
    lo, hi = post.ppf([0.025, 0.975])
    rows.append({"prior": name, "prior_mean": a / (a + b), "posterior_mean": post.mean(),
                 "ci_low": lo, "ci_high": hi})

print(f"Raw rate: {k / n:.3f}\\n")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# Sensitivity analysis: how much does the answer depend on the prior strength?
print("\\nPrior strength -> posterior mean")
for s in (5, 20, 40, 100, 400):
    a, b = beta_prior(typical_rate, s)
    print(f"{s:>4} -> {(a + k) / (a + b + n):.3f}")
'''

LESSON = Lesson(
    id="l1-02-prior-distributions", level=1, order=2,
    title="Prior Distributions: What You Believe Before the Data",
    subtitle="Weak, informative and too-strong priors",
    minutes=15,
    concept_ids=["prior-distributions"],
    simple="""
A **prior** is your starting belief, written as a distribution instead of a single number. It says which values of the unknown quantity you consider plausible *before* looking at the new data.

Two knobs matter:

- **Where it's centered:** what's a typical value?
- **How strong it is:** how many observations' worth of confidence do you have?

A useful trick: a prior's strength is a **pretend sample size**. "Typical rate 15%, worth 40 targets" means you act as if you've already watched 40 targets at 15%.
""",
    problem="You almost always know *something* before the data arrives (league averages, last season). A prior lets you use that knowledge on purpose and in the open.",
    why="A sensible prior stabilizes estimates from tiny samples. A flat prior lets the data speak. Either way, the assumption is written down where others can challenge it.",
    when_useful="""
- Small or noisy samples, where a prior does real work
- Reliable baselines exist (position, conference, previous seasons)
- You want to compare "what if I'm wrong about the prior?" scenarios
""",
    when_not="""
- Large samples: the prior barely matters, so don't agonize over it
- When you'd pick the prior *to get the answer you want* (that's cheating, not modeling)
- When no one can defend the baseline, so use a flat prior and say so
""",
    compare="""
**Flat prior** (Beta(1,1)): every rate equally plausible, so the posterior mode equals the raw rate. **Informative prior:** pulls estimates toward a baseline. **Empirical Bayes:** estimates the prior from all players' data. This is a practical middle path you'll use in projects. **Ridge regression:** the same idea, a pull toward zero.
""",
    sports="""
**Fictional scenario (simulated).** A cornerback has **3 pass breakups on 12 targets** (25%). League-typical PBU rate is about **15%**.

- **Flat prior** Beta(1,1): posterior mean = 4/14 ≈ **0.286**. It basically believes the sample.
- **Informative prior** Beta(6,34), worth 40 targets: posterior mean = 9/52 ≈ **0.173**. It is skeptical but open.
- **Too-strong prior** Beta(60,340), worth 400 targets: posterior mean = 63/412 ≈ **0.153**. Barely moves. It would take years of data to change its mind.

Same data, three conclusions. The prior matters here, which is why you must justify it.
""",
    takeaway="Priors are assumptions, not facts. State them, justify them, and test how much the answer depends on them.",
    visual=visual,
    visual_caption="Increase the targets to 150 and watch all three curves converge. Then drop it to 5 and watch them disagree.",
    math=[
        MathBlock(
            "The Beta distribution (the standard prior for rates)",
            r"f(\theta;\alpha,\beta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)},\qquad 0<\theta<1",
            """
- $\\theta$: a rate between 0 and 1 (completion %, PBU rate, pressure rate)
- $\\alpha,\\beta$: shape parameters, "prior successes" and "prior failures"
- $B(\\alpha,\\beta)$: a constant that makes the area under the curve equal 1
""",
        ),
        MathBlock(
            "Mean and strength",
            r"\mu = \frac{\alpha}{\alpha+\beta},\qquad \kappa = \alpha+\beta \quad\Longleftrightarrow\quad \alpha = \mu\kappa,\;\; \beta = (1-\mu)\kappa",
            """
- $\\mu$: the prior mean ("typical rate")
- $\\kappa$: the prior strength = pretend sample size
- Use the right-hand pair to build a prior from "typical rate + how confident I am"
""",
            "Example: μ = 0.30 and κ = 50 gives α = 15, β = 35.",
        ),
    ],
    code=CODE,
    code_notes="Change `typical_rate` to 0.25 (the sample's own rate) and see what happens. A prior that already agrees with the data has little to correct.",
    walkthrough=[
        ("def beta_prior(mean, strength):", "A helper that converts plain-English beliefs into Beta parameters."),
        ("return mean * strength, (1 - mean) * strength", "alpha = mean x strength, beta = (1 - mean) x strength. Mean 0.15 and strength 40 gives Beta(6, 34)."),
        ("k, n = 3, 12", "The cornerback's data: 3 pass breakups on 12 targets."),
        ('priors = {', "Three different starting beliefs, so we can compare their conclusions side by side."),
        ("post = stats.beta(a + k, b + n - k)", "The same update rule for every prior: add successes to a, failures to b."),
        ("rows.append({...})", "Save each prior's mean, posterior mean and 95% interval as one row of a table."),
        ("pd.DataFrame(rows)", "Turn the rows into a pandas table so the three priors line up for comparison."),
        ("for s in (5, 20, 40, 100, 400):", "Sensitivity analysis: keep the same prior mean, change only its strength, and watch the answer move."),
    ],
    psfal="""
- Build priors **by position group and conference**, using previous seasons as the baseline.
- **Document every prior choice** next to the result, and run the sensitivity loop above before trusting a ranking.
- Try **empirical Bayes**: estimate the prior's mean and spread from the whole roster instead of guessing.
- If two reasonable priors give different player rankings, report that, since it is a finding.
""",
    quiz=[
        Question(
            "q1", "mc", "A prior Beta(6, 34) means:",
            "Mean = 6/40 = 15%, and strength = 6 + 34 = 40 pretend observations.",
            options=["A 15% typical rate worth about 40 observations", "A 6% rate worth 34 observations",
                     "A 34% rate worth 6 observations", "A 50% rate with no information"],
            notes=["Correct.", "Mean is α/(α+β), not α.", "Mean is α/(α+β), not β.", "That would be Beta(1,1)."],
            answer=0),
        Question(
            "q2", "numeric", "You want a prior with mean 0.30 and strength 50. What is α?",
            "α = μ·κ = 0.30 × 50 = 15.", category="Application", answer=15, tolerance=0.01),
        Question(
            "q3", "mc", "You're evaluating a brand-new kind of player with no history. You pick Beta(1,1). What does that say?",
            "Beta(1,1) is uniform: every rate from 0 to 1 is equally plausible, so the data dominates.",
            category="Application",
            options=["Every rate is equally plausible, so the data drives the answer",
                     "The player is probably average", "The player is probably elite",
                     "The model is invalid"],
            notes=["Correct.", "That would need a prior centered on the average.", "Nothing in Beta(1,1) favors high rates.",
                   "It's a perfectly valid prior, just a very weak one."],
            answer=0),
        Question(
            "q4", "short", "Why should you run a sensitivity check on your prior?",
            "To see whether conclusions depend on a subjective choice.",
            keywords=[["sensitiv", "change", "differ", "depend", "robust", "affect", "influence", "vary"],
                      ["conclusion", "result", "posterior", "estimate", "answer", "ranking"]],
            model_answer="Because the prior is a judgment call. If reasonable priors lead to different conclusions, you need to know that and report it."),
        Question(
            "q5", "mc", "When is the choice of prior LEAST important?",
            "With lots of data, the likelihood overwhelms the prior.",
            options=["With very little data", "With a very large sample", "When the prior is extremely strong",
                     "When the data are surprising"],
            notes=["That's when it matters most.", "Correct.", "A very strong prior dominates.", "Surprising data makes the prior matter more."],
            answer=1),
    ],
)
