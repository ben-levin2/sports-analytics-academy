import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import norm

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner=False)
def _data():
    from ..data import make_draft_classes
    return make_draft_classes()


def visual():
    from sklearn.metrics import roc_auc_score
    from ..smalldata import Design, fit_logistic, predict_logistic
    df = _data()
    c1, c2, c3 = st.columns(3)
    last = c1.select_slider("Train on classes 2008 to", [2010, 2011, 2012, 2013, 2015, 2017], value=2010, key="l18_last")
    pm = c2.slider("Prior mean, rating slope", -1.5, 2.0, 0.8, 0.1, key="l18_pm")
    ps = c3.slider("Prior SD, rating slope", 0.1, 3.0, 0.4, 0.1, key="l18_ps")
    train = df[df["class_year"] <= last]
    test = df[(df["class_year"] >= 2014) & (df["class_year"] <= 2021)]
    d = Design(train)
    X, Xt, y = d.linear(train), d.linear(test), train["drafted"]
    m, s = d.priors(True)
    m[1], s[1] = pm, ps
    flat = fit_logistic(X, y, *d.priors(False))
    post = fit_logistic(X, y, m, s)
    grid = np.linspace(-2.5, 4, 400)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(grid, norm.pdf(grid, pm, ps), color=T.FAINT, lw=2, label="Prior")
    ax.plot(grid, norm.pdf(grid, flat["beta"][1], np.sqrt(flat["cov"][1, 1])), color=T.ACCENT, lw=2, label="Data alone")
    ax.plot(grid, norm.pdf(grid, post["beta"][1], np.sqrt(post["cov"][1, 1])), color=T.PRIMARY, lw=2.4, label="Posterior")
    ax.axvline(0, color=T.BORDER, lw=1)
    ax.set_xlabel("Rating slope (log-odds per SD of rating)"), ax.set_yticks([])
    show_fig(fig)
    k = st.columns(3)
    k[0].metric("Training recruits", f"{len(train)}", f"{int(y.sum())} drafted", delta_color="off")
    k[1].metric("Test AUC, flat priors", f"{roc_auc_score(test['drafted'], predict_logistic(flat, Xt)):.3f}")
    k[2].metric("Test AUC, your priors", f"{roc_auc_score(test['drafted'], predict_logistic(post, Xt, moderated=True)):.3f}")


CODE = '''import numpy as np
from sklearn.metrics import roc_auc_score
from academy.data import make_draft_classes
from academy.smalldata import Design, PRIOR, fit_logistic, predict_logistic

df = make_draft_classes()                        # SIMULATED program, 22 recruits a class
test = df[(df["class_year"] >= 2014) & (df["class_year"] <= 2021)]

def compare(last_train_class):
    train = df[df["class_year"] <= last_train_class]
    d = Design(train)
    X, Xt, y = d.linear(train), d.linear(test), train["drafted"]
    flat = fit_logistic(X, y, *d.priors(informative=False))      # prior SD 100 = no real prior
    info = fit_logistic(X, y, *d.priors(informative=True))       # the priors written down in advance
    auc_flat = roc_auc_score(test["drafted"], predict_logistic(flat, Xt))
    auc_info = roc_auc_score(test["drafted"], predict_logistic(info, Xt, moderated=True))
    print(f"Train 2008-{last_train_class} ({len(train)} recruits, {int(y.sum())} drafted): "
          f"flat AUC {auc_flat:.3f} | informative AUC {auc_info:.3f}")
    return info

print("Priors (mean, SD) in log-odds per 1 SD of input:", PRIOR)
for last in (2010, 2012, 2013):
    post = compare(last)

# Read the posterior for the rating slope from the last fit
m, s = post["beta"][1], np.sqrt(post["cov"][1, 1])
print(f"Rating slope: prior 0.80 +/- 0.40  ->  posterior {m:.2f} +/- {s:.2f}")
print(f"Odds multiplier per +1 SD of rating: {np.exp(m):.2f}x")

# How many 'extra recruits' is the prior worth? Compare prior and data precision.
prior_prec = 1 / 0.40 ** 2
data_prec = 1 / s ** 2 - prior_prec
print(f"Share of the rating-slope information from the prior: {prior_prec / (prior_prec + data_prec):.0%}")
'''

LESSON = Lesson(
    id="l1-08-bayesian-logistic", level=1, order=8,
    title="Bayesian Logistic Regression with Informative Priors",
    subtitle="Priors as extra data: the signs and sizes you already know",
    minutes=25,
    concept_ids=["bayesian-logistic"],
    simple="""
**Logistic regression** predicts a yes/no outcome (drafted or not) as a probability.

A **Bayesian** logistic regression puts a prior on every slope. Last lesson the prior was "near zero" (that's ridge). Here we go further and use what you actually know:

- **Sign:** a higher recruiting rating should make a draft pick *more* likely, not less.
- **Size:** one standard deviation of rating probably multiplies the odds by something like 1.5 to 3, not by 50.

Writing that down is an **informative prior**. It acts like extra data: when your own sample is tiny, the prior carries the model; as data piles up, the data takes over.
""",
    problem="A small program might have 60–130 finished recruits and fewer than 15 draft picks. A model with flat priors can't tell a real effect from luck, and can even get signs backwards.",
    why="You rarely start from zero. Coaches, past research and common sense already pin down the direction and rough size of the main effects. A prior is the honest, testable way to say so, and the posterior shows how much the data moved it.",
    when_useful="""
- Rare outcomes (draft picks, awards) with few events
- Inputs whose direction is not in doubt (rating, early production)
- When you want uncertainty on each effect, not just a single number
""",
    when_not="""
- When you don't actually know the direction: use a prior centered at 0 (plain ridge)
- When the prior came from the same data you test on. That's leakage, not knowledge
- When you'd be tempted to keep adjusting the prior until the test result looks good. Write it down first
""",
    compare="""
**Flat priors:** the same as plain logistic regression. **Ridge (L1.7):** a prior centered at 0, so it shrinks toward "no effect". **Informative prior:** centered where you believe the effect is, so it shrinks toward "the usual effect". **Hierarchical model (L5.4):** learns the prior's center and width from groups. **Transfer learning (L5.7):** gets the prior's center from a large related dataset.
""",
    sports="""
**Fictional scenario (simulated).** Same program as L1.7. Priors written down before looking (log-odds per 1 SD of input):
- rating slope: mean **0.8**, SD **0.4** (positive, moderate)
- year-1 percentile slope: mean **0.8**, SD **0.4**
- each position: mean **0**, SD **0.5** (small, direction unknown)

Test on classes 2014–2021:

| Training classes | Recruits (drafted) | Flat priors AUC | Informative priors AUC |
|---|---|---|---|
| 2008–2010 | 66 (8) | 0.639 | **0.866** |
| 2008–2012 | 110 (9) | 0.840 | **0.868** |
| 2008–2013 | 132 (12) | 0.822 | **0.862** |

With 66 recruits, flat priors are close to guessing for some players; the priors carry the model. With more data the gap narrows.

**Reading the rating slope (2008–2013):** prior 0.80 ± 0.40, posterior **0.97 ± 0.29**. The data nudged it up. About **53%** of the information about that slope still came from the prior, which tells you how small this sample really is.
""",
    takeaway="An informative prior is extra data you write down before looking. It helps most when data is scarcest, and the posterior shows how far the real data moved it.",
    visual=visual,
    visual_caption="Grey: your prior. Amber: what the data alone says (wide with few recruits). Blue: the posterior, a compromise weighted by how certain each one is. Try a wrong-sign prior (mean −1, SD 0.3) with 66 recruits and watch the test AUC fall, then widen the prior.",
    math=[
        MathBlock(
            "Step 1: the model",
            r"P(y_i = 1) = p_i = \frac{1}{1 + e^{-\eta_i}}, \qquad \eta_i = \beta_0 + \beta_1\, z^{\text{rating}}_i + \beta_2\, z^{\text{year1}}_i + \sum_k \gamma_k\, \text{pos}_{ik}",
            """
- $y_i$: 1 if recruit $i$ was drafted within 5 years
- $\\eta_i$ (eta): his log-odds; $p_i$ turns it into a probability between 0 and 1
- $z^{\\text{rating}}_i$, $z^{\\text{year1}}_i$: standardized rating and year-1 percentile (0 = average, 1 = one SD above)
- $\\beta_1$, $\\beta_2$: slopes; $e^{\\beta_1}$ is how much the odds multiply per +1 SD of rating
- $\\text{pos}_{ik}$: 1 if he plays position $k$; $\\gamma_k$ is that position's effect vs the most common one
""",
        ),
        MathBlock(
            "Step 2: informative priors",
            r"\beta_1 \sim \text{Normal}(0.8,\; 0.4^2), \quad \beta_2 \sim \text{Normal}(0.8,\; 0.4^2), \quad \gamma_k \sim \text{Normal}(0,\; 0.5^2)",
            """
- $\\text{Normal}(0.8, 0.4^2)$: "the rating slope is about 0.8, and very likely between 0 and 1.6". In odds terms: one SD of rating multiplies the odds by roughly 1 to 5
- The prior mean sets the **sign and size**; the prior SD sets **how sure** you are
- Position effects get a mean-0 prior: you expect small differences but don't claim a direction
""",
        ),
        MathBlock(
            "Step 3: posterior, and why a prior is like extra data",
            r"\log p(\beta \mid y) = \sum_i \log P(y_i \mid \beta) + \sum_j \log \text{Normal}(\beta_j \mid m_j, s_j^2) + \text{const}, \qquad \frac{1}{\text{SD}^2_{\text{post}}} \approx \frac{1}{s^2} + \frac{1}{\text{SD}^2_{\text{data}}}",
            """
- $m_j, s_j$: prior mean and SD for slope $j$
- The **posterior** combines the likelihood (data) and the prior; we find its peak with Newton's method and use the curvature there for uncertainty (the Laplace approximation)
- **Precision** = 1/SD². Precisions add: prior precision + data precision = posterior precision. The prior's share of the total says how much of the answer is prior and how much is data
""",
            "Predictions use the 'moderated' probability: it averages over the slope uncertainty, which pulls extreme probabilities slightly toward the middle.",
        ),
    ],
    code=CODE,
    code_notes="Change the rating prior in academy/smalldata.py's PRIOR to (−0.8, 0.4), a confidently wrong sign, and rerun. With 66 recruits the AUC drops; with 132 the data starts to fight back.",
    walkthrough=[
        ("test = df[(df['class_year'] >= 2014) & (df['class_year'] <= 2021)]", "A fixed test set of later classes whose draft windows have closed."),
        ("def compare(last_train_class):", "Fit both models on classes 2008 to `last_train_class`, so we can see the effect of sample size."),
        ("flat = fit_logistic(X, y, *d.priors(informative=False))", "Prior SD of 100 on every slope: effectively no prior, i.e. plain logistic regression."),
        ("info = fit_logistic(X, y, *d.priors(informative=True))", "The same model with the priors from PRIOR: rating and year-1 slopes centered at +0.8, positions at 0."),
        ("predict_logistic(info, Xt, moderated=True)", "Probabilities that account for the uncertainty in the slopes."),
        ("m, s = post['beta'][1], np.sqrt(post['cov'][1, 1])", "The posterior mean and SD of the rating slope (Laplace approximation)."),
        ("np.exp(m)", "Turn a log-odds slope into an odds multiplier: here one SD of rating multiplies the draft odds by about 2.6."),
        ("prior_prec / (prior_prec + data_prec)", "Precisions add, so this is the share of the slope's information that came from the prior rather than the recruits."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Model 8 (rating plus early production):** its two main inputs have directions nobody disputes. A Bayesian version would put positive priors on both, which mainly helps where production is thin, the case where Model 8 found its +0.087 AUC.
- **Model 7 (draft-outlook candidate, being redesigned):** draft outcomes are rare, so this is exactly the setting where informative priors help. The Small-Data Draft Model project (P3C) tests that honestly against a plain logistic baseline.
- **Your rule:** priors must be written down before the test set is scored. The project's priors are fixed in code (academy/smalldata.py), not tuned on results.
""",
    quiz=[
        Question(
            "q1", "mc", "What makes a prior 'informative'?",
            "It is centered on a believed value with a stated certainty, instead of being flat or centered at zero.",
            concept="bayesian-logistic",
            options=["It encodes a believed sign and size", "It uses the test data", "It is very wide", "It is always centered at 0"],
            notes=["Correct.", "That would be leakage.", "A very wide prior is close to no prior.", "Centered at 0 is plain ridge."],
            answer=0),
        Question(
            "q2", "mc", "When does an informative prior change the answer the most?",
            "With little data, the likelihood is wide and the prior carries more weight. With lots of data, the data dominates.",
            concept="bayesian-logistic",
            options=["When data is scarce", "When data is plentiful", "Always equally", "Only when the prior is wrong"],
            notes=["Correct: 66 recruits moved AUC from 0.639 to 0.866.", "Then the data swamps the prior.",
                   "Its influence shrinks as data grows.", "A right prior helps too, mostly with little data."],
            answer=0),
        Question(
            "q3", "numeric", "A rating slope's posterior mean is 0.7 (log-odds per SD). By what factor do the odds of being drafted multiply per +1 SD of rating? (2 decimals)",
            "e^0.7 ≈ 2.01.",
            concept="bayesian-logistic", category="Application", answer=2.01, tolerance=0.02),
        Question(
            "q4", "numeric", "Prior SD for a slope is 0.5 and the data alone gives SD 0.5. Roughly what share of the posterior's information comes from the prior? (as a decimal)",
            "Precisions: 1/0.25 = 4 from each; the prior's share is 4/(4+4) = 0.5.",
            concept="bayesian-logistic", category="Application", answer=0.5, tolerance=0.02),
        Question(
            "q5", "short", "Your prior for a slope came from fitting the same seasons you'll test on. What's wrong with that?",
            "It uses test outcomes to build the prior, so the test is no longer independent: that's leakage.",
            concept="bayesian-logistic", category="Application",
            keywords=[["leak", "test", "same data", "future", "peek", "double"],
                      ["prior", "independent", "unfair", "optimistic", "contamin", "count"]],
            model_answer="It's leakage: the prior already contains the test outcomes, so the test score will be too optimistic. Priors must come from knowledge or data from before the test period."),
    ],
)
