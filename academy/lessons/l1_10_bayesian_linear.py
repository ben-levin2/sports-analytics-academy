import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig

SIGMA = 0.9
PRIOR_MEAN, PRIOR_SD = [0.0, 0.5], [0.5, 0.25]


@st.cache_data(show_spinner=False)
def _data():
    from ..data import make_draft_classes
    df = make_draft_classes()
    z = ((df["rating"] - df["rating"].mean()) / df["rating"].std()).to_numpy()
    y = ((df["year1_percentile"] - 50) / 25).to_numpy()
    return z, y, (df["class_year"] >= 2014).to_numpy()


def visual():
    from ..smalldata import bayes_linear
    z, y, test = _data()
    n = st.select_slider("Training recruits (earliest classes first)", [4, 8, 16, 40, 130], value=8, key="l110_n")
    idx = np.flatnonzero(~test)[:n]
    X = np.c_[np.ones(n), z[idx]]
    ols = np.linalg.lstsq(X, y[idx], rcond=None)[0]
    post = bayes_linear(X, y[idx], PRIOR_MEAN, PRIOR_SD, SIGMA)
    grid = np.linspace(-2.5, 2.5, 100)
    G = np.c_[np.ones_like(grid), grid]
    mean = G @ post["mean"]
    sd = np.sqrt(np.einsum("ij,jk,ik->i", G, post["cov"], G))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.scatter(z[idx], y[idx], color=T.NEUTRAL_LIGHT, s=26, zorder=3, label="Training recruits")
    ax.fill_between(grid, mean - 2 * sd, mean + 2 * sd, color=T.PRIMARY, alpha=0.2, label="Posterior line (±2 SD)")
    ax.plot(grid, mean, color=T.PRIMARY, lw=2.4, label="Bayesian line")
    ax.plot(grid, ols[0] + ols[1] * grid, color=T.ACCENT, lw=2, label="Ordinary least squares")
    ax.plot(grid, PRIOR_MEAN[1] * grid, color=T.FAINT, lw=1.4, ls="--", label="Prior line")
    ax.set_xlim(-2.5, 2.5), ax.set_xlabel("Recruiting rating (SDs from average)"), ax.set_ylabel("Year-1 production (scaled)")
    ax.legend(fontsize=7, ncol=2)
    show_fig(fig)
    c = st.columns(3)
    c[0].metric("OLS slope", f"{ols[1]:+.2f}")
    c[1].metric("Bayesian slope", f"{post['mean'][1]:+.2f}", f"± {np.sqrt(post['cov'][1, 1]):.2f}", delta_color="off")
    c[2].metric("Prior slope", "+0.50", "± 0.25", delta_color="off")


CODE = '''import numpy as np
from sklearn.linear_model import Ridge
from academy.data import make_draft_classes
from academy.smalldata import bayes_linear, predict_bayes_linear

df = make_draft_classes()                               # SIMULATED program
rating_z = (df["rating"] - df["rating"].mean()) / df["rating"].std()
prod = (df["year1_percentile"] - 50) / 25               # year-1 production, roughly in SD-like units
test = df["class_year"] >= 2014                         # later classes are the test
Xt, yt = np.c_[np.ones(test.sum()), rating_z[test]], prod[test].to_numpy()

SIGMA = 0.9                                             # noise SD we assume, from earlier seasons
PRIOR_MEAN, PRIOR_SD = [0.0, 0.5], [0.5, 0.25]          # intercept ~ N(0, 0.5^2), rating slope ~ N(0.5, 0.25^2)
print("Prior slope: 0.50 +/- 0.25")
for n in (4, 8, 16, 40, 130):
    idx = np.flatnonzero(~test.to_numpy())[:n]
    X, y = np.c_[np.ones(n), rating_z.iloc[idx]], prod.iloc[idx].to_numpy()
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    post = bayes_linear(X, y, PRIOR_MEAN, PRIOR_SD, SIGMA)
    mu, sd = predict_bayes_linear(post, Xt)
    rmse = lambda b: np.sqrt(np.mean((Xt @ b - yt) ** 2))
    cover = np.mean(np.abs(yt - mu) <= 1.645 * sd)
    print(f"n={n:>3}: OLS slope {ols[1]:+.2f}  Bayes slope {post['mean'][1]:+.2f} +/- {np.sqrt(post['cov'][1, 1]):.2f}"
          f"  |  test RMSE OLS {rmse(ols):.3f} vs Bayes {rmse(post['mean']):.3f}  |  90% interval covers {cover:.0%}")

# Ridge IS the posterior mean when the prior is centered at 0 (slope prior N(0, tau^2), tiny penalty on the intercept)
tau = 0.5
idx = np.flatnonzero(~test.to_numpy())[:40]
x, y = rating_z.iloc[idx].to_numpy(), prod.iloc[idx].to_numpy()
ridge = Ridge(alpha=SIGMA ** 2 / tau ** 2).fit(x[:, None], y)
bayes0 = bayes_linear(np.c_[np.ones(40), x], y, [0, 0], [100.0, tau], SIGMA)
print(f"Ridge slope {ridge.coef_[0]:.4f} vs posterior mean with a Normal(0, {tau}^2) prior {bayes0['mean'][1]:.4f}")
'''

LESSON = Lesson(
    id="l1-10-bayesian-linear", level=1, order=10,
    title="Bayesian Linear Regression: The Exact Case",
    subtitle="Priors, data and a posterior you can write down, with no approximation",
    minutes=25,
    concept_ids=["bayesian-linear"],
    simple="""
**Linear regression** fits a straight line to a number you measure (year-1 production, say) from an input (recruiting rating).

The **Bayesian** version puts a prior on the line's intercept and slope and gives you back a full **posterior**: not one line, but a whole cloud of plausible lines, narrow where data is plentiful and wide where it isn't.

This is the one case where everything can be written as a clean formula. With Normal priors and Normal noise, the posterior is **exactly** Normal, and the formula is just "prior precision + data precision". That makes it the best place to see how a prior acts like extra data.
""",
    problem="With 8 or 16 players, an ordinary best-fit line can be wildly off, and it gives you no sense of how off. You want the line plus honest uncertainty, and a way to bring in what you already know.",
    why="You get uncertainty for free (on the slope and on every prediction), small samples are protected by the prior, and as data grows the prior fades by itself. It's also the foundation for Gaussian processes (L3.7) and hierarchical models (L5.4).",
    when_useful="""
- A number to predict (production, rating change, snap share) with few rows
- When you want a prediction interval, not just a point
- When you have a sensible guess for the effect's size, from past seasons or basic football sense
""",
    when_not="""
- Yes/no outcomes: use Bayesian logistic regression (L1.8), which has no exact formula
- When the noise level is badly unknown or clearly not Normal (heavy-tailed outliers): extra care is needed
- Curved relationships: add curves, or use a Gaussian process
""",
    compare="""
**Ordinary least squares:** the best-fit line only. **Ridge (L1.7):** the same fit as a Bayesian posterior *mean* with a prior centered at zero. **Bayesian linear regression:** the whole posterior, so you can see uncertainty and put the prior's center where you like. **Bayesian logistic (L1.8):** the yes/no cousin, solved by approximation. **Gaussian process regression (L3.7):** the same math with a flexible curve instead of a line.
""",
    sports="""
**Fictional scenario (simulated).** Predict a recruit's year-1 production (scaled) from his recruiting rating. The prior, written before looking: slope **0.50 ± 0.25** (higher ratings help, moderately), intercept 0 ± 0.5, noise SD 0.9 (assumed from earlier seasons). Train on the earliest recruits; test on 2014 onward.

| Training recruits | OLS slope | Bayesian slope | Test RMSE: OLS | Test RMSE: Bayes | 90% interval covers |
|---|---|---|---|---|---|
| 4 | +1.33 | +0.64 ± 0.23 | 1.111 | **0.900** | 95% |
| 8 | +1.18 | +0.64 ± 0.22 | 1.042 | **0.910** | 95% |
| 16 | +0.71 | +0.59 ± 0.19 | 0.955 | 0.946 | 93% |
| 40 | +0.59 | +0.56 ± 0.13 | 0.909 | 0.910 | 92% |
| 130 | +0.57 | +0.57 ± 0.07 | 0.910 | 0.910 | 91% |

- With **4 to 8 recruits**, the OLS slope is absurd (+1.33), and the prior steadies it: error falls from 1.111 to 0.900.
- By **40 recruits** the two agree. The prior has faded; the data has taken over.
- The Bayesian 90% prediction intervals contain 91% to 95% of test recruits, so the uncertainty is honest.
- **Ridge is a posterior mean.** With a Normal(0, 0.5²) prior, ridge and the Bayesian posterior mean give the identical slope: 0.5398.
""",
    takeaway="Bayesian linear regression is the exact case: posterior precision = prior precision + data precision. A prior is extra data, it matters most when you have the least, and it gives you honest uncertainty.",
    visual=visual,
    visual_caption="Dashed: your prior line. Amber: the ordinary fit. Blue: the Bayesian line and its uncertainty band. With 4 recruits the amber line is steep and the blue one is held near the prior; with 130 they overlap.",
    math=[
        MathBlock(
            "Step 1: the model and the prior",
            r"y_i = \beta_0 + \beta_1 z_i + \varepsilon_i, \quad \varepsilon_i \sim \text{Normal}(0, \sigma^2), \quad \beta \sim \text{Normal}(m_0,\; S_0)",
            """
- $y_i$: recruit $i$'s year-1 production; $z_i$: his standardized rating
- $\\beta_0, \\beta_1$: intercept and slope; $\\varepsilon_i$ (epsilon): the leftover noise, with standard deviation $\\sigma$ (assumed known here)
- $m_0$: prior means (0 and 0.5); $S_0$: prior covariance (here diagonal, with SDs 0.5 and 0.25)
""",
        ),
        MathBlock(
            "Step 2: the exact posterior",
            r"S_N^{-1} = S_0^{-1} + \frac{1}{\sigma^2} X^\top X, \qquad m_N = S_N\Big(S_0^{-1} m_0 + \frac{1}{\sigma^2} X^\top y\Big)",
            """
- $X$: the table of inputs (a column of 1s and the standardized ratings); $y$: the outcomes
- $S_N^{-1}$: posterior **precision** = prior precision $S_0^{-1}$ + data precision $X^\\top X/\\sigma^2$. Precisions add
- $m_N$: the posterior mean: a precision-weighted blend of the prior mean and what the data says
- $S_N$: the posterior covariance; its diagonal gives each slope's uncertainty
""",
            "This is exact, not an approximation. It's the same 'precisions add' idea as L1.8's Laplace step, here without any approximating.",
        ),
        MathBlock(
            "Step 3: predicting a new recruit",
            r"y_\ast \mid x_\ast \sim \text{Normal}\Big(x_\ast^\top m_N,\;\; \sigma^2 + x_\ast^\top S_N\, x_\ast\Big)",
            """
- $x_\\ast$: the new recruit's inputs; $y_\\ast$: his unknown outcome
- The predictive variance has two parts: $\\sigma^2$, the noise you can never remove, and $x_\\ast^\\top S_N x_\\ast$, your uncertainty about the line
- The second part shrinks as data grows; the first never does
""",
        ),
    ],
    code=CODE,
    code_notes="Change PRIOR_MEAN to [0.0, -0.5] (a confidently wrong sign). With 4 recruits the test error rises; by 130 the data has overruled it.",
    walkthrough=[
        ("rating_z = (df['rating'] - df['rating'].mean()) / df['rating'].std()", "Standardize the input so a slope means 'per 1 SD of rating'."),
        ("prod = (df['year1_percentile'] - 50) / 25", "Rescale the outcome to roughly SD-like units so the prior is easy to read."),
        ("SIGMA = 0.9", "The noise SD, assumed known. A fuller treatment learns it too; this keeps the math exact."),
        ("PRIOR_MEAN, PRIOR_SD = [0.0, 0.5], [0.5, 0.25]", "Priors written down before looking: slope 0.5 ± 0.25, intercept 0 ± 0.5."),
        ("idx = np.flatnonzero(~test.to_numpy())[:n]", "Take the n earliest recruits as training data; test on 2014 onward."),
        ("bayes_linear(X, y, PRIOR_MEAN, PRIOR_SD, SIGMA)", "Compute the exact posterior: precisions add, then solve for the mean."),
        ("predict_bayes_linear(post, Xt)", "Predictive mean and SD for each test recruit: noise plus line uncertainty."),
        ("np.abs(yt - mu) <= 1.645 * sd", "Check the 90% prediction interval: how often does the real outcome land inside it?"),
        ("Ridge(alpha=SIGMA ** 2 / tau ** 2)", "Ridge with alpha = σ²/τ² gives the same slope as the posterior mean under a Normal(0, τ²) prior."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Any continuous early-career number** (a production score, a rating change) that you predict from a few inputs has this problem when classes are small: the plain line swings, the Bayesian line holds steady and tells you how unsure it is.
- **Prior from outside data:** the prior's center and width are a place to bring knowledge from other seasons or programs (L5.7), but never from the same seasons you test on.
- **Pre-set test:** it doesn't replace your rule. A Bayesian line still has to beat a simple baseline on the pre-set test before it's used.
""",
    quiz=[
        Question(
            "q1", "mc", "In Bayesian linear regression with Normal priors and Normal noise, what form is the posterior?",
            "Exactly Normal, with precision = prior precision + data precision.",
            concept="bayesian-linear",
            options=["Exactly Normal", "Uniform", "Only approximately Normal", "It has no closed form"],
            notes=["Correct: no approximation needed.", "The prior and data both shape it.", "That's the logistic case (L1.8).", "It does: that's the point of this lesson."],
            answer=0),
        Question(
            "q2", "numeric", "Prior slope precision is 16 (SD 0.25). The data adds precision 36. What is the posterior SD of the slope? (2 decimals)",
            "Precisions add: 16 + 36 = 52; SD = 1/√52 ≈ 0.14.",
            concept="bayesian-linear", category="Application", answer=0.14, tolerance=0.01),
        Question(
            "q3", "mc", "As you add more training recruits, what happens to the prior's influence?",
            "The data precision grows while the prior's stays fixed, so the prior fades.",
            concept="bayesian-linear",
            options=["It fades", "It grows", "It stays the same", "It flips sign"],
            notes=["Correct: at 130 recruits the OLS and Bayes slopes match.", "The reverse.", "Its share of the total shrinks.", "Priors don't flip."],
            answer=0),
        Question(
            "q4", "mc", "A predicted outcome has two kinds of uncertainty. Which one never shrinks as you add data?",
            "The noise σ² is irreducible; uncertainty about the line shrinks.",
            concept="bayesian-linear", category="Application",
            options=["The noise σ²", "Uncertainty about the slope", "Uncertainty about the intercept", "Both shrink to zero"],
            notes=["Correct.", "That shrinks as data grows.", "That shrinks too.", "The noise doesn't."],
            answer=0),
        Question(
            "q5", "short", "How is ridge regression related to Bayesian linear regression?",
            "Ridge's fit equals the posterior mean when the prior on the slopes is a Normal centered at zero.",
            concept="bayesian-linear", category="Application",
            keywords=[["posterior", "mean", "map", "estimate"], ["normal", "prior", "zero", "centered"]],
            model_answer="Ridge gives the same slopes as the posterior mean of Bayesian linear regression with a Normal prior centered at zero (penalty λ = σ²/τ²). Bayesian regression adds the full uncertainty and lets you center the prior elsewhere."),
    ],
)
