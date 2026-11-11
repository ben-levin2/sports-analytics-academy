import warnings

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from .. import data, theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner=False)
def _setup():
    dev = data.make_development(n_players=200)
    wide = dev.pivot(index="player_id", columns="season", values="production")
    return wide, wide.mean()


def visual():
    wide, pop = _setup()
    c1, c2 = st.columns([1, 2])
    with c1:
        i = st.slider("Player", 0, 40, 7, key="l37_player")
    y = wide.iloc[i].to_numpy()
    x = np.arange(1, 6)
    xs = np.linspace(1, 6.5, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GaussianProcessRegressor(C(60.0) * RBF(6.0) + WhiteKernel(10.0), optimizer=None).fit(
            x[:4, None], y[:4] - pop.to_numpy()[:4])
    pop_xs = np.interp(xs, x, pop.to_numpy(), right=pop.iloc[-1])
    mu, sd = gp.predict(xs[:, None], return_std=True)
    line = np.polyval(np.polyfit(x[:4], y[:4], 1), xs)
    fig, ax = plt.subplots()
    ax.fill_between(xs, pop_xs + mu - 2 * sd, pop_xs + mu + 2 * sd, color=T.BLUE, alpha=0.18)
    ax.plot(xs, pop_xs + mu, color=T.BLUE, lw=2.4, label="GP (population curve + player offset)")
    ax.plot(xs, line, color=T.AMBER, lw=1.8, ls="--", label="His own straight-line trend")
    ax.scatter(x[:4], y[:4], color=T.TEXT, s=40, zorder=5, label="Seasons 1-4 (used)")
    ax.scatter([5], [y[4]], color=T.ROSE, marker="*", s=160, zorder=5, label="Season 5 (actual)")
    ax.set_xlabel("Season"), ax.set_ylabel("Production index"), ax.legend(fontsize=7, loc="lower right")
    with c2:
        show_fig(fig)


CODE = '''import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from academy.data import make_development
warnings.simplefilter("ignore")

dev = make_development(n_players=200)                       # SIMULATED: 200 players x 5 seasons
wide = dev.pivot(index="player_id", columns="season", values="production")
ids = np.random.default_rng(0).permutation(wide.index)
train, test = ids[:120], ids[120:]                          # learn settings on 120 players, test on 80
pop = wide.loc[train].mean().to_numpy()                     # population development curve

# 1. Fit the kernel's settings by maximizing the marginal likelihood (lesson L1.5).
#    Trick: line training players up far apart (x + 100*i) so they don't influence each other.
X = np.array([[s + 100 * i] for i in range(60) for s in range(1, 6)], float)
Y = np.concatenate([wide.loc[p].to_numpy() - pop for p in train[:60]])
fit = GaussianProcessRegressor(C(20.0) * RBF(2.0, (0.5, 20)) + WhiteKernel(9.0)).fit(X, Y)
print("Fitted kernel:", fit.kernel_)

# 2. For each TEST player, use seasons 1-4 to predict season 5, four different ways
x4 = np.arange(1, 5)[:, None]
errors = {"same as last season": [], "his own straight line": [], "population curve + offset": [], "GP": []}
for p in test:
    y = wide.loc[p].to_numpy()
    errors["same as last season"].append(abs(y[4] - y[3]))
    errors["his own straight line"].append(abs(y[4] - np.polyval(np.polyfit(range(1, 5), y[:4], 1), 5)))
    errors["population curve + offset"].append(abs(y[4] - (pop[4] + np.mean(y[:4] - pop[:4]))))
    gp = GaussianProcessRegressor(fit.kernel_, optimizer=None).fit(x4, y[:4] - pop[:4])
    errors["GP"].append(abs(y[4] - (pop[4] + gp.predict([[5]])[0])))

print("\\nAverage error predicting season 5 (lower is better):")
for name, e in errors.items():
    print(f"  {name:<27} {np.mean(e):.2f}")
'''

LESSON = Lesson(
    id="l3-07-gp-regression", level=3, order=7,
    title="Gaussian Processes, Part 3: Fitting Them, and When Simpler Wins",
    subtitle="Learning kernel settings from data, and a fair contest with ordinary regression",
    minutes=25,
    concept_ids=["gp-regression"],
    simple="""
In Part 1 we picked the GP's settings by hand. In practice you **let the data choose them**: the length scale, the signal height and the noise level.

The GP picks the settings under which the observed data is **least surprising**. That's the **marginal likelihood** from L1.5, now choosing kernel settings instead of a prior.

Then comes the important part: **a fair contest**. Does a GP actually predict better than simple methods? We predict each player's **season 5** from seasons 1–4, four ways, on players the model never saw:

1. **Same as last season**
2. **His own straight-line trend**
3. **Population curve + his personal offset**
4. **GP** (population curve + a GP for his personal deviations)
""",
    problem="A flexible model can look impressive and still predict no better than a one-line rule. You need to fit it properly and then test it against simple baselines.",
    why="This is how you decide whether the extra complexity earns its place, which is your PSFAL rule applied to Gaussian processes.",
    when_useful="""
- When curves really do bend differently for different players and you have enough seasons
- When you need honest uncertainty for each player's forecast
- As a smoother, uncertainty-aware alternative to a hand-picked curve shape
""",
    when_not="""
- When a simple rule already ties it (as here): use the simple rule and say so
- Very short histories (1–2 seasons): there's nothing for the GP to learn
- When the main problem is year-to-year noise rather than curve shape
""",
    compare="""
**GP vs ordinary (linear) regression:** a straight-line trend extrapolates confidently and badly when careers flatten. The GP leans back toward the population and does far better.

**GP vs 'same as last season':** in this simulation they **tie**. The GP adds honest uncertainty, but not accuracy.

**GP vs population curve + offset:** that simple mixed-model-like rule (Level 5) is close behind too.

Being more complex is not the same as being better.
""",
    sports="""
**Fictional scenario (simulated).** 200 players, 5 seasons each. Settings learned on 120 players, tested on the other 80. Average error predicting season 5:

| Method | Error |
|---|---|
| Same as last season | **3.30** |
| GP | **3.30** |
| Population curve + offset | 3.64 |
| His own straight line | 5.86 |

**Honest result:** the GP ties the simplest rule. It clearly beats straight-line extrapolation, which overshoots when careers level off, but it doesn't beat "same as last season". This echoes your Models 9 and 9b, where "same as last season" (AUC 0.713) wasn't beaten either.
""",
    takeaway="Fit the GP's settings from data, then make it beat simple baselines on players it hasn't seen. Here it only ties 'same as last season'.",
    visual=visual,
    visual_caption="Pick a player. The amber straight line keeps climbing; the GP bends toward the population curve, with a band. The star is what actually happened in season 5.",
    math=[
        MathBlock(
            "Step 1: the settings to learn",
            r"\theta = (\ell,\; \sigma_f^2,\; \sigma_n^2)",
            """
- $\\ell$: length scale (how many seasons apart still count as related)
- $\\sigma_f^2$: signal height (how far players stray from the population curve)
- $\\sigma_n^2$: noise (season-to-season luck)
""",
        ),
        MathBlock(
            "Step 2: choose them by the marginal likelihood",
            r"\log p(\mathbf{y} \mid \theta) = -\tfrac12\, \mathbf{y}^\top K_\theta^{-1} \mathbf{y} \;-\; \tfrac12 \log |K_\theta| \;-\; \tfrac{n}{2}\log 2\pi",
            """
- $K_\\theta$: kernel table (plus noise) for the observed points, using settings $\\theta$
- First term: rewards settings that **fit** the data
- Second term ($\\log|K_\\theta|$): penalizes settings that are too flexible. This is the built-in Occam's razor from L1.5
- The GP tries many $\\theta$ and keeps the best (scikit-learn does this automatically in `.fit`)
""",
        ),
        MathBlock(
            "Step 3: the fair contest",
            r"\text{MAE} = \frac{1}{N_{\text{test}}}\sum_{i \in \text{test}} \bigl|\, y_{i,5} - \hat y_{i,5} \,\bigr|",
            """
- $y_{i,5}$: test player $i$'s actual season-5 production
- $\\hat y_{i,5}$: a method's prediction from seasons 1–4 only
- Test players were never used to learn the settings: an honest held-out comparison
""",
        ),
    ],
    code=CODE,
    code_notes="Try predicting season 5 from only seasons 1–2 (change the ranges). Which method copes best with less history?",
    walkthrough=[
        ("train, test = ids[:120], ids[120:]", "Split players: learn on 120, judge on 80 the model never saw."),
        ("pop = wide.loc[train].mean().to_numpy()", "The average development curve, from training players only."),
        ("X = np.array([[s + 100 * i] ...])", "A beginner-friendly trick: place each training player's seasons far apart on the x-axis so the GP treats them as separate players sharing the same settings."),
        ("GaussianProcessRegressor(C(20.0) * RBF(...) + WhiteKernel(9.0)).fit(X, Y)", "Fit: scikit-learn tries kernel settings and keeps those with the highest marginal likelihood."),
        ("errors['same as last season'].append(abs(y[4] - y[3]))", "Baseline 1: predict season 5 = season 4."),
        ("np.polyfit(range(1, 5), y[:4], 1)", "Baseline 2: fit a straight line to his four seasons and extend it."),
        ("pop[4] + np.mean(y[:4] - pop[:4])", "Baseline 3: the population's season 5, shifted by how far above or below average he has been."),
        ("GaussianProcessRegressor(fit.kernel_, optimizer=None).fit(...)", "The GP: reuse the learned settings (no refitting) on this player's deviations from the population curve."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Same lesson as Models 9 and 9b:** a sophisticated next-season model must beat "same as last season" before it's used. Here the GP only ties it.
- **What a GP could still add:** honest per-player uncertainty. If a decision depends on the band, not just the best guess, that's a reason to keep exploring, but as exploratory work under your rule.
- **Where to look next:** Level 5's hierarchical models combine the population curve with player offsets more carefully, which is the natural next contender.
""",
    quiz=[
        Question(
            "q1", "mc", "How does scikit-learn choose a GP's length scale and noise by default?",
            "It maximizes the marginal likelihood: the settings under which the observed data is least surprising.",
            options=["By maximizing the marginal likelihood", "By picking the values that look smoothest",
                     "By using the test set", "Randomly"],
            notes=["Correct (lesson L1.5).", "Looks aren't a criterion.", "That would leak the test set.", "It's an optimization, not a guess."],
            answer=0),
        Question(
            "q2", "mc", "Why does a straight-line trend predict season 5 so badly here?",
            "It keeps extending early growth even when careers flatten out.",
            options=["It keeps climbing after careers level off", "It ignores the data",
                     "It uses the population curve", "It's too uncertain"],
            notes=["Correct.", "It fits the data; it just extrapolates badly.", "It doesn't.", "It gives no uncertainty at all."],
            answer=0),
        Question(
            "q3", "numeric", "Errors: GP 3.30, 'same as last season' 3.30. How much better is the GP, in percent?",
            "(3.30 − 3.30) / 3.30 = 0%. It ties.", category="Application", answer=0, tolerance=0.001),
        Question(
            "q4", "short", "The GP ties 'same as last season'. Under your PSFAL rule, what do you do with it?",
            "Keep it exploratory: it didn't beat the simple baseline on the held-out test.",
            category="Application",
            keywords=[["exploratory", "don't use", "not use", "keep", "lab", "not on", "don't put", "reject"],
                      ["baseline", "didn't beat", "not beat", "tie", "simple", "same as last"]],
            model_answer="Keep it exploratory. It didn't beat the simple 'same as last season' baseline on the held-out test, so it doesn't go on the site. Its uncertainty bands might still be worth exploring."),
        Question(
            "q5", "mc", "Why must the kernel settings be learned on training players only?",
            "Otherwise the test players influence the model and the comparison is no longer fair (leakage).",
            category="Application",
            options=["Otherwise test players leak into the model and the contest isn't fair",
                     "Because GPs can't handle many players", "It's faster", "It doesn't matter"],
            notes=["Correct.", "It's about fairness, not capacity.", "Speed isn't the reason.", "It matters for honest results."],
            answer=0),
    ],
)
