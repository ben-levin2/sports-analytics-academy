import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig

SEASONS = np.array([1, 2, 3, 4, 5], float)
PLAYER = np.array([42.0, 51.0, 57.0, 60.0, 61.0])     # SIMULATED production index by season


def rbf(a, b, length, height):
    return height ** 2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / length ** 2)


def posterior(x_obs, y_obs, x_new, length=1.5, height=10.0, noise=2.0, mean=50.0):
    if len(x_obs) == 0:
        K = rbf(x_new, x_new, length, height)
        return np.full(len(x_new), mean), K
    K = rbf(x_obs, x_obs, length, height) + noise ** 2 * np.eye(len(x_obs))
    Ks = rbf(x_new, x_obs, length, height)
    Kss = rbf(x_new, x_new, length, height)
    alpha = np.linalg.solve(K, y_obs - mean)
    mu = mean + Ks @ alpha
    cov = Kss - Ks @ np.linalg.solve(K, Ks.T)
    return mu, cov


def visual():
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Seasons observed so far", 0, 4, 3, key="l35_n")
        length = st.slider("How smooth (length scale, seasons)", 0.5, 4.0, 1.5, 0.25, key="l35_len")
    xs = np.linspace(0.5, 5.5, 120)
    mu, cov = posterior(SEASONS[:n], PLAYER[:n], xs, length)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    rng = np.random.default_rng(3)
    draws = rng.multivariate_normal(mu, cov + 1e-6 * np.eye(len(xs)), 3)
    fig, ax = plt.subplots()
    ax.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color=T.BLUE, alpha=0.18, label="95% band")
    for d in draws:
        ax.plot(xs, d, color=T.CYAN, lw=0.9, alpha=0.7)
    ax.plot(xs, mu, color=T.BLUE, lw=2.4, label="Best guess (posterior mean)")
    ax.scatter(SEASONS[:n], PLAYER[:n], color=T.AMBER, s=60, zorder=5, label="Observed seasons")
    ax.scatter(SEASONS[n:], PLAYER[n:], color=T.MUTED, s=30, marker="x", zorder=5, label="Future (hidden)")
    ax.set_xlabel("Season"), ax.set_ylabel("Production index"), ax.set_ylim(15, 85), ax.legend(fontsize=7, loc="lower right")
    with c2:
        show_fig(fig)
    st.caption("Thin lines are three plausible curves the GP considers. With 0 seasons it only knows the population average.")


CODE = '''import numpy as np

# --- SIMULATED: one fictional player's production index in seasons 1-3 ---
x_obs = np.array([1.0, 2.0, 3.0])
y_obs = np.array([42.0, 51.0, 57.0])
x_new = np.array([4.0, 5.0])                       # seasons we want to predict

def rbf(a, b, length=1.5, height=10.0):
    """Similarity between seasons: 1 season apart = very similar, 4 apart = barely related."""
    return height ** 2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / length ** 2)

mean, noise = 50.0, 2.0                            # population average; game-to-game wobble in the index
K = rbf(x_obs, x_obs) + noise ** 2 * np.eye(3)     # how observed seasons relate to each other
K_s = rbf(x_new, x_obs)                            # how new seasons relate to observed ones
K_ss = rbf(x_new, x_new)                           # how new seasons relate to each other

mu = mean + K_s @ np.linalg.solve(K, y_obs - mean)          # best guess
cov = K_ss - K_s @ np.linalg.solve(K, K_s.T)               # remaining uncertainty
sd = np.sqrt(np.diag(cov))

for s, m, d in zip(x_new, mu, sd):
    print(f"Season {s:.0f}: predicted {m:.1f}  +/- {2 * d:.1f} (95%)")
print("Season 5 is further from the data than season 4, so it's less certain and drifts back toward the average.")
'''

LESSON = Lesson(
    id="l3-05-gp-intro", level=3, order=5,
    title="Gaussian Processes, Part 1: Drawing a Curve With Honest Uncertainty",
    subtitle="The idea behind GPs, using one player's development",
    minutes=20,
    concept_ids=["gp-regression"],
    simple="""
You've seen a player's production for three seasons: 42, 51, 57. What happens in seasons 4 and 5?

Ordinary regression picks **one** curve shape in advance (a straight line, a parabola) and fits it.

A **Gaussian process (GP)** works differently. It considers **many possible smooth curves at once**, keeps the ones that pass close to the data, and reports:
- a **best guess** (the average of the kept curves), and
- an **uncertainty band** (how much the kept curves disagree).

Near the seasons you've seen, the curves agree, so the band is narrow. Far from them, they disagree, so the band widens and the guess drifts back toward the population average.

The only thing you tell a GP up front is **what counts as similar**: here, "seasons close together have similar production". That rule is called the **kernel** (Part 2, L3.6).
""",
    problem="You need a prediction that bends to the data without forcing a shape, and that is honest about being less sure the further it reaches.",
    why="GPs give uncertainty that grows exactly where the data runs out, which is what you want for extrapolating a career or filling in a map (Kriging, L3.8).",
    when_useful="""
- Smooth trends over time, age or distance where the shape isn't known in advance
- Small-to-medium data where honest uncertainty matters
- Maps: predicting at places with no data (L3.8)
""",
    when_not="""
- Large data: a plain GP slows down sharply beyond a few thousand points
- When a straight line or simple average already does as well (often true, see L3.7)
- Far-off extrapolation: a GP just reverts to the average there, which may or may not be sensible
""",
    compare="""
**GP vs ordinary regression:** regression commits to one shape. A GP lets the data choose among smooth shapes, and gives honest uncertainty. But more flexibility doesn't mean better predictions. L3.7 tests this directly.

**GP vs Bayesian inference (Level 1):** it's the same recipe: a **prior** (all smooth curves) plus **data** gives a **posterior** (the curves that fit). In Level 1 the unknown was one rate; here it's a whole curve.
""",
    sports="""
**Fictional scenario (simulated).** A player's production index was 42, 51 and 57 in seasons 1–3. Using a GP that assumes seasons one apart are closely related:

- **Season 4:** predicted about **57** with a band of roughly **±10**.
- **Season 5:** predicted a bit lower, about **53**, with a much wider band (about ±17). It's further from the data, so the GP leans back toward the population average of 50.

Notice what it didn't do: it didn't extend the early climb in a straight line. With 0 observed seasons it would simply say "50, ± a lot".
""",
    takeaway="A GP considers many smooth curves, keeps those that fit, and is honestly less sure the further you go from the data.",
    visual=visual,
    visual_caption="Change the number of observed seasons and the smoothness. The band pinches at observed seasons and widens beyond them.",
    math=[
        MathBlock(
            "Step 1: a GP is a distribution over curves",
            r"f(x) \sim \mathcal{GP}\bigl(m(x),\; k(x, x')\bigr)",
            """
- $f(x)$: the unknown curve (production at season $x$)
- $m(x)$: the mean curve before seeing data (here, a flat 50)
- $k(x, x')$: the **kernel**: how similar production should be at seasons $x$ and $x'$
- "$\\sim \\mathcal{GP}$": the curve is uncertain, and any handful of its values follows a bell curve (a multivariate normal)
""",
        ),
        MathBlock(
            "Step 2: the best guess after seeing data",
            r"\mu_* = m(x_*) + K_{*}\,(K + \sigma^2 I)^{-1}\,(\mathbf{y} - m(\mathbf{x}))",
            """
- $x_*$: a new season to predict; $\\mathbf{x}, \\mathbf{y}$: observed seasons and production
- $K$: kernel values between observed seasons; $K_*$: between the new season and observed seasons
- $\\sigma^2$: noise (season-to-season wobble); $I$: the identity matrix
- In words: start at the average, then move toward the observed values, weighted by how similar the new season is to each observed one
""",
        ),
        MathBlock(
            "Step 3: the uncertainty after seeing data",
            r"\sigma_*^2 = k(x_*, x_*) - K_{*}\,(K + \sigma^2 I)^{-1}\,K_{*}^\top",
            """
- $k(x_*, x_*)$: the uncertainty before seeing any data
- The subtracted part: how much the nearby observations tell you. It's large when $x_*$ is close to observed seasons
- Far from the data the subtraction is small, so uncertainty returns to the starting level
""",
        ),
    ],
    code=CODE,
    code_notes="Change `length=1.5` to 4.0 (very smooth) and to 0.6 (wiggly). With 0.6, season 5 knows almost nothing from seasons 1–3.",
    walkthrough=[
        ("x_obs = np.array([1.0, 2.0, 3.0])", "The seasons we've seen."),
        ("def rbf(a, b, length=1.5, height=10.0):", "The kernel: similarity between two seasons that fades with distance. length = how fast it fades; height = how far curves typically stray from the average."),
        ("K = rbf(x_obs, x_obs) + noise ** 2 * np.eye(3)", "Similarities among observed seasons, plus a little noise on the diagonal because each season's number has some luck in it."),
        ("K_s = rbf(x_new, x_obs)", "How strongly each new season is tied to each observed one."),
        ("np.linalg.solve(K, y_obs - mean)", "Work out how much each observation should pull the curve away from the average."),
        ("mu = mean + K_s @ ...", "Best guess: the average plus those pulls, weighted by similarity."),
        ("cov = K_ss - K_s @ np.linalg.solve(K, K_s.T)", "Uncertainty: what we didn't know before, minus what the data told us."),
        ("sd = np.sqrt(np.diag(cov))", "One standard deviation for each predicted season; ±2 of them gives a ~95% band."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Development curves:** a GP is a natural way to describe how a player's production might evolve with honest uncertainty, the kind of question Model 8 addresses with a different method.
- **But test it:** L3.7 shows a GP only *ties* "same as last season" on simulated data. Under your rule, a GP development curve would need to beat that baseline on a test set fixed in advance.
""",
    quiz=[
        Question(
            "q1", "mc", "What does a Gaussian process give you that a single fitted line doesn't?",
            "A best guess plus uncertainty that grows away from the data, without fixing the curve's shape in advance.",
            options=["A best guess plus uncertainty that grows away from the data", "A guaranteed better prediction",
                     "A straight line through every point", "A cluster label for each season"],
            notes=["Correct.", "Flexibility doesn't guarantee accuracy (see L3.7).", "It doesn't force a line or pass exactly through noisy points.", "That's clustering."],
            answer=0),
        Question(
            "q2", "mc", "Far from any observed season, a GP's prediction tends to:",
            "With no nearby data, it falls back to the prior mean with wide uncertainty.",
            options=["Drift back to the average with a wide band", "Continue the last trend forever",
                     "Become exactly zero", "Become more certain"],
            notes=["Correct.", "That's what a straight line does.", "It reverts to the mean, which here is 50.", "It becomes less certain."],
            answer=0),
        Question(
            "q3", "numeric", "A GP predicts 58 with SD 4 for season 4. What's the upper end of the ~95% band (mean + 2 SD)?",
            "58 + 2 × 4 = 66.", category="Application", answer=66, tolerance=0.01),
        Question(
            "q4", "short", "How is a GP like the Bayesian updating from Level 1?",
            "Prior over curves + data gives a posterior over curves; same recipe, with a curve instead of one rate.",
            keywords=[["prior", "before"], ["posterior", "data", "update", "after"]],
            model_answer="It starts with a prior over possible curves, combines it with the data, and ends with a posterior: the same Bayesian recipe, but the unknown is a whole curve instead of one rate."),
        Question(
            "q5", "mc", "What is the one main thing you choose for a GP up front?",
            "The kernel: the rule for what counts as similar.",
            options=["The kernel (what counts as similar)", "The exact curve shape", "The number of clusters", "The final prediction"],
            notes=["Correct.", "That's what ordinary regression needs; GPs avoid it.", "That's K-means.", "That comes out of the model."],
            answer=0),
    ],
)
