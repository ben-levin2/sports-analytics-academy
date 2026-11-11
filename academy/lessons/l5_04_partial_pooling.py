import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .. import data, theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner=False)
def _pairs():
    df = data.make_pressure_hierarchical()
    edge = df[df["position"] == "EDGE"].sort_values(["player_id", "season"])
    nxt = edge.groupby("player_id").shift(-1)
    p = edge.assign(k2=nxt["successes"], n2=nxt["opportunities"], s2=nxt["season"]).dropna(subset=["k2"])
    return p[p["s2"] == p["season"] + 1].reset_index(drop=True)


def visual():
    p = _pairs()
    raw = p["successes"] / p["opportunities"]
    league = p["successes"].sum() / p["opportunities"].sum()
    se2 = league * (1 - league) / p["opportunities"]
    tau2 = max(raw.var() - se2.mean(), 1e-6)
    scale = st.slider("Between-player spread you assume (× the estimate)", 0.0, 3.0, 1.0, 0.1, key="l54_scale")
    t2 = tau2 * scale ** 2
    B = t2 / (t2 + se2) if t2 > 0 else 0 * se2
    est = league + B * (raw - league)
    target = p["k2"] / p["n2"]
    err = np.average((est - target) ** 2, weights=p["n2"]) * 1e4
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    sc = ax.scatter(raw, est, c=np.log10(p["opportunities"]), cmap="academy_seq", s=10)
    ax.plot([0, 0.3], [0, 0.3], color=T.MUTED, ls="--", lw=1)
    ax.axhline(league, color=T.TEXT, lw=0.8)
    ax.set_xlim(0, 0.3), ax.set_ylim(0, 0.3), ax.set_xlabel("This season's raw rate"), ax.set_ylabel("Pooled estimate")
    plt.colorbar(sc, ax=ax, label="log10(snaps)")
    show_fig(fig)
    st.metric("Error predicting NEXT season (×10⁻⁴, lower = better)", f"{err:.2f}",
              help="0 × = complete pooling, 1 × = estimated spread (partial pooling), large = almost no pooling")


CODE = '''import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from academy.data import make_pressure_hierarchical

df = make_pressure_hierarchical()                                 # SIMULATED
edge = df[df["position"] == "EDGE"].sort_values(["player_id", "season"])
nxt = edge.groupby("player_id").shift(-1)                         # the same player's NEXT season
pairs = edge.assign(k2=nxt["successes"], n2=nxt["opportunities"], s2=nxt["season"]).dropna(subset=["k2"])
pairs = pairs[pairs["s2"] == pairs["season"] + 1]

raw = pairs["successes"] / pairs["opportunities"]                 # this season's raw rate
target = pairs["k2"] / pairs["n2"]                                # what we try to predict
league = pairs["successes"].sum() / pairs["opportunities"].sum()  # the EDGE-wide rate

se2 = league * (1 - league) / pairs["opportunities"]              # luck in each player's rate (L1.6)
tau2 = raw.var() - se2.mean()                                     # real between-player spread (total minus luck)
B = tau2 / (tau2 + se2)                                           # weight on the player's own data
partial = league + B * (raw - league)

w = pairs["n2"]
for name, est in [("no pooling (raw)", raw), ("complete pooling (league)", pd.Series(league, index=pairs.index)),
                  ("partial pooling", partial)]:
    print(f"{name:<26} next-season error {np.average((est - target) ** 2, weights=w) * 1e4:.2f} x 1e-4")
print(f"\\nReal spread between EDGE players: {np.sqrt(tau2):.3f}; weight on own data at 50 snaps "
      f"{tau2 / (tau2 + league * (1 - league) / 50):.2f}, at 400 snaps {tau2 / (tau2 + league * (1 - league) / 400):.2f}")

# Ridge regression with the right penalty gives EXACTLY the same estimates
ridge = Ridge(alpha=league * (1 - league) / tau2, fit_intercept=False)
ridge.fit(np.eye(len(pairs)), (raw - league).to_numpy(), sample_weight=pairs["opportunities"].to_numpy())
print("Ridge with penalty sigma^2 / tau^2 matches partial pooling:", np.allclose(league + ridge.coef_, partial))
'''

LESSON = Lesson(
    id="l5-04-partial-pooling", level=5, order=4,
    title="Hierarchical Models, Part 1: Partial Pooling",
    subtitle="Why players in the same position should share information",
    minutes=20,
    concept_ids=["hierarchical-bayes"],
    simple="""
To estimate an edge rusher's true pressure rate you have three choices:

- **No pooling:** trust only his own numbers. Fine for 500 snaps; wild for 40.
- **Complete pooling:** give every EDGE the position average. Stable, but ignores real differences.
- **Partial pooling:** start from the position average and move toward his own numbers **as far as his data deserves**. Lots of snaps, move far. Few snaps, stay close.

That's exactly the Level 1 shrinkage. A **hierarchical model** makes it a general recipe:
- players are pooled toward their **position**,
- positions (and teams, and experience levels) are pooled toward the **population**,
- and the model **learns from the data** how much pooling each level deserves.

"Players in the same position share information" means a player with little data borrows strength from similar players.
""",
    problem="Every player has a different amount of data. You need estimates that are stable for small samples and faithful for big ones, all at once.",
    why="Partial pooling beats both extremes on average, and a hierarchy lets you pool along several football-meaningful groupings at once.",
    when_useful="""
- Players with very different amounts of data
- Natural groupings: positions, teams, experience, conferences
- Small groups (a team with 3 EDGEs) that need to borrow strength
""",
    when_not="""
- When the grouping doesn't predict anything: pooling toward a meaningless group adds bias
- When the group average itself is driven by WHO is in the group (team rates mostly reflect roster talent). Here pooling teams can backfire; in this simulation, a team's one-season pressure rate predicted its later seasons worse than the league average did
- Huge samples for everyone: pooling barely changes anything
""",
    compare="""
**Ridge regression vs hierarchical modeling:** ridge shrinks every estimate toward zero with a penalty you tune. Partial pooling shrinks toward a group mean by an amount learned from the data. With the penalty set to σ²/τ² (luck ÷ real spread), they give **identical** estimates, as the code proves. The hierarchical version also gives uncertainty and handles several levels at once.

**Hierarchical vs Level 1 shrinkage:** Level 1 is a one-level hierarchy (players within position) with point-estimated priors. Level 5 adds teams and experience, and full uncertainty through MCMC (L5.5).
""",
    sports="""
**Fictional scenario (simulated).** 661 EDGE player-seasons with a next season to predict. Error predicting next season (×10⁻⁴):

| Approach | Error |
|---|---|
| No pooling (raw rate) | 23.96 |
| Complete pooling (EDGE average) | 27.90 |
| **Partial pooling** | **18.43** |

Partial pooling wins clearly. The weight on a player's own data is about **0.39 at 50 snaps** and **0.84 at 400 snaps**: the data earns its weight.

**Honest caution:** pooling the same way at the TEAM level did worse than the plain league average in this simulation, because a team's rate one season mostly reflects which players it had. Grouping must match how the effect really works.
""",
    takeaway="Partial pooling moves each player from his group's average toward his own data, by as much as his data deserves. It beats both extremes when the grouping is meaningful.",
    visual=visual,
    visual_caption="0× = complete pooling (everyone at the line), large = almost no pooling (everyone on the diagonal). The error is lowest near the estimated spread (1×).",
    math=[
        MathBlock(
            "Step 1: two levels of variation",
            r"\theta_j \sim \mathcal{N}(\mu, \tau^2), \qquad \hat\theta_j \mid \theta_j \sim \mathcal{N}(\theta_j, \sigma_j^2)",
            """
- $\\theta_j$: player $j$'s true rate; $\\mu$: the position average
- $\\tau$: how much players **really** differ (between-player spread)
- $\\hat\\theta_j$: his observed rate; $\\sigma_j^2$: its luck, which shrinks with snaps ($\\approx \\mu(1-\\mu)/n_j$)
""",
        ),
        MathBlock(
            "Step 2: the partially pooled estimate",
            r"\tilde\theta_j = \mu + B_j(\hat\theta_j - \mu), \qquad B_j = \frac{\tau^2}{\tau^2 + \sigma_j^2}",
            """
- $B_j$: weight on his own data, between 0 (complete pooling) and 1 (no pooling)
- Big $\\tau$ (players really differ) or small $\\sigma_j$ (lots of snaps) push $B_j$ toward 1
""",
        ),
        MathBlock(
            "Step 3: learning τ from data (and the ridge link)",
            r"\hat\tau^2 = \text{Var}(\hat\theta_j) - \overline{\sigma_j^2}, \qquad \text{ridge penalty } \lambda = \sigma^2/\tau^2",
            """
- Total spread of observed rates minus the average luck = the real spread
- Ridge regression with penalty $\\lambda$ shrinks exactly like partial pooling with that $\\tau$
- A full hierarchical model (L5.5) learns $\\tau$ with uncertainty, by MCMC
""",
        ),
    ],
    code=CODE,
    code_notes="Replace 'EDGE' with 'LB'. Linebackers have fewer pass-rush snaps, so their weights on own data are lower.",
    walkthrough=[
        ("nxt = edge.groupby('player_id').shift(-1)", "Line up each player-season with the same player's next season, the thing we try to predict."),
        ("raw = pairs['successes'] / pairs['opportunities']", "This season's raw pressure rate: the no-pooling estimate."),
        ("league = ...sum() / ...sum()", "The EDGE-wide rate: the complete-pooling estimate."),
        ("se2 = league * (1 - league) / pairs['opportunities']", "How much luck is in each player's rate, given his snaps."),
        ("tau2 = raw.var() - se2.mean()", "Real between-player spread = total spread minus average luck."),
        ("B = tau2 / (tau2 + se2)", "Each player's weight on his own data."),
        ("partial = league + B * (raw - league)", "The partially pooled estimate."),
        ("Ridge(alpha=league * (1 - league) / tau2, ...)", "Ridge regression with the matching penalty: identical numbers, different name."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Model 7 connection:** the redesign is still open (current candidate: draft outlook), but any player-level estimate with uneven data benefits from partial pooling by position.
- **Model 8 connection:** Model 8's gain came where production was thin. That's where partial pooling moves estimates the most.
- **Choose groupings that work:** test whether pooling by team, conference or experience actually predicts better on later seasons before using it.
""",
    quiz=[
        Question(
            "q1", "mc", "Partial pooling means:",
            "Each player's estimate is pulled toward his group's average by an amount that depends on how much data he has.",
            options=["Pull each player toward his group average, less so with more data",
                     "Give everyone the group average", "Use only each player's own data", "Average two random players"],
            notes=["Correct.", "That's complete pooling.", "That's no pooling.", "Not a method."],
            answer=0),
        Question(
            "q2", "numeric", "τ² = 0.0014 and a player's luck σ² = 0.0021. What is his weight on his own data, B?",
            "0.0014 / (0.0014 + 0.0021) = 0.4.", category="Application", answer=0.4, tolerance=0.005),
        Question(
            "q3", "mc", "A player has 600 snaps. Compared with a 40-snap player, his estimate is:",
            "More data means less luck (smaller σ²), so B is closer to 1 and his estimate stays near his own rate.",
            options=["Closer to his own raw rate", "Closer to the group average", "Exactly the group average", "Undefined"],
            notes=["Correct.", "That's the 40-snap player.", "Only with complete pooling.", "It's well defined."],
            answer=0),
        Question(
            "q4", "short", "How are ridge regression and partial pooling related?",
            "With penalty λ = σ²/τ², ridge gives the same shrunken estimates; the hierarchical model learns τ from data and adds uncertainty.",
            keywords=[["same", "identical", "equivalent", "equal", "match"],
                      ["penalty", "lambda", "shrink", "tau", "spread"]],
            model_answer="They give the same estimates when ridge's penalty equals luck ÷ real spread (σ²/τ²). The hierarchical model learns that spread from the data and adds uncertainty."),
        Question(
            "q5", "mc", "Pooling team pressure rates made predictions worse here. The most likely reason:",
            "Team rates mostly reflect which players were on the roster, which changes from year to year, so the 'team effect' doesn't carry forward.",
            category="Application",
            options=["Team rates mostly reflect roster talent, which changes", "Pooling never works",
                     "There were too many teams", "The math is wrong"],
            notes=["Correct.", "It worked for players.", "Not the issue.", "The method is fine; the grouping didn't fit."],
            answer=0),
    ],
)
