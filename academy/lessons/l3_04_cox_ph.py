import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import data, survival as S, theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner=False)
def _model():
    df = data.make_roster_careers()
    X, _ = S.design(df)
    return S.cox_fit(X, df["seasons_observed"], df["contributed"])


def visual():
    m = _model()
    c1, c2 = st.columns([1, 2])
    with c1:
        z = st.slider("Recruit A's rating, in SDs above average", -2.0, 2.0, 1.0, 0.25, key="l34_z")
    hr = float(np.exp(m["beta"][0] * z))
    times = m["H0"]["time"].to_numpy()
    H0 = m["H0"]["cum_hazard"].to_numpy()
    fig, ax = plt.subplots()
    ax.step(np.r_[0, times], np.r_[0, 1 - np.exp(-H0)], where="post", color=T.MUTED, lw=2, label="Average recruit")
    ax.step(np.r_[0, times], np.r_[0, 1 - np.exp(-H0 * hr)], where="post", color=T.BLUE, lw=2.6, label=f"Recruit A ({z:+.2f} SD)")
    ax.set_xlabel("Seasons since enrolling"), ax.set_ylabel("Share who have contributed"), ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    with c2:
        show_fig(fig)
    st.metric("Hazard ratio vs average recruit", f"{hr:.2f}",
              help="Each season, Recruit A is this many times as likely to break through as an average recruit who hasn't yet")


CODE = '''import numpy as np
from academy.data import make_roster_careers
from academy.survival import cox_fit, design

df = make_roster_careers()                                       # SIMULATED
positions = sorted(df["position"].unique())
X, names = design(df, positions=positions)                       # rating (in SDs) + position dummies
model = cox_fit(X, df["seasons_observed"], df["contributed"])    # our own small Cox fitter (NumPy only)

print("Hazard ratios (above 1 = contributes sooner; reference position:", positions[0], ")")
for name, b, se in zip(names, model["beta"], model["se"]):
    lo, hi = np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
    print(f"  {name:<15} {np.exp(b):.2f}   95% CI {lo:.2f}-{hi:.2f}")

# Proportional-hazards check: is the rating effect the same early and late?
early_t = np.minimum(df["seasons_observed"], 1)
early_e = ((df["contributed"] == 1) & (df["seasons_observed"] == 1)).astype(int)
hr_early = np.exp(cox_fit(X[:, :1], early_t, early_e)["beta"][0])
later = df["seasons_observed"] >= 2
hr_late = np.exp(cox_fit(X[later.to_numpy(), :1], df.loc[later, "seasons_observed"] - 1,
                         df.loc[later, "contributed"])["beta"][0])
print(f"\\nRating hazard ratio in season 1: {hr_early:.2f}   in seasons 2+: {hr_late:.2f}")
print("Similar numbers = the 'proportional hazards' assumption looks reasonable.")
'''

LESSON = Lesson(
    id="l3-04-cox-ph", level=3, order=4,
    title="Cox Proportional Hazards: What Speeds Up a Breakthrough?",
    subtitle="Hazard ratios for recruit rating and position",
    minutes=25,
    concept_ids=["cox-ph"],
    simple="""
Kaplan-Meier (L3.3) draws one curve per group. But you want to ask: **how much faster do higher-rated recruits break through, holding position fixed?**

Think of each season as a new chance. For a player who hasn't contributed yet, the **hazard** is his chance of breaking through *this* season. (Again, "hazard" is just the name; it's a good thing here.)

The **Cox model** says each trait multiplies that chance by a fixed amount, every season:
- **Hazard ratio 1.7** for +1 SD of rating: each season, he's 1.7× as likely to break through as an otherwise similar recruit.
- **Hazard ratio 1.0:** the trait makes no difference.
- **Below 1:** the trait slows things down.

It handles censored players exactly like Kaplan-Meier, and never needs to know the baseline shape of the curve.
""",
    problem="Kaplan-Meier can only compare a few groups at a time and can't separate overlapping traits. Cox measures several traits' effects at once, while still handling censoring.",
    why="It gives one easy-to-read number per trait (the hazard ratio) with a confidence interval, and turns into P(contributes by season t) for any recruit.",
    when_useful="""
- Several traits at once (rating, position, program) for a time-to-event outcome
- Censored data such as recent recruiting classes
- Turning a recruit's traits into P(contributes by year 2)
""",
    when_not="""
- When the effect changes over time (rating matters in year 1 but not later): the 'proportional' assumption breaks. Check it
- Very few events (a handful of contributors): estimates are unstable
- Only a yes/no outcome within a window where everyone is fully observed: logistic regression is simpler
""",
    compare="""
**Cox vs logistic regression:** logistic regression needs a fixed window where everyone's outcome is known. Cox uses every season of every player, censored ones included.

**Cox vs Kaplan-Meier:** Kaplan-Meier describes. Cox measures trait effects and can adjust for several traits at once.

**Cox vs splitting by rating tier:** tiers throw away detail within each tier. Cox uses the full rating.
""",
    sports="""
**Fictional scenario (simulated).** The same ~12,400 recruits. A Cox model with rating and position finds:

- **Rating:** hazard ratio about **1.73** per +1 SD of rating. Each season, a recruit 1 SD above average is ~73% more likely to break through than an average recruit who hasn't yet.
- **Positions:** all about **1.0**, with intervals crossing 1. Position doesn't change the pace in this simulation, and the model correctly says so.
- **Proportional-hazards check:** the rating effect is about the same in season 1 as in seasons 2+, so the assumption looks fine here.

Honest caveat for the Level 3 project: turned into P(contributes by year 2), the Cox model beats a simple rating-tier Kaplan-Meier baseline by only about **2%**. It's real, but small.
""",
    takeaway="A hazard ratio says how much a trait multiplies the chance of breaking through each season. Check that the effect really is steady over time.",
    visual=visual,
    visual_caption="Move Recruit A's rating. His whole curve shifts by one constant multiple of the hazard. That's what 'proportional hazards' means.",
    math=[
        MathBlock(
            "Step 1: the hazard",
            r"h(t) = P(\text{contributes in season } t \mid \text{hasn't yet})",
            """
- $h(t)$: the chance of breaking through in season $t$, for a player still waiting
- It can change from season to season (e.g., more players break through in year 2 than in year 1)
""",
        ),
        MathBlock(
            "Step 2: the Cox model",
            r"h(t \mid \mathbf{x}) = h_0(t)\, \exp(\beta_1 x_1 + \beta_2 x_2 + \cdots)",
            """
- $h_0(t)$: the **baseline hazard**: the season-by-season pattern for an 'average' recruit. The model never needs its shape
- $x_1, x_2, \\ldots$: the recruit's traits (rating in SDs, position dummies)
- $\\beta_1, \\beta_2, \\ldots$: how strongly each trait matters (learned from data)
- $\\exp(\\beta_1)$: the **hazard ratio** for a one-unit increase in $x_1$
""",
            "Because x only multiplies h₀(t), the ratio between two players stays the same every season: 'proportional hazards'.",
        ),
        MathBlock(
            "Step 3: from hazard to a probability you can use",
            r"P(T \le t \mid \mathbf{x}) = 1 - \exp\!\bigl(-H_0(t)\, e^{\boldsymbol{\beta}^\top \mathbf{x}}\bigr)",
            """
- $H_0(t)$: the baseline hazard added up through season $t$ (estimated from the data)
- $e^{\\boldsymbol{\\beta}^\\top \\mathbf{x}}$: this recruit's combined hazard ratio
- Result: his probability of contributing by season $t$. This is what the Level 3 project scores with the Brier score (L3.1)
""",
        ),
    ],
    code=CODE,
    code_notes="The proportional-hazards check is a simple version of the real diagnostics. If the two hazard ratios were very different (say 2.5 vs 1.1), the single Cox number would be misleading.",
    walkthrough=[
        ("X, names = design(df, positions=positions)", "Build the trait table: rating converted to SDs from average, plus a 0/1 column for each position except the first (the reference)."),
        ("model = cox_fit(X, df['seasons_observed'], df['contributed'])", "Fit the Cox model. Our fitter finds the β values that best explain who broke through when, among those still waiting."),
        ("np.exp(b)", "Turn each β into a hazard ratio. Above 1 = faster, below 1 = slower."),
        ("np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)", "A 95% confidence interval. If it includes 1, the data can't rule out 'no effect'."),
        ("early_t = np.minimum(df['seasons_observed'], 1)", "Proportional-hazards check, part 1: look only at season 1."),
        ("later = df['seasons_observed'] >= 2", "Part 2: only players still waiting at season 2, restarting the clock there."),
        ("hr_early, hr_late", "If the rating effect is about the same in both, 'proportional' holds."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Model 8 neighbourhood:** Model 8 uses rating plus early production for development. A Cox model of time-to-contribution is a related tool that uses censored recent classes instead of discarding them.
- **Your rule:** the Level 3 project's Production Candidate tab tests whether Cox beats a simple rating-tier baseline for P(contributes by year 2), on later classes only, with a calibration check.
- **Watch for:** rating effects that fade after year 1, and transfers that aren't random. Both break the simple model.
""",
    quiz=[
        Question(
            "q1", "mc", "A hazard ratio of 1.7 for +1 SD of rating means:",
            "Each season, among players still waiting, the higher-rated recruit is 1.7 times as likely to break through.",
            options=["Each season, he's 1.7× as likely to break through as a similar recruit still waiting",
                     "He contributes 1.7 seasons sooner", "He has a 70% chance of contributing",
                     "He plays 1.7× as many snaps"],
            notes=["Correct.", "It's a multiplier on the chance, not a number of seasons.", "It's a ratio, not a probability.", "It's about timing, not snaps."],
            answer=0),
        Question(
            "q2", "numeric", "β for rating (per SD) is 0.55. What is the hazard ratio? (e^0.55)",
            "e^0.55 ≈ 1.733.", category="Application", answer=1.733, tolerance=0.01),
        Question(
            "q3", "mc", "A position's hazard ratio is 1.04 with a 95% CI of 0.94–1.15. What should you conclude?",
            "The interval includes 1, so the data can't distinguish this position from the reference.",
            category="Application",
            options=["No clear evidence the position changes the pace", "That position breaks through 4% faster, definitely",
                     "That position never contributes", "The model failed"],
            notes=["Correct.", "The interval includes 1 (no effect).", "Nothing suggests that.", "The model worked; the effect is just small."],
            answer=0),
        Question(
            "q4", "short", "What does 'proportional hazards' assume, and how could you check it?",
            "That a trait multiplies the hazard by the same amount every season; check by comparing the effect early vs late.",
            keywords=[["same", "constant", "steady", "every season", "doesn't change", "over time"],
                      ["check", "compare", "early", "late", "season 1", "split", "plot", "test"]],
            model_answer="It assumes each trait multiplies the chance of breaking through by the same amount every season. Check it by estimating the effect separately for early and later seasons and seeing whether they match."),
        Question(
            "q5", "mc", "Why use Cox instead of logistic regression for 'contributes by year 3' with recruiting classes through 2025?",
            "Recent classes haven't had 3 seasons. Logistic regression would have to drop them or call them failures; Cox uses what we've seen.",
            category="Application",
            options=["Recent classes are censored; Cox uses them instead of dropping or miscounting them",
                     "Cox is always more accurate", "Logistic regression can't use ratings", "Cox needs no data"],
            notes=["Correct.", "Not always; with complete follow-up they can be similar.", "It can.", "It needs data like anything else."],
            answer=0),
    ],
)
