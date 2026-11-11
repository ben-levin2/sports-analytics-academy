import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .. import data, survival as S, theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


def visual():
    df = data.make_roster_careers()
    tiers = pd.cut(df["recruit_rating"], [0, 0.84, 0.89, 1], labels=["< .84", ".84–.89", ".89+"])
    c1, c2 = st.columns([1, 2])
    with c1:
        wrong = st.toggle("Show the WRONG way (drop censored players)", key="l33_wrong")
    fig, ax = plt.subplots()
    for (tier, g), c in zip(df.groupby(tiers, observed=True), [T.ROSE, T.AMBER, T.GREEN]):
        km = S.kaplan_meier(g["seasons_observed"], g["contributed"])
        ax.step(np.r_[0, km["time"]], np.r_[0, 1 - km["survival"]], where="post", color=c, lw=2.2, label=f"rating {tier}")
        if wrong:
            done = g[g["contributed"] == 1]["seasons_observed"]
            share = [(done <= s).mean() for s in range(0, 6)]
            ax.step(range(0, 6), share, where="post", color=c, lw=1.2, ls="--")
    ax.set_xlabel("Seasons since enrolling"), ax.set_ylabel("Share who have contributed"), ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    with c2:
        show_fig(fig)
    if wrong:
        st.caption("Dashed lines only count players who eventually contributed, so they wrongly reach 100% for every tier.")


CODE = '''import numpy as np
import pandas as pd
from academy.data import make_roster_careers

df = make_roster_careers()       # SIMULATED: ~67 teams x 10 classes. Contribution = 300+ snaps in a season
print(f"{len(df)} recruits. {df['contributed'].mean():.0%} reached the threshold while we watched; the rest are censored.\\n")

def kaplan_meier(t, event):
    """Share who have NOT contributed yet after each season, handling censoring correctly."""
    surv, out = 1.0, []
    for s in range(1, int(t.max()) + 1):
        at_risk = (t >= s).sum()                    # still being watched at season s, no event yet
        events = ((t == s) & (event == 1)).sum()    # first contributed in season s
        surv *= 1 - events / at_risk                # chance of NOT contributing this season, chained
        out.append((s, at_risk, events, surv))
    return pd.DataFrame(out, columns=["season", "at_risk", "contributed", "not_yet"])

km = kaplan_meier(df["seasons_observed"].to_numpy(), df["contributed"].to_numpy())
km["contributed_so_far"] = 1 - km["not_yet"]
print(km.round(3).to_string(index=False))

# The two classic mistakes
naive_drop = df[df["contributed"] == 1]["seasons_observed"].mean()
naive_share = df["contributed"].mean()
print(f"\\nWrong #1 (only count contributors): average {naive_drop:.2f} seasons, too fast, ignores who never got there")
print(f"Wrong #2 (raw share who contributed): {naive_share:.0%}, too low, recent classes haven't had time yet")
print(f"Kaplan-Meier: {1 - km.loc[2, 'not_yet']:.0%} have contributed by season 3")
'''

LESSON = Lesson(
    id="l3-03-survival-analysis", level=3, order=3,
    title="Survival Analysis: How Long Until a Recruit Contributes?",
    subtitle="Time-to-event questions and the problem of players we never saw finish",
    minutes=20,
    concept_ids=["survival-analysis"],
    simple="""
Some questions are about **how long until something happens**: how many seasons until a recruit becomes a contributor (say, 300+ snaps in a season), or until he leaves the roster.

The tricky part: for many players **we don't know the answer yet**.
- A freshman from the 2025 class has only played one season. Maybe he contributes next year, maybe never.
- A player who transferred out after two seasons never got the chance.

These players are **censored**: we know they hadn't contributed *as of the last time we saw them*, but not what came after.

**Survival analysis** uses every player correctly, censored ones included. Its simplest tool, the **Kaplan-Meier curve**, goes season by season and asks: *of the players still being watched and still waiting, what share contributed this season?*
""",
    problem="Throwing away censored players, or treating them as 'never contributed', both give wrong answers. Survival analysis uses the partial information they carry.",
    why="Recruiting classes are always incomplete at the recent end. Without survival methods, the most recent and most relevant classes look worse than they are.",
    when_useful="""
- Time until first contribution, first start, departure or injury return
- Data where recent players haven't finished their story
- Comparing groups, like rating tiers or development programs
""",
    when_not="""
- Everyone's outcome is fully known (all players observed for 5 years): simple percentages work
- When 'how long' doesn't matter, only 'whether' within a fixed, fully observed window
- Departures that are related to the outcome (benched players transfer) bend the assumptions. Treat leaving as its own outcome (a competing risk)
""",
    compare="""
**Kaplan-Meier vs raw percentages:** raw percentages treat censored players as failures, so recent classes look bad. Kaplan-Meier only counts players while they're actually being observed.

**Kaplan-Meier vs averaging the contributors' times:** that ignores everyone who hasn't contributed, so it looks too fast.

**Kaplan-Meier vs Cox model (L3.4):** Kaplan-Meier describes one group at a time. Cox measures how traits like recruit rating speed things up.
""",
    sports="""
**Fictional scenario (simulated).** About 12,400 recruits from ~67 teams over 10 classes (2016–2025). Contribution = 300+ snaps in a season.

- Only **50%** reached the threshold while we watched. That sounds low, but the 2024 and 2025 classes have barely started.
- Averaging only the contributors' times gives about **2.2 seasons**. That's too fast, because it ignores everyone still waiting.
- Kaplan-Meier says about **54%** have contributed by season 3, and the curves split clearly by rating tier.

Same data, three answers. Only Kaplan-Meier handles censoring honestly.
""",
    takeaway="Censored players carry real information: they hadn't contributed yet. Survival analysis counts them while they're being watched and stops when they're not.",
    visual=visual,
    visual_caption="Each curve is the share of a rating tier who have contributed by each season. Turn on the wrong way to see how dropping censored players inflates every tier.",
    math=[
        MathBlock(
            "Step 1: the survival function",
            r"S(t) = P(T > t)",
            """
- $T$: the season in which a player first contributes (the time to the event)
- $t$: a specific season, like 2
- $S(t)$: the chance he has **not yet** contributed after season $t$. "Survival" is just a name; nobody is dying
- $1 - S(t)$: the chance he **has** contributed by season $t$, the curves in the visual
""",
        ),
        MathBlock(
            "Step 2: the Kaplan-Meier estimate",
            r"\hat S(t) = \prod_{s \le t} \left(1 - \frac{d_s}{n_s}\right)",
            """
- $n_s$: players **at risk** in season $s$: still being watched and haven't contributed yet
- $d_s$: players who first contributed in season $s$
- $\\frac{d_s}{n_s}$: this season's contribution rate among those still waiting
- $\\prod$: multiply season by season ("didn't contribute in season 1" AND "didn't in season 2" ...)
""",
            "A censored player stays in n until his last observed season, then quietly leaves the count. That's the whole trick.",
        ),
        MathBlock(
            "Step 3: uncertainty (Greenwood's formula)",
            r"\widehat{\text{Var}}\bigl[\hat S(t)\bigr] = \hat S(t)^2 \sum_{s \le t} \frac{d_s}{n_s\,(n_s - d_s)}",
            """
- Gives the width of a confidence band around the curve
- Later seasons have fewer players at risk ($n_s$ small), so the band widens
""",
        ),
    ],
    code=CODE,
    code_notes="Filter to one class, e.g. `df = df[df['class_year'] == 2023]`, and rerun. The curve stops early because nobody from 2023 has been watched for 5 seasons.",
    walkthrough=[
        ("df = make_roster_careers()", "Each row is a recruit: how many seasons we watched him, and whether he contributed."),
        ("for s in range(1, int(t.max()) + 1):", "Walk forward one season at a time."),
        ("at_risk = (t >= s).sum()", "Players still being watched in season s who haven't contributed yet. Censored players drop out after their last season."),
        ("events = ((t == s) & (event == 1)).sum()", "Players whose first contributing season is s."),
        ("surv *= 1 - events / at_risk", "Multiply in this season's 'didn't contribute' chance. That's the Kaplan-Meier product."),
        ("km['contributed_so_far'] = 1 - km['not_yet']", "Flip it around: the share who HAVE contributed by each season."),
        ("df[df['contributed'] == 1]['seasons_observed'].mean()", "Mistake 1: averaging only the players who got there. It ignores everyone still waiting."),
        ("df['contributed'].mean()", "Mistake 2: the raw share. It counts recent freshmen as failures."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Roster data:** with ~67 teams over 10 years, every recent class is censored. Time-to-contribution should be estimated with Kaplan-Meier, not raw percentages.
- **Model 8 connection:** Model 8 forecasts development from rating and early production. "Time until contribution" is a related outcome that handles players who haven't finished yet.
- **Departures:** treat "left the roster" as its own outcome too (the Level 3 project lets you switch), because transfers aren't random.
""",
    quiz=[
        Question(
            "q1", "mc", "A 2025 recruit has played one season without reaching 300 snaps. In survival analysis he is:",
            "We know he hasn't contributed yet, but not what happens next: he's censored.",
            options=["Censored: not contributed yet, future unknown", "A failure who never contributes",
                     "Removed from the data", "Counted as contributing in season 2"],
            notes=["Correct.", "We don't know that yet.", "Removing him throws away real information.", "We can't assume that."],
            answer=0),
        Question(
            "q2", "numeric", "100 recruits are at risk in season 1 and 20 contribute. Of the 70 still at risk in season 2, 14 contribute. What is S(2), the share NOT yet contributed?",
            "(1 − 20/100) × (1 − 14/70) = 0.8 × 0.8 = 0.64.", category="Application", answer=0.64, tolerance=0.005),
        Question(
            "q3", "mc", "Why does averaging only contributors' times give an answer that's too fast?",
            "It leaves out everyone who hasn't contributed yet, and those are the slower cases.",
            options=["It leaves out the slower players who haven't contributed yet",
                     "It counts redshirts twice", "It uses the wrong threshold", "It's actually too slow"],
            notes=["Correct.", "Redshirts aren't the issue.", "The threshold is the same.", "It's biased toward fast."],
            answer=0),
        Question(
            "q4", "short", "The raw share of recruits who contributed looks lower for recent classes. Why, and what fixes it?",
            "Recent classes haven't had time (censoring); Kaplan-Meier fixes it.",
            category="Application",
            keywords=[["time", "recent", "censor", "not yet", "haven't", "yet", "short"],
                      ["kaplan", "survival", "at risk", "km"]],
            model_answer="Recent classes haven't had enough seasons yet, so they're censored, not failing. Kaplan-Meier counts players only while they're observed, which fixes the bias."),
        Question(
            "q5", "mc", "Benched players transfer more often. What's the risk of treating every transfer as ordinary censoring?",
            "Censoring assumes leaving is unrelated to the outcome. If benched players leave, the remaining players look better than the whole class really was.",
            category="Application",
            options=["The contribution rate looks too good, because the likely non-contributors left",
                     "No risk; censoring always works", "The curve will reach 100%", "Transfers are counted twice"],
            notes=["Correct. Model departure as its own outcome.", "It assumes leaving is unrelated to the outcome.",
                   "That happens only with the wrong method.", "They aren't."],
            answer=0),
    ],
)
