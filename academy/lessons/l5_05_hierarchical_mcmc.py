import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .. import data, hierarchical as H, theme as T
from ..models import Lesson, MathBlock, Question
from .. import ui
from ..ui import show_fig


@st.cache_data(show_spinner="Running the hierarchical sampler…")
def _fit():
    d = H.prepare(data.make_pressure_hierarchical())
    f = H.gibbs(d)
    return H.diagnostics(f), {k: f[k] for k in ("sd_player", "sd_team", "sd_season")}


def visual():
    diag, sds = _fit()
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6))
    for ax, (k, ch), c in zip(axes, sds.items(), [T.BLUE, T.AMBER, T.GREEN]):
        for chain in ch:
            ax.plot(chain, lw=0.6, color=c, alpha=0.8)
        ax.set_title(k.replace("_", " "), fontsize=9)
    show_fig(fig)
    ui.table(diag)
    st.caption("Three chains per parameter (after warm-up). True simulated values: player spread 0.30, team spread 0.15, "
               "season wobble 0.08, experience steps 0.12 / 0.24 / 0.36.")


CODE = '''import numpy as np
from academy.data import make_pressure_hierarchical
from academy.hierarchical import diagnostics, expit, gibbs, prepare

df = make_pressure_hierarchical()                         # SIMULATED: position, team, experience and talent effects
d = prepare(df)                                           # adds experience + log-odds rate and its luck
fit = gibbs(d, iters=900, burn=300, chains=3)             # our NumPy MCMC sampler (Gibbs steps)

print(diagnostics(fit).round(3).to_string(index=False))

# Turn log-odds back into pressure rates for an average player, by position and experience
beta = fit["beta"].reshape(-1, len(fit["names"]))
for i, pos in enumerate(fit["positions"]):
    base = beta[:, i]
    print(f"{pos:<5} rookie {expit(base).mean():.3f}   3rd+ season {expit(base + beta[:, -1]).mean():.3f}")

# Partial pooling in action: low-snap players' effects are pulled toward 0 the most
u = fit["u"].reshape(-1, len(fit["players"])).mean(0)
snaps = d.groupby("player_id")["opportunities"].sum().reindex(fit["players"]).to_numpy()
print(f"\\nAvg |player effect|, under 100 total snaps: {np.abs(u[snaps < 100]).mean():.3f}; "
      f"800+ snaps: {np.abs(u[snaps >= 800]).mean():.3f}")

try:
    import pymc  # noqa: F401
    print("\\nPyMC is installed: see the project page for the same model written in PyMC.")
except ImportError:
    print("\\nPyMC not installed (optional): pip install -r requirements-optional.txt")
'''

LESSON = Lesson(
    id="l5-05-hierarchical-mcmc", level=5, order=5,
    title="Hierarchical Models, Part 2: Positions, Teams and Experience, Fitted by MCMC",
    subtitle="A full hierarchical Bayesian player model on NumPy",
    minutes=25,
    concept_ids=["hierarchical-bayes", "mcmc"],
    simple="""
Now we put several groupings into one model. On the log-odds scale, a player's pressure rate in a season is:

> **position average** + **experience step** + **team effect** + **player effect** + **season wobble** + **luck**

- **Position averages** and **experience steps** are shared by everyone in that group.
- **Team effects** and **player effects** are *partially pooled*: each is pulled toward 0 with a spread the model learns. Teams with few players, and players with few snaps, are pulled hardest.
- **Season wobble** is real year-to-year change; **luck** is sampling noise from the snap count.

No formula solves this, so we fit it with **MCMC** (L5.2–L5.3): a Gibbs sampler that updates each piece in turn, run as 3 chains, checked with R-hat.

PyMC is the standard library for this. It's optional here; the academy's own NumPy sampler does the job for learning.
""",
    problem="A player's numbers mix several sources: position, scheme (team), experience, talent and luck. A hierarchical model separates them, pooling each where data is thin.",
    why="It answers questions like 'how much does a third season add?' or 'how much do teams differ?' with uncertainty, and it gives each player an estimate that borrows strength from all the groups he belongs to.",
    when_useful="""
- Several meaningful groupings at once (position, team, experience)
- Uneven data across players and teams
- When you want the size of each source of variation, with uncertainty
""",
    when_not="""
- When the extra groupings don't carry real effects: the model adds noise and can lose to simpler shrinkage (the Level 5 project shows this)
- Without convergence checks
- When speed matters more than uncertainty: a mixed-effects model (L5.6) is much faster
""",
    compare="""
**Hierarchical Bayes vs ordinary regression:** ordinary regression gives every player and team its own free number, which overfits small groups. The hierarchical model learns how much groups really differ and shrinks accordingly.

**Hierarchical Bayes vs mixed-effects (L5.6):** same structure. Mixed-effects estimates the spreads as single numbers (fast). Bayes gives them uncertainty and full posteriors (slower, needs MCMC).

**Hierarchical vs Level 1 shrinkage:** Level 1 pooled players toward their position only.
""",
    sports="""
**Fictional scenario (simulated).** About 3,400 player-seasons with modest team and experience effects built in. The sampler recovers them:

| Quantity | Truth | Posterior mean |
|---|---|---|
| Experience step after 1 / 2 / 3+ seasons | 0.12 / 0.24 / 0.36 | **0.13 / 0.26 / 0.36** |
| Player spread | 0.30 | **0.28** |
| Team spread | 0.15 | **0.13** |
| Season wobble | 0.08 | **0.10** |

All R-hats are close to 1.00 after a standard fix (re-centering the effects each round). Before the fix, position averages had R-hat about 1.10.

Low-snap players' estimated effects are pulled much closer to 0 than high-snap players'. That's partial pooling at work.
""",
    takeaway="A hierarchical model splits a player's rate into position, experience, team, talent and luck, pooling each where data is thin. Fit it with MCMC and check convergence.",
    visual=visual,
    visual_caption="Trace plots: three chains per spread parameter, overlapping. That's what convergence looks like. The table lists every parameter with R-hat and ESS.",
    math=[
        MathBlock(
            "Step 1: the model for each player-season",
            r"y_{is} = \beta_{\text{pos}(i)} + \gamma_{\text{exp}(is)} + \tau_{\text{team}(i)} + u_i + \delta_{is} + \varepsilon_{is}",
            """
- $y_{is}$: log-odds of player $i$'s pressure rate in season $s$
- $\\beta$: position averages; $\\gamma$: experience steps (0 for rookies)
- $\\tau$: team effect; $u_i$: player talent
- $\\delta_{is} \\sim \\mathcal{N}(0, \\sigma_s^2)$: season wobble; $\\varepsilon_{is} \\sim \\mathcal{N}(0, v_{is})$: luck, known from the snap count
""",
        ),
        MathBlock(
            "Step 2: partial pooling of teams and players",
            r"\tau_t \sim \mathcal{N}(0, \sigma_{\text{team}}^2), \qquad u_i \sim \mathcal{N}(0, \sigma_{\text{player}}^2)",
            """
- Effects are drawn from a shared distribution whose spread is learned from data
- Small $\\sigma_{\\text{team}}$ means teams barely differ, so team effects are shrunk hard
- Priors on the spreads are weak (inverse-gamma), letting the data decide
""",
        ),
        MathBlock(
            "Step 3: Gibbs sampling, one piece at a time",
            r"u_i \mid \text{rest} \sim \mathcal{N}\!\left(\frac{\sum_s w_{is} r_{is}}{\sum_s w_{is} + 1/\sigma_{\text{player}}^2},\; \frac{1}{\sum_s w_{is} + 1/\sigma_{\text{player}}^2}\right)",
            """
- $r_{is}$: what's left of $y_{is}$ after the other effects; $w_{is} = 1/(v_{is} + \\sigma_s^2)$: how much each season counts
- The $1/\\sigma_{\\text{player}}^2$ term is the pull toward 0: partial pooling, in one line
- The sampler cycles through all such updates thousands of times (MCMC)
""",
        ),
    ],
    code=CODE,
    code_notes="Run `gibbs(d, iters=300, burn=100)` and compare the R-hats. Short runs are cheaper but less trustworthy.",
    walkthrough=[
        ("d = prepare(df)", "Add each player's experience (earlier seasons in the data) and convert rates to log-odds with their luck."),
        ("fit = gibbs(d, iters=900, burn=300, chains=3)", "Run the MCMC sampler: 3 chains of 900 rounds, dropping the first 300 as warm-up."),
        ("diagnostics(fit)", "Every parameter's posterior mean, spread, R-hat and effective sample size."),
        ("beta = fit['beta'].reshape(...)", "Gather all draws of the position averages and experience steps."),
        ("expit(base).mean()", "Convert log-odds back to a pressure rate, averaging over posterior draws."),
        ("u = fit['u']...mean(0)", "Each player's estimated talent effect (posterior mean)."),
        ("np.abs(u[snaps < 100]).mean()", "Low-snap players' effects are smaller: pulled toward 0 by partial pooling."),
        ("import pymc", "PyMC is optional. If it's installed, the project page shows the same model written in PyMC."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Model 7 connection:** whatever the redesigned Model 7 becomes (current candidate: draft outlook), a hierarchical structure (players within positions, teams and experience levels) is a natural way to borrow strength across uneven data.
- **Your rule:** the Level 5 project tests this model against "same as last season" and Level 1 shrinkage on later seasons. It only counts if it beats **both**.
""",
    quiz=[
        Question(
            "q1", "mc", "In this model, why are team effects 'partially pooled'?",
            "They're drawn from a shared distribution with a learned spread, so teams with little data are pulled toward 0.",
            concept="hierarchical-bayes",
            options=["They share a learned spread, so thin-data teams are pulled toward 0",
                     "Every team gets the same effect", "Teams are ignored", "Each team is fit separately with no link"],
            notes=["Correct.", "That's complete pooling.", "They're included.", "That's no pooling."],
            answer=0),
        Question(
            "q2", "numeric", "An experience step of 0.36 on the log-odds scale multiplies the odds by e^0.36. What is that multiplier?",
            "e^0.36 ≈ 1.433, so about 43% higher odds of a pressure.", category="Application",
            concept="hierarchical-bayes", answer=1.433, tolerance=0.005),
        Question(
            "q3", "mc", "Before re-centering, the position averages had R-hat ≈ 1.10. What did that mean?",
            "The chains hadn't converged for those parameters; results weren't trustworthy yet.",
            concept="mcmc",
            options=["Not converged; results weren't trustworthy yet", "The model was wrong",
                     "Positions don't matter", "Perfect convergence"],
            notes=["Correct.", "R-hat checks the sampler.", "Unrelated.", "1.10 is too high."],
            answer=0),
        Question(
            "q4", "short", "Why do low-snap players get smaller estimated talent effects?",
            "Their data is mostly luck, so partial pooling pulls their effect toward the group (0).",
            category="Application", concept="hierarchical-bayes",
            keywords=[["luck", "noise", "few snaps", "little data", "uncertain", "small sample"],
                      ["pool", "shrink", "pulled", "toward 0", "toward zero", "toward the group", "average"]],
            model_answer="Their numbers are mostly luck, so the model trusts them less and partial pooling pulls their effect toward 0 (the group average)."),
        Question(
            "q5", "mc", "When might this hierarchical model LOSE to Level 1 shrinkage?",
            "When the extra groupings (team, experience) carry no real effect: extra parameters add noise.",
            concept="hierarchical-bayes",
            options=["When team and experience effects aren't really there", "Never; more structure always helps",
                     "When there are many players", "When R-hat is 1.00"],
            notes=["Correct (the project shows this).", "More structure can add noise.", "More players usually helps it.", "Convergence doesn't guarantee better predictions."],
            answer=0),
    ],
)
