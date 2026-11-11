import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .. import data, hierarchical as H, theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner="Fitting the mixed model…")
def _fit():
    import statsmodels.formula.api as smf
    d = H.prepare(data.make_pressure_hierarchical())
    d = d[d["opportunities"] >= 100].copy()
    d["exp"] = d["experience"].astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("y ~ 0 + C(position) + C(exp)", d, groups=d["player_id"]).fit()
    blup = pd.Series({k: v.iloc[0] for k, v in m.random_effects.items()})
    resid = d["y"] - m.fittedvalues + d["player_id"].map(blup)
    raw = resid.groupby(d["player_id"]).mean()
    n = d.groupby("player_id").size()
    return raw, blup.reindex(raw.index), n, float(np.sqrt(m.cov_re.iloc[0, 0])), float(np.sqrt(m.scale))


def visual():
    raw, blup, n, sd_p, sd_r = _fit()
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for k, c in [(1, T.ROSE), (2, T.AMBER), (3, T.BLUE)]:
        m = (n == k) if k < 3 else (n >= 3)
        ax.scatter(raw[m], blup[m], s=8, color=c, alpha=0.6, label=f"{k}{'+' if k == 3 else ''} season(s)")
    ax.plot([-1.2, 1.2], [-1.2, 1.2], color=T.MUTED, ls="--", lw=1)
    ax.set_xlabel("Player's raw average (vs fixed effects)"), ax.set_ylabel("Mixed-model estimate (BLUP)")
    ax.legend(fontsize=7)
    show_fig(fig)
    st.caption(f"Player spread {sd_p:.2f}, season-to-season residual {sd_r:.2f}. Players with fewer seasons sit "
               "further below the diagonal: their effects are shrunk more.")


CODE = '''import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from academy.data import make_pressure_hierarchical
from academy.hierarchical import prepare
warnings.simplefilter("ignore")

d = prepare(make_pressure_hierarchical())                    # SIMULATED; log-odds rate in column y
d = d[d["opportunities"] >= 100].copy()                      # mixedlm has no per-row luck weights, so drop tiny samples
d["exp"] = d["experience"].astype(str)

# FIXED effects: position and experience (shared by everyone in the group)
# RANDOM effect: a player intercept (each player's own talent, partially pooled)
model = smf.mixedlm("y ~ 0 + C(position) + C(exp)", d, groups=d["player_id"]).fit()
print(model.params.round(3).to_string())

sd_player = np.sqrt(model.cov_re.iloc[0, 0])
sd_resid = np.sqrt(model.scale)
icc = sd_player ** 2 / (sd_player ** 2 + sd_resid ** 2)
print(f"\\nplayer spread {sd_player:.3f}   season-to-season residual {sd_resid:.3f}")
print(f"ICC (share of variation that's stable player talent): {icc:.2f}")

# The random effects (BLUPs) are shrunk: fewer seasons -> more shrinkage
blup = pd.Series({k: v.iloc[0] for k, v in model.random_effects.items()})
raw = (d["y"] - model.fittedvalues + d["player_id"].map(blup)).groupby(d["player_id"]).mean()
n = d.groupby("player_id").size()
for label, mask in [("1 season", n == 1), ("3+ seasons", n >= 3)]:
    print(f"typical shrink factor, {label}: {(blup[mask] / raw[mask]).median():.2f}")
'''

LESSON = Lesson(
    id="l5-06-mixed-effects", level=5, order=6,
    title="Mixed-Effects Models: Fixed Effects, Random Effects, and the Hierarchical Link",
    subtitle="The fast, frequentist cousin of hierarchical Bayes, with statsmodels",
    minutes=20,
    concept_ids=["mixed-effects"],
    simple="""
A **mixed-effects model** is a regression with two kinds of terms:

- **Fixed effects:** things you want a specific number for, shared by everyone in a category. *How much higher is an EDGE's pressure rate than a DT's? How much does a second season add?*
- **Random effects:** a separate number for each member of a large group (each player, each team), **partially pooled** toward zero with a spread the model learns. You care about the spread and the shrunken estimates, not about each one as a free parameter.

It's the **same structure** as the hierarchical model in L5.5. The difference is how it's fitted: mixed-effects models estimate the spreads as single best numbers (fast, no MCMC), while hierarchical Bayes gives them full uncertainty.

The random-effect estimates are called **BLUPs** (best linear unbiased predictions). They're shrunk exactly like Level 1 shrinkage: less data, more shrinkage.
""",
    problem="Repeated measurements on the same players (several seasons each) aren't independent. Ordinary regression ignores that; mixed models handle it and pool players properly.",
    why="It's fast, standard and easy to report: fixed-effect sizes with standard errors, plus how much players really differ (the ICC).",
    when_useful="""
- Repeated seasons per player, players within teams
- When you mainly need fixed-effect sizes (position, experience) and a quick player spread
- As a fast check before a slower full Bayesian model
""",
    when_not="""
- Very few groups (e.g., 3 positions as a random effect): estimating a spread from 3 numbers is unreliable, so make it fixed
- When the uncertainty in the spreads matters for decisions: use hierarchical Bayes
- When noise differs a lot between rows (snap counts): basic mixedlm can't use per-row luck weights; filter or use the Bayesian model
""",
    compare="""
**Mixed-effects vs hierarchical Bayes:** same model structure. Mixed-effects is fast and gives point estimates of the spreads. Bayes gives full uncertainty, handles per-row luck weights, but needs MCMC.

**Random vs fixed effects:** fixed = a free number per category (few categories, you care about each). Random = shrunken numbers from a shared distribution (many members, you care about the spread).

**Mixed-effects vs ridge:** a random intercept is ridge regression on player dummies, with the penalty learned from data (L5.4).
""",
    sports="""
**Fictional scenario (simulated).** Player-seasons with 100+ pass-rush snaps, fitted with `statsmodels`:

- **Fixed effects:** the experience steps come out at about **0.13 / 0.29 / 0.39** (truth 0.12 / 0.24 / 0.36). Close, slightly larger than the Bayesian model's, which also models team effects that this simple version leaves out.
- **Player spread** about **0.33**, season-to-season residual about **0.31**: an ICC of about **0.54**, meaning roughly half the variation between player-seasons is stable talent.
- **Shrinkage:** a one-season player's effect is typically shrunk to about **0.54** of his raw value; a 3+ season player's to about **0.78**.
""",
    takeaway="Mixed models = fixed effects for categories you care about + random effects pooled with a learned spread. Same structure as hierarchical Bayes, fitted faster without full uncertainty.",
    visual=visual,
    visual_caption="Each dot is a player: his raw average vs his mixed-model estimate. Players with fewer seasons are pulled further toward 0.",
    math=[
        MathBlock(
            "Step 1: the model",
            r"y_{is} = \mathbf{x}_{is}^\top \boldsymbol\beta + u_i + e_{is}, \qquad u_i \sim \mathcal{N}(0, \sigma_u^2), \quad e_{is} \sim \mathcal{N}(0, \sigma_e^2)",
            """
- $y_{is}$: player $i$'s log-odds rate in season $s$
- $\\mathbf{x}_{is}^\\top\\boldsymbol\\beta$: **fixed effects** (position, experience)
- $u_i$: **random effect**, player $i$'s talent, drawn from a shared distribution
- $e_{is}$: everything else (season wobble + luck)
""",
        ),
        MathBlock(
            "Step 2: the shrunken player estimate (BLUP)",
            r"\hat u_i = \frac{\sigma_u^2}{\sigma_u^2 + \sigma_e^2 / n_i}\;\bar r_i",
            """
- $\\bar r_i$: player $i$'s average leftover after fixed effects
- $n_i$: his number of seasons
- The fraction is the same partial-pooling weight as L5.4 and Level 1
""",
        ),
        MathBlock(
            "Step 3: how much is stable talent? (ICC)",
            r"\text{ICC} = \frac{\sigma_u^2}{\sigma_u^2 + \sigma_e^2}",
            """
- Intraclass correlation: share of the variation that's consistent player-to-player differences
- Near 1: players are very consistent. Near 0: mostly season-to-season noise
""",
        ),
    ],
    code=CODE,
    code_notes="Change the filter to opportunities >= 300. With less noisy rows, the residual spread falls and the ICC rises.",
    walkthrough=[
        ("d = d[d['opportunities'] >= 100]", "Keep rows with enough snaps, because basic mixedlm treats every row as equally noisy."),
        ("smf.mixedlm('y ~ 0 + C(position) + C(exp)', d, groups=d['player_id'])", "Fixed effects for position and experience; a random intercept for each player."),
        (".fit()", "Estimate everything: fixed effects, the player spread and the residual spread (by restricted maximum likelihood)."),
        ("model.cov_re.iloc[0, 0]", "The estimated variance of player talent (σ_u²)."),
        ("model.scale", "The residual variance (σ_e²): season wobble plus luck."),
        ("icc = sd_player ** 2 / (...)", "Share of variation that's stable talent."),
        ("model.random_effects", "Each player's shrunken talent estimate (BLUP)."),
        ("(blup[mask] / raw[mask]).median()", "How much each group's estimates were shrunk: fewer seasons, more shrinkage."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Fast first pass:** a mixed model gives position and experience effects with standard errors in seconds, a good check before any full Bayesian model.
- **Model 7 connection:** for any redesigned player model, the ICC tells you how much of a stat is stable talent vs noise, which sets how much any forecast can hope to achieve.
- **Your rule:** fixed-effect sizes are descriptive. A forecasting model built from them still has to beat simple baselines on later seasons.
""",
    quiz=[
        Question(
            "q1", "mc", "Which should be a FIXED effect in a player model?",
            "Position: few categories, and you want a specific number for each.",
            options=["Position (EDGE, DT, LB)", "Each of 1,500 players", "Each player-season's luck", "Nothing"],
            notes=["Correct.", "Many members, so that's a random effect.", "That's the residual.", "Fixed effects are useful."],
            answer=0),
        Question(
            "q2", "numeric", "σ_u² = 0.11 and σ_e² = 0.095. What is the ICC?",
            "0.11 / (0.11 + 0.095) ≈ 0.537.", category="Application", answer=0.537, tolerance=0.005),
        Question(
            "q3", "mc", "How is a mixed-effects model related to the hierarchical Bayesian model?",
            "Same structure; mixed-effects estimates spreads as single numbers (fast), Bayes gives full uncertainty via MCMC.",
            options=["Same structure; mixed models give point estimates of the spreads, Bayes gives full uncertainty",
                     "Completely unrelated", "Mixed models use MCMC", "Bayes has no random effects"],
            notes=["Correct.", "They're closely related.", "They don't need MCMC.", "It does (partially pooled effects)."],
            answer=0),
        Question(
            "q4", "short", "A player with one season gets shrunk more than one with four seasons. Why?",
            "Less data means his average is noisier, so the model trusts it less (smaller weight σ_u²/(σ_u²+σ_e²/n)).",
            keywords=[["less data", "fewer season", "one season", "noisier", "noise", "uncertain", "small"],
                      ["shrink", "trust", "weight", "pool", "pulled"]],
            model_answer="One season is a noisier estimate of his talent, so the model trusts it less: the weight σ_u²/(σ_u² + σ_e²/n) is smaller when n is small, so he's shrunk more."),
        Question(
            "q5", "mc", "Why filter out rows with fewer than 100 snaps before fitting mixedlm here?",
            "Basic mixedlm assumes equal noise per row; tiny samples are far noisier and would distort the fit.",
            category="Application",
            options=["Basic mixedlm assumes every row is equally noisy", "Small players don't count",
                     "statsmodels can't handle small numbers", "To make the ICC bigger on purpose"],
            notes=["Correct (the Bayesian model handles it with per-row luck).", "They count; the method can't weight them.", "It can.", "That's not the reason."],
            answer=0),
    ],
)
