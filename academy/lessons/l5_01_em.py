import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import stats

from .. import data, theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner=False)
def _x():
    df = data.make_defensive_profiles()
    return df[(df["position_group"] == "DL") & (df["snaps"] >= 150)]["pressure_rate"].to_numpy()


def visual():
    x = _x()
    it = st.slider("EM rounds", 0, 15, 0, key="l51_it")
    w, mu, sd = np.array([0.5, 0.5]), np.array([0.02, 0.20]), np.array([0.05, 0.05])
    for _ in range(it):
        dens = w * stats.norm.pdf(x[:, None], mu, sd)
        r = dens / dens.sum(1, keepdims=True)
        nk = r.sum(0)
        w, mu = nk / len(x), (r * x[:, None]).sum(0) / nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk)
    xs = np.linspace(-0.02, 0.22, 300)
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.hist(x, bins=40, density=True, color=T.GRID, alpha=0.9)
    for k, c in enumerate([T.BLUE, T.ROSE]):
        ax.plot(xs, w[k] * stats.norm.pdf(xs, mu[k], sd[k]), color=c, lw=2.4,
                label=f"group {k + 1}: {w[k]:.0%} of players, mean {mu[k]:.3f}")
    ax.set_xlabel("Pressure rate (DL seasons, 150+ snaps)"), ax.set_yticks([]), ax.legend(fontsize=7)
    show_fig(fig)


CODE = '''import numpy as np
from scipy import stats
from academy.data import make_defensive_profiles

df = make_defensive_profiles()                               # SIMULATED
dl = df[(df["position_group"] == "DL") & (df["snaps"] >= 150)]
x = dl["pressure_rate"].to_numpy()

# Two hidden groups of linemen (e.g. pass-rush specialists vs everyone else). Start with rough guesses:
share = np.array([0.5, 0.5])                                 # how common each group is
mean = np.array([0.02, 0.20])                                # each group's typical pressure rate
sd = np.array([0.05, 0.05])                                  # each group's spread

for round_ in range(1, 51):
    # E-step: for each player, how likely is he to belong to each group?
    dens = share * stats.norm.pdf(x[:, None], mean, sd)
    resp = dens / dens.sum(axis=1, keepdims=True)            # 'responsibilities', each row sums to 1
    # M-step: re-estimate each group from those soft memberships
    n_k = resp.sum(axis=0)
    share = n_k / len(x)
    mean = (resp * x[:, None]).sum(axis=0) / n_k
    sd = np.sqrt((resp * (x[:, None] - mean) ** 2).sum(axis=0) / n_k)
    if round_ in (1, 2, 5, 50):
        loglik = np.log(dens.sum(axis=1)).sum()
        print(f"round {round_:>2}: log-lik {loglik:7.1f}  shares {share.round(2)}  means {mean.round(3)}")

edge = (dl["true_archetype"] == "Edge rusher").to_numpy()   # hidden truth (simulation only)
print(f"\\nAvg P(high group) for true edge rushers: {resp[edge, 1].mean():.2f}; for others: {resp[~edge, 1].mean():.2f}")
print(f"Players EM is unsure about (30-70%): {((resp[:, 1] > 0.3) & (resp[:, 1] < 0.7)).mean():.0%}")
'''

LESSON = Lesson(
    id="l5-01-em", level=5, order=1,
    title="Expectation-Maximization: Fitting Models With Hidden Groups",
    subtitle="Soft memberships, and the guess-and-refit loop behind HMMs and mixtures",
    minutes=20,
    concept_ids=["em"],
    simple="""
Suppose defensive linemen secretly come in **two kinds**: pass-rush specialists and everyone else. You only see pressure rates, not the labels. If you knew who belonged to which group, estimating each group's average would be easy. If you knew the averages, guessing each player's group would be easy. You know neither.

**Expectation-maximization (EM)** breaks the deadlock by alternating:

1. **E-step:** with the current guesses, work out each player's **chance** of belonging to each group (a *soft* membership, like 80% / 20%).
2. **M-step:** re-estimate each group's average, spread and size, letting each player count in proportion to those chances.
3. Repeat. Each round can only improve the fit, so it settles down.

You've already used EM twice without the name. **K-means** (L2.2) is a hard-assignment cousin. **Baum-Welch** (L4.2) is EM for hidden Markov models.
""",
    problem="Many football questions involve hidden labels (player type, hot/cold, injured/healthy). EM fits models with hidden labels without ever needing to know them.",
    why="It handles uncertainty about group membership honestly (soft memberships) and works for many hidden-variable models: mixtures, HMMs, missing data.",
    when_useful="""
- Mixture models: hidden player types behind one stat
- Hidden Markov models (Baum-Welch)
- Filling in missing data under a model
""",
    when_not="""
- When groups overlap so much that soft memberships hover around 50%: the 'types' may not really exist
- When the answer depends on the starting guess: EM can get stuck. Try several starts
- When you need full uncertainty about the parameters themselves: use MCMC (L5.2) instead
""",
    compare="""
**EM vs K-means:** K-means gives each player one group (hard). EM gives probabilities (soft), and learns each group's spread and size.

**EM vs MCMC:** EM finds the single best set of parameters. MCMC maps out the whole range of plausible parameters (uncertainty).

**EM vs gradient descent:** both climb the likelihood. EM's steps often have simple formulas and never make the fit worse.
""",
    sports="""
**Fictional scenario (simulated).** About 1,000 DL seasons. Fit a two-group mixture to pressure rate alone, starting from deliberately poor guesses (2% and 20%).

- EM settles on two groups with typical pressure rates of about **7%** and **12.5%**, split roughly **63% / 37%**.
- True edge rushers get, on average, about a **0.8** probability of belonging to the high-pressure group; other linemen about 0.2.
- About **one in five** players stays uncertain (30–70%). That's honest: pressure rate alone can't place them, which is why Level 2 used several stats.
""",
    takeaway="EM alternates 'guess the hidden labels softly' and 'refit the groups', climbing to the best fit. It powers mixtures and HMMs.",
    visual=visual,
    visual_caption="Start at round 0 with poor guesses, then step the rounds forward. Each round moves the two curves toward the shape of the histogram.",
    math=[
        MathBlock(
            "Step 1: the mixture model",
            r"p(x) = \pi_1\, \mathcal{N}(x;\mu_1,\sigma_1^2) + \pi_2\, \mathcal{N}(x;\mu_2,\sigma_2^2)",
            """
- $x$: a player's pressure rate
- $\\pi_k$: share of players in group $k$ (the shares add to 1)
- $\\mu_k, \\sigma_k$: group $k$'s average and spread
- The overall curve is a weighted blend of two bell curves
""",
        ),
        MathBlock(
            "Step 2: E-step (soft memberships)",
            r"r_{ik} = \frac{\pi_k\, \mathcal{N}(x_i;\mu_k,\sigma_k^2)}{\sum_j \pi_j\, \mathcal{N}(x_i;\mu_j,\sigma_j^2)}",
            """
- $r_{ik}$: the **responsibility**: the chance player $i$ belongs to group $k$, given the current guesses
- Bayes' rule again (L1.1): prior share × likelihood, then rescale so each player's chances add to 1
""",
        ),
        MathBlock(
            "Step 3: M-step (refit the groups)",
            r"\pi_k = \frac{1}{N}\sum_i r_{ik}, \qquad \mu_k = \frac{\sum_i r_{ik}\, x_i}{\sum_i r_{ik}}, \qquad \sigma_k^2 = \frac{\sum_i r_{ik}(x_i - \mu_k)^2}{\sum_i r_{ik}}",
            """
- Weighted versions of share, average and spread, where each player counts by his responsibility
- Repeat Steps 2 and 3. The log-likelihood never goes down
""",
        ),
    ],
    code=CODE,
    code_notes="Change the starting means to [0.10, 0.11] (almost identical). Does EM still separate the groups? Then try three groups.",
    walkthrough=[
        ("x = dl['pressure_rate'].to_numpy()", "One stat per player: pressure rate. We pretend we don't know anyone's type."),
        ("share, mean, sd = ...", "Deliberately rough starting guesses for the two groups."),
        ("dens = share * stats.norm.pdf(x[:, None], mean, sd)", "For every player, how well each group explains his rate, times how common that group is."),
        ("resp = dens / dens.sum(axis=1, keepdims=True)", "E-step: turn that into probabilities that add to 1 per player."),
        ("n_k = resp.sum(axis=0)", "How many players each group 'owns', counting fractions."),
        ("mean = (resp * x[:, None]).sum(axis=0) / n_k", "M-step: each group's average, weighting players by membership."),
        ("np.log(dens.sum(axis=1)).sum()", "The log-likelihood: how well the whole mixture fits. Watch it rise, then settle."),
        ("resp[edge, 1].mean()", "Check against the hidden truth (simulation only): do true edge rushers land in the high group?"),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Hidden player types:** a mixture fitted by EM gives each player a *probability* of being a type, which is more honest than hard labels for borderline players.
- **Missing data:** EM can fill gaps under a model, but only if the gaps aren't caused by the outcome (e.g., injured players' missing games).
- **Your rule:** a type label goes on your site only if it beats a simple baseline on a test set fixed in advance (see the Level 2 project result).
""",
    quiz=[
        Question(
            "q1", "mc", "What does the E-step compute?",
            "Each player's probability of belonging to each group, using the current parameter guesses.",
            options=["Each player's chance of belonging to each group", "The final group averages",
                     "The number of groups", "Which players to delete"],
            notes=["Correct.", "That's the M-step.", "You choose that.", "EM deletes nothing."],
            answer=0),
        Question(
            "q2", "numeric", "Group A: share 0.5, density at x = 4.0. Group B: share 0.5, density 1.0. What is the responsibility of group A? (decimal)",
            "0.5×4 / (0.5×4 + 0.5×1) = 2 / 2.5 = 0.8.", category="Application", answer=0.8, tolerance=0.001),
        Question(
            "q3", "mc", "How is EM related to K-means?",
            "K-means is like EM with hard (0/1) memberships and equal round spreads.",
            options=["K-means is a hard-assignment cousin of EM", "They're unrelated",
                     "EM is a type of K-means with labels", "K-means uses probabilities"],
            notes=["Correct.", "They're closely related.", "EM uses no labels.", "K-means is hard 0/1."],
            answer=0),
        Question(
            "q4", "short", "About 20% of players have memberships between 30% and 70%. What does that tell you, and what could help?",
            "The groups overlap on this stat; more stats (more information) would separate them.",
            category="Application",
            keywords=[["overlap", "unsure", "uncertain", "ambiguous", "can't tell", "not separ"],
                      ["more stat", "more feature", "other stat", "more data", "more information", "several", "add"]],
            model_answer="Those players sit where the groups overlap, so one stat can't place them. Adding more stats (as in Level 2) or more data would separate them better."),
        Question(
            "q5", "mc", "Where have you already used EM in this academy?",
            "Baum-Welch, which fits hidden Markov models in L4.2, is EM.",
            options=["Baum-Welch for HMMs (L4.2)", "Kaplan-Meier (L3.3)", "The Brier score (L3.1)", "PCA (L2.1)"],
            notes=["Correct.", "That's a counting method.", "That's a scoring rule.", "PCA has an exact formula."],
            answer=0),
    ],
)
