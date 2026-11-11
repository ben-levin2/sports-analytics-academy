import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .. import data, sequence as Q, theme as T
from ..models import Lesson, MathBlock, Question
from .. import ui
from ..ui import show_fig


@st.cache_data(show_spinner="Fitting hidden Markov models…")
def _fits():
    df = data.make_game_logs()
    rs = Q.running_mean_residuals(df).dropna(subset=["resid"])
    seqs = [g["resid"].to_numpy() for _, g in rs.groupby(["player_id", "season"])]
    return {k: Q.fit_hmm(seqs, k) for k in (1, 2, 3)}


def visual():
    fits = _fits()
    rows = [{"states": k, "log-likelihood": round(m["loglik"], 1), "BIC (lower = better)": round(Q.bic(m), 1)}
            for k, m in fits.items()]
    ui.table(pd.DataFrame(rows))
    m = fits[2]
    xs = np.linspace(-30, 30, 300)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    from scipy import stats
    for k, (c, name) in enumerate([(T.ROSE, "hot"), (T.BLUE, "cold")]):
        ax.plot(xs, stats.norm.pdf(xs, m["means"][k], m["sds"][k]), color=c, lw=2.2,
                label=f"{name}: mean {m['means'][k]:+.1f}, SD {m['sds'][k]:.1f}")
    ax.set_xlabel("Game grade vs season-to-date average"), ax.set_yticks([]), ax.legend(fontsize=8)
    show_fig(fig)
    st.caption(f"Learned transitions: stay hot {m['A'][0, 0]:.2f}, stay cold {m['A'][1, 1]:.2f}. "
               "The two bell curves overlap heavily: one game rarely tells you the state.")


CODE = '''import numpy as np
from academy.data import make_game_logs
from academy.sequence import bic, fit_hmm, running_mean_residuals, viterbi

df = make_game_logs()                                    # SIMULATED game logs (true_state is hidden truth)

# The trap: fit on RAW grades and the 'states' become good vs bad players
raw = [g.sort_values("week")["grade"].to_numpy() for _, g in df.groupby(["player_id", "season"])]
m_raw = fit_hmm(raw, K=2)
print("RAW grades -> state means", m_raw["means"].round(1), " stay probabilities",
      np.diag(m_raw["A"]).round(2), " (too sticky: it's picking up player level)")

# The fix: each game vs the player's own season-to-date average (earlier games only)
res = running_mean_residuals(df).dropna(subset=["resid"])
seqs = [g["resid"].to_numpy() for _, g in res.groupby(["player_id", "season"])]
m1, m2 = fit_hmm(seqs, K=1), fit_hmm(seqs, K=2)          # Baum-Welch (an EM algorithm)
print("\\nRESIDUALS -> state means", m2["means"].round(1), " stay probabilities", np.diag(m2["A"]).round(2))
print(f"BIC with 1 state: {bic(m1):.0f}   with 2 states: {bic(m2):.0f}   (lower is better)")

# Viterbi: the single most likely hot/cold path for each player-season; compare with the hidden truth
hits = []
for key, g in res.groupby(["player_id", "season"]):
    path = viterbi(g["resid"].to_numpy(), m2["pi"], m2["A"], m2["means"], m2["sds"])
    truth = df.set_index(["player_id", "season", "week"]).loc[[(*key, w) for w in g["week"]], "true_state"]
    hits.append(np.mean((path == 0) == (truth.to_numpy() == "hot")))
print(f"Viterbi path matches the hidden truth in {np.mean(hits):.0%} of games (50% = coin flip)")
'''

LESSON = Lesson(
    id="l4-02-hmm-fitting", level=4, order=2,
    title="Hidden Markov Models, Part 2: Learning the Model, and Is the Hot Hand Real?",
    subtitle="Baum-Welch, Viterbi, and choosing between 1, 2 or 3 states",
    minutes=25,
    concept_ids=["hmms"],
    simple="""
In Part 1 we set the HMM's numbers by hand: how sticky form is, how good a hot game is. Now the model **learns them from data**, using an algorithm called **Baum-Welch**:

1. Guess the numbers.
2. Use them to estimate, for every game, how likely each state was.
3. Re-estimate the numbers from those soft guesses.
4. Repeat until nothing changes.

This guess-and-refit loop is an **EM algorithm**, which you'll meet properly in Level 5.

Two more tools:
- **Viterbi** finds the single most likely hot/cold path through a season.
- **Comparing 1, 2 or 3 states** answers the real question: does the data even support a "hot hand", or is one state (no streaks, just luck) enough?
""",
    problem="Hand-set numbers are guesses. You need to learn them from many player-seasons, and to check whether hidden states exist at all.",
    why="It turns 'is there a hot hand?' into a testable comparison, and learns how sticky form is and how big the effect is.",
    when_useful="""
- Many sequences (player-seasons) sharing the same hot/cold behavior
- Testing whether streakiness exists beyond luck
- Labeling phases of a season (Viterbi) for film study or reporting
""",
    when_not="""
- Raw grades without removing player level: you'll 'discover' good vs bad players instead of form
- Very short sequences: Baum-Welch needs many games to separate states from noise
- Trusting a Viterbi path as fact: it's the most likely path, not the true one
""",
    compare="""
**Baum-Welch vs fitting by hand:** the data chooses the numbers by maximizing the likelihood, the same idea as fitting the prior in L1.5.

**1 state vs 2 states:** one state means "his level plus luck, no streaks". Compare with BIC, which adds a penalty for extra parameters, or with held-out likelihood. A tiny improvement isn't proof of a hot hand.

**Viterbi vs the forward filter:** the filter gives P(hot) using only the games so far (for real-time use). Viterbi looks at the whole season afterwards and picks one path.
""",
    sports="""
**Fictional scenario (simulated).** 160 players, 2019–2025, about 10 observed games per season, with hot/cold form built in.

- **The trap:** fit on raw grades and the two states become "good players" and "weaker players". They're unrealistically sticky (about 0.97), because a player's level doesn't change week to week.
- **The fix:** measure each game against the player's own season-to-date average. Now the model recovers the built-in stickiness, about **0.80** both ways (the truth: 0.80 hot, 0.85 cold).
- **Is two states worth it?** BIC favours 2 states over 1, but only modestly.
- **Viterbi** matches the hidden truth in only about **63%** of games (50% would be a coin flip). The states overlap a lot, so one game says little.

The forecasting test in the project is the real judge, and there the gain over a plain average is under 1%.
""",
    takeaway="Learn the HMM from many seasons, remove player level first, and check whether more states really earn their keep.",
    visual=visual,
    visual_caption="The table compares 1, 2 and 3 states. The curves show what the learned hot and cold games look like. Their heavy overlap is why single games mislead.",
    math=[
        MathBlock(
            "Step 1: the E-step (soft guesses of the states)",
            r"\gamma_t(k) = P(S_t = k \mid x_1, \ldots, x_T)",
            """
- $\\gamma_t(k)$: the chance game $t$ was in state $k$, using the **whole** season (earlier and later games)
- Computed with a forward pass (L4.1) plus a backward pass
""",
        ),
        MathBlock(
            "Step 2: the M-step (re-estimate the numbers)",
            r"\hat\mu_k = \frac{\sum_t \gamma_t(k)\, x_t}{\sum_t \gamma_t(k)}, \qquad \hat A_{ij} = \frac{\text{expected } i \to j \text{ switches}}{\text{expected time in } i}",
            """
- $\\hat\\mu_k$: average game in state $k$, weighting each game by how likely it was in that state
- $\\hat A_{ij}$: how often state $i$ was followed by state $j$, counted softly
- Repeat E and M until the likelihood stops improving
""",
        ),
        MathBlock(
            "Step 3: compare models with BIC",
            r"\text{BIC} = -2 \log L + p \log n",
            """
- $\\log L$: log-likelihood of the data under the fitted model (higher = better fit)
- $p$: number of parameters. Each extra state adds transitions, a mean and an SD
- $n$: number of games
- Lower BIC is better. The $p \\log n$ term stops you adding states just to chase noise
""",
        ),
    ],
    code=CODE,
    code_notes="Try `fit_hmm(seqs, K=3)` and compare its BIC. Does a third state ('normal') help, or is it just splitting noise?",
    walkthrough=[
        ("raw = [...grade...]", "One array of raw grades per player-season."),
        ("m_raw = fit_hmm(raw, K=2)", "Fit a 2-state HMM on raw grades. That's the trap."),
        ("np.diag(m_raw['A'])", "The 'stay' probabilities come out near 1: the states are really players' levels, which don't change week to week."),
        ("running_mean_residuals(df)", "The fix: each game minus that player's average of EARLIER games this season."),
        ("fit_hmm(seqs, K=1), fit_hmm(seqs, K=2)", "Baum-Welch fits a 1-state (no hot hand) and a 2-state (hot/cold) model."),
        ("bic(m1), bic(m2)", "Compare fit with a penalty for extra parameters. Lower wins."),
        ("viterbi(g['resid'], ...)", "The most likely hot/cold path for one player-season."),
        ("np.mean((path == 0) == (truth == 'hot'))", "How often the path matches the hidden truth. Only possible with simulated data."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Partial game logs:** with only parts of seasons, Baum-Welch has little to learn from. Treat any hot/cold finding as a learning exercise.
- **Before believing a hot hand:** remove player level, compare 1 vs 2 states, and then check whether forecasts beat the season-to-date average on later seasons (the Level 4 project). A better BIC alone doesn't count.
""",
    quiz=[
        Question(
            "q1", "mc", "Baum-Welch learns the HMM's numbers by:",
            "Alternating soft state guesses (E-step) and re-estimating the numbers from them (M-step) until stable.",
            options=["Alternating soft guesses of the states and re-estimating the numbers",
                     "Asking scouts which games were hot", "Picking the numbers that look best on a chart",
                     "Using the test season"],
            notes=["Correct (an EM algorithm).", "It uses no labels.", "It maximizes the likelihood, not looks.", "That would be leakage."],
            answer=0),
        Question(
            "q2", "mc", "An HMM fit on raw grades gives 'stay' probabilities near 0.97. What is most likely happening?",
            "The states are capturing player levels, which barely change, rather than week-to-week form.",
            category="Application",
            options=["The states are capturing good vs weaker players, not form", "Form is extremely sticky",
                     "The model is perfect", "There's a bug in the data"],
            notes=["Correct. Subtract player level first.", "That's the misleading reading.", "It's answering the wrong question.", "The data is fine; the setup is the issue."],
            answer=0),
        Question(
            "q3", "numeric", "Model A: log L = −10,700, 5 parameters. n = 2,500 games. What is its BIC? (ln 2500 ≈ 7.824)",
            "BIC = −2 × (−10,700) + 5 × 7.824 = 21,400 + 39.1 ≈ 21,439.", category="Application", answer=21439.1, tolerance=1.0),
        Question(
            "q4", "short", "BIC prefers 2 states over 1, but only slightly. Does that prove a hot hand? What would you check next?",
            "No; check whether hot/cold forecasts beat a simple average on later seasons, and whether the effect is big enough to matter.",
            category="Application",
            keywords=[["no", "not", "doesn't", "weak", "slight"],
                      ["forecast", "predict", "baseline", "later season", "test", "out-of-sample", "held-out", "average"]],
            model_answer="No. A small BIC edge is weak evidence. Check whether hot/cold forecasts beat the season-to-date average on later seasons, and whether the gain is large enough to matter."),
        Question(
            "q5", "mc", "You need P(hot) BEFORE next week's game. Which tool?",
            "The forward filter uses only games so far; Viterbi looks back over the whole season.",
            options=["The forward filter", "Viterbi", "Baum-Welch", "BIC"],
            notes=["Correct.", "Viterbi uses all games, including future ones, so it's hindsight.", "That fits the model; it isn't the live estimate.", "BIC compares models."],
            answer=0),
    ],
)
