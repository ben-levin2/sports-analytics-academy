import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


def visual():
    c1, c2 = st.columns([1, 2])
    with c1:
        rate = st.slider("QB's TRUE completion rate (never changes)", 0.50, 0.75, 0.62, 0.01, key="l16_rate")
        seasons = st.slider("Simulated seasons", 5, 60, 25, key="l16_seasons")
        attempts = st.slider("Attempts per season", 50, 500, 400, 25, key="l16_att")
    rng = np.random.default_rng(1)
    x = rng.random((seasons, attempts)) < rate
    running = x.cumsum(1) / np.arange(1, attempts + 1)
    t = np.arange(1, attempts + 1)
    sd = np.sqrt(rate * (1 - rate) / t)
    fig, ax = plt.subplots()
    for r in running:
        ax.plot(t, r, color=T.BLUE, alpha=0.25, lw=1)
    ax.plot(t, rate + 2 * sd, color=T.AMBER, lw=1.6, ls="--", label="True rate ± 2 SD")
    ax.plot(t, rate - 2 * sd, color=T.AMBER, lw=1.6, ls="--")
    ax.axhline(rate, color=T.TEXT, lw=1)
    ax.set_ylim(0.2, 1.0), ax.set_xlabel("Attempts so far this season"), ax.set_ylabel("Running completion %")
    ax.legend(loc="upper right")
    with c2:
        show_fig(fig)
    final = running[:, -1]
    st.caption(f"Same true rate in every season, yet final completion % ranged from {final.min():.1%} to "
               f"{final.max():.1%} purely by chance.")


CODE = '''import numpy as np

rng = np.random.default_rng(7)

# --- 1. Bernoulli process: each pass is an independent coin flip with P(complete) = 0.62 ---
true_rate, attempts, seasons = 0.62, 400, 5000
passes = rng.random((seasons, attempts)) < true_rate      # True = completion
running = passes.cumsum(axis=1) / np.arange(1, attempts + 1)

for n in (50, 200, 400):
    spread = running[:, n - 1].std()
    theory = np.sqrt(true_rate * (1 - true_rate) / n)
    print(f"after {n:>3} attempts: SD of completion % = {spread:.3f} (theory {theory:.3f})")

# --- 2. Streaks happen by luck: longest run of completions in a 35-attempt game ---
def longest_run(row):
    best = cur = 0
    for completed in row:
        cur = cur + 1 if completed else 0
        best = max(best, cur)
    return best

games = rng.random((20000, 35)) < true_rate
runs = np.array([longest_run(g) for g in games])
print(f"\\nP(8+ straight completions in a game) = {(runs >= 8).mean():.2f}  (no 'hot hand' needed)")

# --- 3. A Markov chain: hidden 'hot' / 'cold' form that persists from game to game ---
P = np.array([[0.8, 0.2],      # from HOT:  stay hot 80%, turn cold 20%
              [0.3, 0.7]])     # from COLD: turn hot 30%, stay cold 70%
state, hot_games = 0, 0
for game in range(20_000):
    state = rng.choice(2, p=P[state])
    hot_games += state == 0
print(f"Long-run share of HOT games: {hot_games / 20_000:.3f}  (theory 0.3/(0.2+0.3) = 0.600)")
'''

LESSON = Lesson(
    id="l1-06-stochastic-processes", level=1, order=6,
    title="Stochastic Processes: How Luck Unfolds Over a Season",
    subtitle="Bernoulli processes, random fluctuation and Markov chains",
    minutes=20,
    concept_ids=["stochastic-processes"],
    simple="""
A **stochastic process** is a sequence of random outcomes unfolding over time: pass after pass, game after game, season after season.

The key idea: **even if a player's true ability never changes, his numbers wander.** A 62% passer will look like a 70% passer for stretches, and like a 55% passer for others, purely by chance. How far his numbers wander, and how quickly they settle down, follows predictable rules.

Some processes also have **memory**. In a **Markov chain**, what happens next depends on the current *state* (e.g., "hot" or "cold" form), but not on the full history before it.
""",
    problem="Without a model of randomness, every streak looks like a story. Stochastic processes tell you how much variation luck alone produces, so you can tell real change from noise.",
    why="It sets realistic expectations (how much can a stat move in 50 attempts?), it explains why the Bayesian shrinkage in L1.1–L1.5 is needed, and it is the foundation for HMMs (Level 4), Gaussian processes (Level 3) and MCMC (Level 5).",
    when_useful="""
- Judging whether a streak or slump is unusual
- Deciding how many attempts before a stat is trustworthy
- Simulating seasons to see the range of plausible outcomes
- Modeling states that persist (form, injuries, game situations)
""",
    when_not="""
- Plays are not truly independent (same opponent, weather, game script). The simple coin-flip model understates swings
- Abilities change mid-season (injury, scheme change), so a fixed-rate model is wrong
- A Markov chain assumes only the current state matters. Long memory needs richer models (e.g., LSTMs in Level 4)
""",
    compare="""
**Independent (Bernoulli) process:** no memory at all. Each pass ignores the last one. **Markov chain:** one step of memory through a state. **Hidden Markov model (Level 4):** the state (hot/cold) is never observed directly and must be inferred from performance. **Gaussian process (Level 3):** a smooth random *curve* rather than a sequence of discrete steps.
""",
    sports="""
**Fictional scenario (simulated).** A QB's true completion rate is exactly **62%** every single pass. Simulate thousands of 400-attempt seasons:

- After **50 attempts**, the standard deviation of his completion % is about **6.9 points**. Swings between roughly 48% and 76% are normal.
- After **400 attempts**, it falls to about **2.4 points**, because the spread shrinks with the square root of attempts.
- In a 35-attempt game, he completes **8 or more passes in a row about 23% of the time**, with no hot hand at all.

So "he's 9-for-9 to start the game" is not, by itself, evidence that he's playing above his level.
""",
    takeaway="Numbers wander even when ability is fixed. Know how much luck alone can do before you read meaning into a streak.",
    visual=visual,
    visual_caption="Each blue line is one simulated season of the same QB. The amber band is where the season-to-date rate should usually sit. Notice the funnel: early-season numbers are wild, late-season numbers settle.",
    math=[
        MathBlock(
            "Step 1: a Bernoulli process",
            r"X_t \sim \text{Bernoulli}(\theta),\qquad t = 1, 2, \ldots, n",
            """
- $X_t$: outcome of pass $t$ (1 = complete, 0 = incomplete)
- $\\theta$: the true completion rate, fixed
- Each $X_t$ is independent of the others
""",
        ),
        MathBlock(
            "Step 2: how much the running rate wanders",
            r"\hat\theta_n = \frac{1}{n}\sum_{t=1}^{n} X_t,\qquad \text{SD}(\hat\theta_n) = \sqrt{\frac{\theta(1-\theta)}{n}}",
            """
- $\\hat\\theta_n$: completion % after $n$ attempts
- $\\sum_{t=1}^{n} X_t$: total completions so far
- $\\text{SD}$: the typical distance between $\\hat\\theta_n$ and the true $\\theta$
- Because of the $\\sqrt{n}$, four times the attempts only halves the noise
""",
            "Example: θ = 0.62, n = 50 gives SD = √(0.62 × 0.38 / 50) ≈ 0.069.",
        ),
        MathBlock(
            "Step 3: a Markov chain",
            r"P(S_{t+1} = j \mid S_t = i, S_{t-1}, \ldots, S_1) = P(S_{t+1} = j \mid S_t = i) = P_{ij}",
            """
- $S_t$: the state in game $t$ (e.g., hot or cold)
- $P_{ij}$: probability of moving from state $i$ to state $j$, a **transition probability**
- The first equals sign is the **Markov property**: only the current state matters
""",
            "Long-run share of hot games = P(cold→hot) / (P(hot→cold) + P(cold→hot)) = 0.3 / (0.2 + 0.3) = 0.6.",
        ),
    ],
    code=CODE,
    code_notes="Change the transition matrix so hot states are stickier (0.95 stay hot). The long-run share of hot games rises, and streaks last longer. This is exactly the structure an HMM tries to detect in Level 4.",
    walkthrough=[
        ("rng.random((seasons, attempts)) < true_rate", "A table of 5,000 seasons x 400 passes. Each cell is True (complete) with probability 0.62."),
        ("passes.cumsum(axis=1) / np.arange(1, attempts + 1)", "Running completions divided by attempts so far: the season-to-date completion % after every pass."),
        ("running[:, n - 1].std()", "How much that completion % varies across the 5,000 seasons after n attempts."),
        ("np.sqrt(true_rate * (1 - true_rate) / n)", "The formula's prediction for that spread. Simulation and theory match."),
        ("def longest_run(row):", "Walk through one game's passes, counting the current run of completions and remembering the longest."),
        ("(runs >= 8).mean()", "Share of simulated games with a run of 8+ completions: the luck-only baseline for a 'hot streak'."),
        ("P = np.array([[0.8, 0.2], [0.3, 0.7]])", "Transition matrix. Row = today's state, column = tomorrow's state."),
        ("state = rng.choice(2, p=P[state])", "Pick the next state using the current state's row. That one line is the Markov property."),
    ],
    psfal="""
- **Stability thresholds:** before ranking players on a rate stat, use the √n rule to decide the minimum sample where it means something.
- **Null models for streaks:** compare a player's best stretch with simulated luck-only seasons before calling it a breakout.
- **Simulation as a baseline:** simulate seasons under "nothing changed" to see if an observed jump in production is unusual.
- Markov structure underlies later PSFAL-style work: game-state models, hot/cold HMMs, and MCMC fitting of hierarchical models.
""",
    quiz=[
        Question(
            "q1", "mc", "A QB's true completion rate never changes. His season-to-date completion % will:",
            "Random variation makes the running rate wander, with swings that shrink as attempts accumulate.",
            options=["Wander, with swings that shrink as attempts accumulate", "Stay exactly at his true rate",
                     "Drift steadily upward", "Swing just as much in week 12 as in week 1"],
            notes=["Correct.", "Only in the long run, and never exactly.", "Nothing pushes it in one direction.",
                   "More attempts means less wandering, by the √n rule."],
            answer=0),
        Question(
            "q2", "numeric", "True rate θ = 0.62, n = 100 attempts. What is SD(completion %)? (decimal)",
            "√(0.62 × 0.38 / 100) = √0.002356 ≈ 0.0485.",
            category="Application", answer=0.0485, tolerance=0.002),
        Question(
            "q3", "mc", "The Markov property says that the next state depends on:",
            "Only the current state matters for what comes next, not the full history.",
            options=["Only the current state", "The entire history of states", "Nothing; states are independent",
                     "Only the very first state"],
            notes=["Correct.", "That's a process with long memory, not a Markov chain.",
                   "That's an independent (Bernoulli) process.", "The starting state fades in importance over time."],
            answer=0),
        Question(
            "q4", "short", "A receiver catches his first 9 targets of a game. Why shouldn't you immediately call it a hot streak?",
            "Runs like this happen by chance alone fairly often; compare with a luck-only baseline.",
            category="Application",
            keywords=[["chance", "luck", "random", "noise", "variance", "happen"],
                      ["baseline", "simulat", "expect", "compare", "often", "common", "sample", "small"]],
            model_answer="Runs of that length happen by chance alone fairly often, especially with a high catch rate. You'd need to compare against a simulated luck-only baseline before claiming real change."),
        Question(
            "q5", "mc", "Hot→cold has probability 0.2 and cold→hot has probability 0.3. What is the long-run share of hot games?",
            "π_hot = 0.3 / (0.2 + 0.3) = 0.6.",
            category="Application",
            options=["0.40", "0.50", "0.60", "0.80"],
            notes=["That's the share of cold games.", "Only if the two switch probabilities were equal.", "Correct.",
                   "That's the probability of staying hot, not the long-run share."],
            answer=2),
    ],
)
