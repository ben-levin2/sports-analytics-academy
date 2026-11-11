import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import data, sequence as Q, theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


def visual():
    df = data.make_game_logs()
    keys = df.groupby(["player_id", "season"]).size()
    keys = keys[keys >= 10].index[:40]
    c1, c2 = st.columns([1, 2])
    with c1:
        i = st.slider("Player-season", 0, len(keys) - 1, 3, key="l41_ps")
        stay_hot = st.slider("P(stay hot)", 0.5, 0.95, 0.80, 0.05, key="l41_hot")
        stay_cold = st.slider("P(stay cold)", 0.5, 0.95, 0.85, 0.05, key="l41_cold")
    g = df.set_index(["player_id", "season"]).loc[keys[i]].sort_values("week")
    y = g["grade"].to_numpy()
    level = y.mean()
    A = np.array([[stay_hot, 1 - stay_hot], [1 - stay_cold, stay_cold]])
    alpha, _ = Q.forward(y - level, np.array([0.4, 0.6]), A, np.array([6.0, -4.0]), np.array([7.0, 7.0]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 3.8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    hot = (g["true_state"] == "hot").to_numpy()
    ax1.plot(g["week"], y, color=T.MUTED, lw=1)
    ax1.scatter(g["week"][hot], y[hot], color=T.ROSE, s=40, label="truly hot (hidden)")
    ax1.scatter(g["week"][~hot], y[~hot], color=T.BLUE, s=40, label="truly cold (hidden)")
    ax1.axhline(level, color=T.TEXT, lw=0.8, ls=":"), ax1.set_ylabel("Grade"), ax1.legend(fontsize=7)
    ax2.fill_between(g["week"], alpha[:, 0], color=T.ROSE, alpha=0.4, step="mid")
    ax2.set_ylim(0, 1), ax2.set_ylabel("P(hot)"), ax2.set_xlabel("Week")
    with c2:
        show_fig(fig)
    st.caption("Top: game grades, colored by the hidden truth (which you'd never see in real data). "
               "Bottom: the model's belief that he's hot, updated game by game.")


CODE = '''import numpy as np
from scipy import stats

# --- SIMULATED: one fictional QB's game grades in one season (12 games) ---
grades = np.array([58, 71, 69, 74, 63, 55, 52, 60, 68, 73, 70, 57], float)
level = grades.mean()                       # his normal level this season
x = grades - level                          # how far above/below normal each game was

# The model (set by hand here; L4.2 learns these from data)
start = np.array([0.4, 0.6])                # P(hot), P(cold) before game 1
A = np.array([[0.80, 0.20],                 # from HOT:  stay hot 80%, turn cold 20%
              [0.15, 0.85]])                # from COLD: turn hot 15%, stay cold 85%
means = np.array([6.0, -4.0])               # typical game when hot / cold (vs his normal level)
sd = 7.0                                    # game-to-game noise in either state

belief = start.copy()
for week, xi in enumerate(x, start=1):
    if week > 1:
        belief = belief @ A                 # PREDICT: form may have changed since last game
    fit = stats.norm.pdf(xi, means, sd)     # how well each state explains this game
    belief = belief * fit                   # UPDATE: Bayes' rule (lesson L1.1)
    belief = belief / belief.sum()          # make the two probabilities add to 1
    print(f"week {week:>2}: grade {grades[week - 1]:>4.0f}   P(hot) = {belief[0]:.2f}")

nxt = belief @ A
print(f"\\nBefore next game: P(hot) = {nxt[0]:.2f}, expected grade = {level + nxt @ means:.1f}")
'''

LESSON = Lesson(
    id="l4-01-hmm-intro", level=4, order=1,
    title="Hidden Markov Models, Part 1: Is He Hot or Cold Right Now?",
    subtitle="Hidden states, transitions, and updating your belief game by game",
    minutes=20,
    concept_ids=["hmms"],
    simple="""
Fans talk about a QB being **hot** or **cold**. You can't see "form" directly. You only see game results, and those include luck.

A **hidden Markov model (HMM)** puts that idea into numbers:
- There are a few **hidden states** (here: hot and cold).
- Form **tends to stick**: a hot QB usually stays hot next week, but sometimes cools off. This is the Markov chain from L1.6.
- Each state produces games that are **better or worse on average**, with plenty of luck either way.

After every game, the model updates **how likely it is he's hot right now**. A great game nudges it up; a bad game nudges it down; and how much it moves depends on how big the surprise was.
""",
    problem="Single games are noisy. An HMM separates lasting changes in form from one-off luck, and says how confident it is.",
    why="It gives a live probability of 'hot' that updates game by game, and a next-game forecast that leans on recent form only as much as the evidence supports.",
    when_useful="""
- Performance that seems to come in streaks or phases
- Sequences where you want a probability of 'which phase are we in?'
- Few states, simple patterns, modest data
""",
    when_not="""
- When 'form' doesn't really persist: then the model mostly chases noise (L1.6 showed how often streaks happen by luck)
- Without first removing each player's normal level: the 'states' turn into 'good players vs bad players' (L4.2 shows this trap)
- Long, complex patterns: an LSTM (L4.3–L4.4) can learn more, given far more data
""",
    compare="""
**HMM vs a moving average (EWMA):** both lean on recent games. The HMM does it through an explicit hidden state with a probability attached, while EWMA just down-weights older games.

**HMM vs the Markov chain (L1.6):** in L1.6 you could see the state. Here it is hidden, and you infer it from noisy games.

**HMM vs Bayesian updating (L1.1):** each game is a small Bayes update: belief (prior) × how well each state explains the game (likelihood) → new belief (posterior).
""",
    sports="""
**Fictional scenario (simulated).** A QB's 12 game grades, measured against his own season average.

The model assumes: hot games run about +6 and cold games about −4, with ±7 of luck. A hot QB stays hot 80% of the time; a cold QB stays cold 85% of the time.

- After a run of strong games, P(hot) climbs to almost 0.9.
- One bad game doesn't flip it straight to cold. The model knows form sticks, and one game is noisy.
- Before the next game, it blends the two states into one forecast.

In the visual you can see the hidden truth (only possible because the data is simulated) and how well the model's belief tracks it.
""",
    takeaway="An HMM tracks a hidden state you can't see, like form, by updating a probability after every game, while remembering that form tends to stick.",
    visual=visual,
    visual_caption="Pick a player-season and change how sticky form is. Stickier form makes the belief move more slowly. It needs more evidence to switch.",
    math=[
        MathBlock(
            "Step 1: the hidden chain",
            r"P(S_t = j \mid S_{t-1} = i) = A_{ij}",
            """
- $S_t$: the hidden state in game $t$ (hot or cold)
- $A_{ij}$: the chance of moving from state $i$ to state $j$ between games (the **transition matrix**)
- Same Markov property as L1.6: only last game's state matters
""",
        ),
        MathBlock(
            "Step 2: what each state produces",
            r"x_t \mid S_t = k \;\sim\; \mathcal{N}(\mu_k,\; \sigma_k^2)",
            """
- $x_t$: the observed game (grade minus his normal level)
- $\\mu_k$: the typical game in state $k$ (e.g., +6 hot, −4 cold)
- $\\sigma_k$: the luck in that state
- $\\mathcal{N}$: a bell curve. This is the **emission** distribution: how a hidden state shows up in the box score
""",
        ),
        MathBlock(
            "Step 3: the forward filter (predict, then update)",
            r"\alpha_t(j) \;\propto\; \underbrace{\Bigl[\textstyle\sum_i \alpha_{t-1}(i)\,A_{ij}\Bigr]}_{\text{predict}} \times \underbrace{\mathcal{N}(x_t;\,\mu_j,\sigma_j^2)}_{\text{update}}",
            """
- $\\alpha_t(j)$: the probability he's in state $j$ after seeing games 1 to $t$
- Predict: carry last game's belief forward through the transition matrix
- Update: multiply by how well each state explains this game, then rescale so the probabilities add to 1
""",
        ),
    ],
    code=CODE,
    code_notes="Change the transition matrix to [[0.5, 0.5], [0.5, 0.5]] (form doesn't stick). The belief now jumps around with every game, because there's no memory to lean on.",
    walkthrough=[
        ("x = grades - level", "Measure each game against his own season level, so 'hot' means hot for him, not 'a good player'."),
        ("start = np.array([0.4, 0.6])", "Before any games: 40% chance he starts hot."),
        ("A = np.array([[0.80, 0.20], [0.15, 0.85]])", "Transition matrix: how sticky hot and cold are from one game to the next."),
        ("means = np.array([6.0, -4.0]); sd = 7.0", "What a typical hot or cold game looks like, and how much luck there is."),
        ("belief = belief @ A", "Predict: before seeing this game, allow for the chance that his form changed."),
        ("fit = stats.norm.pdf(xi, means, sd)", "Likelihood: how well 'hot' and 'cold' each explain this game's grade."),
        ("belief = belief * fit; belief / belief.sum()", "Update with Bayes' rule, then rescale so P(hot) + P(cold) = 1."),
        ("nxt = belief @ A; level + nxt @ means", "Next-game forecast: the chance of each state next week, times what each state usually produces."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Game-level data covers only parts of some seasons,** so any hot/cold analysis is a learning exercise until coverage is complete. The Level 4 project checks sample sufficiency explicitly.
- **Your Models 9 and 9b lesson applies within seasons too:** a hot/cold model must beat the plain season-to-date average on later seasons before it means anything.
""",
    quiz=[
        Question(
            "q1", "mc", "What makes the 'hot/cold' state 'hidden'?",
            "You only see game results, which include luck; the underlying form is never observed directly.",
            options=["You only see noisy game results, never the form itself", "The data is secret",
                     "Coaches hide it", "It's only visible after the season"],
            notes=["Correct.", "Nothing is secret; it's just not directly measurable.", "Not the reason.", "It's never directly visible."],
            answer=0),
        Question(
            "q2", "numeric", "Belief before a game: P(hot) = 0.6. P(stay hot) = 0.8 and P(cold→hot) = 0.15. What is P(hot) before seeing the next game?",
            "0.6 × 0.8 + 0.4 × 0.15 = 0.48 + 0.06 = 0.54.", category="Application", answer=0.54, tolerance=0.005),
        Question(
            "q3", "mc", "One bad game after five great ones. With sticky form, P(hot) will most likely:",
            "It drops somewhat but stays fairly high, because one noisy game isn't strong evidence of a lasting change.",
            category="Application",
            options=["Drop a bit but stay fairly high", "Drop straight to zero", "Not change at all", "Rise"],
            notes=["Correct.", "One game rarely overturns strong evidence when form is sticky.", "Every game moves it somewhat.", "A bad game is evidence against hot."],
            answer=0),
        Question(
            "q4", "short", "Why subtract each player's season level before running the HMM?",
            "Otherwise the states capture good vs bad players instead of hot vs cold form.",
            keywords=[["level", "average", "normal", "baseline", "own"],
                      ["good player", "better player", "talent", "ability", "form", "players differ", "instead"]],
            model_answer="Otherwise the 'hot' state just picks out good players and 'cold' picks out weaker ones. Subtracting each player's level makes the states mean form relative to his own normal."),
        Question(
            "q5", "mc", "In the forward filter, the 'update' step multiplies by:",
            "How well each state explains the new game: the likelihood (emission probability).",
            options=["How well each state explains the new game", "The number of games played",
                     "The player's recruiting rating", "A random number"],
            notes=["Correct (the likelihood).", "Not used.", "Not used here.", "The filter is deterministic given the data."],
            answer=0),
    ],
)
