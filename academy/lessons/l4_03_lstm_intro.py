import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def run_cell(xs, forget_bias, input_w=1.0):
    """A one-number LSTM cell with hand-set weights, to show what the forget gate does."""
    c, h, cs, fs = 0.0, 0.0, [], []
    for x in xs:
        f = sigmoid(forget_bias)                 # forget gate: how much old memory to keep
        i = sigmoid(2.0 * abs(x) - 1.0)          # input gate: write more when the game is surprising
        g = np.tanh(input_w * x)                 # candidate memory: this game's signal
        c = f * c + i * g
        o = sigmoid(1.0)
        h = o * np.tanh(c)
        cs.append(c)
        fs.append(f)
    return np.array(cs), np.array(fs)


def visual():
    xs = np.array([0, 0, 1.5, 1.2, 1.4, 0, 0, 0, -1.3, -1.5, 0, 0], float)
    c1, c2 = st.columns([1, 2])
    with c1:
        fb = st.slider("Forget-gate setting", -2.0, 4.0, 1.5, 0.25, key="l43_fb",
                       help="High = keep memory a long time; low = forget quickly")
    cs, fs = run_cell(xs, fb)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.bar(range(1, 13), xs, color=[T.ROSE if v > 0 else T.BLUE if v < 0 else T.GRID for v in xs], alpha=0.5,
           label="game vs normal (input)")
    ax.plot(range(1, 13), cs, color=T.AMBER, lw=2.6, marker="o", label="cell memory")
    ax.axhline(0, color=T.MUTED, lw=0.8)
    ax.set_xlabel("Game"), ax.set_ylabel("Value"), ax.legend(fontsize=8)
    with c2:
        show_fig(fig)
    st.caption(f"Forget gate keeps {fs[0]:.0%} of memory each game. High settings remember the hot streak for weeks; "
               "low settings drop it almost immediately.")


CODE = '''import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))                   # squashes any number into 0..1 (a gate's 'how open')

rng = np.random.default_rng(0)
n_in, n_hid = 1, 4                                # 1 input per game, 4 memory units
W = {g: rng.normal(0, 0.5, (n_hid, n_in + n_hid)) for g in "figo"}   # one weight table per gate
b = {g: np.zeros(n_hid) for g in "figo"}
b["f"] += 1.0                                     # common trick: start with the forget gate fairly open

def lstm_step(x, h, c):
    z = np.concatenate([x, h])                    # this game's input + last game's output
    f = sigmoid(W["f"] @ z + b["f"])              # forget gate: keep how much old memory?
    i = sigmoid(W["i"] @ z + b["i"])              # input gate: write how much new information?
    g = np.tanh(W["g"] @ z + b["g"])              # candidate memory: what this game suggests
    o = sigmoid(W["o"] @ z + b["o"])              # output gate: reveal how much memory?
    c = f * c + i * g                             # update the long-term memory (cell state)
    h = o * np.tanh(c)                            # short-term output passed to the next game
    return h, c, f

# --- SIMULATED season: grades vs his normal level, scaled to roughly -1..1 ---
season = np.array([0.1, 0.9, 1.0, 0.8, 1.1, -0.2, -0.9, -1.0, -0.8, 0.0, 0.3, 0.2])
h, c = np.zeros(n_hid), np.zeros(n_hid)
for week, x in enumerate(season, start=1):
    h, c, f = lstm_step(np.array([x]), h, c)
    print(f"week {week:>2}: input {x:+.1f}  memory {np.round(c, 2)}  forget gate (avg) {f.mean():.2f}")
print("\\nThese weights are RANDOM, so the memory means nothing yet. Training (L4.4) is what makes it useful.")
'''

LESSON = Lesson(
    id="l4-03-lstm-intro", level=4, order=3,
    title="LSTMs, Part 1: A Neural Network With a Memory",
    subtitle="Recurrent networks and the forget, input and output gates",
    minutes=25,
    concept_ids=["lstms"],
    simple="""
An HMM remembers the past through **one** hidden state (hot or cold). What if what matters is more complicated, like how a QB played in his last 3 games against strong defenses, before an injury?

A **recurrent neural network** reads a sequence one game at a time and carries a running **memory** forward. A plain one forgets quickly. An **LSTM** (long short-term memory) adds **gates**, little valves between 0 (shut) and 1 (open) that the network learns to control:

- **Forget gate:** how much of the old memory to keep.
- **Input gate:** how much of this game to write into memory.
- **Output gate:** how much of the memory to use right now.

With the right gate settings, an LSTM can remember a hot streak for weeks, or drop it immediately when a game says form has changed. It **learns** those settings from data.
""",
    problem="Some patterns depend on long, complicated histories that a single hidden state can't hold. LSTMs can learn what to remember and what to forget.",
    why="They can learn rich sequence patterns directly from data without you specifying states or rules.",
    when_useful="""
- Long sequences (many games, many seasons, play-by-play) with lots of players
- Several inputs per step (grade, opponent strength, snaps, injuries)
- Patterns too complex to write down as a few states
""",
    when_not="""
- Short sequences and small data, like ~10 games per player-season. It will memorize noise
- When you need to explain *why* to a coach: the gates are hard to interpret
- Before beating simple baselines: compare with the season average and EWMA first (L4.4)
""",
    compare="""
**LSTM vs HMM:** an HMM has a few named states and clear probabilities, needs little data and is easy to explain. An LSTM has a flexible learned memory, needs lots of data and is hard to explain.

**LSTM vs plain recurrent network:** plain ones forget long-ago games because their memory fades quickly; the LSTM's forget gate can keep memory almost unchanged.

**LSTM vs autoencoder (L2.3):** both are neural networks, but an LSTM reads things in order and carries memory between steps.
""",
    sports="""
**Fictional scenario (simulated).** A season of game grades measured against a QB's normal level: a hot stretch in weeks 2–5, then a cold stretch in weeks 7–9.

With a **high** forget-gate setting, the cell's memory builds through the hot stretch and lingers for a few games after it ends. With a **low** setting it tracks only the latest game.

The code runs a real 4-unit LSTM step by step with **random** weights. Its memory numbers mean nothing yet. That's the point: an LSTM is only as good as its training, which needs data, and the data is the catch (L4.4).
""",
    takeaway="An LSTM carries a memory through a sequence and learns gates that decide what to keep, add and use. It's powerful, data-hungry, and hard to explain.",
    visual=visual,
    visual_caption="Bars are game-by-game inputs; the amber line is the cell's memory. Move the forget gate: high keeps the hot streak in memory for weeks; low lets it go almost immediately.",
    math=[
        MathBlock(
            "Step 1: gates are numbers between 0 and 1",
            r"f_t = \sigma(W_f [x_t, h_{t-1}] + b_f), \quad i_t = \sigma(W_i [x_t, h_{t-1}] + b_i), \quad o_t = \sigma(W_o [x_t, h_{t-1}] + b_o)",
            """
- $x_t$: this game's input; $h_{t-1}$: last game's output
- $W, b$: learned weights and offsets (one set per gate)
- $\\sigma$: the sigmoid, which squashes any number into 0–1 (0 = gate shut, 1 = wide open)
- $f_t$ forget gate, $i_t$ input gate, $o_t$ output gate
""",
        ),
        MathBlock(
            "Step 2: update the memory",
            r"\tilde c_t = \tanh(W_c [x_t, h_{t-1}] + b_c), \qquad c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t",
            """
- $\\tilde c_t$: what this game suggests writing into memory
- $c_t$: the **cell state**, the long-term memory
- $\\odot$: multiply unit by unit
- In words: keep a share ($f_t$) of the old memory, and add a share ($i_t$) of the new suggestion
""",
        ),
        MathBlock(
            "Step 3: produce this step's output",
            r"h_t = o_t \odot \tanh(c_t)",
            """
- $h_t$: what the network passes to the prediction layer and to the next game
- The output gate decides how much of the memory to reveal now
""",
            "If f is close to 1 and i close to 0, memory passes through almost unchanged. That's how LSTMs remember for a long time.",
        ),
    ],
    code=CODE,
    code_notes="Change `b['f'] += 1.0` to `b['f'] -= 3.0` and rerun: the forget gate shuts and the memory resets every game.",
    walkthrough=[
        ("def sigmoid(z):", "The squashing function that turns any number into a 0–1 'how open' value."),
        ("W = {g: rng.normal(...) for g in 'figo'}", "Random starting weights for the four parts: forget, input, candidate (g) and output."),
        ("b['f'] += 1.0", "A standard trick: start with the forget gate fairly open so early training doesn't wipe memory."),
        ("z = np.concatenate([x, h])", "Each gate looks at this game AND what the network said after the previous game."),
        ("f = sigmoid(...); i = sigmoid(...); o = sigmoid(...)", "The three gates, each between 0 and 1 for every memory unit."),
        ("g = np.tanh(...)", "The candidate memory: this game's suggested update, squashed to between −1 and 1."),
        ("c = f * c + i * g", "The heart of the LSTM: keep some old memory, add some new."),
        ("h = o * np.tanh(c)", "The output passed onward: a gated, squashed view of the memory."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Data reality:** game-level data covering only parts of some seasons, with about 10 games per player-season, is far too little for an LSTM. Treat LSTM work as a learning exercise.
- **Where it could fit later:** play-by-play sequences across many seasons, which give thousands of steps per player.
- **Your rule:** an LSTM must beat the season-to-date average and an EWMA on later seasons first. L4.4 runs that comparison.
""",
    quiz=[
        Question(
            "q1", "mc", "What does the forget gate control?",
            "How much of the old memory (cell state) is kept at each step.",
            options=["How much old memory to keep", "How many games to read", "Which players to drop", "The learning rate"],
            notes=["Correct.", "The sequence length is separate.", "It drops nothing from the data.", "That's a training setting."],
            answer=0),
        Question(
            "q2", "numeric", "Old memory c = 2.0, forget gate f = 0.7, input gate i = 0.5, candidate g = 0.4. What is the new memory? (c = f·c + i·g)",
            "0.7 × 2.0 + 0.5 × 0.4 = 1.4 + 0.2 = 1.6.", category="Application", answer=1.6, tolerance=0.001),
        Question(
            "q3", "mc", "Why might an LSTM do badly with ~10 games per player-season?",
            "It has many weights to learn and too little data, so it memorizes noise instead of patterns.",
            category="Application",
            options=["Too little data for many weights, so it memorizes noise", "LSTMs can't handle numbers",
                     "10 is too many games", "It always predicts zero"],
            notes=["Correct.", "They handle numbers fine.", "It's too few.", "Not true."],
            answer=0),
        Question(
            "q4", "short", "Name one advantage of an HMM over an LSTM for hot/cold form.",
            "Easier to interpret (named states, probabilities), needs far less data.",
            keywords=[["interpret", "explain", "understand", "clear", "named", "simple", "less data", "small data", "fewer"],
                      ["state", "probab", "data", "coach", "hot", "parameters", "need"]],
            model_answer="It's much easier to explain (two named states with probabilities) and needs far less data, which suits ~10 games per season."),
        Question(
            "q5", "mc", "The code's LSTM uses random weights. Its memory values are:",
            "Meaningless until the network is trained on data.",
            options=["Meaningless until it's trained", "Already accurate forecasts", "The true hot/cold states", "Recruiting ratings"],
            notes=["Correct.", "Untrained networks forecast nothing useful.", "It has learned nothing yet.", "Unrelated."],
            answer=0),
    ],
)
