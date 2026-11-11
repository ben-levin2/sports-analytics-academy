import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import data, theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


def _windows(d, past=6, future=3):
    X, Y = [], []
    for _, g in d.groupby(["player_id", "season"]):
        y = g.sort_values("week")["grade"].to_numpy(float)
        for t in range(past, len(y) - future + 1):
            X.append(y[t - past:t])
            Y.append(y[t:t + future])
    return np.array(X), np.array(Y)


@st.cache_data(show_spinner=False)
def _fit():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    df = data.make_game_logs()
    Xtr, Ytr = _windows(df[df["season"] <= 2023])
    Xte, Yte = _windows(df[df["season"] >= 2024])
    enc = PCA(2).fit(Xtr - Xtr.mean(1, keepdims=True))

    def ctx(X):
        return np.c_[X.mean(1), enc.transform(X - X.mean(1, keepdims=True))]

    dec = Ridge(1.0).fit(ctx(Xtr), Ytr)
    return Xte, Yte, dec.predict(ctx(Xte))


def visual():
    Xte, Yte, P = _fit()
    i = st.slider("Example window", 0, len(Xte) - 1, 12, key="l45_i")
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.plot(range(1, 7), Xte[i], marker="o", color=T.MUTED, label="past 6 games (input sequence)")
    ax.plot(range(7, 10), Yte[i], marker="o", color=T.TEXT, label="next 3 games (actual)")
    ax.plot(range(7, 10), P[i], marker="s", color=T.BLUE, lw=2.2, label="seq2seq forecast")
    ax.plot(range(7, 10), np.repeat(Xte[i].mean(), 3), color=T.AMBER, ls="--", label="repeat the 6-game average")
    ax.axvline(6.5, color=T.GRID, lw=1), ax.set_xlabel("Game in window"), ax.set_ylabel("Grade"), ax.legend(fontsize=7)
    show_fig(fig)


CODE = '''import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from academy.data import make_game_logs

df = make_game_logs()                                         # SIMULATED

def make_pairs(d, past=6, future=3):
    """Input sequence: 6 games. Output sequence: the next 3 games."""
    X, Y = [], []
    for _, g in d.groupby(["player_id", "season"]):
        y = g.sort_values("week")["grade"].to_numpy(float)
        for t in range(past, len(y) - future + 1):
            X.append(y[t - past:t])
            Y.append(y[t:t + future])
    return np.array(X), np.array(Y)

Xtr, Ytr = make_pairs(df[df["season"] <= 2023])               # earlier seasons
Xte, Yte = make_pairs(df[df["season"] >= 2024])               # later seasons

# ENCODER: squeeze 6 games into a 3-number context: overall level + 2 shape numbers (trend, recent swing)
shape = PCA(2).fit(Xtr - Xtr.mean(axis=1, keepdims=True))
def encode(X):
    return np.c_[X.mean(axis=1), shape.transform(X - X.mean(axis=1, keepdims=True))]

# DECODER: turn the context into all 3 future games at once
decoder = Ridge(alpha=1.0).fit(encode(Xtr), Ytr)
pred = decoder.predict(encode(Xte))
base = np.repeat(Xte.mean(axis=1, keepdims=True), 3, axis=1)  # baseline: repeat the 6-game average

for h in range(3):
    print(f"game +{h + 1}: baseline error {np.mean(np.abs(Yte[:, h] - base[:, h])):.2f}   "
          f"seq2seq error {np.mean(np.abs(Yte[:, h] - pred[:, h])):.2f}")
print(f"\\nOverall: baseline {np.mean(np.abs(Yte - base)):.2f}   seq2seq {np.mean(np.abs(Yte - pred)):.2f}  "
      f"({len(Xtr)} training pairs)")
'''

LESSON = Lesson(
    id="l4-05-seq2seq", level=4, order=5,
    title="Sequence-to-Sequence Models: Forecasting the Next Few Games at Once",
    subtitle="Encoders, context vectors and decoders",
    minutes=20,
    concept_ids=["seq2seq"],
    simple="""
So far each forecast was **one** next game. Sometimes you want a **sequence out**: the next three games, the rest of the season, a whole development path.

A **sequence-to-sequence (seq2seq)** model has two parts:
- An **encoder** reads the input sequence (the last 6 games) and squeezes it into a short summary, the **context**.
- A **decoder** turns that context into the output sequence (the next 3 games).

Translation apps work this way: encode a sentence, decode it in another language. With neural networks, both parts are usually LSTMs.

Here we build the simplest honest version on NumPy and scikit-learn. The encoder is a level plus two "shape" numbers (from PCA, L2.1), and the decoder is a linear model that writes all three future games at once. The idea is identical; only the parts are simpler.
""",
    problem="You need several future steps that hang together, not one step at a time, which would compound errors when each forecast feeds the next.",
    why="It forecasts a whole path in one go and separates two jobs: summarizing the past (encoder) and producing the future (decoder).",
    when_useful="""
- Multi-week or rest-of-season forecasts
- Inputs and outputs of different lengths (6 games in, 3 out)
- With neural versions: large data where complex encoders pay off
""",
    when_not="""
- When the honest answer is 'about his recent average' for every future game (often true with noisy grades)
- Small data with neural encoders and decoders: even more weights than a single LSTM
- When you only need the next game: a one-step model is simpler
""",
    compare="""
**Seq2seq vs repeating a one-step forecast:** feeding each forecast back in compounds its errors. Seq2seq predicts the whole path directly.

**Seq2seq vs an autoencoder (L2.3):** both squeeze into a code. An autoencoder rebuilds its own input; seq2seq produces a different, future sequence.

**Linear vs neural seq2seq:** same structure. Neural encoders and decoders can capture more, but need far more data.
""",
    sports="""
**Fictional scenario (simulated).** Pairs of 6 games → next 3 games, trained on 2019–2023 and tested on 2024–2025.

- The baseline, "repeat the 6-game average for all three games", errs by about **7.33** grade points per game.
- The linear seq2seq errs by about **7.18**: roughly **2% better**, and best for the very next game, where recent form matters most.
- With only about 400 training pairs, a neural seq2seq would have far more weights than examples. That's not a fair fight.
""",
    takeaway="Seq2seq = encoder (summarize the past) + decoder (write the future). The structure matters more than the parts, and simple parts are fine for small data.",
    visual=visual,
    visual_caption="Pick a window. Grey: the 6 input games. White: what actually happened next. Blue: the seq2seq forecast. Amber: just repeating the average.",
    math=[
        MathBlock(
            "Step 1: encode the past into a context",
            r"\mathbf{c} = \text{Enc}(x_{t-5}, \ldots, x_t)",
            """
- $x_{t-5}, \\ldots, x_t$: the last 6 games (the input sequence)
- $\\mathbf{c}$: the **context**, a short summary (here 3 numbers: level, trend, recent swing)
- $\\text{Enc}$: the encoder (here: an average plus PCA; in neural seq2seq, an LSTM)
""",
        ),
        MathBlock(
            "Step 2: decode the context into the future",
            r"(\hat x_{t+1}, \hat x_{t+2}, \hat x_{t+3}) = \text{Dec}(\mathbf{c})",
            """
- $\\hat x_{t+1}, \\ldots$: the forecast for each of the next 3 games
- $\\text{Dec}$: the decoder (here, one linear model per future game; in neural seq2seq, another LSTM)
""",
        ),
        MathBlock(
            "Step 3: judge each horizon separately",
            r"\text{Err}_h = \frac{1}{N}\sum_{n=1}^{N} \bigl| x_{t+h}^{(n)} - \hat x_{t+h}^{(n)} \bigr|,\quad h = 1, 2, 3",
            """
- $h$: how many games ahead
- Forecasts usually get worse as $h$ grows, and the gain over a simple average usually shrinks
""",
        ),
    ],
    code=CODE,
    code_notes="Change future=3 to future=5 (and the 3s in the code). How quickly does the seq2seq edge over the plain average disappear?",
    walkthrough=[
        ("def make_pairs(d, past=6, future=3):", "Build training examples: 6 games in, the next 3 games out."),
        ("make_pairs(df[df['season'] <= 2023])", "Training pairs from earlier seasons only; test pairs from 2024–2025."),
        ("shape = PCA(2).fit(Xtr - Xtr.mean(...))", "Learn the 2 most common shapes of a 6-game stretch (like 'trending up'), after removing each window's level."),
        ("def encode(X): ...", "The encoder: each window becomes 3 numbers, its level plus 2 shape scores. That's the context."),
        ("Ridge(alpha=1.0).fit(encode(Xtr), Ytr)", "The decoder: a linear model that maps the context to all 3 future games at once."),
        ("np.repeat(Xte.mean(...), 3, axis=1)", "Baseline: forecast every future game as the recent average."),
        ("for h in range(3):", "Compare errors for 1, 2 and 3 games ahead."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Multi-step questions** like rest-of-season production or year-2 and year-3 development paths are natural seq2seq problems.
- **Your rule still applies:** compare every horizon with 'repeat the recent average' on later seasons. A small edge on game +1 that vanishes by game +3 isn't a season forecast.
""",
    quiz=[
        Question(
            "q1", "mc", "In a seq2seq model, the encoder:",
            "Reads the input sequence and summarizes it into a context.",
            options=["Summarizes the input sequence into a context", "Produces the forecast directly",
                     "Removes bad games", "Ranks players"],
            notes=["Correct.", "That's the decoder's job.", "It removes nothing.", "Not its role."],
            answer=0),
        Question(
            "q2", "mc", "Why not just repeat a one-game forecast three times, feeding each forecast back in?",
            "Errors compound: each forecast's mistake becomes the next step's input.",
            options=["Errors compound step by step", "It's always better", "It needs more data", "It can't be computed"],
            notes=["Correct.", "Not usually.", "It needs less, actually.", "It can; it's just error-prone."],
            answer=0),
        Question(
            "q3", "numeric", "Baseline error 7.33, seq2seq error 7.18. What's the improvement in percent? (decimal or %)",
            "(7.33 − 7.18) / 7.33 ≈ 0.0205, so about 2%.", category="Application", answer=0.0205, tolerance=0.001),
        Question(
            "q4", "short", "Why is a linear encoder and decoder a sensible choice for ~400 training pairs?",
            "Few parameters suit small data; neural versions would overfit.",
            category="Application",
            keywords=[["few", "small", "little", "400", "not enough"],
                      ["overfit", "parameter", "weights", "simple", "noise", "memoriz"]],
            model_answer="With only about 400 examples, a model with few parameters is less likely to overfit. Neural encoders and decoders would have more weights than examples."),
        Question(
            "q5", "mc", "Forecast errors usually ___ as you predict further ahead.",
            "Further-ahead games depend less on recent form and more on luck, so errors usually grow and gains shrink.",
            options=["Grow", "Shrink", "Stay exactly the same", "Become zero"],
            notes=["Correct.", "Rarely.", "Not usually.", "Never."],
            answer=0),
    ],
)
