import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import data, sequence as Q, theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner="Training…")
def _compare():
    df = data.make_game_logs()
    Xtr, ytr = Q.windows(df[df["season"] <= 2023])
    Xte, yte = Q.windows(df[df["season"] >= 2024])
    preds = {"6-game average": Xte.mean(1), "Last game": Xte[:, -1],
             "EWMA (0.3)": np.array([Q.ewma_forecast(x, 0.3) for x in Xte]),
             "Windowed neural net (no PyTorch)": Q.windowed_mlp(Xtr, ytr, Xte)}
    if Q.torch_available():
        preds["Real LSTM (PyTorch)"] = Q.torch_lstm(Xtr, ytr, Xte)
    return {k: float(np.mean(np.abs(yte - v))) for k, v in preds.items()}, len(Xtr), len(Xte)


def visual():
    if st.button("Train and compare forecasters", key="l44_run"):
        st.session_state["l44_res"] = _compare()
    res = st.session_state.get("l44_res")
    if not Q.torch_available():
        st.info(Q.TORCH_NOTE)
    if not res:
        return
    mae, ntr, nte = res
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    names = list(mae)
    ax.barh(names, [mae[n] for n in names], color=[T.BLUE if "neural" in n.lower() or "LSTM" in n else T.MUTED for n in names])
    ax.set_xlim(min(mae.values()) - 0.5, max(mae.values()) + 0.3), ax.set_xlabel("Average error, next game (lower = better)")
    show_fig(fig)
    st.caption(f"Trained on {ntr} six-game windows (2019–2023), tested on {nte} windows (2024–2025).")


CODE = '''import numpy as np
from academy.data import make_game_logs
from academy.sequence import ewma_forecast, torch_available, windowed_mlp, windows, TORCH_NOTE

df = make_game_logs()                                       # SIMULATED game logs
Xtr, ytr = windows(df[df["season"] <= 2023], past=6)        # train: earlier seasons
Xte, yte = windows(df[df["season"] >= 2024], past=6)        # test: later seasons only
print(f"{len(Xtr)} training windows, {len(Xte)} test windows (that's very little for a neural net)\\n")

preds = {
    "6-game average": Xte.mean(axis=1),
    "last game": Xte[:, -1],
    "EWMA (alpha 0.3)": np.array([ewma_forecast(x, 0.3) for x in Xte]),
    "windowed neural net": windowed_mlp(Xtr, ytr, Xte),     # runs on scikit-learn, NOT an LSTM
}

if torch_available():                                       # the real LSTM only if PyTorch is installed
    from academy.sequence import torch_lstm
    preds["real LSTM (PyTorch)"] = torch_lstm(Xtr, ytr, Xte)
else:
    print(TORCH_NOTE, "\\n")

for name, p in preds.items():
    print(f"{name:<22} average error {np.mean(np.abs(yte - p)):.2f}")
'''

LESSON = Lesson(
    id="l4-04-lstm-training", level=4, order=4,
    title="LSTMs, Part 2: Training One, and an Honest Contest",
    subtitle="How neural nets learn, how much data they need, and HMMs vs LSTMs",
    minutes=25,
    concept_ids=["lstms"],
    simple="""
An LSTM starts with random weights (L4.3). **Training** shows it thousands of examples, "here are 6 games, here's the next one", and nudges every weight a tiny bit to shrink the error. Repeat many times.

The catch is **data**. An LSTM has hundreds or thousands of weights. With fewer examples than that, it learns the quirks of the training seasons (luck) instead of real patterns, and does worse on new seasons. That's called **overfitting**.

So the honest test is the same as always: train on **earlier seasons**, test on **later seasons**, and compare with **simple baselines**: the recent average, the last game, and a weighted recent average (EWMA).

This academy runs without PyTorch, the library for real LSTMs. If PyTorch is installed, the code trains a **real LSTM**. If not, it trains a small **windowed neural net** with scikit-learn instead, clearly labelled as **not an LSTM**, so you can still see the data-hunger problem.
""",
    problem="A neural sequence model can look sophisticated and still lose to a moving average. You need a fair, time-respecting contest to know.",
    why="This is where you decide whether a deep model has earned its complexity on *your* amount of data.",
    when_useful="""
- Tens of thousands of sequences or more, with long histories
- Several inputs per step and patterns too complex to write by hand
- When simpler forecasters have clearly plateaued
""",
    when_not="""
- Small game logs: here, under a thousand training windows
- When a moving average ties it: choose the simple one
- When results must be explained to staff in plain terms
""",
    compare="""
**HMM vs LSTM, side by side:**

| | HMM | LSTM |
|---|---|---|
| Memory | a few named states | a learned vector of numbers |
| Data needed | small | large |
| Interpretable | yes ("P(hot) = 0.8") | hardly |
| Uncertainty | built in | needs extra work |
| Best when | the structure is simple and known | the patterns are complex and data is plentiful |

**Neural net vs EWMA:** an EWMA has one setting (how fast to forget). A neural net has hundreds. With small data, one well-chosen setting often wins.
""",
    sports="""
**Fictional scenario (simulated).** About 900 six-game training windows (2019–2023), and about 500 test windows (2024–2025). Average error on the next game:

| Forecaster | Error |
|---|---|
| EWMA (0.3) | **7.13** (best) |
| Windowed neural net | 7.22 |
| 6-game average | 7.30 |
| Last game | 8.55 (worst; one game is too noisy) |

**Honest result:** the neural net doesn't beat a simple weighted average. It fits the training seasons better (6.9) than the test seasons (7.2), the classic sign of a model learning noise. If you install PyTorch, the real LSTM row appears. Expect it to land in the same neighbourhood at this data size, not to run away with it.
""",
    takeaway="Neural sequence models need lots of data. On small game logs, a weighted recent average is the forecaster to beat, and it usually wins.",
    visual=visual,
    visual_caption="Train the models and compare their errors on later seasons. Blue bars are the neural models. If PyTorch isn't installed you'll see an install note, and everything else still runs.",
    math=[
        MathBlock(
            "Step 1: what training minimizes",
            r"L(\theta) = \frac{1}{N}\sum_{n=1}^{N} \bigl(y_n - \hat y_n(\theta)\bigr)^2",
            """
- $\\theta$: all the network's weights
- $y_n$: the actual next-game grade in training example $n$; $\\hat y_n(\\theta)$: the network's forecast
- $N$: number of training windows
""",
        ),
        MathBlock(
            "Step 2: gradient descent (nudging the weights)",
            r"\theta \leftarrow \theta - \eta\, \nabla_\theta L(\theta)",
            """
- $\\nabla_\\theta L$: the gradient: which direction makes the error bigger, for every weight
- $\\eta$ (eta): the learning rate, the size of each nudge
- For an LSTM the gradient flows back through every game in the sequence ("backpropagation through time")
""",
        ),
        MathBlock(
            "Step 3: the overfitting gap",
            r"\text{gap} = \text{Error}_{\text{test}} - \text{Error}_{\text{train}}",
            """
- A big positive gap means the model memorized training noise
- More weights and less data make the gap bigger
- Defences: more data, fewer weights, penalties (weight decay), early stopping, and always a held-out later season
""",
        ),
    ],
    code=CODE,
    code_notes="Change `past=6` to `past=3` or `past=10`. Longer windows give the network more to look at, but fewer training examples.",
    walkthrough=[
        ("windows(df[df['season'] <= 2023], past=6)", "Cut every player-season into 'previous 6 games → next game' examples, from earlier seasons only."),
        ("windows(df[df['season'] >= 2024], past=6)", "Test examples from later seasons the models never saw."),
        ("Xte.mean(axis=1)", "Baseline: the average of the previous 6 games."),
        ("ewma_forecast(x, 0.3)", "Baseline: a weighted average where recent games count more."),
        ("windowed_mlp(Xtr, ytr, Xte)", "A small scikit-learn neural net on the 6-game window. Works without PyTorch, but it is NOT an LSTM."),
        ("if torch_available():", "Only if PyTorch is installed, train a real LSTM. Otherwise print the install note and carry on."),
        ("np.mean(np.abs(yte - p))", "Average error on the later seasons: the fair comparison."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Same story as Models 9 and 9b:** sophisticated sequence models must beat simple recency rules on later seasons. Here they don't.
- **PSFAL's game-level data** covers only parts of some seasons, so LSTM work is a learning exercise. Revisit it only with far more sequences (e.g., play-level data across many seasons).
- **PyTorch stays optional** for this academy; nothing on your PSFAL side depends on it.
""",
    quiz=[
        Question(
            "q1", "mc", "What is overfitting?",
            "Learning the quirks (noise) of the training data, so the model does worse on new data.",
            options=["Learning training-set noise, so new data goes worse", "Training for too few steps",
                     "Using too much data", "Predicting the average"],
            notes=["Correct.", "That's underfitting.", "More data helps prevent it.", "That's a baseline, not overfitting."],
            answer=0),
        Question(
            "q2", "numeric", "Train error 6.9, test error 7.22. What is the overfitting gap?",
            "7.22 − 6.9 = 0.32.", category="Application", answer=0.32, tolerance=0.005),
        Question(
            "q3", "mc", "On ~900 training windows, the neural net loses to an EWMA. What should you do?",
            "Use the simpler forecaster; the neural net hasn't earned its complexity on this data.",
            category="Application",
            options=["Use the EWMA; the net hasn't earned its complexity", "Make the network bigger",
                     "Test on the training seasons instead", "Report the neural net anyway"],
            notes=["Correct, and consistent with your PSFAL rule.", "More weights with the same data usually overfits more.", "That's leakage.", "That hides an honest failure."],
            answer=0),
        Question(
            "q4", "short", "Give two reasons an HMM might be preferred over an LSTM for weekly game grades.",
            "Needs much less data; easier to interpret; gives probabilities of states.",
            keywords=[["less data", "small data", "little data", "fewer", "data"],
                      ["interpret", "explain", "probab", "named", "simple", "understand"]],
            model_answer="It needs far less data than an LSTM, and it's easy to explain: two named states with probabilities, like 'P(hot) = 0.8'."),
        Question(
            "q5", "mc", "PyTorch isn't installed. What happens in this academy?",
            "Everything runs; the real-LSTM lab is skipped with an install note, and a clearly-labelled scikit-learn fallback runs instead.",
            options=["Everything runs; the real-LSTM part shows an install note", "The app crashes",
                     "Lessons are hidden", "It installs PyTorch automatically"],
            notes=["Correct.", "PyTorch is optional by design.", "Nothing is hidden.", "Nothing is installed without your approval."],
            answer=0),
    ],
)
