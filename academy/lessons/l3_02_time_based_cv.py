import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


def visual():
    seasons = list(range(2016, 2026))
    c1, c2 = st.columns([1, 2])
    with c1:
        mode = st.radio("Validation scheme", ["Random split", "Rolling origin (time-based)"], key="l32_mode")
        st.caption("Blue = train, amber = test, dark = unused.")
    rng = np.random.default_rng(4)
    if mode.startswith("Random"):
        grid = rng.integers(1, 3, size=(5, len(seasons)))         # each fold mixes past and future
        title = "Random split: future seasons leak into training"
    else:
        grid = np.zeros((7, len(seasons)), int)
        for i, T_ in enumerate(range(3, 10)):
            grid[i, :T_] = 1
            grid[i, T_] = 2
        title = "Rolling origin: always train on the past, test on the next season"
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.imshow(grid, cmap=ListedColormap([T.PANEL, T.BLUE, T.AMBER]), vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(seasons)), [str(s)[2:] for s in seasons])
    ax.set_yticks(range(len(grid)), [f"fold {i + 1}" for i in range(len(grid))])
    ax.grid(False), ax.set_xlabel("Season ('16–'25)"), ax.set_title(title, fontsize=10)
    with c2:
        show_fig(fig)


CODE = '''import numpy as np
from sklearn.model_selection import KFold
from academy.data import make_qb_completions

df = make_qb_completions()                         # SIMULATED QB seasons, 2016-2025
df = df[df["opportunities"] >= 50].copy()
df["rate"] = df["successes"] / df["opportunities"]
df = df.reset_index(drop=True)

def predict_from(history, rows):
    """Predict each row's rate as that player's mean rate in `history` (league mean if none)."""
    means = history.groupby("player_id")["rate"].mean()
    return rows["player_id"].map(means).fillna(history["rate"].mean()).to_numpy()

# --- 1. Random 5-fold split: training folds contain FUTURE seasons of the same player ---
errs = []
for train_idx, test_idx in KFold(5, shuffle=True, random_state=0).split(df):
    train, test = df.loc[train_idx], df.loc[test_idx]
    errs.append(np.mean((predict_from(train, test) - test["rate"]) ** 2))
random_mse = np.mean(errs)

# --- 2. Rolling origin: train on seasons < T, test on season T ---
errs, n_test = [], 0
for T in range(2019, 2026):
    train, test = df[df["season"] < T], df[df["season"] == T]
    assert train["season"].max() < T               # leakage check: nothing from T or later
    errs.append(np.mean((predict_from(train, test) - test["rate"]) ** 2) * len(test))
    n_test += len(test)
rolling_mse = np.sum(errs) / n_test

# --- 3. A leaky FEATURE: career average computed on ALL seasons before any split ---
df["career_rate"] = df.groupby("player_id")["rate"].transform("mean")    # includes the future AND the row itself
test_rows = df[df["season"] >= 2019]
feature_mse = np.mean((test_rows["career_rate"] - test_rows["rate"]) ** 2)

print(f"Rolling-origin (honest) error:     {rolling_mse:.5f}")
print(f"Random-split error:                {random_mse:.5f}  ({1 - random_mse / rolling_mse:.0%} too optimistic)")
print(f"Leaky career-average feature:      {feature_mse:.5f}  ({1 - feature_mse / rolling_mse:.0%} too optimistic)")
'''

LESSON = Lesson(
    id="l3-02-time-based-cv", level=3, order=2,
    title="Time-Based Cross-Validation: Never Let the Future Leak In",
    subtitle="Rolling-origin validation and temporal leakage",
    minutes=20,
    concept_ids=["time-based-cv"],
    simple="""
To test a model honestly you hide some data, train on the rest, and check predictions on the hidden part.

With sports data, **how** you hide it matters enormously. If you hide random rows, the model trains on a player's *2024* season and is then "tested" on his *2022* season. It has peeked at the future. Real life never lets you do that.

**Time-based cross-validation** hides data the way time does: train only on seasons *before* T, test on season T, then move T forward one season and repeat. This is called **rolling-origin** validation.
""",
    problem="Random splits quietly let future information into training (temporal leakage). The model looks better on paper than it will ever be when you actually use it next season.",
    why="It's the only test that copies how the model will really be used: fit on what you know now, predict what comes next. It also shows whether performance holds up season after season or depends on one lucky year.",
    when_useful="""
- Any forecast of a future season, game or outcome
- Player data where the same player appears in several seasons
- Settings where rules, schemes or recruiting change over time
""",
    when_not="""
- Data with no time order and no repeated players (rare in sports)
- Very few seasons: with 3 seasons you only get 1–2 test folds. Report that honestly rather than switching to random splits
- If you then tune the model over and over on the same test seasons, they stop being a fair test. Keep a final holdout locked, as PSFAL does
""",
    compare="""
**Random K-fold:** fine when rows are truly independent; optimistic for player-seasons. **Group K-fold (by player):** keeps each player in one fold, which removes same-player leakage but still lets the model train on later seasons. **Rolling origin (expanding window):** train on everything before T. **Sliding window:** train on only the last few seasons before T, which helps when old seasons no longer resemble the present. **Locked chronological holdout:** the final, touch-once test. This is your PSFAL practice.
""",
    sports="""
**Fictional scenario (simulated).** Predict each QB's completion % as his average in *other* seasons.

- **Random 5-fold split:** "other seasons" includes his *future* seasons. The model sees how he turned out.
- **Rolling origin:** for each test season from 2019 to 2025, only *earlier* seasons are allowed.

On the simulated data, the random split makes the error look about **5% smaller** than the honest rolling-origin estimate. That's a small but real flattering bias, and it can be the whole margin by which a model "beats" its baseline.

The bigger danger is a **leaky feature**. Compute each player's career average over *all* seasons before splitting, and the error looks about **80% smaller**, because the "prediction" already contains the answer. Nothing about the model changed, only what it was allowed to see.

**Leakage hides in more places than the split:**
- career averages that include later seasons
- priors or cut-offs fitted on all years
- "final" recruiting ratings updated after the season
- features computed after the outcome

The Level 1 project checks the split, the prior, the top-third cutoff and the qualifying volume on every fold.
""",
    takeaway="Test the way you'll use it: train on the past, predict the next season, and check that nothing from the future slipped in.",
    visual=visual,
    visual_caption="Switch between the schemes. In a random split, amber test seasons sit among blue training seasons, including later ones. In rolling origin, the test season is always to the right of everything used for training.",
    math=[
        MathBlock(
            "Step 1: one fold of rolling origin",
            r"\mathcal{D}_{\text{train}}^{(T)} = \{\,\text{rows with season} < T\,\},\qquad \mathcal{D}_{\text{test}}^{(T)} = \{\,\text{rows with season} = T\,\}",
            """
- $T$: the test season
- $\\mathcal{D}_{\\text{train}}^{(T)}$: every row from seasons strictly before $T$. This is everything the model may see
- $\\mathcal{D}_{\\text{test}}^{(T)}$: the rows of season $T$ only
""",
        ),
        MathBlock(
            "Step 2: average the error across folds",
            r"\widehat{\text{Err}} = \frac{\sum_{T=T_0}^{T_{\max}} \sum_{i \in \mathcal{D}_{\text{test}}^{(T)}} L\bigl(y_i, \hat f^{(T)}(x_i)\bigr)}{\sum_{T=T_0}^{T_{\max}} |\mathcal{D}_{\text{test}}^{(T)}|}",
            """
- $T_0$: the first test season (you need some seasons to train on first)
- $\\hat f^{(T)}$: the model refit using only $\\mathcal{D}_{\\text{train}}^{(T)}$
- $L(y, \\hat y)$: the loss for one prediction, e.g. squared error $(y - \\hat y)^2$
- $|\\mathcal{D}_{\\text{test}}^{(T)}|$: number of test rows in season $T$. Dividing by the total gives the average loss per test row
""",
        ),
        MathBlock(
            "Step 3: the leakage check",
            r"\max\{\text{season} : \text{row} \in \mathcal{D}_{\text{train}}^{(T)}\} < T \quad \text{for every fold, and for every input the model uses}",
            """
- Applies to the training rows *and* anything derived from data: priors, averages, thresholds, scalers
- If any of them was computed with season $T$ or later, the test is contaminated
""",
            "In code it's one line, `assert train['season'].max() < T`. The hard part is remembering to check every derived input, not just the rows.",
        ),
    ],
    code=CODE,
    code_notes="Change `range(2019, 2026)` to start at 2017. You get more test folds, but the early folds train on only one season. Notice how the estimate changes.",
    walkthrough=[
        ("df = df[df['opportunities'] >= 50].copy()", "Keep seasons with at least 50 attempts so each rate means something."),
        ("def predict_from(history, rows):", "A deliberately simple model: predict a player's rate as his average in whatever history it's allowed to see."),
        ("rows['player_id'].map(means).fillna(...)", "Look up each player's historical average. Players with no history get the league average."),
        ("KFold(5, shuffle=True, random_state=0)", "The tempting mistake: shuffle all rows and cut them into 5 random folds, ignoring time."),
        ("train, test = df.loc[train_idx], df.loc[test_idx]", "Each training fold contains a mix of past AND future seasons, often from the same players being tested."),
        ("for T in range(2019, 2026):", "Rolling origin: each season from 2019 to 2025 takes a turn as the test season."),
        ("train, test = df[df['season'] < T], df[df['season'] == T]", "Train on strictly earlier seasons only, and test on season T."),
        ("assert train['season'].max() < T", "The leakage check. If anything from season T or later got in, the code stops."),
        ("np.sum(errs) / n_test", "Average error per test row across all folds, so busier seasons count for more."),
        ("df.groupby('player_id')['rate'].transform('mean')", "The classic leaky feature: a career average built from every season, including the one being predicted."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.** This is how the ideas carry over:

- **Your rule** (beat a simple baseline on a test set fixed in advance) assumes the test set is *after* the training data. Model 8's pre-set test is exactly that.
- **Models 9 and 9b** were next-season tests. A random split could have made them look like they beat "same as last season". The chronological test is why you can trust that they failed.
- **Level 1 project:** every Production Candidate run now uses rolling origin and checks for leakage in the rows, the prior, the top-third cutoff and the qualifying volume, on every fold.
""",
    quiz=[
        Question(
            "q1", "mc", "Why can a random train/test split mislead you with player-season data?",
            "The model can train on seasons that come after the ones it is tested on, which is information it would never have in real use.",
            options=["It lets future seasons into training", "It makes the training set too small",
                     "Random numbers are unreliable", "It always underestimates accuracy"],
            notes=["Correct.", "Size isn't the issue here; the timing is.", "The randomness is fine; the problem is the order.",
                   "It usually OVERestimates accuracy."],
            answer=0),
        Question(
            "q2", "numeric", "Data covers 2016–2025. Rolling origin uses test seasons 2019 to 2025. How many folds is that?",
            "2019, 2020, ..., 2025 is 7 test seasons, so 7 folds.",
            category="Application", answer=7, tolerance=0),
        Question(
            "q3", "mc", "Which of these is temporal leakage when predicting 2023 production?",
            "A career average that includes 2024 and 2025 seasons uses information from after the prediction date.",
            category="Application",
            options=["Using a career average computed through 2025", "Using the player's 2022 snap count",
                     "Using his recruiting rating at signing", "Using his position"],
            notes=["Correct: it includes future seasons.", "2022 is before 2023, so it's fine.",
                   "Known at signing, before 2023, so it's fine (an updated rating would not be).", "Known in advance."],
            answer=0),
        Question(
            "q4", "short", "The prior in the Level 1 project is fitted from data. Why must it be refit inside each fold?",
            "A prior fitted on all seasons contains the test season's information, so it would leak the future.",
            category="Application",
            keywords=[["leak", "future", "test season", "later", "contamin", "peek"],
                      ["fold", "before", "earlier", "only", "past", "refit", "each"]],
            model_answer="Because a prior fitted on all seasons already contains the test season (and later ones). Refitting it on the earlier seasons only, inside each fold, keeps the future out."),
        Question(
            "q5", "mc", "A model beats 'same as last season' in a random split but loses in rolling origin. What's the right conclusion?",
            "The honest, time-respecting test is the one that matches real use. The random-split win was most likely leakage.",
            options=["It does not beat the baseline; keep it exploratory",
                     "It beats the baseline; random splits use more data",
                     "Average the two results", "Try more random seeds until it wins"],
            notes=["Correct, and consistent with your rule for PSFAL.", "More data doesn't fix peeking at the future.",
                   "Averaging an honest and a leaky test is still leaky.", "That's tuning until it wins, which is another form of leakage."],
            answer=0),
    ],
)
