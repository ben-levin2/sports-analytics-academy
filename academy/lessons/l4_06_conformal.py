import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import data, sequence as Q, theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


def _forecasts(d):
    out = []
    for _, g in d.groupby(["player_id", "season"]):
        y = g.sort_values("week")["grade"].to_numpy(float)
        for t in range(2, len(y)):
            out.append((y[t], y[:t].mean(), y[:t].std()))
    return np.array(out)


@st.cache_data(show_spinner=False)
def _data():
    df = data.make_game_logs()
    return _forecasts(df[df["season"] == 2023]), _forecasts(df[df["season"] >= 2024])


def visual():
    calib, test = _data()
    level = st.slider("Target coverage", 0.5, 0.95, 0.8, 0.05, key="l46_level")
    r = np.abs(calib[:, 0] - calib[:, 1])
    q = Q.conformal_halfwidth(r, level)
    cov = float(np.mean(np.abs(test[:, 0] - test[:, 1]) <= q))
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.hist(r, bins=40, color=T.BLUE, alpha=0.7)
    ax.axvline(q, color=T.AMBER, lw=2.4, label=f"half-width = {q:.1f} grade points")
    ax.set_xlabel("|actual − forecast| on the calibration season (2023)"), ax.set_ylabel("Games"), ax.legend(fontsize=8)
    show_fig(fig)
    m1, m2 = st.columns(2)
    m1.metric("Target coverage", f"{level:.0%}")
    m2.metric("Actual coverage on 2024–25", f"{cov:.1%}", f"{cov - level:+.1%}", delta_color="off")


CODE = '''import numpy as np
from academy.data import make_game_logs

df = make_game_logs()                                          # SIMULATED

def forecasts(d):
    """For every game from the 3rd on: (actual, season-to-date average, season-to-date SD)."""
    out = []
    for _, g in d.groupby(["player_id", "season"]):
        y = g.sort_values("week")["grade"].to_numpy(float)
        for t in range(2, len(y)):
            out.append((y[t], y[:t].mean(), y[:t].std()))
    return np.array(out)

calib = forecasts(df[df["season"] == 2023])                    # calibration season (earlier)
test = forecasts(df[df["season"] >= 2024])                     # test seasons (later)

# Split conformal: look at how wrong the forecast was on the calibration season...
scores = np.abs(calib[:, 0] - calib[:, 1])
n = len(scores)
k = int(np.ceil((n + 1) * 0.80)) - 1                           # the slightly-above-80% position
half_width = np.sort(scores)[k]                                # ...and use that error size as the interval

covered = np.abs(test[:, 0] - test[:, 1]) <= half_width
print(f"Conformal 80% interval: forecast +/- {half_width:.1f} grade points")
print(f"Coverage on later seasons: {covered.mean():.1%}  (target 80%)")

# The naive alternative: +/- 1.28 x the player's own SD so far (assumes a bell curve and enough games)
naive = np.abs(test[:, 0] - test[:, 1]) <= 1.2816 * test[:, 2]
print(f"Naive 'own SD' interval coverage: {naive.mean():.1%}  (badly too narrow early in the season)")
'''

LESSON = Lesson(
    id="l4-06-conformal", level=4, order=6,
    title="Conformal Prediction: Honest Ranges Around Any Forecast",
    subtitle="Prediction intervals with a coverage guarantee, and when the guarantee breaks",
    minutes=20,
    concept_ids=["conformal-prediction"],
    simple="""
A forecast like "68 next game" is incomplete. Staff need a **range**: "between 59 and 77, 80% of the time."

**Conformal prediction** builds that range from your model's **past mistakes**, with almost no assumptions:

1. Run the forecaster on a **calibration** set it wasn't fitted on (here, last season).
2. Record how wrong it was each time.
3. Find the size of error that covers 80% of those mistakes.
4. Use that size as ± around every new forecast.

It works with **any** forecaster, from a season average to an LSTM. The guarantee: if future games behave like the calibration games, about 80% of future outcomes land inside the range.

The fine print, "behave like", is called **exchangeability**. If the future looks different (a rule change, a new scheme, early-season games vs late-season games), coverage can drift. So you always **check** it on later data.
""",
    problem="Point forecasts hide how wrong they can be. Naive ranges (assuming a bell curve, or a player's own short-run SD) are often far too narrow.",
    why="It gives honest ranges for any model, in a few lines of code, with a guarantee you can verify.",
    when_useful="""
- Wrapping any forecast (next game, next season, P1's rates) with a range
- Models that give no uncertainty at all (most machine learning)
- Reporting to staff: '80% of the time between X and Y'
""",
    when_not="""
- When the future clearly differs from the calibration data (era changes): re-calibrate on recent data
- Tiny calibration sets (a few dozen errors): the range itself is unstable
- When you need a different range for each player: basic split conformal gives everyone the same width (adaptive versions exist)
""",
    compare="""
| Interval | What it's about | Typical width |
|---|---|---|
| **Confidence interval** | an average or parameter (the true season mean) | narrow |
| **Credible interval** (L1.4) | a parameter, Bayesian | narrow |
| **Prediction interval** | one future outcome (the next game) | wide, because it includes game-to-game luck |

Conformal produces **prediction** intervals. **Conformal vs model-based intervals** (e.g., GP bands, L3.5): model-based ones are only right if the model's assumptions are; conformal only assumes the future resembles the calibration data.
""",
    sports="""
**Fictional scenario (simulated).** The forecaster is the plain season-to-date average. Calibrate on the 2023 season, test on 2024–2025.

- **Conformal 80% interval:** about **±11 grade points**. Coverage on later seasons: about **76%**, a little short of 80%.
- **Naive interval** (±1.28 × the player's own SD so far): only about **62%** coverage. With 2–3 games, a player's own SD is far too unreliable.

**Honest note:** 76% vs the 80% target shows the guarantee isn't magic. The test seasons differ slightly from the single calibration season. The Level 4 project re-calibrates every season and lands near 80%.
""",
    takeaway="Conformal prediction turns your model's past mistakes into honest ranges for any forecast. Always check the coverage on later data.",
    visual=visual,
    visual_caption="The histogram shows how wrong the forecast was on the calibration season. The amber line is the conformal half-width. Change the target and compare it with the coverage actually achieved on later seasons.",
    math=[
        MathBlock(
            "Step 1: conformity scores on the calibration set",
            r"s_i = \bigl| y_i - \hat y_i \bigr|, \quad i = 1, \ldots, n",
            """
- $y_i$: what actually happened in calibration game $i$; $\\hat y_i$: the forecast for it
- $s_i$: how wrong the forecast was (the "score")
- $n$: number of calibration forecasts
""",
        ),
        MathBlock(
            "Step 2: the conformal quantile",
            r"\hat q = s_{(k)}, \qquad k = \bigl\lceil (n+1)(1-\alpha) \bigr\rceil",
            """
- $s_{(k)}$: the $k$-th smallest score
- $1 - \\alpha$: the target coverage (0.80 here, so $\\alpha = 0.20$)
- $\\lceil \\cdot \\rceil$: round up. The "+1" is a small correction that makes the guarantee exact
""",
        ),
        MathBlock(
            "Step 3: the interval and its guarantee",
            r"\bigl[\hat y_{\text{new}} - \hat q,\;\; \hat y_{\text{new}} + \hat q\bigr], \qquad P(\text{covered}) \ge 1 - \alpha",
            """
- Apply the same ± to every new forecast
- The guarantee holds **if** calibration and future cases are exchangeable (come from the same kind of process)
- Coverage is on average over many forecasts, not for each individual player
""",
        ),
    ],
    code=CODE,
    code_notes="Change the 0.80 to 0.95. The interval widens a lot: certainty is expensive when single games are this noisy.",
    walkthrough=[
        ("def forecasts(d):", "For each game from the 3rd on, record the actual grade, the season-to-date average and SD."),
        ("calib = forecasts(df[df['season'] == 2023])", "The calibration season: earlier, and not used for anything else."),
        ("scores = np.abs(calib[:, 0] - calib[:, 1])", "How wrong the forecast was each time."),
        ("k = int(np.ceil((n + 1) * 0.80)) - 1", "The position just above 80% in the sorted errors (the '+1' correction; −1 because Python counts from 0)."),
        ("half_width = np.sort(scores)[k]", "That error size becomes the ± for every future forecast."),
        ("covered = np.abs(test[:, 0] - test[:, 1]) <= half_width", "Check: did the later seasons' outcomes fall inside the range?"),
        ("1.2816 * test[:, 2]", "The naive alternative: ±1.28 × the player's own SD so far (the 80% bell-curve rule). It's far too narrow."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Any PSFAL-style forecast** could carry a conformal range calibrated on the previous season, with coverage re-checked on the next. That includes a Model 8-type development forecast.
- **Your rule extended:** a forecast range only goes on your site if its coverage holds on a test season fixed in advance, as the calibration gates in this academy require.
""",
    quiz=[
        Question(
            "q1", "mc", "Conformal prediction builds intervals from:",
            "The model's own errors on a separate calibration set.",
            options=["The model's errors on a separate calibration set", "The training data's average",
                     "A bell-curve assumption", "Expert opinion"],
            notes=["Correct.", "Not the average itself.", "It avoids that assumption.", "No opinions involved."],
            answer=0),
        Question(
            "q2", "numeric", "n = 99 calibration errors, target 80%. Which sorted position k = ⌈(n+1) × 0.8⌉ gives the half-width?",
            "⌈100 × 0.8⌉ = 80, so the 80th smallest error.", category="Application", answer=80, tolerance=0),
        Question(
            "q3", "mc", "A prediction interval for next game's grade is wider than a confidence interval for his season mean because:",
            "It must include game-to-game luck, not just uncertainty about the average.",
            options=["It includes game-to-game luck, not just uncertainty about the average",
                     "It uses less data", "It's computed wrongly", "Confidence intervals ignore data"],
            notes=["Correct.", "Not the reason.", "Both can be right; they answer different questions.", "They don't."],
            answer=0),
        Question(
            "q4", "short", "Conformal 80% intervals cover only 70% on a new season after a rule change. Why, and what would you do?",
            "Exchangeability broke (the future differs from calibration); re-calibrate on recent data and re-check coverage.",
            category="Application",
            keywords=[["exchangeab", "differ", "changed", "shift", "rule", "not like", "new era", "different"],
                      ["recalibrat", "re-calibrat", "recent", "update", "re-check", "new calibration", "calibrate again"]],
            model_answer="The new season no longer resembles the calibration season (exchangeability broke). Re-calibrate on recent post-change data and check coverage again before trusting the ranges."),
        Question(
            "q5", "mc", "Which statement about the conformal guarantee is TRUE?",
            "It holds on average across forecasts, if future cases resemble the calibration cases.",
            options=["It holds on average across forecasts, if the future resembles the calibration data",
                     "It guarantees 80% for every single player", "It needs a normal distribution",
                     "It only works with neural networks"],
            notes=["Correct.", "It's an average across forecasts.", "It needs no distribution shape.", "It works with any forecaster."],
            answer=0),
    ],
)
