import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import theme as T
from .. import validation as V
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


def _sim(seed=8, n=2000):
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1, n)
    p_true = 1 / (1 + np.exp(-(-0.9 + 1.4 * skill)))
    y = rng.binomial(1, p_true)
    logit = -0.9 + 1.4 * (skill + rng.normal(0, 0.5, n))
    return logit, y


def visual():
    logit, y = _sim()
    c1, c2 = st.columns([1, 2])
    with c1:
        temp = st.slider("Confidence knob (temperature)", 0.3, 3.0, 1.0, 0.1, key="l31_temp",
                         help="Below 1 = overconfident (pushes forecasts toward 0% and 100%). Above 1 = underconfident.")
    p = 1 / (1 + np.exp(-logit / temp))
    t = V.reliability_table(p, y, bins=10)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color=T.MUTED, ls="--", lw=1.2, label="Perfect calibration")
    ax.plot(t["predicted"], t["observed"], marker="o", color=T.BLUE, lw=2.2, label="This model")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast: P(top-third season)"), ax.set_ylabel("How often it actually happened")
    ax.legend(loc="upper left")
    with c2:
        show_fig(fig)
    m = st.columns(4)
    m[0].metric("Brier score", f"{V.brier(p, y):.3f}", help="Lower is better")
    m[1].metric("Brier skill vs 'always say base rate'", f"{V.brier_skill(p, y, y.mean()):+.1%}")
    m[2].metric("Calibration error (ECE)", f"{V.ece(p, y, 10):.3f}")
    m[3].metric("AUC (discrimination)", f"{V.auc(p, y):.3f}", help="Never changes with the knob: ranking is identical")


CODE = '''import numpy as np
from sklearn.metrics import roc_auc_score

# --- SIMULATED: 2,000 player-seasons. Event = "top-third season next year" (1 = yes) ---
rng = np.random.default_rng(8)
skill = rng.normal(0, 1, 2000)                          # hidden true quality
p_true = 1 / (1 + np.exp(-(-0.9 + 1.4 * skill)))        # true chance of a top-third season
y = rng.binomial(1, p_true)                              # what actually happened
logit = -0.9 + 1.4 * (skill + rng.normal(0, 0.5, 2000)) # a model's noisy read of quality

def brier(p, y):
    return np.mean((p - y) ** 2)

def reliability(p, y, bins=5):
    order = np.argsort(p)
    groups = np.array_split(order, bins)                 # equal-size bins, low to high forecasts
    return [(p[g].mean(), y[g].mean()) for g in groups]

base = brier(np.full(len(y), y.mean()), y)               # reference: always forecast the base rate
print(f"Base rate {y.mean():.3f}, reference Brier {base:.3f}\\n")

for name, temp in [("Overconfident", 0.5), ("Calibrated", 1.0), ("Underconfident", 2.0)]:
    p = 1 / (1 + np.exp(-logit / temp))                  # same ranking, different confidence
    print(f"{name:<15} Brier {brier(p, y):.3f}  skill {1 - brier(p, y) / base:+.1%}  "
          f"AUC {roc_auc_score(y, p):.3f}")

p = 1 / (1 + np.exp(-logit / 0.5))
print("\\nOverconfident model, reliability table (forecast -> what happened):")
for pred, obs in reliability(p, y):
    print(f"  said {pred:.2f}  ->  happened {obs:.2f}")
'''

LESSON = Lesson(
    id="l3-01-calibration-brier", level=3, order=1,
    title="Calibration and the Brier Score: Do Your 80% Forecasts Happen 80% of the Time?",
    subtitle="Judging probability forecasts, and why ranking well is not enough",
    minutes=20,
    concept_ids=["calibration-brier"],
    simple="""
A model that gives **probabilities** makes promises. If it says "80% chance of a top-third season" for 50 different players, about **40 of them** should have one.

- **Calibration:** do the forecasts mean what they say? Yes means 80% happens about 80% of the time.
- **Brier score:** one number that grades probability forecasts. It's the average squared gap between the forecast and what happened (1 = yes, 0 = no). **Lower is better.**
- **Discrimination (AUC):** a different question: does the model *rank* the yes-cases above the no-cases?

A model can rank players perfectly and still be badly calibrated, or be calibrated but useless at ranking. You need to check both.
""",
    problem="A probability is only useful if you can take it at face value. Staff will treat '85% chance he's a starter' very differently from '55%'. If those numbers are inflated, decisions built on them go wrong, even when the player ranking is right.",
    why="Calibration tells you whether forecasts can be trusted as probabilities. The Brier score rewards both calibration and sharpness in one number, and comparing it with a simple reference forecast tells you whether the model adds anything.",
    when_useful="""
- Any model that outputs a probability: transfer risk, contribution by year 2, top-third season
- Deciding whether a probability can be shown to coaches as-is
- Comparing models that both rank well (same AUC) but differ in honesty
""",
    when_not="""
- You only need a ranking (a draft board order). AUC answers that directly
- Very small test sets: reliability bins of 10–20 players are mostly noise
- Very rare events: the Brier score is dominated by the many easy 'no' cases. Look at the reliability diagram and a skill score instead
""",
    compare="""
**Calibration vs discrimination:** discrimination (AUC) asks *who is more likely*. Calibration asks *how likely, exactly*. Rescaling every forecast toward 0% and 100% leaves the ranking, and so the AUC, untouched, while the calibration gets worse. That's what the knob in the visual does.

**Brier score vs log loss:** both are *proper scores*, so the best strategy is always to report your true belief. Log loss punishes a confident miss (saying 99% when the answer is no) far more harshly. Brier is easier to read and more forgiving.

**Brier score vs accuracy:** accuracy turns probabilities into yes/no at 50% and throws away the confidence. Brier keeps it.
""",
    sports="""
**Fictional scenario (simulated).** A model gives 2,000 player-seasons a probability of a **top-third season next year**. About a third actually have one. Compare three versions of the same model:

| Version | What it does | AUC | Brier |
|---|---|---|---|
| Overconfident | pushes forecasts toward 0% and 100% | same | worse |
| Calibrated | forecasts mean what they say | same | best |
| Underconfident | squeezes forecasts toward 50% | same | worse |

All three **rank** players identically, so AUC can't tell them apart. Only calibration and the Brier score can.

This is your Model 9 lesson in another form. "Same as last season" got AUC 0.713 for a top-third season, and richer inputs didn't beat it. Before believing a forecast's *probabilities*, check calibration too. The Level 1 project now does this automatically.
""",
    takeaway="AUC asks 'who is more likely'. Calibration asks 'how likely, exactly'. A forecast you'll act on needs both.",
    visual=visual,
    visual_caption="Turn the knob below 1 and the curve bends away from the diagonal: the model claims more certainty than it has. The AUC never moves, because the ranking never changes.",
    math=[
        MathBlock(
            "Step 1: the Brier score",
            r"\text{BS} = \frac{1}{N}\sum_{i=1}^{N} (p_i - y_i)^2",
            """
- $N$: number of forecasts (players)
- $p_i$: forecast probability for player $i$, e.g. 0.8
- $y_i$: what happened: 1 if he had a top-third season, 0 if not
- $(p_i - y_i)^2$: the squared miss. Saying 0.8 when it happened costs 0.04, and when it didn't costs 0.64
""",
            "0 is perfect. Always saying 50% scores exactly 0.25.",
        ),
        MathBlock(
            "Step 2: compare with a simple reference (skill score)",
            r"\text{BSS} = 1 - \frac{\text{BS}_{\text{model}}}{\text{BS}_{\text{reference}}}",
            """
- $\\text{BS}_{\\text{reference}}$: the Brier score of a no-skill forecast, usually 'always predict the base rate'
- $\\text{BSS} > 0$: the model beats the reference. $\\text{BSS} = 0$: no better. Below 0: worse than just quoting the base rate
""",
        ),
        MathBlock(
            "Step 3: measure calibration directly",
            r"\text{ECE} = \sum_{k=1}^{K} \frac{n_k}{N}\,\bigl|\,\bar p_k - \bar y_k\,\bigr|",
            """
- Sort forecasts into $K$ bins (e.g., 5 groups of equal size)
- $n_k$: players in bin $k$
- $\\bar p_k$: average forecast in the bin. $\\bar y_k$: share who actually had the event
- $|\\bar p_k - \\bar y_k|$: how far that bin's promise is from reality
- ECE (expected calibration error): the average gap, weighted by bin size
""",
            "Plotting each (p̄ₖ, ȳₖ) point gives the reliability diagram. A perfectly calibrated model sits on the diagonal.",
        ),
    ],
    code=CODE,
    code_notes="Change the 0.5 in the last block to 2.0 and rerun. The underconfident model never says anything bold: its forecasts bunch near the base rate.",
    walkthrough=[
        ("skill = rng.normal(0, 1, 2000)", "Each simulated player-season gets a hidden quality number."),
        ("p_true = 1 / (1 + np.exp(-(...)))", "Turn quality into a true probability of a top-third season (the logistic curve)."),
        ("y = rng.binomial(1, p_true)", "Flip a weighted coin for each player: 1 = top-third season, 0 = not. That's the outcome."),
        ("logit = ... + rng.normal(0, 0.5, 2000)", "The model sees quality with some error, like any real model."),
        ("def brier(p, y): return np.mean((p - y) ** 2)", "The Brier score in one line: the average squared gap between forecast and outcome."),
        ("groups = np.array_split(order, bins)", "Sort forecasts low to high and cut them into equal-size bins for the reliability table."),
        ("base = brier(np.full(len(y), y.mean()), y)", "The reference: a forecaster who always quotes the base rate. Any useful model must beat it."),
        ("p = 1 / (1 + np.exp(-logit / temp))", "Same model, different confidence. Dividing by 0.5 doubles every log-odds (overconfident), and dividing by 2 halves it."),
        ("roc_auc_score(y, p)", "AUC: identical for all three versions, because the order of the players never changes."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.** This is how the ideas carry over:

- **Model 8** is judged on AUC (+0.087 over baseline). AUC proves it *ranks* well. Before any of its probabilities are shown as "X% chance", check its reliability diagram and Brier skill on the same pre-set test.
- **Models 9 and 9b:** "same as last season" scored AUC 0.713 for a top-third season. A new candidate should report Brier skill against that baseline's calibrated probabilities, not only AUC.
- **Level 1 project:** its Production Candidate check now includes a calibration gate: P(top-third next season) must have calibration error ≤ 0.05 and positive Brier skill against the base rate.
""",
    quiz=[
        Question(
            "q1", "mc", "A model says '80% chance of a top-third season' for 100 players. It is well calibrated if:",
            "Calibration means forecasts match observed frequencies: about 80 of the 100 should have the event.",
            options=["About 80 of them have a top-third season", "All 100 of them do",
                     "It ranks those 100 above everyone else", "Its AUC is above 0.80"],
            notes=["Correct.", "That would mean 80% was an understatement.",
                   "That's discrimination, not calibration.", "AUC measures ranking, not whether 80% means 80%."],
            answer=0),
        Question(
            "q2", "numeric", "Three forecasts: 0.9 (happened), 0.2 (did not happen), 0.6 (did not happen). What is the Brier score?",
            "((0.9−1)² + (0.2−0)² + (0.6−0)²) / 3 = (0.01 + 0.04 + 0.36) / 3 ≈ 0.137.",
            category="Application", answer=0.137, tolerance=0.003),
        Question(
            "q3", "mc", "You push every forecast toward 0% or 100% without changing the order of players. What happens?",
            "A monotone rescaling keeps the ranking, so AUC is unchanged, while calibration and usually the Brier score get worse.",
            options=["AUC stays the same; calibration gets worse", "AUC improves; calibration stays the same",
                     "Both get better", "Both get worse"],
            notes=["Correct.", "AUC only depends on the order of the players.",
                   "Overconfidence doesn't add information.", "AUC can't change, because the ranking is identical."],
            answer=0),
        Question(
            "q4", "short", "Your model's Brier skill score against 'always predict the base rate' is −3%. What does that mean, and what would you do?",
            "Negative skill means it is worse than quoting the base rate, so it shouldn't be used as a probability forecast.",
            category="Application",
            keywords=[["worse", "no better", "negative", "base rate", "reference", "nothing"],
                      ["recalibrat", "don't use", "not use", "reject", "fix", "simpler", "baseline", "exploratory", "drop"]],
            model_answer="It is worse than just quoting the base rate, so its probabilities add nothing. Don't use them. Recalibrate it or fall back to the simpler baseline, and keep it labelled exploratory."),
        Question(
            "q5", "mc", "Model A: AUC 0.75, badly overconfident. Model B: AUC 0.74, well calibrated. You'll show coaches 'chance he contributes by year 2'. Which?",
            "For probabilities that people will read at face value, calibration matters. A tiny AUC edge doesn't make up for misleading numbers. (You could also recalibrate A and re-test it.)",
            category="Application",
            options=["Model B, or recalibrate A and re-test it", "Model A, because its AUC is higher",
                     "Either; AUC is all that matters", "Neither can ever be used"],
            notes=["Correct.", "0.01 of AUC doesn't justify inflated probabilities.",
                   "AUC ignores whether the stated chances are honest.", "B is usable as it stands."],
            answer=0),
    ],
)
