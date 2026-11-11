import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner="Scoring the challenger against each baseline…")
def _gains(seed):
    from ..data import make_draft_classes
    from ..smalldata import evaluate, fast_auc
    r = evaluate(make_draft_classes(seed=seed), boot=0, baseline="plain")["rows"]
    out = {"Constant rate": 0.5, "Last class's rate": 0.5,
           "Plain logistic (overfit)": fast_auc(r["plain"], r["y"]), "Rating only": fast_auc(r["rating_only"], r["y"])}
    return fast_auc(r["bayes"], r["y"]), out


def visual():
    seed = st.select_slider("Simulated program", list(range(1, 9)) + [29], value=29, key="l311_seed")
    pick = st.radio("Judge the challenger against", ["Constant rate", "Last class's rate", "Plain logistic (overfit)",
                                                      "Rating only"], horizontal=True, key="l311_base")
    ch, base = _gains(seed)
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    names = list(base)
    gains = [ch - base[n] for n in names]
    ax.barh(names, gains, color=[T.PRIMARY if n == pick else T.FAINT for n in names])
    ax.axvline(0.02, color=T.ACCENT, ls="--", lw=1.2)
    ax.axvline(0, color=T.BORDER, lw=1)
    ax.invert_yaxis(), ax.set_xlabel("AUC gain of the same challenger (dashed: a +0.02 bar)")
    show_fig(fig)
    c = st.columns(3)
    c[0].metric("Challenger AUC", f"{ch:.3f}")
    c[1].metric(f"{pick} AUC", f"{base[pick]:.3f}")
    c[2].metric("Gain", f"{ch - base[pick]:+.3f}")


CODE = '''import numpy as np
from academy.data import make_draft_classes
from academy.smalldata import evaluate, fast_auc

def gains(seed):
    """One SIMULATED program: the same challenger (Bayesian logistic) judged against four baselines."""
    res = evaluate(make_draft_classes(seed=seed), boot=0, baseline="plain")
    r = res["rows"]
    r["constant"] = 0.10                                   # 'everyone has a 10% chance'
    base_auc = {"Constant rate": 0.5,                      # a constant can't rank anyone: AUC is 0.5
                "Plain logistic (overfit)": fast_auc(r["plain"], r["y"]),
                "Rating only": fast_auc(r["rating_only"], r["y"])}
    challenger = fast_auc(r["bayes"], r["y"])
    brier = {"Constant rate": np.mean((r["constant"] - r["y"]) ** 2),
             "Plain logistic (overfit)": np.mean((r["plain"] - r["y"]) ** 2),
             "Rating only": np.mean((r["rating_only"] - r["y"]) ** 2)}
    b_ch = np.mean((r["bayes"] - r["y"]) ** 2)
    return {k: (challenger - v, 1 - b_ch / brier[k]) for k, v in base_auc.items()}, challenger

g, auc = gains(29)
print(f"Challenger (Bayesian logistic) test AUC: {auc:.3f}")
for name, (dauc, dbrier) in g.items():
    print(f"  vs {name:<26} AUC gain {dauc:+.3f}   Brier improvement {dbrier:+.1%}")

# Is that pattern luck? Repeat on 8 more simulated programs.
many = [gains(s)[0] for s in range(1, 9)]
print("Average AUC gain over 8 programs:")
for name in g:
    vals = [m[name][0] for m in many]
    print(f"  vs {name:<26} {np.mean(vals):+.3f}  (range {min(vals):+.3f} to {max(vals):+.3f})")
'''

LESSON = Lesson(
    id="l3-11-strong-baselines", level=3, order=11,
    title="Weak Baselines Make Fake Wins",
    subtitle="Judge a model against the best simple alternative, not the easiest one",
    minutes=15,
    concept_ids=["strong-baselines"],
    simple="""
"Our model beats the baseline" sounds convincing. It means very little until you know **which** baseline.

Beating "everyone has a 10% chance" is easy: that baseline can't tell players apart at all. Beating a clumsy model that overfits is easy too. Beating the **best simple thing** someone could do in five minutes (sort by recruiting rating, say) is the real test.

**The rule:** before comparing, list the simple baselines a sensible person would actually use, and judge your model against the **strongest** one.
""",
    problem="A model can show a large, real-looking gain that comes entirely from a poor comparison. The model is no better than what you already had.",
    why="The question a staff cares about is 'does this beat what we'd do anyway?' Only the strongest simple baseline answers it.",
    when_useful="""
- Every model comparison, every time
- Especially rare outcomes (draft picks), where a constant rate looks respectable on Brier but can't rank at all
- When a fancy model's win seems surprisingly large
""",
    when_not="""
- Never skip it. The only judgment call is which simple baselines are fair: they must use only information available at prediction time
- Don't pick the strongest baseline by peeking at the test set and then tuning it; define the candidates first
""",
    compare="""
**Constant rate / last class's rate:** the floor; anything that ranks players beats it. **A complex model with no penalty:** can be weaker than one input on its own, because it overfits. **Rating only, same as last season:** the strong simple baselines; one sensible input, nothing fitted beyond that. **Your challenger:** must clear the strongest one by a bar you wrote down first (L6.3).
""",
    sports="""
**A common mistake, simulated.** One fictional program, 22 recruits a class. The challenger is a Bayesian logistic model (L1.8), test AUC **0.844**. Same model, same test, four baselines:

| Baseline | AUC gain | Brier improvement |
|---|---|---|
| Constant rate (10%) | +0.344 | +32.0% |
| Plain logistic (overfit) | +0.066 | +14.2% |
| **Rating only** | **−0.018** | **+0.4%** |

(A "last class's rate" is also a constant for each class, so it ranks no one either: AUC 0.5.)

Against a constant it looks like a breakthrough. Against rating alone it loses on ranking. Nothing about the model changed.

**Is this luck?** Over 8 more simulated programs the average gain was +0.312 vs a constant, +0.105 vs the overfit plain model and **+0.040** vs rating only (range −0.017 to +0.098). The challenger has a real edge on average, but it's about a third of what the overfit baseline suggested, and sometimes it's gone.
""",
    takeaway="A win means nothing until you name the baseline. Always compare with the strongest simple one.",
    visual=visual,
    visual_caption="The challenger never changes; only the yardstick does. Switch programs: the gain over a constant is always huge, the gain over rating only is small and sometimes negative.",
    math=[
        MathBlock(
            "Step 1: gain depends on the baseline",
            r"\Delta = \text{AUC}(\text{model}) - \text{AUC}(\text{baseline})",
            """
- $\\Delta$ (delta): the reported gain
- The model's AUC is fixed; changing the baseline changes $\\Delta$ by exactly the baseline's AUC difference
""",
        ),
        MathBlock(
            "Step 2: why a constant gets AUC 0.5",
            r"\text{AUC} = P\big(s_i > s_j \mid y_i = 1,\ y_j = 0\big) + \tfrac12 P\big(s_i = s_j \mid y_i = 1,\ y_j = 0\big)",
            """
- $s_i$: the score given to player $i$; $y_i = 1$ if he was drafted
- AUC: the chance a random drafted player is scored above a random undrafted one, counting ties as half
- A constant gives everyone the same score, so every pair is a tie: AUC $= \\tfrac12$
""",
        ),
        MathBlock(
            "Step 3: judge against the strongest",
            r"\Delta^\star = \text{AUC}(\text{model}) - \max_{b \in \mathcal{B}} \text{AUC}(b)",
            """
- $\\mathcal{B}$: the list of simple baselines, written down before testing (constant, rating only, ...)
- $\\max$: take the best of them. $\\Delta^\\star$ is the only gain worth reporting
""",
            "Picking the max on the test set can only make the bar harder, never easier. That's why it's a fair way to choose.",
        ),
    ],
    code=CODE,
    code_notes="Add a fifth baseline: year-1 percentile alone. Fit it like rating only (one input) and see whether it's even stronger.",
    walkthrough=[
        ("res = evaluate(make_draft_classes(seed=seed), boot=0, baseline='plain')", "Run the class-by-class test from the P3C project on one simulated program; it returns every model's test predictions."),
        ("r['constant'] = 0.10", "The weakest baseline: the same 10% for everyone."),
        ("'Constant rate': 0.5", "A constant ranks nobody, so its AUC is 0.5 by definition."),
        ("fast_auc(r['rating_only'], r['y'])", "The strong simple baseline: a logistic on recruiting rating alone."),
        ("challenger - v", "The same challenger's gain against each baseline."),
        ("1 - b_ch / brier[k]", "Brier improvement (concept 25): how much smaller the challenger's squared error is."),
        ("many = [gains(s)[0] for s in range(1, 9)]", "Repeat on 8 more programs so one lucky test doesn't decide the lesson."),
    ],
    psfal="""
**The academy is separate from PSFAL; this lesson uses simulated data only.**

The general habit: for any model you build anywhere, write the list of simple baselines first (a constant rate, the one obvious input, "same as last time"), and report the gain over the strongest. The P3C project now does this automatically: its Production Candidate gates use the stronger of plain logistic and rating only.
""",
    quiz=[
        Question(
            "q1", "mc", "A model beats 'everyone has a 10% chance' by +0.30 AUC. What have you learned?",
            "Very little: a constant can't rank anyone (AUC 0.5), so almost any sensible model beats it.",
            concept="strong-baselines",
            options=["Very little; that baseline can't rank anyone", "The model is excellent", "The model is calibrated", "The model will beat rating only"],
            notes=["Correct.", "It may be, but this comparison can't show it.", "AUC says nothing about calibration.", "Not implied at all."],
            answer=0),
        Question(
            "q2", "numeric", "Every player gets the same predicted probability. What is the AUC?",
            "Every drafted/undrafted pair is a tie, counted as half: AUC = 0.5.",
            concept="strong-baselines", category="Application", answer=0.5, tolerance=0.001),
        Question(
            "q3", "mc", "Why can a plain logistic regression be a WEAK baseline with small data?",
            "With many inputs and few events it overfits, so it can rank worse than one sensible input alone.",
            concept="strong-baselines",
            options=["It overfits, so it can be worse than one good input", "Logistic regression is always weak",
                     "It can't output probabilities", "It uses future data"],
            notes=["Correct: here 0.778 vs 0.862 for rating only.", "With enough data it's a fine baseline.", "It outputs probabilities.", "Not inherently."],
            answer=0),
        Question(
            "q4", "numeric", "Your model's AUC is 0.80. Baselines: constant 0.50, rating only 0.78, plain logistic 0.71. What gain should you report?",
            "Against the strongest baseline: 0.80 − 0.78 = 0.02.",
            concept="strong-baselines", category="Application", answer=0.02, tolerance=0.001),
        Question(
            "q5", "short", "How should you choose the baselines for a comparison?",
            "List the sensible simple alternatives (using only information available at prediction time) before testing, then judge against the strongest.",
            concept="strong-baselines", category="Application",
            keywords=[["before", "in advance", "first", "list"], ["strong", "best", "simple", "sensible"]],
            model_answer="Write down the simple alternatives a sensible person would use (constant rate, the one obvious input, same as last season) before testing, using only information available at prediction time, then report the gain over the strongest of them."),
    ],
)
