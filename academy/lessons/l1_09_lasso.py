import warnings

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner=False)
def _paths():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from ..data import make_draft_classes
    from ..smalldata import Design
    warnings.filterwarnings("ignore")
    df = make_draft_classes()
    train = df[df["class_year"] <= 2013]
    test = df[(df["class_year"] >= 2014) & (df["class_year"] <= 2021)]
    d = Design(train)
    X, Xt = d.linear(train)[:, 1:], d.linear(test)[:, 1:]
    Cs = np.logspace(-1.4, 1, 21)
    out = {}
    for pen in ("l1", "l2"):
        coefs, aucs = [], []
        for C in Cs:
            m = LogisticRegression(penalty=pen, C=C, solver="saga", max_iter=20000, tol=1e-6).fit(X, train["drafted"])
            coefs.append(m.coef_[0])
            aucs.append(roc_auc_score(test["drafted"], m.predict_proba(Xt)[:, 1]))
        out[pen] = (np.array(coefs), np.array(aucs))
    return Cs, out, ["Rating", "Year-1"] + list(d.positions)


def visual():
    Cs, out, names = _paths()
    i = st.select_slider("Penalty strength (left = strong, right = weak)", options=list(range(len(Cs))), value=9,
                         format_func=lambda k: f"C = {Cs[k]:.2g}", key="l19_c")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, pen, title in zip(axes, ("l2", "l1"), ("Ridge (L2): shrinks", "Lasso (L1): shrinks and drops")):
        coefs, _ = out[pen]
        for j in range(coefs.shape[1]):
            ax.plot(Cs, coefs[:, j], color=T.PRIMARY if j == 0 else T.ACCENT if j == 1 else T.FAINT,
                    lw=2.2 if j < 2 else 1)
        ax.axvline(Cs[i], color=T.MUTED, ls="--", lw=1)
        ax.axhline(0, color=T.BORDER, lw=1)
        ax.set_xscale("log"), ax.set_xlabel("C"), ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Slope (log-odds per SD)")
    show_fig(fig)
    c = st.columns(4)
    for col, pen, label in ((c[0], "l1", "Lasso"), (c[2], "l2", "Ridge")):
        coefs, aucs = out[pen]
        col.metric(f"{label}: inputs kept", f"{int(np.sum(np.abs(coefs[i]) > 1e-9))} of {coefs.shape[1]}")
        (c[1] if pen == "l1" else c[3]).metric(f"{label}: test AUC", f"{aucs[i]:.3f}")


CODE = '''import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from academy.data import make_draft_classes
from academy.smalldata import Design

warnings.filterwarnings("ignore")                          # saga prints convergence chatter we don't need

def fit(df, penalty, C):
    train = df[df["class_year"] <= 2013]                   # 132 finished recruits (SIMULATED program)
    d = Design(train)
    X = d.linear(train)[:, 1:]                             # standardized rating, year-1, 7 position dummies
    m = LogisticRegression(penalty=penalty, C=C, solver="saga", max_iter=20000, tol=1e-6).fit(X, train["drafted"])
    return m, d

df = make_draft_classes()
test = df[(df["class_year"] >= 2014) & (df["class_year"] <= 2021)]
names = ["rating", "year1"] + fit(df, "l1", 1)[1].positions

print("Penalty   C     inputs kept   test AUC   test Brier")
for C in [0.05, 0.1, 0.2, 0.5, 1, 10]:
    for pen in ["l1", "l2"]:
        m, d = fit(df, pen, C)
        p = m.predict_proba(d.linear(test)[:, 1:])[:, 1]
        kept = int(np.sum(np.abs(m.coef_[0]) > 1e-9))
        print(f"{pen.upper()}     {C:>5}   {kept:>2} of 9      {roc_auc_score(test['drafted'], p):.3f}      "
              f"{np.mean((p - test['drafted']) ** 2):.4f}")

m, _ = fit(df, "l1", 0.5)
print("Lasso (C = 0.5) kept:", {n: round(float(c), 2) for n, c in zip(names, m.coef_[0]) if abs(c) > 1e-9})

# Is the selection stable? Refit on 8 other simulated programs.
kept_rating, kept_n = 0, []
for seed in range(1, 9):
    other = make_draft_classes(seed=seed)
    mm, _ = fit(other, "l1", 0.5)
    kept_rating += abs(mm.coef_[0][0]) > 1e-9
    kept_n.append(int(np.sum(np.abs(mm.coef_[0]) > 1e-9)))
print(f"Over 8 programs the rating slope survived {kept_rating} times; inputs kept: {kept_n}")
'''

LESSON = Lesson(
    id="l1-09-lasso", level=1, order=9,
    title="Lasso (L1): A Penalty That Drops Inputs",
    subtitle="Ridge shrinks every slope; Lasso shrinks and sets weak ones to exactly zero",
    minutes=20,
    concept_ids=["lasso"],
    simple="""
Ridge (L1.7) pulls every slope toward zero but never reaches it. Every input stays in the model, just quieter.

**Lasso** (also **L1**) uses a different penalty: the sum of the **absolute** slopes, not the squares. That small change has a big effect. As the penalty grows, weak slopes hit **exactly zero** and leave the model. Lasso fits and **selects inputs** in one step.

That's useful with small data and many candidate inputs (position dummies, testing numbers, measurements): you get a short list of what the data actually supports.
""",
    problem="With few rows and many inputs, most inputs are noise. You want a model that keeps the few that matter and can say which they are.",
    why="A shorter model is easier to explain and often generalizes better. And the list of survivors is itself a result: it tells you which inputs the data can support.",
    when_useful="""
- Many candidate inputs, few rows, and you suspect most do little
- When you need a short, explainable list of inputs
- As a screen before a more careful model
""",
    when_not="""
- When inputs move together (rating and year-1 production, say): Lasso tends to keep one and drop the other, somewhat arbitrarily, and the choice can flip between samples
- When you believe every input matters a little: ridge is the better prior
- Don't read 'dropped' as 'has no effect'. It means the data couldn't support it
""",
    compare="""
**Ridge (L1.7):** shrinks smoothly, keeps everything, stable with correlated inputs. **Lasso:** shrinks and drops; a Laplace prior (a sharp peak at zero with heavy tails). **Elastic net:** a mix of both. **Bayesian logistic (L1.8):** priors you choose, with uncertainty on every slope; Lasso only gives a single best guess.
""",
    sports="""
**Fictional scenario (simulated).** Same program as L1.7: 132 finished recruits, 12 drafted, 9 inputs (rating, year-1 percentile, 7 position dummies). Test on classes 2014–2021.

| C | Lasso inputs kept | Lasso AUC | Ridge AUC | Lasso Brier | Ridge Brier |
|---|---|---|---|---|---|
| 0.05 (very strong) | 0 of 9 | 0.500 | 0.865 | 0.0964 | 0.0836 |
| 0.2 | 2 of 9 | 0.876 | 0.864 | 0.0789 | 0.0733 |
| 0.5 | 2 of 9 | 0.875 | 0.858 | 0.0693 | 0.0694 |
| 1 | 3 of 9 | 0.872 | 0.851 | 0.0674 | 0.0684 |
| 10 (almost none) | 8 of 9 | 0.833 | 0.832 | 0.0708 | 0.0705 |

- At C = 0.5 Lasso kept just **rating (0.96)** and **year-1 production (0.31)** and zeroed all seven position dummies. That is the right answer here, since the positions matter little.
- Too strong a penalty zeroes **everything** (C = 0.05): every player gets the same probability, AUC 0.500.
- Lasso ranks slightly better than ridge here (0.875 vs 0.858) with similar probabilities (Brier 0.0693 vs 0.0694).
- **Stability:** over 8 other simulated programs the rating slope survived all 8 times, and Lasso kept between 1 and 3 inputs each time.
""",
    takeaway="Lasso is ridge's sharper cousin: same idea (a penalty is a prior), but it zeroes out inputs the data can't support. Pick the strength on training data, and treat the survivors as a hypothesis, not a proof.",
    visual=visual,
    visual_caption="Left: ridge slopes shrink toward zero but stay. Right: Lasso slopes hit zero and stay there; blue is rating, amber is year-1 production, grey are the 7 positions. Move the slider and compare how many inputs each keeps.",
    math=[
        MathBlock(
            "Step 1: the Lasso objective",
            r"\hat\beta_{\text{lasso}} = \arg\max_\beta \; \sum_i \log P(y_i \mid \beta) \; - \; \lambda \sum_{j \ge 1} |\beta_j|",
            """
- The first term is the log-likelihood (how well the slopes explain who was drafted)
- $|\\beta_j|$: the **absolute** value of slope $j$ (ridge uses $\\beta_j^2$)
- $\\lambda$: penalty strength; scikit-learn's $C = 1/\\lambda$, so small C = strong penalty
- The intercept is not penalized
""",
        ),
        MathBlock(
            "Step 2: why slopes hit exactly zero",
            r"\hat\beta_j = \text{sign}(b_j)\,\max\!\big(|b_j| - \lambda,\; 0\big)",
            """
- $b_j$: what the slope would be with no penalty (in a simple, uncorrelated setup)
- Lasso subtracts $\\lambda$ from its size. If $|b_j| \\le \\lambda$, the result is **0**: the input is out
- Ridge instead divides: $\\hat\\beta_j = b_j/(1+\\lambda)$, which never reaches 0
""",
            "This soft-threshold formula is exact for uncorrelated inputs and a good picture for the rest.",
        ),
        MathBlock(
            "Step 3: the penalty is a prior",
            r"\text{Laplace}(\beta_j \mid 0, b) \propto e^{-|\beta_j|/b}, \qquad \lambda = 1/b",
            """
- Ridge = a Normal prior (smooth bell, no sharp peak). Lasso = a **Laplace** prior (a sharp peak at 0 and heavier tails)
- The sharp peak puts a lot of belief on 'exactly zero'; the heavy tails let a clearly real slope stay large
- Maximizing the log posterior gives the Lasso fit (the MAP estimate)
""",
        ),
    ],
    code=CODE,
    code_notes="Try penalty='elasticnet' with l1_ratio=0.5 (needs solver='saga'). It sits between ridge and Lasso: some slopes drop, the rest shrink smoothly.",
    walkthrough=[
        ("warnings.filterwarnings('ignore')", "saga (the solver that supports L1 without penalizing the intercept) prints convergence chatter; hide it."),
        ("train = df[df['class_year'] <= 2013]", "Only finished classes, as in L1.7."),
        ("d.linear(train)[:, 1:]", "Standardized rating and year-1, plus position dummies. Standardizing matters: the penalty must treat inputs fairly."),
        ("LogisticRegression(penalty=penalty, C=C, solver='saga', ...)", "The same model with an L1 or L2 penalty. saga leaves the intercept unpenalized."),
        ("kept = int(np.sum(np.abs(m.coef_[0]) > 1e-9))", "Count slopes that are not exactly zero: the inputs the model kept."),
        ("print(f'{pen.upper()} ...')", "Compare Lasso and ridge on the same test set at each strength."),
        ("for seed in range(1, 9):", "Selection can be unstable, so refit on 8 other simulated programs and see whether the same inputs survive."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **A screen, not a verdict.** With many candidate inputs and a small sample, Lasso gives a short list to examine, but a dropped input isn't proven useless. Check what survives against your pre-set test, as with any model.
- **Correlated inputs:** early production measures that move together may swap in and out between samples. If a list of survivors changes across seasons, trust the pattern, not the exact list.
""",
    quiz=[
        Question(
            "q1", "mc", "What is the main difference between Lasso (L1) and ridge (L2)?",
            "Lasso's absolute-value penalty can set slopes to exactly zero; ridge only shrinks them.",
            concept="lasso",
            options=["Lasso can drop inputs by setting slopes to exactly zero", "Ridge drops inputs, Lasso keeps them",
                     "Lasso needs more data", "They are identical"],
            notes=["Correct.", "Reversed.", "Neither needs more data by design.", "They differ in the penalty shape."],
            answer=0),
        Question(
            "q2", "numeric", "An unpenalized slope would be 0.80. With a Lasso threshold of λ = 0.30, what slope remains? (2 decimals)",
            "sign × max(|b| − λ, 0) = 0.80 − 0.30 = 0.50.",
            concept="lasso", category="Application", answer=0.5, tolerance=0.01),
        Question(
            "q3", "mc", "Lasso prior vs ridge prior: which statement is right?",
            "Ridge is a Normal prior; Lasso is a Laplace prior with a sharp peak at zero.",
            concept="lasso",
            options=["Ridge = Normal prior, Lasso = Laplace prior", "Ridge = Laplace prior, Lasso = Normal prior",
                     "Both are Normal priors", "Neither is a prior"],
            notes=["Correct.", "Reversed.", "The Laplace's sharp peak is what produces zeros.", "Both penalties are priors."],
            answer=0),
        Question(
            "q4", "mc", "Rating and year-1 production move together. What can happen with Lasso?",
            "It can keep one and drop the other, and which one it keeps can change between samples.",
            concept="lasso", category="Application",
            options=["It may keep one and drop the other, unstably", "It keeps both with equal slopes",
                     "It drops both", "It fails to run"],
            notes=["Correct.", "That is closer to what ridge does.", "Not necessarily.", "It runs fine."],
            answer=0),
        Question(
            "q5", "short", "In the lesson, the strongest penalty (C = 0.05) gave a Lasso with AUC 0.500. Why?",
            "Every slope was shrunk to exactly zero, so all players got the same probability and nothing was ranked.",
            concept="lasso", category="Application",
            keywords=[["zero", "all", "every", "no input", "dropped"], ["same", "constant", "rank", "nothing", "equal"]],
            model_answer="The penalty was so strong that every slope became exactly zero, so every player got the same probability. Nothing can be ranked, which is an AUC of 0.5."),
    ],
)
