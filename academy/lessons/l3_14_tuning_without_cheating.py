import warnings

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig

CS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100]


@st.cache_data(show_spinner="Running the tuning comparison…")
def _curves(seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from ..data import make_draft_classes
    from ..smalldata import Design
    warnings.filterwarnings("ignore")
    df = make_draft_classes(seed=seed)
    train = df[df["class_year"] <= 2013]
    test = df[(df["class_year"] >= 2014) & (df["class_year"] <= 2021)]
    d = Design(train)
    X, Xt = d.linear(train)[:, 1:], d.linear(test)[:, 1:]
    y, yt, yrs = train["drafted"].to_numpy(), test["drafted"].to_numpy(), train["class_year"].to_numpy()
    cv, te = [], []
    for C in CS:
        a = []
        for k in range(2010, 2014):
            tr, va = yrs < k, yrs == k
            if y[va].sum() and y[tr].sum():
                a.append(roc_auc_score(y[va], LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]))
        cv.append(float(np.mean(a)))
        te.append(roc_auc_score(yt, LogisticRegression(C=C, max_iter=2000).fit(X, y).predict_proba(Xt)[:, 1]))
    return cv, te


def visual():
    seed = st.select_slider("Simulated program (each is a different random draw)", list(range(1, 13)), value=1, key="l314_seed")
    cv, te = _curves(seed)
    bc, bt = int(np.argmax(cv)), int(np.argmax(te))
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    ax.plot(CS, cv, color=T.ACCENT, lw=2.2, marker="o", label="Cross-validation inside training classes")
    ax.plot(CS, te, color=T.PRIMARY, lw=2.4, marker="o", label="Later classes (the real test)")
    ax.scatter([CS[bc]], [cv[bc]], s=90, facecolor="none", edgecolor=T.ACCENT, lw=2, zorder=5)
    ax.scatter([CS[bt]], [te[bt]], s=90, facecolor="none", edgecolor=T.PRIMARY, lw=2, zorder=5)
    ax.set_xscale("log"), ax.set_xlabel("C (weaker penalty to the right)"), ax.set_ylabel("AUC"), ax.legend(fontsize=7)
    show_fig(fig)
    c = st.columns(3)
    c[0].metric("CV picks C =", f"{CS[bc]:g}", f"CV score {cv[bc]:.3f}", delta_color="off")
    c[1].metric("Its real test AUC", f"{te[bc]:.3f}")
    c[2].metric("Best possible (peeking)", f"{te[bt]:.3f}", f"C = {CS[bt]:g}", delta_color="off")


CODE = '''import warnings
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from academy.data import make_draft_classes
from academy.smalldata import Design

warnings.filterwarnings("ignore")


def load(seed):
    df = make_draft_classes(seed=seed)                          # SIMULATED program, 22 recruits a class
    train = df[df["class_year"] <= 2013]                        # 132 finished recruits
    test = df[(df["class_year"] >= 2014) & (df["class_year"] <= 2021)]
    d = Design(train)
    X, Xt = d.linear(train)[:, 1:], d.linear(test)[:, 1:]       # rating, year-1, position dummies
    return X, train["drafted"].to_numpy(), train["class_year"].to_numpy(), Xt, test["drafted"].to_numpy()


def ridge(C):
    return LogisticRegression(C=C, max_iter=2000)


def boost(n):
    return GradientBoostingClassifier(n_estimators=n, learning_rate=0.05, max_depth=2, subsample=0.8,
                                      min_samples_leaf=5, random_state=0)


def rolling_cv(make, setting, X, y, years, first):
    """Train on classes before year k, score class k, for every k from `first`. Mean AUC over those folds."""
    aucs = []
    for k in range(first, years.max() + 1):
        tr, va = years < k, years == k
        if y[va].sum() and y[tr].sum():
            aucs.append(roc_auc_score(y[va], make(setting).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]))
    return float(np.mean(aucs))


def nested_cv(make, grid, X, y, years):
    """Score the WHOLE procedure (tune, then fit). Tuning only ever sees the outer training classes."""
    out = []
    for k in range(2011, years.max() + 1):
        tr, va = years < k, years == k
        if y[va].sum():
            inner = [rolling_cv(make, s, X[tr], y[tr], years[tr], first=years.min() + 2) for s in grid]
            best = grid[int(np.argmax(inner))]
            out.append(roc_auc_score(y[va], make(best).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]))
    return float(np.mean(out))


GRIDS = {"ridge": (ridge, [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100], 1),      # (model, settings, sensible default)
         "boost": (boost, [5, 10, 20, 50, 100, 300], 100)}

# Part 1: three ways to pick the setting, averaged over 12 simulated programs.
rows = {name: [] for name in GRIDS}
for seed in range(1, 13):
    X, y, years, Xt, yt = load(seed)
    for name, (make, grid, default) in GRIDS.items():
        cv = [rolling_cv(make, s, X, y, years, first=2010) for s in grid]               # honest: training classes only
        test = [roc_auc_score(yt, make(s).fit(X, y).predict_proba(Xt)[:, 1]) for s in grid]
        rows[name].append((test[int(np.argmax(cv))], max(test), test[grid.index(default)], np.mean(test), max(cv)))
print("Model  CV-chosen | peeked at test | default | grid average | CV score of the winner")
for name, r in rows.items():
    a = np.mean(r, axis=0)
    print(f"{name:<6} {a[0]:.3f}      {a[1]:.3f}          {a[2]:.3f}     {a[3]:.3f}          {a[4]:.3f}")

# Part 2: how much does a plain CV score overstate, and does nesting fix it? (ridge)
make, grid, _ = GRIDS["ridge"]
res = []
for seed in range(1, 13):
    X, y, years, Xt, yt = load(seed)
    cv = [rolling_cv(make, s, X, y, years, first=2010) for s in grid]
    best = grid[int(np.argmax(cv))]
    res.append((max(cv), nested_cv(make, grid, X, y, years), roc_auc_score(yt, make(best).fit(X, y).predict_proba(Xt)[:, 1])))
a = np.mean(res, axis=0)
print(f"Ridge: plain CV score of the winner {a[0]:.3f} | nested CV {a[1]:.3f} | true test AUC {a[2]:.3f}")
'''

LESSON = Lesson(
    id="l3-14-tuning-without-cheating", level=3, order=14,
    title="Tuning Without Cheating: Cross-Validation and Nested CV",
    subtitle="How to pick a model's settings honestly, and why small data limits what tuning can do",
    minutes=25,
    concept_ids=["tuning-cv"],
    simple="""
Most models have **settings** you must choose: how strong the penalty is (L1.7, L1.9), how many trees to grow (L3.13), how deep they go. These are called **hyperparameters**.

The tempting way to choose is to try several and keep whichever scores best on your test set. That quietly turns the test set into a tuning set, so the score you report is **too good**. The best of several noisy scores is almost always lucky.

The honest way is **cross-validation inside the training data**: hold out a training class, fit on the classes before it, score it, repeat, and choose the setting with the best average. The test set is touched once, at the end. And to measure the **whole** procedure (tuning included) fairly, you use **nested** cross-validation: tuning is repeated inside every outer fold, so it never sees the class being scored.
""",
    problem="You must pick settings, but every time you look at a score to choose, that score stops being an honest estimate. With only 12 draft picks, the temptation and the damage are both large.",
    why="It's the most common way a good-looking model turns out to be a fluke. Doing it properly also tells you something uncomfortable and useful: on tiny data, tuning often buys very little, and a sensible default can be just as good.",
    when_useful="""
- Any model with settings that matter (penalty strength, trees, depth, prior width)
- When you have enough rows and events for the validation scores to mean something
- When you must report how well the whole pipeline works, not just one fitted model
""",
    when_not="""
- Don't tune a lot on a handful of events: the validation score is too noisy to choose reliably
- Don't use ordinary shuffled folds on seasons: use rolling-origin (train on earlier classes, score the next), as in L3.2
- Don't report the best cross-validation score as your result: it is optimistic by construction
""",
    compare="""
**Peeking at the test set:** fast and wrong; the score is inflated. **Plain cross-validation:** picks the setting honestly, but its best score is still optimistic. **Nested cross-validation:** repeats the tuning inside each outer fold, so it scores the whole procedure fairly. **A fixed default or an informative prior (L1.8):** no tuning at all, which on small data is often the strongest choice. **Time-based CV (L3.2):** the fold design these all rely on.
""",
    sports="""
**Fictional scenario (simulated).** Same program as L1.7 and L3.13: train on the 132 finished recruits (classes 2008–2013, about 12 drafted), test on 2014–2021. Settings are tuned with rolling-origin cross-validation inside the training classes only. Averages over **12 simulated programs**, test AUC:

| Model | Setting picked by CV | Peeked at the test set | Fixed sensible default | Average of all settings | CV score of the winner |
|---|---|---|---|---|---|
| Ridge logistic (8 penalty strengths) | 0.806 | 0.823 | 0.810 | 0.806 | 0.857 |
| Boosting (6 tree counts) | 0.757 | 0.788 | 0.752 | 0.761 | 0.775 |

- **Peeking inflates the result:** +0.017 for ridge and +0.031 for boosting. The more settings you try, the bigger the inflation.
- **Cross-validation did not beat a sensible default here.** The CV-picked ridge (0.806) is no better than C = 1 (0.810) or a random pick (0.806). With about 12 events, the validation score is too noisy to choose reliably. The picked penalty strength scattered from 0.01 to 100 across the 12 programs.
- **The winner's CV score is optimistic.** For ridge, plain CV says 0.857 but the real test AUC is 0.806. **Nested CV says 0.821**, much closer to the truth, because it scores the tuning procedure on classes the tuning never saw.
- Tuning isn't useless. It still beats peeking in honesty. But on small data, the better lever is **fewer knobs and sensible defaults or priors**, not a larger search.
""",
    takeaway="Choose settings inside the training classes, touch the test set once, and report a nested estimate for the whole pipeline. If validation can't separate the settings, don't pretend it can: use a sensible default or a prior.",
    visual=visual,
    visual_caption="Amber: cross-validation score inside the training classes. Blue: the real test AUC. Rings mark each curve's best setting. Slide through the 12 simulated programs: the two curves often peak at different places, and the amber peak is nearly always higher than what you actually get.",
    math=[
        MathBlock(
            "Step 1: choose settings by validation score",
            r"\hat\lambda = \arg\max_{\lambda \in \Lambda} \; \frac{1}{K}\sum_{k=1}^{K} \text{AUC}\big(f_{\lambda}^{(<k)},\; \text{class } k\big)",
            """
- $\\Lambda$ (capital lambda): the list of settings you try (the grid), such as the 8 penalty strengths
- $f_\\lambda^{(<k)}$: the model with setting $\\lambda$ trained only on classes **before** class $k$
- The inner part scores it on class $k$; the average is over $K$ validation classes
- $\\hat\\lambda$ (lambda-hat): the setting with the best average. This uses training classes only
""",
        ),
        MathBlock(
            "Step 2: why the winner's score is too high",
            r"\mathbb{E}\Big[\max_{\lambda}\; \widehat{\text{AUC}}_\lambda\Big] \;\ge\; \max_{\lambda}\; \mathbb{E}\big[\widehat{\text{AUC}}_\lambda\big]",
            """
- $\\widehat{\\text{AUC}}_\\lambda$: the noisy validation estimate for setting $\\lambda$
- Each estimate is the truth plus noise. The maximum of several noisy numbers is, on average, **above** the best true value
- The more settings you try and the noisier each estimate (few events), the larger the gap
""",
            "This is the 'winner's curse', and it is why 0.857 became 0.806.",
        ),
        MathBlock(
            "Step 3: nested cross-validation",
            r"\text{Nested} = \frac{1}{J}\sum_{j=1}^{J} \text{AUC}\Big(f_{\hat\lambda_j}^{(<j)},\; \text{class } j\Big), \quad \hat\lambda_j \text{ chosen using only classes} < j",
            """
- The **outer** loop picks a class $j$ to score; the **inner** loop (Step 1) tunes using only classes before $j$
- $\\hat\\lambda_j$: the setting the inner loop picked for that outer fold
- The class being scored never influenced its own tuning, so the result estimates the whole procedure, tuning included
- Cost: you refit many more models, which is fine on small data
""",
        ),
    ],
    code=CODE,
    code_notes="Shrink the ridge grid to just [0.1, 1, 10]. Notice the peeking gap shrinks too: fewer settings tried means less luck to harvest.",
    walkthrough=[
        ("warnings.filterwarnings('ignore')", "Hide scikit-learn convergence chatter so the table is readable."),
        ("def load(seed):", "Build one simulated program and split it: classes up to 2013 for training, 2014–2021 for the final test."),
        ("d.linear(train)[:, 1:]", "The same inputs as the other lessons: rating, year-1 percentile, position dummies."),
        ("def rolling_cv(make, setting, X, y, years, first):", "For each class k, fit only on earlier classes and score class k. Time order is respected (L3.2)."),
        ("if y[va].sum() and y[tr].sum():", "AUC needs at least one drafted player in the scored class and in the training part; skip folds that have none."),
        ("def nested_cv(make, grid, X, y, years):", "For each outer class, tune using only the classes before it, then score it. This measures the whole procedure."),
        ("GRIDS = {...}", "The settings to try for each model, plus a sensible default chosen before looking (ridge C = 1, boosting 100 trees)."),
        ("cv = [rolling_cv(...) for s in grid]", "Validation score for every setting, using training classes only."),
        ("test = [roc_auc_score(yt, ...)]", "The same models on the final test classes. Used to compare, never to choose."),
        ("test[int(np.argmax(cv))]", "The honest pick: the setting CV chose, scored on the test classes."),
        ("max(test)", "The cheating pick: the best test score over all settings. The gap to the line above is the inflation."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Write the tuning plan down before scoring:** which settings, which grid, which fold design. Part of the pre-set bar (L6.3).
- **Report the nested number**, not the best cross-validation score, when you describe how well a pipeline works. If you can only afford one, the honest one is lower.
- **With few events per season, lean on defaults and priors.** A tuned model that can't beat a sensible default on a pre-set test hasn't earned its extra moving parts.
""",
    quiz=[
        Question(
            "q1", "mc", "Why is choosing a model's settings by their test-set score a mistake?",
            "The test set stops being an independent check: the best of several noisy scores is optimistically biased.",
            concept="tuning-cv",
            options=["The reported score is inflated because you selected the winner on it", "Test sets are too small to score",
                     "It uses too much memory", "It is only a problem for boosting"],
            notes=["Correct: that is the winner's curse.", "Size isn't the issue; independence is.", "Not a computing issue.", "It affects any tuned model."],
            answer=0),
        Question(
            "q2", "mc", "What does nested cross-validation add over plain cross-validation?",
            "Tuning is repeated inside every outer fold, so it scores the whole tune-then-fit procedure on data it never saw.",
            concept="tuning-cv",
            options=["It evaluates the whole procedure, tuning included, fairly", "It makes the model more accurate",
                     "It removes the need for a test set", "It tunes faster"],
            notes=["Correct.", "It estimates accuracy more honestly; it doesn't improve the model.", "You still need an untouched test.", "It is slower, not faster."],
            answer=0),
        Question(
            "q3", "numeric", "In the lesson, plain CV said 0.857 for the winning ridge but the real test AUC was 0.806. How much did plain CV overstate it? (3 decimals)",
            "0.857 − 0.806 = 0.051.",
            concept="tuning-cv", category="Application", answer=0.051, tolerance=0.002),
        Question(
            "q4", "mc", "The CV-picked ridge (0.806) did no better than a fixed default (0.810). What is the best reading?",
            "With only about 12 events the validation score is too noisy to separate the settings, so a sensible default is competitive.",
            concept="tuning-cv", category="Application",
            options=["Validation was too noisy to choose reliably on so few events", "Cross-validation is broken",
                     "The default was secretly tuned on the test set", "Ridge can't be tuned"],
            notes=["Correct: tuning can't beat noise.", "It worked as designed; the data is just small.",
                   "The default was set before looking.", "It can be tuned, but the gain was small here."],
            answer=0),
        Question(
            "q5", "short", "Why should seasons use rolling-origin folds for tuning rather than ordinary shuffled folds?",
            "Shuffled folds train on future classes to predict past ones, which leaks the future and overstates performance.",
            concept="tuning-cv", category="Application",
            keywords=[["future", "later", "leak", "time", "order"], ["train", "earlier", "past", "before", "predict"]],
            model_answer="Shuffled folds let the model train on later classes to score earlier ones, which leaks the future and inflates the score. Rolling-origin folds train only on earlier classes and score the next one, matching how the model will really be used."),
    ],
)
