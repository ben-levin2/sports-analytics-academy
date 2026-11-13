import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig

TREES = [1, 2, 3, 5, 10, 20, 35, 50, 75, 100, 200, 300, 500, 1000]


@st.cache_data(show_spinner="Fitting boosted trees…")
def _curve(depth, rate):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from ..data import make_draft_classes
    from ..smalldata import Design
    warnings.filterwarnings("ignore")
    df = make_draft_classes()
    train = df[df["class_year"] <= 2013]
    test = df[(df["class_year"] >= 2014) & (df["class_year"] <= 2021)]
    d = Design(train)
    X, Xt = d.linear(train)[:, 1:], d.linear(test)[:, 1:]
    g = GradientBoostingClassifier(n_estimators=max(TREES), learning_rate=rate, max_depth=depth, subsample=0.8,
                                   min_samples_leaf=5, random_state=0).fit(X, train["drafted"])
    tr = [roc_auc_score(train["drafted"], p[:, 1]) for p in g.staged_predict_proba(X)]
    te = [roc_auc_score(test["drafted"], p[:, 1]) for p in g.staged_predict_proba(Xt)]
    return [tr[n - 1] for n in TREES], [te[n - 1] for n in TREES]


def visual():
    c1, c2 = st.columns(2)
    depth = c1.select_slider("Tree depth", [1, 2, 3, 4], value=2, key="l313_depth")
    rate = c2.select_slider("Learning rate", [0.02, 0.05, 0.1, 0.3], value=0.05, key="l313_rate")
    tr, te = _curve(depth, rate)
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    ax.plot(TREES, tr, color=T.ACCENT, lw=2.2, label="Training AUC")
    ax.plot(TREES, te, color=T.PRIMARY, lw=2.4, label="Test AUC")
    best = int(np.argmax(te))
    ax.scatter([TREES[best]], [te[best]], color=T.PRIMARY, zorder=4)
    ax.set_xscale("log"), ax.set_xlabel("Number of trees"), ax.set_ylabel("AUC"), ax.legend(fontsize=8)
    show_fig(fig)
    st.caption(f"Training AUC reaches {tr[-1]:.3f} at {TREES[-1]} trees; test AUC peaks at {te[best]:.3f} with {TREES[best]} trees "
               f"and ends at {te[-1]:.3f}. Choosing the peak by looking at the test set is cheating: pick the number of trees "
               "with cross-validation inside the training classes.")


CODE = '''import warnings
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from academy.data import make_draft_classes
from academy.smalldata import Design, evaluate

warnings.filterwarnings("ignore")
df = make_draft_classes()                                   # SIMULATED program, 22 recruits a class
train = df[df["class_year"] <= 2013]                        # 132 finished recruits
test = df[(df["class_year"] >= 2014) & (df["class_year"] <= 2021)]
d = Design(train)
X, Xt = d.linear(train)[:, 1:], d.linear(test)[:, 1:]       # rating, year-1, position dummies

print("Trees  train AUC  test AUC  test Brier")
for n in [1, 5, 20, 50, 100, 300, 1000]:
    g = GradientBoostingClassifier(n_estimators=n, learning_rate=0.05, max_depth=2, subsample=0.8,
                                   min_samples_leaf=5, random_state=0).fit(X, train["drafted"])
    p_tr, p_te = g.predict_proba(X)[:, 1], g.predict_proba(Xt)[:, 1]
    print(f"{n:>5}   {roc_auc_score(train['drafted'], p_tr):.3f}      {roc_auc_score(test['drafted'], p_te):.3f}"
          f"     {np.mean((p_te - test['drafted']) ** 2):.4f}")

# Does boosting catch up with more data? Average test AUC over 4 simulated programs at each class size.
print("Recruits per class: plain logistic | rating only | Bayesian logistic | boosting")
for per in [22, 60, 150, 400]:
    auc = [evaluate(make_draft_classes(per_class=per, seed=s), boot=0, baseline="plain")["summary"]["auc"]
           for s in (1, 2, 3, 4)]
    m = sum(auc) / 4
    print(f"{per:>5}: {m['plain']:.3f} | {m['rating_only']:.3f} | {m['bayes']:.3f} | {m['boost']:.3f}")
'''

LESSON = Lesson(
    id="l3-13-gradient-boosting", level=3, order=13,
    title="Gradient Boosting (XGBoost-Style): Many Small Trees, Added One at a Time",
    subtitle="What it is, why small data punishes it, and how to tame it",
    minutes=25,
    concept_ids=["gradient-boosting"],
    simple="""
A **decision tree** asks a few yes/no questions ("year-1 percentile above 80?") and ends in a prediction. One shallow tree is crude.

**Gradient boosting** builds hundreds of tiny trees, one after another. Each new tree is trained to fix what the earlier ones still get wrong, and its contribution is turned down by a **learning rate** so no single tree dominates. Add them up and you get a flexible model that finds curves and interactions on its own.

**XGBoost** (and LightGBM, CatBoost) are fast, engineered versions of exactly this idea, with extra penalties built in. This academy uses scikit-learn's `GradientBoostingClassifier`, the same family, so you get the lesson without adding a package.
""",
    problem="Models like Bayesian logistic regression assume a shape. Boosting assumes almost nothing and learns the shape, which is powerful with a lot of data and a trap with a little.",
    why="Boosting is the standard choice for large tabular data, so you should know it, and know exactly why it needs more rows than the models in this level.",
    when_useful="""
- Hundreds or thousands of rows (or many more)
- Messy tabular inputs with interactions you can't list in advance
- When you can afford careful tuning with cross-validation
""",
    when_not="""
- A few hundred rows with few events (draft picks, awards): it memorizes
- When you must explain direction and size of effects simply
- When probabilities matter as much as ranking: boosted probabilities often need calibration (L3.1)
""",
    compare="""
**Single tree:** readable but unstable. **Random forest:** many deep trees averaged; a different recipe. **Gradient boosting:** many shallow trees added in sequence. **XGBoost / LightGBM:** the same idea, faster and with built-in regularization. **Monotonic GAM (L3.9):** smooth, directional, readable, usually safer on small data. **Bayesian logistic (L1.8):** priors protect small samples; boosting has no such protection except the settings you choose.
""",
    sports="""
**Fictional scenario (simulated).** Same program as L1.7: train on the 132 finished recruits from 2008–2013 (12 drafted), test on 2014–2021. Depth-2 trees, learning rate 0.05:

| Trees | Training AUC | Test AUC | Test Brier |
|---|---|---|---|
| 1 | 0.834 | 0.789 | 0.0950 |
| 20 | 0.925 | **0.879** | 0.0738 |
| 100 | 0.987 | 0.839 | 0.0747 |
| 300 | 1.000 | 0.814 | 0.0770 |
| 1000 | 1.000 | 0.739 | 0.0790 |

**Training AUC climbs to a perfect 1.000 while test AUC falls from 0.879 to 0.739.** That is memorization: with 12 draft picks, the model learns the individual players, not the pattern.

The best test result (20 trees) looks great, but finding it meant looking at the test set. The honest move is to choose the number of trees with cross-validation inside the training classes.

**Does it catch up with more data?** Average test AUC over 4 simulated programs:

| Recruits per class | Plain logistic | Rating only | Bayesian logistic | Boosting |
|---|---|---|---|---|
| 22 | 0.722 | 0.779 | 0.799 | 0.713 |
| 60 | 0.764 | 0.779 | 0.814 | 0.777 |
| 150 | 0.806 | 0.796 | 0.830 | 0.809 |
| 400 | 0.839 | 0.805 | 0.842 | 0.841 |

With 22 a class, boosting is no better than a plain logistic and below rating alone. By 400 a class it matches the others, but even then it doesn't beat the Bayesian model, because the truth here is smooth and simple.
""",
    takeaway="Boosting is powerful but data-hungry. With a small sample, keep trees shallow, the learning rate low, the number of trees chosen inside training, and judge it against the strongest simple baseline.",
    visual=visual,
    visual_caption="Training AUC (amber) rises toward 1; test AUC (blue) peaks and then falls. Deeper trees or a faster learning rate make the gap open sooner.",
    math=[
        MathBlock(
            "Step 1: an additive model of trees",
            r"F_M(x) = F_0 + \nu \sum_{m=1}^{M} h_m(x), \qquad p(x) = \frac{1}{1 + e^{-F_M(x)}}",
            """
- $F_M(x)$: the model's log-odds for inputs $x$ after $M$ trees
- $F_0$: the starting guess (the overall base rate on the log-odds scale)
- $h_m(x)$: the $m$-th small tree (a handful of yes/no splits)
- $\\nu$ (nu): the **learning rate**, a number like 0.05 that turns each tree's contribution down
""",
        ),
        MathBlock(
            "Step 2: each tree fixes what's left",
            r"r_i = y_i - p_i, \qquad h_m \approx \text{a small tree fitted to the residuals } r_i",
            """
- $r_i$: the **residual** for recruit $i$: what happened ($y_i$ is 1 or 0) minus what the model currently predicts ($p_i$)
- This residual is the gradient of the log-loss, which is where 'gradient' boosting gets its name
- A new tree learns to push predictions toward the recruits the model currently gets most wrong
""",
            "Recruits with large residuals (a drafted player scored low) pull the next tree toward them. With 12 draft picks, that means chasing individual players.",
        ),
        MathBlock(
            "Step 3: what controls overfitting",
            r"\text{complexity} \uparrow \text{ with } M \;(\text{trees}),\; d \;(\text{depth}),\; \nu \;(\text{rate});\qquad \text{regularize with } \nu \downarrow,\; d \downarrow,\; \text{subsample} < 1,\; \text{min leaf size} \uparrow",
            """
- More trees, deeper trees and a larger $\\nu$ let the model fit the training data more closely
- **Subsample** (0.8): each tree sees a random 80% of rows, which adds noise that reduces memorization
- **Minimum leaf size** (5): a leaf must hold at least 5 recruits, so no leaf can isolate one player
- XGBoost adds explicit L1 and L2 penalties on the leaf values: the same ideas as L1.7 and L1.9
""",
        ),
    ],
    code=CODE,
    code_notes="Change max_depth=2 to max_depth=4 and min_samples_leaf=5 to 1 in the first loop. Training AUC hits 1.000 with far fewer trees.",
    walkthrough=[
        ("train = df[df['class_year'] <= 2013]", "Only finished classes: 132 recruits, 12 drafted."),
        ("d.linear(train)[:, 1:]", "The same inputs as the other models: rating, year-1 percentile, position dummies. Trees don't need standardizing, but this keeps the comparison fair."),
        ("GradientBoostingClassifier(n_estimators=n, learning_rate=0.05, max_depth=2, ...)", "n small trees of depth 2, each scaled by 0.05."),
        ("subsample=0.8, min_samples_leaf=5", "Two built-in brakes: each tree sees 80% of rows, and no leaf can hold fewer than 5 recruits."),
        ("roc_auc_score(train['drafted'], p_tr)", "AUC on the data the model was trained on, which only shows how well it memorized."),
        ("roc_auc_score(test['drafted'], p_te)", "AUC on later classes: the real result."),
        ("for per in [22, 60, 150, 400]:", "Repeat the comparison as the program gets bigger, using the same class-by-class test as the P3C project."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Boosting becomes reasonable when rows are plentiful**, like play-level data with thousands of snaps. For one program's draft picks it's almost always the wrong tool; this lesson shows why.
- **In the P3C project** boosting is a fourth challenger, tested under the same pre-set bar as the others (and a stricter bootstrap check because there are now four).
- **Choosing the number of trees** is part of the model. Do it by cross-validation on training seasons only, never by watching the test set.
""",
    quiz=[
        Question(
            "q1", "mc", "How does gradient boosting build its model?",
            "It adds many small trees one at a time, each fitted to what the previous trees still get wrong.",
            concept="gradient-boosting",
            options=["Small trees added in sequence, each fixing earlier errors", "One very deep tree",
                     "Many deep trees averaged independently", "A straight line plus a penalty"],
            notes=["Correct.", "That's a single tree.", "That's a random forest.", "That's regularized regression."],
            answer=0),
        Question(
            "q2", "mc", "In the lesson, training AUC reached 1.000 while test AUC fell. What is this called?",
            "Overfitting: the model memorizes individual training recruits instead of learning the pattern.",
            concept="gradient-boosting", category="Application",
            options=["Overfitting", "Underfitting", "Leakage", "Good calibration"],
            notes=["Correct.", "Training AUC was perfect; that's the opposite.", "No future data was used.", "Calibration is about probabilities."],
            answer=0),
        Question(
            "q3", "mc", "Which setting makes boosting LESS likely to overfit?",
            "A lower learning rate (with enough trees), shallower trees, subsampling and a larger minimum leaf size.",
            concept="gradient-boosting",
            options=["Shallower trees and a lower learning rate", "More trees of greater depth", "A higher learning rate",
                     "A minimum leaf size of 1"],
            notes=["Correct.", "That adds complexity.", "That fits training data faster.", "A leaf of one recruit memorizes him."],
            answer=0),
        Question(
            "q4", "mc", "Why is picking 20 trees because it had the best test AUC a mistake?",
            "You chose a setting using the test set, so the test result is no longer an independent check: that's leakage into tuning.",
            concept="gradient-boosting", category="Application",
            options=["The test set was used to tune, so it's no longer a fair test", "20 is too few trees",
                     "Test AUC can't be computed", "Boosting must always use 1000 trees"],
            notes=["Correct: tune with cross-validation inside training classes.", "The number isn't the problem; how it was chosen is.",
                   "It can.", "No such rule."],
            answer=0),
        Question(
            "q5", "short", "How does XGBoost relate to scikit-learn's gradient boosting?",
            "It's the same idea (shallow trees added in sequence), engineered to be faster and with extra built-in regularization.",
            concept="gradient-boosting",
            keywords=[["same", "idea", "family", "similar", "version"], ["fast", "regulariz", "penalt", "engineer", "efficient"]],
            model_answer="XGBoost is the same family of model: shallow trees added one at a time to fix earlier errors. It is a faster, more engineered implementation with extra regularization, so scikit-learn's version teaches the same lesson without a new dependency."),
    ],
)
