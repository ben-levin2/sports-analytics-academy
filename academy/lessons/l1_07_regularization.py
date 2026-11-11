import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner=False)
def _path():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from ..data import make_draft_classes
    from ..smalldata import Design
    df = make_draft_classes()
    train = df[df["class_year"] <= 2013]
    test = df[(df["class_year"] >= 2014) & (df["class_year"] <= 2021)]
    d = Design(train)
    X, Xt = d.linear(train)[:, 1:], d.linear(test)[:, 1:]
    Cs = np.logspace(-2, 3, 26)
    coefs, aucs, briers = [], [], []
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=5000).fit(X, train["drafted"])
        p = m.predict_proba(Xt)[:, 1]
        coefs.append(m.coef_[0])
        aucs.append(roc_auc_score(test["drafted"], p))
        briers.append(np.mean((p - test["drafted"]) ** 2))
    return Cs, np.array(coefs), np.array(aucs), np.array(briers), d.positions


def visual():
    Cs, coefs, aucs, briers, positions = _path()
    i = st.select_slider("Penalty strength (left = strong prior, right = almost none)", options=list(range(len(Cs))),
                         value=12, format_func=lambda k: f"C = {Cs[k]:.2g}  (prior SD {np.sqrt(Cs[k]):.2g})", key="l17_c")
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(Cs, coefs[:, 0], color=T.PRIMARY, lw=2.2, label="Rating")
    ax.plot(Cs, coefs[:, 1], color=T.ACCENT, lw=2.2, label="Year-1 percentile")
    for j in range(2, coefs.shape[1]):
        ax.plot(Cs, coefs[:, j], color=T.FAINT, lw=1)
    ax.axvline(Cs[i], color=T.MUTED, ls="--", lw=1)
    ax.axhline(0, color=T.BORDER, lw=1)
    ax.set_xscale("log"), ax.set_xlabel("C (bigger = weaker penalty = wider prior)"), ax.set_ylabel("Slope (log-odds per SD)")
    ax.set_title("Grey lines: the 7 position effects", fontsize=9)
    show_fig(fig)
    c = st.columns(3)
    c[0].metric("Test AUC (ranking)", f"{aucs[i]:.3f}")
    c[1].metric("Test Brier (probabilities)", f"{briers[i]:.4f}", help="Lower is better")
    c[2].metric("Best Brier on this path", f"{briers.min():.4f}", f"at C = {Cs[briers.argmin()]:.2g}", delta_color="off")


CODE = '''import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from academy.data import make_draft_classes
from academy.smalldata import Design, fit_logistic

df = make_draft_classes()                        # SIMULATED: one program, 22 recruits a class
train = df[df["class_year"] <= 2013]             # 6 finished classes = 132 recruits
test = df[(df["class_year"] >= 2014) & (df["class_year"] <= 2021)]
d = Design(train)                                # standardize with TRAINING stats only
X, Xt = d.linear(train)[:, 1:], d.linear(test)[:, 1:]   # drop the 1s column; sklearn adds an intercept
y, yt = train["drafted"], test["drafted"]
print(f"Training rows: {len(train)}, drafted: {y.sum()}, inputs: {X.shape[1]}")

# 1. The same model with a weaker and weaker penalty. C = 1 / lambda.
for C in [0.01, 0.1, 0.5, 1, 10, 1000]:
    m = LogisticRegression(C=C, max_iter=5000).fit(X, y)
    p = m.predict_proba(Xt)[:, 1]
    auc, brier = roc_auc_score(yt, p), np.mean((p - yt) ** 2)
    size = np.sqrt(np.sum(m.coef_ ** 2))
    print(f"C = {C:>7}: test AUC {auc:.3f}  Brier {brier:.4f}  size of slopes {size:4.2f}")

# 2. Ridge IS a prior: penalty lambda/2 * b^2  <=>  b ~ Normal(0, sd = 1/sqrt(lambda)) = sqrt(C)
C = 0.5
sk = LogisticRegression(C=C, max_iter=5000).fit(X, y)
sd = np.r_[100.0, np.full(X.shape[1], np.sqrt(C))]      # vague prior on the intercept, like sklearn
bayes = fit_logistic(d.linear(train), y, prior_sd=sd)
print("sklearn ridge rating slope:", round(float(sk.coef_[0, 0]), 4))
print("Normal-prior MAP rating slope:", round(float(bayes["beta"][1]), 4))
'''

LESSON = Lesson(
    id="l1-07-regularization", level=1, order=7,
    title="Regularization: Why a Penalty Helps When Data Is Thin",
    subtitle="Ridge (L2) regression, and why the penalty is really a prior",
    minutes=20,
    concept_ids=["regularization"],
    simple="""
When you have few rows and several inputs, a model can "explain" the training data too well. It finds patterns that are just luck, and those patterns fail on new players.

**Regularization** adds a small cost for big coefficients. The model now has two goals: fit the data, and keep its slopes modest unless the data really insists.

The most common version is **ridge** (also called **L2**): the cost is the sum of the squared slopes. It doesn't delete inputs; it pulls every slope toward zero, and pulls the weakly supported ones the most.

The key idea for this academy: **that penalty is exactly a prior.** Ridge is the same as saying, before seeing data, "each slope is probably near zero, give or take this much." You already know priors from Lessons L1.1–L1.4.
""",
    problem="With small data, plain (unpenalized) models overfit: they chase noise, give extreme slopes to rare categories, and make overconfident probabilities that fail on the next class.",
    why="A penalty trades a little bias for a large cut in variance. With 132 recruits and only 12 drafted, that trade almost always pays off. It also gives you a clean way to bring in outside knowledge (next lesson).",
    when_useful="""
- Few rows, or few "yes" outcomes (draft picks, all-conference seasons, transfers)
- Many inputs relative to rows, especially category dummies like position
- Inputs that move together (rating and offer count, say), where plain slopes swing wildly
""",
    when_not="""
- Lots of data: the penalty barely matters (it still rarely hurts)
- When you need exact, unbiased slope estimates for one key input. Ridge pulls them toward zero on purpose
- Inputs on different scales: always standardize first, or the penalty treats them unfairly
""",
    compare="""
**Plain logistic regression:** no penalty. Fine with lots of data; overfits with little. **Ridge (L2):** shrinks every slope smoothly, keeps all inputs. **Lasso (L1):** can push slopes to exactly zero, so it also selects inputs; less stable with very small data. **Bayesian regression (L1.4, next lesson):** the same as ridge when the prior is Normal with mean 0, but you can also center the prior somewhere other than 0 and get a full uncertainty. **Hierarchical models (L5.4):** learn how strong the shrinkage should be from groups in the data.
""",
    sports="""
**Fictional scenario (simulated).** One program, 22 recruits a class. Predict who gets drafted from rating, year-1 production percentile and position. Train on the six finished classes 2008–2013 (**132 recruits, only 12 drafted**) and test on 2014–2021.

| Penalty (C = 1/λ) | Test AUC | Test Brier | Size of slopes |
|---|---|---|---|
| 0.01 (very strong) | 0.864 | 0.0928 | 0.13 |
| 0.1 | 0.866 | 0.0781 | 0.63 |
| 1 | 0.851 | **0.0684** | 1.46 |
| 1000 (almost none) | 0.823 | 0.0717 | 2.74 |

Two things to read here:
- **Almost no penalty is worst at ranking** (AUC 0.823). The plain model gives large slopes to positions based on one or two drafted players.
- **Too much penalty ranks fine but gets the probabilities wrong** (Brier 0.0928, the worst). Every slope is near zero, so everyone gets roughly the base rate.

A moderate penalty (C around 1 here) gives the best probabilities. That middle ground is the whole skill.
""",
    takeaway="With little data, a modest penalty beats no penalty. The penalty is a prior that says 'slopes are probably not huge'; how strong it should be is a choice you test on later data.",
    visual=visual,
    visual_caption="Move the slider from strong penalty (left) to almost none (right). Rating and year-1 slopes grow quickly; the grey position effects stay near 0 until the penalty is weak, then fan out. That fan-out is overfitting.",
    math=[
        MathBlock(
            "Step 1: plain logistic regression",
            r"\hat\beta = \arg\max_\beta \sum_{i=1}^{n} \Big[y_i \log p_i + (1-y_i)\log(1-p_i)\Big], \qquad p_i = \frac{1}{1+e^{-x_i^\top\beta}}",
            """
- $y_i$: 1 if recruit $i$ was drafted, 0 if not
- $x_i$: his inputs (a 1 for the intercept, standardized rating, standardized year-1 percentile, position dummies)
- $\\beta$: the slopes; $x_i^\\top\\beta$ is his log-odds of being drafted
- $p_i$: his predicted probability
- The sum is the **log-likelihood**: how well the slopes explain who was and wasn't drafted
""",
        ),
        MathBlock(
            "Step 2: add the ridge penalty",
            r"\hat\beta_{\text{ridge}} = \arg\max_\beta \; \underbrace{\sum_{i} \log P(y_i \mid \beta)}_{\text{fit the data}} \; - \; \underbrace{\frac{\lambda}{2}\sum_{j \ge 1} \beta_j^2}_{\text{keep slopes small}}",
            """
- $\\lambda$ (lambda): penalty strength, 0 or more. $\\lambda = 0$ is plain regression
- $\\beta_j^2$: the squared slope of input $j$; the intercept ($j = 0$) is usually not penalized
- scikit-learn uses $C = 1/\\lambda$, so **small C = strong penalty**
""",
        ),
        MathBlock(
            "Step 3: the penalty is a Normal prior",
            r"\log \text{Normal}(\beta_j \mid 0, \sigma^2) = -\frac{\beta_j^2}{2\sigma^2} + \text{const} \quad\Longrightarrow\quad \lambda = \frac{1}{\sigma^2},\; \sigma = \sqrt{C}",
            """
- $\\text{Normal}(0, \\sigma^2)$: a prior saying "this slope is probably within about $\\pm 2\\sigma$ of zero"
- $\\sigma$: the prior's standard deviation. Small $\\sigma$ = strong belief the slope is small
- Adding the log prior to the log-likelihood gives the log posterior. Maximizing it (the **MAP** estimate) is exactly ridge
""",
            "The code checks this: scikit-learn's ridge slope (0.9506) and our Normal-prior fit (0.9504) agree.",
        ),
    ],
    code=CODE,
    code_notes="Change the training classes to `<= 2010` (66 recruits). The gap between strong and weak penalties grows. Then try `<= 2018`: more data, and the penalty matters less.",
    walkthrough=[
        ("train = df[df['class_year'] <= 2013]", "Only finished classes: by 2021 every one of these recruits has had his full five-year draft window."),
        ("d = Design(train)", "Standardize rating and year-1 percentile using training stats only, and make position dummies. A penalty is only fair when inputs share a scale."),
        ("LogisticRegression(C=C, max_iter=5000)", "scikit-learn's logistic regression uses a ridge penalty by default. C is 1/lambda, so C=0.01 is a strong penalty and C=1000 is almost none."),
        ("p = m.predict_proba(Xt)[:, 1]", "Predicted draft probability for every test recruit."),
        ("roc_auc_score(yt, p), np.mean((p - yt) ** 2)", "AUC checks ranking (are drafted players scored higher?); Brier checks the probabilities themselves (concept 25)."),
        ("np.sqrt(np.sum(m.coef_ ** 2))", "One number for how big the slopes are. It grows as the penalty weakens."),
        ("sd = np.r_[100.0, np.full(X.shape[1], np.sqrt(C))]", "The same penalty written as a prior: each slope ~ Normal(0, sd = sqrt(C)); the intercept gets a very wide prior, like sklearn."),
        ("fit_logistic(d.linear(train), y, prior_sd=sd)", "Our own Bayesian fit (Newton's method on the log posterior). Its answer matches sklearn's ridge to 3 decimals."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Models 9 and 9b** added richer inputs and still lost to "same as last season". With small samples, extra inputs without a penalty mostly add noise. A penalty is the first thing to try before concluding an input is useless, and the honest result may still be "no gain".
- **Model 8** keeps a small set of inputs (rating plus early production). If you ever add more, fit them with a penalty and test on the pre-set test set, per your rule.
- **The penalty strength is a choice.** Pick it on training data only (e.g. time-based CV inside the training seasons), never on your fixed test set.
""",
    quiz=[
        Question(
            "q1", "mc", "What does a ridge (L2) penalty do to a logistic regression?",
            "It adds a cost for large slopes, pulling every slope toward zero, the weakly supported ones most.",
            concept="regularization",
            options=["Pulls slopes toward zero", "Removes the intercept", "Adds more training rows", "Makes every slope exactly zero"],
            notes=["Correct.", "The intercept is usually not penalized at all.", "It changes the fit, not the data.",
                   "That's what a huge penalty approaches; ridge alone never makes a slope exactly zero."],
            answer=0),
        Question(
            "q2", "mc", "Ridge with penalty λ is the same as a Bayesian model with which prior on each slope?",
            "Penalty λ/2·β² is minus the log of a Normal(0, σ²) density with σ² = 1/λ.",
            concept="regularization",
            options=["Normal(0, 1/λ)", "Uniform over all values", "Normal(λ, 1)", "No prior at all"],
            notes=["Correct: mean 0, variance 1/λ.", "A flat prior adds no penalty; that's plain regression.",
                   "The prior is centered at 0, and λ sets its width, not its center.", "No prior gives plain regression."],
            answer=0),
        Question(
            "q3", "numeric", "In scikit-learn, C = 0.25. What is the standard deviation of the equivalent Normal prior on each slope?",
            "σ = √C = √0.25 = 0.5.",
            concept="regularization", category="Application", answer=0.5, tolerance=0.01),
        Question(
            "q4", "mc", "In the lesson, the strongest penalty (C = 0.01) kept a high AUC but had the worst Brier score. Why?",
            "The slopes were shrunk almost to zero: the ranking survived, but every probability sat near the base rate.",
            concept="regularization", category="Application",
            options=["Probabilities were squashed toward the base rate", "It leaked future data",
                     "It had fewer test players", "AUC and Brier always disagree"],
            notes=["Correct: good order, poor probabilities.", "The penalty doesn't change what data is used.",
                   "Every penalty used the same test set.", "They often agree; here they measure different failures."],
            answer=0),
        Question(
            "q5", "short", "Why should you standardize rating and year-1 percentile before using a ridge penalty?",
            "The penalty depends on slope size, and slope size depends on the input's units. Without a common scale, the penalty hits inputs unequally.",
            concept="regularization", category="Application",
            keywords=[["scale", "units", "standard", "same size", "comparable"],
                      ["penal", "unfair", "unequal", "shrink", "slope"]],
            model_answer="Because the penalty works on slope size, and slope size depends on units (a rating of 0.86 vs a percentile of 75). Standardizing puts inputs on one scale so the penalty treats them equally."),
    ],
)
