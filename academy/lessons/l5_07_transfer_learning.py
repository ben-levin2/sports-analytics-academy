import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner=False)
def _curves(slope_sd):
    from sklearn.metrics import roc_auc_score
    from ..data import make_draft_classes
    from ..smalldata import Design, fit_logistic, predict_logistic
    source = pd.concat([make_draft_classes(seed=500 + s) for s in range(10)])
    source = source[source["class_year"] <= 2017]
    ours = make_draft_classes(seed=29, shift=0.6)
    test = ours[(ours["class_year"] >= 2014) & (ours["class_year"] <= 2021)]
    d = Design(source)
    src = fit_logistic(d.linear(source), source["drafted"], *d.priors(True))
    p_src = predict_logistic(src, d.linear(test))
    rows = []
    for last in range(2009, 2014):
        train = ours[ours["class_year"] <= last]
        X, Xt = d.linear(train), d.linear(test)
        scratch = predict_logistic(fit_logistic(X, train["drafted"], *d.priors(False)), Xt)
        sd = np.r_[1.5, np.full(len(src["beta"]) - 1, slope_sd)]
        tuned = predict_logistic(fit_logistic(X, train["drafted"], src["beta"], sd), Xt, moderated=True)
        for name, p in (("From scratch", scratch), ("Source as is", p_src), ("Transfer", tuned)):
            rows.append({"recruits": len(train), "model": name, "auc": roc_auc_score(test["drafted"], p),
                         "gap": p.mean() - test["drafted"].mean()})
    return pd.DataFrame(rows)


def visual():
    sd = st.select_slider("How tightly to hold the source's slopes (prior SD)", [0.1, 0.3, 1.0, 3.0], value=0.3, key="l57_sd")
    r = _curves(sd)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    colors = {"From scratch": T.ACCENT, "Source as is": T.FAINT, "Transfer": T.PRIMARY}
    for name, g in r.groupby("model"):
        axes[0].plot(g["recruits"], g["auc"], marker="o", color=colors[name], lw=2, label=name)
        axes[1].plot(g["recruits"], g["gap"], marker="o", color=colors[name], lw=2, label=name)
    axes[0].set_xlabel("Our training recruits"), axes[0].set_ylabel("Test AUC"), axes[0].legend(fontsize=7)
    axes[1].axhline(0, color=T.BORDER, lw=1)
    axes[1].set_xlabel("Our training recruits"), axes[1].set_ylabel("Mean forecast minus actual")
    show_fig(fig)


CODE = '''import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from academy.data import make_draft_classes
from academy.smalldata import Design, PRIOR, fit_logistic, predict_logistic

# SOURCE: a large related problem. 10 other programs, finished classes 2008-2017.
source = pd.concat([make_draft_classes(seed=500 + s) for s in range(10)])
source = source[source["class_year"] <= 2017]
# TARGET: our program. It develops players better (draft odds shifted up), and we only have 3 finished classes.
ours = make_draft_classes(seed=29, shift=0.6)
train = ours[ours["class_year"] <= 2010]
test = ours[(ours["class_year"] >= 2014) & (ours["class_year"] <= 2021)]
print(f"Source: {len(source)} recruits ({int(source['drafted'].sum())} drafted). Target: {len(train)} ({int(train['drafted'].sum())} drafted)")

d = Design(source)                                     # one shared scale, set by the big source
Xs, X, Xt = d.linear(source), d.linear(train), d.linear(test)
yt = test["drafted"]

def report(name, p):
    print(f"{name:<32} AUC {roc_auc_score(yt, p):.3f}  mean forecast {p.mean():.3f}  (actual {yt.mean():.3f})")

scratch = fit_logistic(X, train["drafted"], *d.priors(informative=False))
report("1. From scratch (target only)", predict_logistic(scratch, Xt))

src = fit_logistic(Xs, source["drafted"], *d.priors(informative=True))
report("2. Source model, used as is", predict_logistic(src, Xt))

# 3. Transfer: the source's slopes become the PRIOR for our program; the intercept is re-learned
mean = src["beta"].copy()
sd = np.r_[1.5, np.full(len(mean) - 1, 0.3)]           # loose on the base rate, tight on the slopes
tuned = fit_logistic(X, train["drafted"], mean, sd)
report("3. Transfer (source as prior)", predict_logistic(tuned, Xt, moderated=True))

# 4. Pool everything as if it were one program
both = pd.concat([source, train])
pooled = fit_logistic(d.linear(both), both["drafted"], *d.priors(informative=True))
report("4. Pool source + target", predict_logistic(pooled, Xt))
print(f"Base rate (intercept): source {src['beta'][0]:.2f}, transfer {tuned['beta'][0]:.2f} log-odds")
'''

LESSON = Lesson(
    id="l5-07-transfer-learning", level=5, order=7,
    title="Transfer Learning: Borrow From a Big Problem, Adapt to a Small One",
    subtitle="Source models as priors, fine-tuning, and when to re-learn the base rate",
    minutes=25,
    concept_ids=["transfer-learning"],
    simple="""
Your program has 66 finished recruits. The whole conference has thousands. The relationships are probably similar (higher-rated recruits get drafted more everywhere) even if the details differ (your staff may develop players better than average).

**Transfer learning** means: learn on the big related problem (the **source**), then adapt to your small one (the **target**).

The simplest honest version uses what you learned in Level 1: take the source model's slopes and make them the **prior** for your own model. Your few recruits can then adjust them, and especially re-learn the things that are truly different, like the overall draft rate.
""",
    problem="Small targets can't support a model trained from scratch, but a model copied from elsewhere may be off for your setting, usually in its base rate.",
    why="It uses large data for the hard part (the shape of the relationships) and small data for the part that's local (the level). That's often the best you can do with one program's history.",
    when_useful="""
- One program, many similar programs (conference or national data)
- A new position group, scheme or league with little history
- When the inputs mean the same thing in source and target
""",
    when_not="""
- When the source differs in what matters (a different sport, or inputs measured differently): transfer can hurt ("negative transfer")
- When the target is big enough to learn on its own
- When the source includes your test period. That's leakage, exactly as with any prior
""",
    compare="""
**From scratch:** only your data; noisy when small. **Source as is:** borrows everything, including mistakes about your base rate. **Pooling (concatenate the data):** the source swamps your program. **Transfer as a prior / fine-tuning:** start from the source, let your data pull it where it disagrees. **Hierarchical model (L5.4):** learns how much programs differ from each other, a more formal cousin of this idea. In deep learning, transfer means reusing a network's layers and retraining the last ones; the logic is the same.
""",
    sports="""
**Fictional scenario (simulated).** Source: 10 other programs, **2,200** finished recruits (267 drafted). Target: our program, only **66** finished recruits (12 drafted). Our staff develops players better, so our draft odds are higher than the conference's, but rating and production mean the same thing. Test on our 2014–2021 classes (actual draft rate **0.153**).

| Approach | Test AUC | Mean forecast |
|---|---|---|
| From scratch (our 66 only) | 0.573 | 0.131 |
| Source model as is | 0.809 | 0.101 |
| **Transfer (source as prior)** | **0.809** | **0.156** |
| Pool source + ours | 0.808 | 0.102 |

- From scratch can barely rank players: 66 recruits is too few.
- The source ranks well but **under-forecasts** our draft rate (0.101 vs 0.153): it doesn't know our staff is better.
- Pooling has the same problem: 2,200 source recruits drown out our 66.
- Transfer keeps the source's ranking *and* re-learns our base rate (intercept −2.90 → −2.22).
""",
    takeaway="Borrow the shape from the big problem; re-learn what's local. The source's answer is a prior, not a replacement for your own data.",
    visual=visual,
    visual_caption="Left: ranking (AUC) as our own data grows. Right: how far each model's average forecast is from our real draft rate (0 = right on). Tighten the prior and transfer behaves like the source; loosen it and it behaves like training from scratch.",
    math=[
        MathBlock(
            "Step 1: learn on the source",
            r"\hat\beta^{\,S} = \arg\max_\beta \; \sum_{i \in S} \log P(y_i \mid x_i, \beta) + \log p(\beta)",
            """
- $S$: the source data (other programs, finished classes before our test period)
- $\\hat\\beta^{\\,S}$: the source's fitted intercept and slopes, from a Bayesian logistic regression (L1.8)
""",
        ),
        MathBlock(
            "Step 2: use it as the prior for the target",
            r"\beta_0 \sim \text{Normal}(\hat\beta^{\,S}_0,\; 1.5^2), \qquad \beta_j \sim \text{Normal}(\hat\beta^{\,S}_j,\; 0.3^2) \;\; (j \ge 1)",
            """
- $\\beta_0$: our intercept (our base draft rate). A **wide** prior (SD 1.5) lets our data re-learn it
- $\\beta_j$: our slopes. A **tight** prior (SD 0.3) keeps them near the source unless our data strongly disagrees
- Choosing which parts to hold tight and which to re-learn is the main design decision in transfer
""",
        ),
        MathBlock(
            "Step 3: fit on the target",
            r"\hat\beta^{\,T} = \arg\max_\beta \; \sum_{i \in T} \log P(y_i \mid x_i, \beta) \; - \; \sum_j \frac{(\beta_j - \hat\beta^{\,S}_j)^2}{2 s_j^2}",
            """
- $T$: our program's training recruits
- The second term is a ridge penalty (L1.7) that pulls toward the **source's** values instead of toward zero
- $s_j$: the prior SDs from Step 2. $s_j \\to 0$ copies the source; $s_j \\to \\infty$ trains from scratch
""",
        ),
    ],
    code=CODE,
    code_notes="Set shift=0.0 for our program (it's just like the others). Now 'source as is' is also well calibrated, and transfer's advantage shrinks to almost nothing. Transfer helps most when the target differs in a way it can re-learn.",
    walkthrough=[
        ("source = pd.concat([make_draft_classes(seed=500 + s) for s in range(10)])", "The big related problem: 10 other simulated programs."),
        ("source = source[source['class_year'] <= 2017]", "Only source classes finished before our test period, so the source can't leak our test outcomes."),
        ("ours = make_draft_classes(seed=29, shift=0.6)", "Our program: same meaning of inputs, but better player development (higher draft odds)."),
        ("d = Design(source)", "One shared scale for the inputs, set by the large source, so slopes mean the same thing in both."),
        ("scratch = fit_logistic(X, train['drafted'], *d.priors(informative=False))", "Approach 1: our 66 recruits alone, no prior."),
        ("src = fit_logistic(Xs, source['drafted'], *d.priors(informative=True))", "Learn on the source with the usual Level 1 priors."),
        ("sd = np.r_[1.5, np.full(len(mean) - 1, 0.3)]", "The transfer prior: loose on the intercept (re-learn our base rate), tight on the slopes (keep the source's shape)."),
        ("tuned = fit_logistic(X, train['drafted'], mean, sd)", "Approach 3: fine-tune on our recruits, starting from the source."),
        ("pooled = fit_logistic(d.linear(both), both['drafted'], ...)", "Approach 4: pretend it's all one program. The source outnumbers us 33 to 1."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Your roster data covers ~67 teams.** For one team's question (e.g. Penn State's own draft outcomes), the other 66 are a natural source; transfer lets Penn State's own history re-learn the base rate.
- **Model 7 (draft-outlook candidate, being redesigned):** draft data for one program is thin. Learning slopes on all programs and re-learning the program's level is one design worth testing, not assuming.
- **Leakage rule:** the source must stop before the target's test period, or the prior already knows the answers. Your rule (beat a simple baseline on a pre-set test) applies unchanged.
""",
    quiz=[
        Question(
            "q1", "mc", "In transfer learning, what are the 'source' and the 'target'?",
            "The source is the large related problem you learn from first; the target is the small problem you care about.",
            concept="transfer-learning",
            options=["Source = big related data; target = your small problem", "Source = test set; target = training set",
                     "Source = priors; target = posteriors only", "They're the same dataset split in two"],
            notes=["Correct.", "Test data must never be the source.", "Priors are one way to transfer, not the definition.", "The point is that they're different problems."],
            answer=0),
        Question(
            "q2", "mc", "In the lesson, why did the source model used as is under-forecast our draft rate?",
            "It learned the conference's base rate; our program develops players better, so its true rate is higher.",
            concept="transfer-learning", category="Application",
            options=["Our base rate differs from the source's", "The source had too little data",
                     "The slopes had the wrong sign", "AUC was too low"],
            notes=["Correct: 0.101 vs 0.153.", "The source had 2,200 recruits.", "Its ranking was good (AUC 0.809).", "Its AUC was fine; its level was off."],
            answer=0),
        Question(
            "q3", "mc", "You use a source model's slopes as a prior with SD 0.3. What happens as you make that SD very large?",
            "A very wide prior carries almost no information, so the fit approaches training from scratch.",
            concept="transfer-learning",
            options=["It becomes training from scratch", "It becomes the source model as is", "It becomes pooling", "Nothing changes"],
            notes=["Correct.", "That's what a very SMALL SD does.", "Pooling is a different approach.", "The prior's width is the key dial."],
            answer=0),
        Question(
            "q4", "short", "When can transfer learning make things worse?",
            "When the source differs from the target in what matters (different meaning of inputs or different relationships): negative transfer.",
            concept="transfer-learning", category="Application",
            keywords=[["differ", "different", "unrelated", "not similar", "mismatch", "negative"],
                      ["relationship", "input", "meaning", "setting", "source", "slope"]],
            model_answer="When the source is different in the ways that matter, for example inputs measured differently or relationships that don't hold for your program. Then borrowing pulls you toward the wrong answer (negative transfer)."),
        Question(
            "q5", "mc", "Your source data includes seasons from your test period. What's the problem?",
            "The prior then contains information from the test period, so the test is no longer fair: leakage.",
            concept="transfer-learning", category="Application",
            options=["Leakage: the prior already knows the test outcomes", "None, more data is always better",
                     "The model will be too simple", "AUC can't be computed"],
            notes=["Correct.", "Not when it's future data.", "Complexity isn't the issue.", "It can, but it will be too optimistic."],
            answer=0),
    ],
)
