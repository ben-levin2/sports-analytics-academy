import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner=False)
def _data():
    from ..data import make_draft_classes
    return make_draft_classes()


def visual():
    from ..smalldata import Design, fit_logistic, fit_monotone_gam, gam_curve
    df = _data()
    c1, c2 = st.columns(2)
    last = c1.select_slider("Train on classes 2008 to", [2011, 2013, 2015, 2017, 2021], value=2013, key="l39_last")
    lam = c2.select_slider("Wiggle penalty", [0.01, 0.05, 0.2, 1.0, 5.0], value=0.05, key="l39_lam")
    train = df[df["class_year"] <= last]
    d = Design(train)
    y = train["drafted"].to_numpy()
    grid = np.linspace(-2, 2, 81)
    y1 = d.my + d.sy * grid
    truth = 2.4 / (1 + np.exp(-(y1 - 75) / 7))
    truth = truth - 2.4 / (1 + np.exp(-(d.my - 75) / 7))
    line = fit_logistic(d.linear(train), y, *d.priors(False))["beta"][2] * grid
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(grid, truth, color=T.NEUTRAL_LIGHT, lw=2, ls="--", label="Truth (simulated)")
    ax.plot(grid, line, color=T.FAINT, lw=1.6, label="Straight line")
    ax.plot(grid, gam_curve(fit_monotone_gam(d, train, y, lam=lam, monotone=False), "year1", grid),
            color=T.ACCENT, lw=2, label="Free curve")
    ax.plot(grid, gam_curve(fit_monotone_gam(d, train, y, lam=lam, monotone=True), "year1", grid),
            color=T.PRIMARY, lw=2.4, label="Monotonic curve")
    ax.set_xlabel("Year-1 production (SDs from the average recruit)"), ax.set_ylabel("Effect on draft log-odds")
    show_fig(fig)
    st.caption(f"{len(train)} training recruits, {int(y.sum())} drafted.")


CODE = '''import numpy as np
from sklearn.metrics import roc_auc_score
from academy.data import make_draft_classes
from academy.smalldata import Design, fit_logistic, predict_logistic, fit_monotone_gam, predict_gam, gam_curve

df = make_draft_classes()                        # SIMULATED program, 22 recruits a class
train = df[df["class_year"] <= 2013]             # 132 finished recruits
test = df[(df["class_year"] >= 2014) & (df["class_year"] <= 2021)]
d = Design(train)
y = train["drafted"].to_numpy()

line = fit_logistic(d.linear(train), y, *d.priors(False))           # straight lines
free = fit_monotone_gam(d, train, y, lam=0.05, monotone=False)    # curves, any direction
mono = fit_monotone_gam(d, train, y, lam=0.05, monotone=True)     # curves that can only rise

def score(p):
    return f"AUC {roc_auc_score(test['drafted'], p):.3f}  Brier {np.mean((p - test['drafted']) ** 2):.4f}"

print("Straight-line logistic: ", score(predict_logistic(line, d.linear(test))))
print("Free smooth curves:     ", score(predict_gam(free, test)))
print("Monotonic smooth curves:", score(predict_gam(mono, test)))

# What does each model say about year-1 production? (log-odds vs an average producer)
grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])     # SDs of year-1 percentile
for name, m in [("free", free), ("monotonic", mono)]:
    print(f"{name:>9} year-1 curve at -2,-1,0,+1,+2 SD:", np.round(gam_curve(m, "year1", grid), 2))

# One test set can flatter either model. Repeat on 8 simulated programs.
gaps = []
for seed in range(1, 9):
    other = make_draft_classes(seed=seed)
    tr, te = other[other["class_year"] <= 2013], other[(other["class_year"] >= 2014) & (other["class_year"] <= 2021)]
    dd, yy = Design(tr), tr["drafted"].to_numpy()
    a_free = roc_auc_score(te["drafted"], predict_gam(fit_monotone_gam(dd, tr, yy, lam=0.05, monotone=False), te))
    a_mono = roc_auc_score(te["drafted"], predict_gam(fit_monotone_gam(dd, tr, yy, lam=0.05, monotone=True), te))
    gaps.append(a_mono - a_free)
print(f"8 programs: monotonic minus free AUC, average {np.mean(gaps):+.3f}, range {min(gaps):+.3f} to {max(gaps):+.3f}")
'''

LESSON = Lesson(
    id="l3-09-monotonic-gam", level=3, order=9,
    title="GAMs and Monotonic Constraints: Curves That Only Go One Way",
    subtitle="Smooth, directional effects without trees",
    minutes=25,
    concept_ids=["monotonic-gam"],
    simple="""
A straight line says "each extra bit of production adds the same amount". Real effects bend: going from the 50th to the 70th percentile in year 1 may barely matter for the draft, while going from the 80th to the 95th matters a lot.

A **generalized additive model (GAM)** replaces each straight line with a **smooth curve**, then adds the curves up. It stays easy to read: one picture per input.

The catch with small data is that free curves wiggle. They can even say a *worse* season makes a player *more* draftable, purely from luck. A **monotonic constraint** fixes that: you tell the model "this curve may bend, but it can only go up." It keeps the flexibility and drops the nonsense.
""",
    problem="Straight lines miss real bends; free curves overfit small data into shapes nobody believes. You want flexible curves that still respect a direction you're sure of.",
    why="Monotonic curves are as readable as a line, as flexible as a curve, and much harder to fool with noise. They are also easy to defend to coaches: 'more production never lowers the estimate.'",
    when_useful="""
- Inputs with a known direction but unknown shape (rating, production percentiles, athletic testing)
- Small or noisy data where free curves wiggle
- Models you must explain: one curve per input
""",
    when_not="""
- When the true effect really does rise then fall (age, body weight for some positions). A one-way constraint would hide it
- When effects depend on each other strongly (interactions). Additive curves don't capture "rating matters only for QBs" unless you build that in
- When you have so much data that a free curve is already stable
""",
    compare="""
**Straight-line logistic:** one slope per input; can't bend. **Free GAM (penalized splines):** bends anywhere, wiggles with small data. **Monotonic GAM:** bends, but only in one direction. **Tree models (gradient boosting):** bend and interact automatically, but are step-shaped, hard to read and data-hungry; with ~130 recruits they overfit badly, so this academy doesn't use them here. **Gaussian processes (L3.7):** smooth curves with uncertainty, but no direction built in.
""",
    sports="""
**Fictional scenario (simulated).** Same program, train on 2008–2013 (132 recruits), test on 2014–2021. The hidden truth is S-shaped: year-1 production barely matters until about the 75th percentile, then matters a lot.

| Model | Test AUC | Test Brier |
|---|---|---|
| Straight-line logistic | 0.822 | 0.0718 |
| Free smooth curves | **0.884** | **0.0575** |
| Monotonic smooth curves | 0.867 | 0.0598 |

On this one test set, the **free** curves scored best. Now look at what they learned about year 1 (log-odds vs an average producer, at −2, −1, 0, +1, +2 SD):

- **Free:** 3.52, 1.29, 0, 0.68, 3.46. It says the *worst* year-1 producers are as draftable as the best. That's a U-shape built from a handful of lucky players.
- **Monotonic:** −0.16, −0.14, 0, 0.72, 2.55. Flat at the bottom, rising steeply at the top. That's the true shape.

**Over 8 simulated programs**, the monotonic version averaged **+0.013 AUC** over the free one (range −0.019 to +0.067). The constraint is mainly about trust and stability; one good test score is not proof of a sensible model.
""",
    takeaway="Use smooth curves where effects bend, and lock their direction when you're sure of it. A curve that wins one test but says something absurd is a warning, not a win.",
    visual=visual,
    visual_caption="Dashed: the true year-1 effect. With a small wiggle penalty and few classes, the free curve (amber) bends both ways; the monotonic curve (blue) can't. Raise the penalty or add classes and they converge.",
    math=[
        MathBlock(
            "Step 1: an additive model",
            r"\log\frac{p_i}{1-p_i} = \beta_0 + f_1\big(z^{\text{rating}}_i\big) + f_2\big(z^{\text{year1}}_i\big) + \sum_k \gamma_k\, \text{pos}_{ik}",
            """
- $p_i$: probability recruit $i$ is drafted; the left side is his log-odds
- $f_1$, $f_2$: smooth curves, one per input, learned from data (a straight line is the special case $f(z) = \\beta z$)
- "Additive": the curves are simply added. You can draw each one on its own
""",
        ),
        MathBlock(
            "Step 2: build each curve from smooth ramps",
            r"f(z) = a\,z + \sum_{j=1}^{3} w_j\, r_j(z), \qquad r_j(z) = s \log\!\Big(1 + e^{(z - \kappa_j)/s}\Big)",
            """
- $r_j(z)$: a smooth **ramp**: flat to the left of its knot $\\kappa_j$ (kappa), then rising with slope 1
- $\\kappa_j$: knots at −1, 0 and +1 SD; $s = 0.5$ sets how gently each ramp bends
- $a$: the base slope; $w_j$: how much extra slope kicks in after knot $j$
- Adding ramps with different weights gives a curve that can bend at each knot
""",
        ),
        MathBlock(
            "Step 3: the monotonic constraint and the wiggle penalty",
            r"a \ge 0,\; w_j \ge 0 \;\;\Rightarrow\;\; f'(z) = a + \sum_j w_j\, \sigma\!\Big(\frac{z-\kappa_j}{s}\Big) \ge 0, \qquad \text{maximize } \log L - \frac{\lambda}{2}\sum_j w_j^2",
            """
- $f'(z)$: the curve's slope; $\\sigma$ is the logistic function, always between 0 and 1
- If every weight is 0 or more, every term in the slope is 0 or more, so the curve **can never go down**
- $\\lambda$: the wiggle penalty, a ridge penalty on the ramp weights (L1.7); larger = closer to a straight line
- Fitting with "weights ≥ 0" is a bounded optimization (scipy's L-BFGS-B)
""",
            "Drop the bounds and you get an ordinary penalized spline: the 'free' model in this lesson.",
        ),
    ],
    code=CODE,
    code_notes="Change lam=0.05 to lam=2.0 in all three fits. The free curve calms down because the penalty does part of the constraint's job.",
    walkthrough=[
        ("line = fit_logistic(d.linear(train), y, *d.priors(False))", "The straight-line baseline: one slope per input, no prior."),
        ("fit_monotone_gam(d, train, y, lam=0.05, monotone=False)", "Smooth curves built from ramps, with a small wiggle penalty and no direction rule."),
        ("fit_monotone_gam(d, train, y, lam=0.05, monotone=True)", "The same curves, but every ramp weight must be 0 or more, so each curve can only rise."),
        ("predict_gam(free, test)", "Add up the curves for each test recruit and turn the log-odds into a probability."),
        ("gam_curve(m, 'year1', grid)", "Read off the fitted year-1 curve at a few points, relative to an average producer."),
        ("for seed in range(1, 9):", "One test set can flatter either model by luck, so repeat the comparison on 8 other simulated programs."),
        ("gaps.append(a_mono - a_free)", "Positive means the monotonic version ranked test recruits better on that program."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Model 8's inputs** (rating and early production) both have a known direction. A monotonic curve for each is a natural way to let their effects bend without risking a backwards wiggle on a small sample.
- **Explaining to staff:** one curve per input, guaranteed to rise, is easier to defend than a tree ensemble or a free spline with a dip in the middle.
- **Your rule still applies:** a GAM goes nowhere near your site unless it beats the simple baseline on the pre-set test. The P3C project runs exactly that test.
""",
    quiz=[
        Question(
            "q1", "mc", "What does 'additive' mean in a generalized additive model?",
            "Each input gets its own curve and the curves are added on the log-odds scale.",
            concept="monotonic-gam",
            options=["One curve per input, added together", "Inputs multiply each other", "Trees are added one by one", "Rows are added to the data"],
            notes=["Correct.", "That's an interaction, which a plain GAM doesn't include.", "That's boosting, not a GAM.", "Unrelated."],
            answer=0),
        Question(
            "q2", "mc", "How does the model in this lesson guarantee a curve never goes down?",
            "The curve's slope is a sum of non-negative weights times non-negative ramp slopes, so it can't be negative.",
            concept="monotonic-gam",
            options=["All ramp weights and the base slope are kept at 0 or more", "It uses a bigger penalty",
                     "It sorts the data first", "It removes the intercept"],
            notes=["Correct.", "A penalty smooths but doesn't forbid a dip.", "Sorting doesn't change the fit.", "The intercept only shifts the curve."],
            answer=0),
        Question(
            "q3", "mc", "The free curve beat the monotonic one on the lesson's test set but said the worst year-1 producers are as draftable as the best. What's the right conclusion?",
            "One test result can be luck; a model whose shape contradicts known football is not trustworthy. Check across repeats.",
            concept="monotonic-gam", category="Application",
            options=["Distrust it and check over many samples", "Use the free curve; it scored higher",
                     "Use a straight line instead", "Collect a new test set until monotonic wins"],
            notes=["Correct: across 8 programs the monotonic version averaged slightly higher.", "One score isn't proof.",
                   "A line misses the real bend.", "Testing until you like the answer is a form of leakage."],
            answer=0),
        Question(
            "q4", "mc", "For which input would a one-way (monotonic) constraint be a mistake?",
            "If the true effect rises then falls, forcing one direction hides that peak.",
            concept="monotonic-gam", category="Application",
            options=["An effect that rises then falls, like age at signing", "Recruiting rating", "Year-1 production percentile", "40-yard dash speed for a DB"],
            notes=["Correct.", "Direction is clear.", "Direction is clear.", "Faster is better for a DB; direction is clear (in the 'faster' sense)."],
            answer=0),
        Question(
            "q5", "short", "Why does this academy use monotonic GAMs here instead of a tree model like gradient boosting?",
            "With ~130 recruits, trees overfit and give step-shaped, hard-to-read effects; GAM curves are smooth, readable and can be forced to one direction.",
            concept="monotonic-gam", category="Application",
            keywords=[["small", "little data", "few", "overfit", "130"],
                      ["smooth", "read", "explain", "direction", "monoton", "step"]],
            model_answer="Because with a small sample trees overfit and produce step-shaped, hard-to-explain effects. A monotonic GAM gives one smooth, readable curve per input that can't go the wrong way."),
    ],
)
