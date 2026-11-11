import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import optimize as O, theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner=False)
def _setup():
    f = O.tuning_objective()
    xs = np.linspace(-4, 4, 200)
    return [f([x, 1.0]) for x in xs], xs


def visual():
    curve, xs = _setup()
    f = O.tuning_objective()
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Evaluations so far", 3, 9, 3, key="l61_n")
        show_truth = st.toggle("Show the true curve (normally unknown)", key="l61_truth")
    X = np.array([[-3.0], [1.5], [3.5]])
    y = np.array([f([x[0], 1.0]) for x in X])
    for _ in range(n - 3):
        gp = O.surrogate(X, y)
        mu, sd = gp.predict(xs[:, None], return_std=True)
        x_new = xs[np.argmax(O.expected_improvement(mu, sd, y.min()))]
        X = np.vstack([X, [[x_new]]])
        y = np.append(y, f([x_new, 1.0]))
    gp = O.surrogate(X, y)
    mu, sd = gp.predict(xs[:, None], return_std=True)
    ei = O.expected_improvement(mu, sd, y.min())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 3.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color=T.BLUE, alpha=0.18)
    a1.plot(xs, mu, color=T.BLUE, lw=2, label="surrogate (GP) guess")
    if show_truth:
        a1.plot(xs, curve, color=T.MUTED, ls="--", lw=1.2, label="true error curve")
    a1.scatter(X[:, 0], y, color=T.AMBER, s=45, zorder=5, label="evaluated")
    a1.set_ylabel("Next-season error"), a1.legend(fontsize=7)
    a2.plot(xs, ei, color=T.GREEN, lw=2), a2.set_ylabel("EI"), a2.set_xlabel("log2(prior-strength multiplier)")
    a2.axvline(xs[np.argmax(ei)], color=T.GREEN, ls=":", lw=1)
    with c2:
        show_fig(fig)
    st.caption(f"Best so far: {y.min():.3f} at {X[np.argmin(y), 0]:+.2f}. The green peak is where it will look next.")


CODE = '''import numpy as np
from academy.optimize import expected_improvement, surrogate, tuning_objective

f = tuning_objective()          # error of Level 1 shrinkage on later seasons, for a given setting
# Setting to tune: x = log2(multiplier on the prior strength). 0 = keep the fitted strength.
def error_at(x):
    return f([x, 1.0])          # 1.0 = use position-specific priors

grid = np.linspace(-4, 4, 400).reshape(-1, 1)     # where we're allowed to look
X = np.array([[-3.0], [1.5], [3.5]])              # three starting tries
y = np.array([error_at(x[0]) for x in X])

for step in range(1, 7):
    gp = surrogate(X, y)                                          # 1. fit a GP to the tries so far
    mu, sd = gp.predict(grid, return_std=True)                    # 2. its guess and uncertainty everywhere
    ei = expected_improvement(mu, sd, y.min())                    # 3. how promising each spot is
    x_next = grid[np.argmax(ei)]                                  # 4. try the most promising spot
    X = np.vstack([X, [x_next]])
    y = np.append(y, error_at(x_next[0]))
    print(f"try {step + 3}: x = {x_next[0]:+.2f}   error = {y[-1]:.3f}   best so far = {y.min():.3f}")

print(f"\\nBest setting found: multiplier 2^{X[np.argmin(y), 0]:+.2f} = {2 ** X[np.argmin(y), 0]:.2f}x the fitted strength")
'''

LESSON = Lesson(
    id="l6-01-bayes-opt-intro", level=6, order=1,
    title="Bayesian Optimization, Part 1: Tuning a Model When Every Try Is Expensive",
    subtitle="A Gaussian-process guess plus 'expected improvement' decides where to look next",
    minutes=20,
    concept_ids=["bayesian-optimization"],
    simple="""
Models have **settings** you choose: how strong the prior is, how fast to forget old games, how smooth a kernel is. To judge a setting honestly you run a full time-based validation (L3.2). On real data that can take minutes per try.

So you want to find a good setting in **as few tries as possible**.

**Bayesian optimization** does this:
1. Try a few settings.
2. Fit a **Gaussian process** (L3.5) to those results: a smooth guess of how error depends on the setting, *with uncertainty*.
3. Pick the next setting by **expected improvement**: the spot where the guess is good, **or** where it's uncertain enough that it might be great.
4. Try it, update the guess, repeat.

It balances **exploiting** (look near the best so far) and **exploring** (look where you know little), and it only makes sense when each try is expensive.
""",
    problem="Grid search and random search waste tries on settings that are obviously bad. When each try costs minutes, you want every try to be informative.",
    why="It typically finds good settings in fewer evaluations, by learning from every earlier try.",
    when_useful="""
- Expensive evaluations: full rolling-origin validation, simulations, training a big model
- A handful of continuous settings (1–10)
- A smooth relationship between settings and error
""",
    when_not="""
- Cheap evaluations: just try many settings (random search or a fine grid)
- Problems with an exact solver (like a cap-constrained lineup with a simple score, L6.2)
- Many settings (dozens+) or very jagged objectives: the surrogate can't keep up
""",
    compare="""
**Bayesian optimization vs grid search:** a grid spends tries evenly, including obviously bad regions, and gets lucky only if the best setting is on the grid. BO concentrates tries where they matter.

**BO vs random search:** random search is surprisingly strong with many tries, and trivial to run in parallel. BO wins most when tries are scarce.

**BO vs Gaussian process regression (L3.5–L3.7):** BO *uses* a GP. The new idea is the acquisition rule (expected improvement) that decides where to look.
""",
    sports="""
**Fictional scenario (simulated).** Tune one setting of the Level 1 shrinkage model: a multiplier on the prior strength (on a log₂ scale from −4 to +4). Each try runs the full season-by-season validation.

- Start with three tries spread out.
- After a few more tries, BO homes in on a multiplier close to **1×**, the strength empirical Bayes already chose (L1.5).

**Honest note:** the best setting here is "leave it as fitted". BO confirms that cheaply; it doesn't discover something better. That's a useful result too.
""",
    takeaway="Bayesian optimization fits a GP to the settings tried so far and picks the next try by expected improvement, which is worth it only when each try is expensive.",
    visual=visual,
    visual_caption="Add evaluations one at a time. The band shrinks where it has looked; the green expected-improvement curve shows where it will look next. Turn on the true curve to see how good the guess is.",
    math=[
        MathBlock(
            "Step 1: the surrogate",
            r"f(x) \approx \mathcal{GP}\;\Rightarrow\; \mu(x),\; \sigma(x)",
            """
- $f(x)$: the expensive error for setting $x$ (a full validation run)
- $\\mu(x)$: the GP's best guess of the error at $x$; $\\sigma(x)$: its uncertainty
- Fitted to the settings tried so far, exactly as in L3.5–L3.7
""",
        ),
        MathBlock(
            "Step 2: expected improvement (minimizing error)",
            r"\text{EI}(x) = (f^* - \mu(x))\,\Phi(z) + \sigma(x)\,\phi(z), \qquad z = \frac{f^* - \mu(x)}{\sigma(x)}",
            """
- $f^*$: the best (lowest) error found so far
- $\\Phi, \\phi$: the bell curve's cumulative and density functions
- First term: rewards spots predicted to beat the best (**exploit**)
- Second term: rewards uncertain spots that *might* beat it (**explore**)
""",
        ),
        MathBlock(
            "Step 3: the loop",
            r"x_{\text{next}} = \arg\max_x \text{EI}(x) \;\to\; \text{evaluate } f(x_{\text{next}}) \;\to\; \text{refit the GP}",
            """
- $\\arg\\max$: the setting with the highest expected improvement
- Stop when the budget runs out or EI becomes tiny everywhere
""",
        ),
    ],
    code=CODE,
    code_notes="Change the three starting tries to [[3.0], [3.5], [4.0]] (all on one side). How many extra tries does BO need to find the good region?",
    walkthrough=[
        ("f = tuning_objective()", "The 'expensive' function: run Level 1 shrinkage season by season and return its error for a setting."),
        ("def error_at(x): return f([x, 1.0])", "Tune one setting (the prior-strength multiplier); keep position-specific priors on."),
        ("X = np.array([[-3.0], [1.5], [3.5]])", "Three starting tries, spread out."),
        ("gp = surrogate(X, y)", "Fit a Gaussian process to the tries so far: a smooth guess with uncertainty."),
        ("mu, sd = gp.predict(grid, return_std=True)", "The guess and its uncertainty at every allowed setting."),
        ("ei = expected_improvement(mu, sd, y.min())", "Score each setting: how much it might beat the best so far."),
        ("x_next = grid[np.argmax(ei)]", "Try the most promising setting next."),
        ("y = np.append(y, error_at(x_next[0]))", "Run the expensive evaluation and add the result."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Tuning PSFAL-style models:** settings like prior strength, pooling level or a recency weight are natural BO targets when each validation run is slow.
- **Guardrail:** tune only on training seasons. Tuning on the locked holdout turns it into training data, which breaks your rule of a test set fixed in advance.
""",
    quiz=[
        Question(
            "q1", "mc", "Bayesian optimization is most useful when:",
            "Each evaluation is expensive, so you want to find good settings in as few tries as possible.",
            options=["Each try is expensive and you have few settings to tune", "Each try takes a millisecond",
                     "You have hundreds of settings", "The answer has an exact formula"],
            notes=["Correct.", "Then just try lots of settings.", "BO struggles in high dimensions.", "Then use the formula or an exact solver."],
            answer=0),
        Question(
            "q2", "mc", "Expected improvement is high where:",
            "The surrogate predicts low error (exploit) or is very uncertain (explore).",
            options=["The guess is good or very uncertain", "The setting was already tried",
                     "The guess is bad and certain", "Always at the edges"],
            notes=["Correct.", "Tried spots have little uncertainty left.", "That's the least promising place.", "Not necessarily."],
            answer=0),
        Question(
            "q3", "numeric", "Best error so far 7.00. At a candidate, μ = 7.00 and σ = 0.10. With ξ = 0, what is EI? (EI = σ·φ(0) when μ = f*, φ(0) ≈ 0.3989)",
            "z = 0, so EI = σ·φ(0) = 0.10 × 0.3989 ≈ 0.0399.", category="Application", answer=0.0399, tolerance=0.001),
        Question(
            "q4", "short", "BO says the best prior strength is about the one empirical Bayes already fitted. Is that a wasted effort?",
            "No: it's a cheap confirmation that the default is near-optimal, which is a useful finding.",
            category="Application",
            keywords=[["no", "not wasted", "useful", "valuable", "worth"],
                      ["confirm", "check", "already", "default", "near", "optimal", "evidence"]],
            model_answer="No. It's a cheap confirmation that the fitted setting is already near the best, which is useful evidence that no extra tuning is needed."),
        Question(
            "q5", "mc", "Why tune settings only on training seasons?",
            "Tuning on the test seasons leaks them into the model, so the final test is no longer fair.",
            options=["Tuning on the test set leaks it, so the final test isn't fair", "Training seasons are bigger",
                     "BO can't read test data", "It's faster"],
            notes=["Correct.", "Size isn't the reason.", "It could; it shouldn't.", "Speed isn't the reason."],
            answer=0),
    ],
)
