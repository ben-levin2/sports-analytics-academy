import streamlit as st
from scipy import stats

from ..models import Lesson, MathBlock, Question
from ..ui import show_fig
from ._viz import sequential_fig

WEEKS = [(3, 28), (5, 35), (2, 30), (6, 32)]   # (pressures, pass-rush snaps) per week, SIMULATED
PRIOR = (10, 90)


def visual():
    upto = st.slider("Weeks of data included", 0, 4, 4, key="l14_weeks")
    k = sum(w[0] for w in WEEKS[:upto])
    n = sum(w[1] for w in WEEKS[:upto])
    post = stats.beta(PRIOR[0] + k, PRIOR[1] + n - k)
    show_fig(sequential_fig(PRIOR[0], PRIOR[1], WEEKS, upto))
    lo, hi = post.ppf([0.025, 0.975])
    m1, m2, m3 = st.columns(3)
    m1.metric("Snaps observed", n)
    m2.metric("Posterior mean", f"{post.mean():.1%}")
    m3.metric("95% credible interval", f"{lo:.1%} – {hi:.1%}")


CODE = '''from scipy import stats

# --- SIMULATED: a fictional defensive lineman. Weekly (pressures, pass-rush snaps) ---
weeks = [(3, 28), (5, 35), (2, 30), (6, 32)]

prior_mean, prior_strength = 0.10, 100            # "DL average ~10% pressure rate"
a, b = prior_mean * prior_strength, (1 - prior_mean) * prior_strength

# Update one week at a time: today's posterior is tomorrow's prior
for i, (k, n) in enumerate(weeks, start=1):
    a, b = a + k, b + (n - k)
    post = stats.beta(a, b)
    lo, hi = post.ppf([0.025, 0.975])
    print(f"After week {i}: mean {post.mean():.3f}   95% CI ({lo:.3f}, {hi:.3f})")

# The same answer in one shot using all the data at once
K = sum(k for k, n in weeks)
N = sum(n for k, n in weeks)
one_shot = stats.beta(prior_mean * prior_strength + K,
                      (1 - prior_mean) * prior_strength + N - K)
print(f"\\nAll-at-once mean: {one_shot.mean():.3f}   (matches the sequential result)")

# Decisions, not just numbers: how likely is he above a 12% starter-level rate?
p_above = one_shot.sf(0.12)
print(f"P(true pressure rate > 12%) = {p_above:.2f}")
print(f"Raw rate for comparison:      {K / N:.3f}")
'''

LESSON = Lesson(
    id="l1-04-posterior-distributions", level=1, order=4,
    title="Posterior Distributions: Putting It All Together",
    subtitle="Sequential updating, credible intervals and decisions",
    minutes=20,
    concept_ids=["posterior-distributions"],
    simple="""
The **posterior** is your belief *after* seeing the data. It's not one number. It's a whole curve showing how plausible each value is.

Three superpowers follow from having a curve:

1. **Updating is repeatable.** Today's posterior becomes tomorrow's prior. Learn a bit each week.
2. **Uncertainty is visible.** A narrow curve means we know a lot. A wide curve means we don't.
3. **Decisions get probabilities.** Instead of "he's about 11.6%", you can say "there's a 40% chance he's above a starter's 12%."
""",
    problem="A single estimate hides how sure you are. Decision makers (recruiting boards, coaches) need to know the *risk* behind a number.",
    why="Posterior curves let you compare players fairly even when one has 600 snaps and the other 40, and let you ask direct questions like 'probability he beats the position average'.",
    when_useful="""
- Players with different amounts of data
- Ongoing evaluation that updates weekly or yearly
- Decisions that depend on thresholds or risk
""",
    when_not="""
- This closed-form shortcut only works for simple models. Complex models need MCMC (Level 5)
- The posterior is only as good as the prior and likelihood. A wrong model gives confident nonsense
- If the decision only needs a rough ranking, a simple shrunk estimate may be enough
""",
    compare="""
**Three kinds of interval (often confused):**

| Interval | What it is about | Plain-English reading |
|---|---|---|
| **Credible interval** (Bayesian) | the true rate θ | "Given my model and data, 95% chance θ is in here." |
| **Confidence interval** (frequentist) | the *procedure* | "If I repeated this season many times, 95% of intervals built this way would contain θ." |
| **Prediction interval** | a *future observation* | "His pressure rate *next game* will land in here 95% of the time." Wider, because next game adds luck on top of not knowing θ. |

**Bayesian regression vs ordinary regression:** ordinary (least-squares) regression gives one best-fit slope and a confidence interval. Bayesian regression puts a prior on the slope and returns a posterior for it. With plenty of data they agree closely. With little data, or many predictors, the prior stabilizes the estimates (ridge regression is the same idea in disguise). Bayesian is not automatically better: it adds assumptions you must defend.
""",

    sports="""
**Fictional scenario (simulated).** Track a defensive lineman's **pressure rate** week by week. The prior is Beta(10, 90), meaning DLs average about 10% and we treat that as worth 100 snaps.

Weekly (pressures, snaps): (3, 28), (5, 35), (2, 30), (6, 32).

After four weeks: 16 pressures in 125 snaps (raw 12.8%). The posterior is Beta(26, 199), with mean 26/225 ≈ **11.6%**. Pulled toward the 10% prior, but also visibly tighter than where we started.

Updating week by week gives the *same* final answer as updating once with all four weeks. Order doesn't matter in this model, only the total evidence.
""",
    takeaway="The posterior is a full picture of what you know: the best guess, how sure you are, and the odds for any decision threshold.",
    visual=visual,
    visual_caption="Move the slider through the four weeks. The curve narrows as evidence accumulates and the mean drifts from the prior toward the data.",
    math=[
        MathBlock(
            "The posterior for a rate",
            r"\theta \mid k,n \;\sim\; \text{Beta}(\alpha+k,\;\beta+n-k)",
            """
- $\\alpha,\\beta$: the prior's pretend successes and failures
- $k$: pressures (successes), $n$: snaps (opportunities)
- Nothing else is needed: the whole posterior is two numbers
""",
        ),
        MathBlock(
            "Sequential updating is the same as batch updating",
            r"\text{Beta}(\alpha,\beta)\xrightarrow{(k_1,n_1)}\text{Beta}(\alpha+k_1,\beta+n_1-k_1)\xrightarrow{(k_2,n_2)}\text{Beta}(\alpha+k_1+k_2,\;\beta+(n_1+n_2)-(k_1+k_2))",
            """
- $(k_1,n_1)$, $(k_2,n_2)$: successes and opportunities in week 1 and week 2
- The final parameters only depend on the **totals**
""",
        ),
        MathBlock(
            "Credible interval and threshold probability",
            r"P(\theta_{lo} < \theta < \theta_{hi} \mid D) = 0.95,\qquad P(\theta > c \mid D) = 1 - F_{\text{posterior}}(c)",
            """
- $\\theta_{lo},\\theta_{hi}$: the 2.5% and 97.5% quantiles of the posterior
- $c$: a decision threshold (e.g., a 12% starter-level pressure rate)
- $F_{\\text{posterior}}$: the posterior's cumulative distribution function
""",
        ),
    ],
    code=CODE,
    code_notes="Change the threshold in `one_shot.sf(0.12)` to 0.10 and then 0.15. Notice how the probability moves. That's the decision-ready output a single number can't give.",
    walkthrough=[
        ("weeks = [(3, 28), (5, 35), (2, 30), (6, 32)]", "Four weeks of data: (pressures, pass-rush snaps)."),
        ("a, b = prior_mean * prior_strength, ...", "Start from the prior Beta(10, 90): about 10%, worth 100 snaps."),
        ("for i, (k, n) in enumerate(weeks, start=1):", "Loop over the weeks in order, one update per week."),
        ("a, b = a + k, b + (n - k)", "This week's posterior becomes next week's prior. That is sequential updating."),
        ("post.ppf([0.025, 0.975])", "A 95% credible interval each week. Watch it narrow as snaps accumulate."),
        ("K = sum(k for k, n in weeks)", "Total pressures across all weeks..."),
        ("one_shot = stats.beta(...)", "...and one update with the totals. It matches the week-by-week answer exactly."),
        ("p_above = one_shot.sf(0.12)", "sf = 'survival function' = probability the true rate is ABOVE 12%. This is the decision-ready number."),
    ],
    psfal="""
- Build a **player report** that shows the posterior mean, a 95% credible interval and *P(above position average)* instead of a bare rate.
- **Rank by probability of exceeding a threshold**, or by a lower credible bound, when you want to avoid over-trusting small samples.
- Update the same player model after every game week. The sequential structure fits a season naturally.
- When the model gets too complex for closed forms (e.g., hierarchical player models), move to MCMC in Level 5.
""",
    quiz=[
        Question(
            "q1", "mc", "You update a posterior one week at a time, then repeat using all weeks at once (same prior, Beta-Binomial). What happens?",
            "In this model the posterior depends only on total successes and total opportunities.",
            options=["The results are identical", "Weekly updating gives a narrower posterior",
                     "All-at-once gives a higher mean", "The order of weeks changes the result"],
            notes=["Correct.", "Both use the same total evidence.", "The totals are the same, so the means are too.", "Order doesn't matter here, only totals."],
            answer=0),
        Question(
            "q2", "numeric", "Prior Beta(10, 90). A DL records 16 pressures in 125 snaps. What is the posterior mean? (decimal or %)",
            "(10 + 16) / (100 + 125) = 26 / 225 ≈ 0.1156.",
            category="Application", answer=0.1156, tolerance=0.002),
        Question(
            "q3", "mc", "A 95% credible interval of (0.09, 0.15) means:",
            "A credible interval is a direct probability statement about the parameter, given the model and data.",
            options=["Given the model and data, there's a 95% probability the true rate is in this range",
                     "95% of future games will fall in this range",
                     "If we repeated the season many times, 95% of intervals would be exactly this one",
                     "The player's rate will always be between 9% and 15%"],
            notes=["Correct.", "That describes game-to-game outcomes, not the true rate.", "That's the frequentist confidence interval reading.", "Rates can still drift, and a 5% chance remains outside the interval."],
            answer=0),
        Question(
            "q4", "short", "Why report a posterior distribution instead of just one number?",
            "A distribution shows uncertainty and enables probability-based decisions.",
            keywords=[["uncertain", "spread", "range", "interval", "width", "confiden", "sure", "narrow", "wide"],
                      ["decision", "compare", "risk", "rank", "probab", "threshold", "trust"]],
            model_answer="One number hides how sure you are. A posterior shows the uncertainty, so you can compare players with different sample sizes and compute decision probabilities."),
        Question(
            "q5", "mc",
            "DL-A: posterior mean 12.0% with a wide interval (few snaps). DL-B: posterior mean 11.5% with a narrow interval (many snaps). What's the best reading?",
            "Means alone are misleading. DL-A could be better or worse than the mean suggests, while DL-B's estimate is firmer. The right choice depends on your question and risk tolerance.",
            category="Application",
            options=["DL-A is better because 12.0 > 11.5", "DL-B is better because narrower is always better",
                     "DL-A is uncertain, DL-B is firmly near 11.5%, so the decision depends on risk and more data",
                     "They are identical"],
            notes=["A tiny difference in means isn't meaningful when the intervals overlap this much.", "Narrow means more certain, not better.", "Correct.", "They differ in both the mean and the certainty."],
            answer=2),
    ],
)
