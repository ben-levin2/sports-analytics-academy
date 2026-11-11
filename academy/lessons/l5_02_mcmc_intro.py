import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import stats

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig

K, N = 14, 20


def log_post(theta):
    if not 0 < theta < 1:
        return -np.inf
    return stats.binom.logpmf(K, N, theta) + stats.beta.logpdf(theta, 62, 38)


def sample(n, step, seed=1, start=0.5):
    rng = np.random.default_rng(seed)
    t, out, acc = start, [], 0
    for _ in range(n):
        p = t + rng.normal(0, step)
        if np.log(rng.random()) < log_post(p) - log_post(t):
            t, acc = p, acc + 1
        out.append(t)
    return np.array(out), acc / n


def visual():
    n = st.select_slider("Number of MCMC steps", [100, 500, 2000, 10000], 2000, key="l52_n")
    draws, acc = sample(n, 0.05)
    exact = stats.beta(62 + K, 38 + N - K)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.0), gridspec_kw={"width_ratios": [3, 2]})
    a1.plot(draws, color=T.BLUE, lw=0.7)
    a1.set_xlabel("Step"), a1.set_ylabel("θ"), a1.set_title("The chain wandering", fontsize=9)
    xs = np.linspace(0.45, 0.8, 200)
    a2.hist(draws[len(draws) // 10:], bins=30, density=True, color=T.BLUE, alpha=0.5, label="MCMC draws")
    a2.plot(xs, exact.pdf(xs), color=T.AMBER, lw=2.2, label="exact answer")
    a2.set_yticks([]), a2.legend(fontsize=7), a2.set_title("Draws vs exact posterior", fontsize=9)
    show_fig(fig)
    st.caption(f"Acceptance rate {acc:.0%}. With more steps, the histogram matches the exact curve more closely.")


CODE = '''import numpy as np
from scipy import stats

# --- SIMULATED: the Level 1 QB, 14 completions in 20 attempts, prior Beta(62, 38) ---
k, n = 14, 20

def log_post(theta):
    """log(prior x likelihood): all MCMC ever needs. No normalizing constant required."""
    if not 0 < theta < 1:
        return -np.inf
    return stats.binom.logpmf(k, n, theta) + stats.beta.logpdf(theta, 62, 38)

rng = np.random.default_rng(1)
theta, draws, accepted = 0.5, [], 0
for step in range(20000):
    proposal = theta + rng.normal(0, 0.05)                       # take a small random step
    if np.log(rng.random()) < log_post(proposal) - log_post(theta):
        theta = proposal                                         # accept: move there
        accepted += 1
    draws.append(theta)                                          # (if rejected, stay and record again)

draws = np.array(draws[2000:])                                   # drop the warm-up ('burn-in')
exact = stats.beta(62 + k, 38 + n - k)                           # Level 1's exact answer, for checking
print(f"acceptance rate: {accepted / 20000:.0%}")
print(f"posterior mean    MCMC {draws.mean():.4f}   exact {exact.mean():.4f}")
print(f"95% interval      MCMC {np.quantile(draws, [0.025, 0.975]).round(3)}   exact {exact.ppf([0.025, 0.975]).round(3)}")
'''

LESSON = Lesson(
    id="l5-02-mcmc-intro", level=5, order=2,
    title="MCMC, Part 1: Exploring a Posterior by Taking a Random Walk",
    subtitle="The Metropolis algorithm, checked against an answer we already know",
    minutes=20,
    concept_ids=["mcmc"],
    simple="""
In Level 1, the posterior had a neat formula (Beta-Binomial). Most real models don't. A model with position, team, experience and player effects has thousands of unknowns tied together, and there's no formula.

**Markov chain Monte Carlo (MCMC)** gets around that by **sampling**. A "walker" moves around the space of possible values:

1. From where it is, propose a **small random step**.
2. If the new spot is **more plausible** (higher prior × likelihood), move there.
3. If it's less plausible, move there only **sometimes**, with a chance equal to how much less plausible it is.
4. Record where you are. Repeat thousands of times.

The places the walker visits, and how often, **form the posterior**. Averages of the visits give posterior means; percentiles give credible intervals.

The trick: you never need the hard-to-compute normalizing constant (the evidence from L1.5). Only *ratios* of plausibility matter.
""",
    problem="Posteriors for realistic models have no formula. MCMC gives you samples from them anyway, so you can still compute means, intervals and probabilities.",
    why="It's the engine behind almost all modern Bayesian modeling, including the hierarchical player model in this level and PyMC.",
    when_useful="""
- Any Bayesian model without a closed-form posterior
- Models with many related unknowns (hierarchical models)
- When you need full uncertainty, not just a best guess
""",
    when_not="""
- When a formula exists (Beta-Binomial): use it. It's exact and instant
- Huge models where you only need a quick best guess: optimization (or EM) is faster
- Without checking convergence (L5.3): an unconverged chain gives confident nonsense
""",
    compare="""
**MCMC vs exact formulas:** formulas are exact but only exist for simple models. MCMC is approximate but works for almost any model.

**MCMC vs EM (L5.1):** EM climbs to one best answer. MCMC wanders around the plausible answers, mapping uncertainty.

**MCMC vs the Markov chains of L1.6:** same idea, a process where the next step depends only on the current one. Here the chain is designed so that it spends time in each region in proportion to the posterior.
""",
    sports="""
**Fictional scenario (simulated).** The Level 1 QB again: 14 of 20, with prior Beta(62, 38). We already know the exact answer, which makes it a perfect test for MCMC.

- 20,000 steps, with the first 2,000 dropped as warm-up.
- Acceptance rate about **67%**.
- Posterior mean **0.632** from MCMC vs **0.633** exact; 95% intervals match to two decimals.

MCMC is doing nothing magic, just recovering the answer by walking. The point is that the same walk works when there's no formula to check against.
""",
    takeaway="MCMC samples a posterior by a clever random walk that only needs prior × likelihood. It matches exact answers where they exist and works where they don't.",
    visual=visual,
    visual_caption="Left: the walker's path. Right: its visits as a histogram vs the exact posterior. More steps give a closer match.",
    math=[
        MathBlock(
            "Step 1: you only need the posterior up to a constant",
            r"p(\theta \mid D) \;\propto\; p(D \mid \theta)\, p(\theta)",
            """
- $p(\\theta \\mid D)$: the posterior we want to sample
- $p(D \\mid \\theta)\\,p(\\theta)$: likelihood × prior, easy to compute for any $\\theta$
- $\\propto$: "proportional to". The missing constant $p(D)$ cancels out in Step 3
""",
        ),
        MathBlock(
            "Step 2: propose a nearby value",
            r"\theta' = \theta + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, s^2)",
            """
- $\\theta$: where the walker is now; $\\theta'$: the proposed new spot
- $s$: the step size (0.05 in the code). Too small and it crawls; too big and most steps are rejected (L5.3)
""",
        ),
        MathBlock(
            "Step 3: accept or stay",
            r"P(\text{accept}) = \min\!\left(1,\; \frac{p(D \mid \theta')\,p(\theta')}{p(D \mid \theta)\,p(\theta)}\right)",
            """
- If the new spot is more plausible, the ratio is above 1, so always move
- If less plausible, move with probability equal to the ratio
- In code we compare logs: accept if $\\log u <$ log-posterior difference, where $u$ is uniform on 0–1
""",
            "This rule (Metropolis) guarantees that, in the long run, time spent at each value matches the posterior.",
        ),
    ],
    code=CODE,
    code_notes="Change the step size 0.05 to 0.001 and to 1.0, and watch the acceptance rate and the match to the exact answer. L5.3 explains what goes wrong.",
    walkthrough=[
        ("def log_post(theta):", "The log of prior × likelihood. Impossible values (outside 0–1) get −infinity, so they're never accepted."),
        ("proposal = theta + rng.normal(0, 0.05)", "Propose a small random step from the current value."),
        ("np.log(rng.random()) < log_post(proposal) - log_post(theta)", "The Metropolis rule in log form: always accept uphill moves, and sometimes accept downhill ones."),
        ("theta = proposal; accepted += 1", "Move, and count the acceptance."),
        ("draws.append(theta)", "Record the current value every step, even after a rejection. Staying put is part of the answer."),
        ("draws = np.array(draws[2000:])", "Drop the warm-up, when the walker was still travelling from its arbitrary start."),
        ("exact = stats.beta(62 + k, 38 + n - k)", "Level 1's exact posterior, used only to check that MCMC works."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Any hierarchical player model** (position, team, experience, player) needs MCMC or an approximation. This is the engine.
- **Check against known answers first:** before trusting MCMC on a new PSFAL-style model, test it where you know the answer, exactly as this lesson does.
- **PyMC is optional here** (`requirements-optional.txt`); the academy's own NumPy sampler is enough to learn the ideas.
""",
    quiz=[
        Question(
            "q1", "mc", "Why doesn't MCMC need the evidence p(D) from L1.5?",
            "Acceptance uses a ratio of posteriors, so the constant cancels.",
            options=["It only uses ratios, so the constant cancels", "It estimates it first",
                     "It assumes p(D) = 1", "It ignores the prior"],
            notes=["Correct.", "It never needs to.", "It doesn't need any value for it.", "It uses the prior."],
            answer=0),
        Question(
            "q2", "numeric", "Current log-posterior −10.0, proposal −10.7. What is the acceptance probability? (e^−0.7)",
            "exp(−10.7 − (−10.0)) = e^−0.7 ≈ 0.497.", category="Application", answer=0.497, tolerance=0.005),
        Question(
            "q3", "mc", "A proposal is rejected. What does the chain record for that step?",
            "It records the current value again. Staying is part of the correct sampling.",
            options=["The current value again", "Nothing", "The rejected proposal", "The prior mean"],
            notes=["Correct.", "Skipping would bias the result.", "It wasn't accepted.", "Not used."],
            answer=0),
        Question(
            "q4", "short", "Why test an MCMC sampler on a problem with a known exact answer first?",
            "To check the sampler works before trusting it where you can't check.",
            keywords=[["check", "verify", "test", "confirm", "make sure", "trust"],
                      ["work", "correct", "right", "match", "bug", "exact", "known"]],
            model_answer="To verify that the sampler and code are correct by matching a known answer, before trusting it on models where no exact answer exists."),
        Question(
            "q5", "mc", "What is burn-in?",
            "The early steps, while the chain is still travelling from its arbitrary start, which are discarded.",
            options=["Early steps discarded while the chain leaves its starting point", "The final steps",
                     "Steps that were rejected", "A type of prior"],
            notes=["Correct.", "Those are kept.", "Rejections are recorded, not discarded.", "Not a prior."],
            answer=0),
    ],
)
