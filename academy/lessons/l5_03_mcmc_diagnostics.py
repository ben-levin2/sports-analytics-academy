import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import theme as T
from ..hierarchical import ess, rhat
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig
from .l5_02_mcmc_intro import sample


def visual():
    step = st.select_slider("Step size", [0.002, 0.01, 0.05, 0.2, 0.8], 0.05, key="l53_step")
    starts = [0.1, 0.5, 0.9, 0.3]
    runs = [sample(2000, step, seed=i, start=s) for i, s in enumerate(starts)]
    chains = np.array([r[0][1000:] for r in runs])
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for (d, _), c in zip(runs, T.CYCLE):
        ax.plot(d, color=c, lw=0.7)
    ax.axvline(1000, color=T.MUTED, ls="--", lw=1)
    ax.set_xlabel("Step (dashed = end of warm-up)"), ax.set_ylabel("θ"), ax.set_ylim(0, 1)
    show_fig(fig)
    m = st.columns(3)
    m[0].metric("Acceptance", f"{np.mean([r[1] for r in runs]):.0%}")
    m[1].metric("R-hat", f"{rhat(chains):.3f}", help="Want ≤ 1.01 (strict) / ≤ 1.05 (this academy)")
    m[2].metric("Effective sample size", f"{ess(chains):.0f}", help=f"Out of {chains.size} recorded draws")


CODE = '''import numpy as np
from scipy import stats
from academy.hierarchical import ess, rhat

k, n = 14, 20
def log_post(t):
    return -np.inf if not 0 < t < 1 else stats.binom.logpmf(k, n, t) + stats.beta.logpdf(t, 62, 38)

def run_chain(start, step, iters, seed):
    rng = np.random.default_rng(seed)
    t, out, acc = start, [], 0
    for _ in range(iters):
        p = t + rng.normal(0, step)
        if np.log(rng.random()) < log_post(p) - log_post(t):
            t, acc = p, acc + 1
        out.append(t)
    return np.array(out), acc / iters

# Four chains from deliberately different starting points, for three step sizes
for step in (0.002, 0.05, 0.8):
    runs = [run_chain(s, step, 2000, i) for i, s in enumerate([0.1, 0.5, 0.9, 0.3])]
    chains = np.array([r[0][1000:] for r in runs])        # keep the second half of each chain
    print(f"step {step:<6} acceptance {np.mean([r[1] for r in runs]):.0%}   "
          f"R-hat {rhat(chains):.3f}   ESS {ess(chains):.0f} of {chains.size} draws")
print("\\nToo small: chains crawl and never meet (huge R-hat).")
print("Too big: most proposals are rejected (sticky chains, fewer effective draws).")
'''

LESSON = Lesson(
    id="l5-03-mcmc-diagnostics", level=5, order=3,
    title="MCMC, Part 2: Has the Chain Converged? R-hat, ESS and Step Size",
    subtitle="How to tell a trustworthy MCMC run from confident nonsense",
    minutes=20,
    concept_ids=["mcmc"],
    simple="""
MCMC always gives you numbers, even when they're wrong. The walker may not have explored the whole posterior yet, or may be stuck in one corner. So you **check**:

- **Run several chains** from very different starting points. If they've all converged, they should end up wandering the **same** region.
- **R-hat** compares the chains: about **1.00** means they agree. Above about **1.01** (strict) or **1.05** (this academy's teaching threshold) means "not converged; run longer or fix the sampler".
- **Effective sample size (ESS):** consecutive draws are similar, so 1,000 draws may only be worth 100 independent ones. ESS tells you how many you really have.
- **Step size** matters: too small and the walker crawls; too big and nearly every step is rejected. Acceptance around 20–70% is usually healthy for this simple sampler.
""",
    problem="An unconverged chain produces precise-looking but wrong posteriors. Diagnostics are how you catch that.",
    why="They're the difference between a result you can defend and one you can't. Every serious Bayesian analysis reports them.",
    when_useful="""
- Every MCMC run, without exception
- Comparing samplers or settings (step size, number of iterations)
- Deciding whether to run longer or reparameterize the model
""",
    when_not="""
- As proof the model is right: diagnostics say the sampler explored the model's posterior, not that the model is good (that's validation's job)
- Single chains: R-hat needs several
""",
    compare="""
**R-hat vs ESS:** R-hat asks "do chains agree?" ESS asks "how much information is in the draws?" You need both: chains can agree but be so sticky that ESS is tiny.

**Diagnostics vs validation:** diagnostics check the **sampler**; time-based validation (L3.2) checks the **model**. A perfectly converged chain on a bad model is still a bad model.
""",
    sports="""
**Fictional scenario (simulated).** The same QB posterior as L5.2, with four chains started at 0.1, 0.5, 0.9 and 0.3:

| Step size | Acceptance | R-hat | ESS |
|---|---|---|---|
| 0.002 (too small) | 95% | **4.06** | 26 |
| 0.05 (good) | 67% | **1.008** | ~435 |
| 0.8 (too big) | 7% | 1.034 | ~213 |

Tiny steps get accepted constantly but never travel, so the chains don't meet. Huge steps are almost all rejected. The middle setting is the one to trust.

The hierarchical sampler in this level had the same problem at first: its position averages had R-hat 1.10. A standard fix (re-centering the effects each round) brought them to 1.00.
""",
    takeaway="Always run several chains and check R-hat and ESS before believing an MCMC result. Diagnostics test the sampler, not the model.",
    visual=visual,
    visual_caption="Four chains from different starts. Try the tiny step size: the chains never meet and R-hat explodes. Try the big one: flat stretches where proposals keep getting rejected.",
    math=[
        MathBlock(
            "Step 1: between-chain vs within-chain spread",
            r"W = \frac{1}{m}\sum_{j=1}^{m} s_j^2, \qquad B = \frac{n}{m-1}\sum_{j=1}^{m} (\bar\theta_j - \bar\theta)^2",
            """
- $m$: number of chains; $n$: draws per chain
- $s_j^2$: variance within chain $j$; $W$: their average
- $\\bar\\theta_j$: chain $j$'s mean; $\\bar\\theta$: the overall mean; $B$: how much chain means disagree
""",
        ),
        MathBlock(
            "Step 2: R-hat",
            r"\hat R = \sqrt{\frac{\frac{n-1}{n}W + \frac{1}{n}B}{W}}",
            """
- If chains agree, $B$ is small and $\\hat R \\approx 1$
- If they're exploring different regions, $B$ is large and $\\hat R > 1$
- 'Split' R-hat (used here) also splits each chain in half, to catch chains that are still drifting
""",
        ),
        MathBlock(
            "Step 3: effective sample size",
            r"\text{ESS} = \frac{N}{1 + 2\sum_{t=1}^{\infty}\rho_t}",
            """
- $N$: total draws
- $\\rho_t$: autocorrelation at lag $t$ (how similar draws $t$ steps apart are)
- Highly correlated draws make the denominator big and the ESS small
""",
        ),
    ],
    code=CODE,
    code_notes="Increase iters from 2000 to 20000 for the tiny step size. R-hat improves slowly. A better step size is a far cheaper fix than more iterations.",
    walkthrough=[
        ("def run_chain(start, step, iters, seed):", "The same Metropolis sampler as L5.2, packaged so we can run several chains."),
        ("[run_chain(s, step, 2000, i) for i, s in enumerate([0.1, 0.5, 0.9, 0.3])]", "Four chains, deliberately started far apart."),
        ("chains = np.array([r[0][1000:] for r in runs])", "Keep the second half of each chain (the first half is warm-up)."),
        ("rhat(chains)", "Do the four chains agree? About 1.00 = yes."),
        ("ess(chains)", "How many independent draws the correlated chains are really worth."),
        ("np.mean([r[1] for r in runs])", "Acceptance rate: a quick sign of a bad step size."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Report diagnostics with any Bayesian result:** R-hat, ESS and the number of chains, just as you'd report the test set and baseline.
- **Your rule extended:** the Level 5 project's Production Candidate gate fails if any parameter has R-hat above 1.05 in any season fold.
""",
    quiz=[
        Question(
            "q1", "mc", "R-hat = 1.35. What should you conclude?",
            "The chains disagree: not converged. Run longer, adjust the sampler or reparameterize.",
            options=["Not converged; don't trust the results yet", "Converged; R-hat above 1 is good",
                     "The model is wrong", "The prior is wrong"],
            notes=["Correct.", "Values well above 1 mean disagreement.", "R-hat checks the sampler, not the model.", "Not what R-hat measures."],
            answer=0),
        Question(
            "q2", "mc", "Acceptance rate is 97%. What's the most likely problem?",
            "The step size is too small: almost every tiny step is accepted, but the chain barely moves.",
            category="Application",
            options=["Step size too small; the chain crawls", "Step size too big", "Perfect sampling", "Too many chains"],
            notes=["Correct.", "Big steps cause LOW acceptance.", "High acceptance with tiny moves is inefficient.", "More chains are fine."],
            answer=0),
        Question(
            "q3", "numeric", "10,000 draws with 1 + 2Σρ = 25. What is the ESS?",
            "10,000 / 25 = 400.", category="Application", answer=400, tolerance=0.5),
        Question(
            "q4", "short", "Chains have converged (R-hat 1.00). Does that mean the model predicts well?",
            "No: diagnostics check the sampler; prediction needs out-of-sample validation against baselines.",
            keywords=[["no", "not", "doesn't"],
                      ["validat", "test", "baseline", "predict", "out-of-sample", "later season", "model"]],
            model_answer="No. R-hat only says the sampler explored the model's posterior. Whether the model predicts well needs out-of-sample (time-based) validation against baselines."),
        Question(
            "q5", "mc", "Why start chains from very different values?",
            "If they all end up in the same region despite different starts, that's evidence of convergence.",
            options=["Agreement from different starts is evidence of convergence", "It makes sampling faster",
                     "It changes the posterior", "It's required by the prior"],
            notes=["Correct.", "It doesn't speed things up.", "The posterior doesn't change.", "Unrelated."],
            answer=0),
    ],
)
