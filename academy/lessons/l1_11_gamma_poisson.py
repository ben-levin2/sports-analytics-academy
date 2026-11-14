import numpy as np
import streamlit as st
from scipy import stats

import matplotlib.pyplot as plt

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


def _fig(a0, b0, x, n):
    """Prior, data-only and posterior curves for a rate, shown per 100 snaps."""
    prior, like, post = stats.gamma(a0, scale=1 / b0), stats.gamma(x + 1, scale=1 / max(n, 1)), stats.gamma(a0 + x, scale=1 / (b0 + n))
    hi = max(d.ppf(0.999) for d in (prior, like, post)) * 100
    grid = np.linspace(0, hi, 500)
    fig, ax = plt.subplots()
    for d, color, ls, lw, label in ((prior, T.MUTED, "--", 2, f"Prior Gamma({a0:.3g}, {b0:g})"),
                                    (like, T.AMBER, "-", 2, f"Data only ({x} in {n} snaps)"),
                                    (post, T.BLUE, "-", 2.6, "Posterior")):
        ax.plot(grid, d.pdf(grid / 100) / 100, color=color, ls=ls, lw=lw, label=label)
    ax.fill_between(grid, post.pdf(grid / 100) / 100, color=T.BLUE, alpha=0.18)
    ax.axvline(100 * post.mean(), color=T.BLUE, lw=1, ls=":")
    ax.set_xlabel("True tackle rate (tackles per 100 snaps)"), ax.set_ylabel("Plausibility"), ax.set_yticks([])
    ax.legend(loc="upper right")
    return fig


def visual():
    c1, c2 = st.columns([1, 2])
    with c1:
        rate = st.slider("Prior: typical tackles per 100 snaps", 1.0, 15.0, 6.5, 0.5, key="l111_rate")
        strength = st.slider("Prior strength (pretend snaps)", 10, 1500, 150, 10, key="l111_strength")
        n = st.slider("Snaps observed", 1, 1000, 15, key="l111_n")
        x = st.slider("Tackles observed", 0, 150, 3, key="l111_x")
    a0, b0 = rate / 100 * strength, float(strength)
    with c2:
        show_fig(_fig(a0, b0, x, n))
    m = st.columns(4)
    m[0].metric("Raw rate", f"{100 * x / n:.1f}")
    m[1].metric("Prior mean", f"{rate:.1f}")
    m[2].metric("Posterior mean", f"{100 * (a0 + x) / (b0 + n):.1f}")
    m[3].metric("Weight on prior", f"{b0 / (b0 + n):.0%}")
    st.caption("All rates are tackles per 100 snaps. Simulated, not real data. Weight on prior = k/(k+n), with k = prior strength in snaps.")


CODE = '''import numpy as np
from scipy import stats

# --- SIMULATED example: two fictional safeties ---
prior_rate, prior_strength = 0.065, 150            # 6.5 tackles per 100 snaps, worth about 150 snaps of evidence
a0, b0 = prior_rate * prior_strength, prior_strength   # Gamma(a, b): a = pseudo-tackles, b = pseudo-snaps
safeties = {"Safety A": (3, 15), "Safety B": (120, 700)}   # (tackles, snaps)

rng = np.random.default_rng(0)
for name, (x, n) in safeties.items():
    a1, b1 = a0 + x, b0 + n                        # Gamma-Poisson update: add events to a, exposure to b
    post = stats.gamma(a1, scale=1 / b1)           # SciPy uses scale = 1 / rate
    lo, hi = post.ppf([0.10, 0.90]) * 100          # 80% credible interval, per 100 snaps
    draws = post.rvs(20000, random_state=rng)      # simulated posterior draws of the true rate
    p_above = (draws > prior_rate).mean()          # P(true rate above the position average)
    next100 = rng.poisson(draws * 100)             # draw a rate, then a Poisson count for the next 100 snaps
    print(f"{name}: raw {100 * x / n:.1f} | posterior {100 * a1 / b1:.2f} per 100 snaps"
          f" | prior weight {b0 / (b0 + n):.0%}")
    print(f"   80% interval ({lo:.1f}, {hi:.1f}) | P(above average) {p_above:.2f}"
          f" | expected next 100 snaps {next100.mean():.1f} tackles")

print("Change the prior strength for Safety A:")
for s in (30, 150, 600):
    a, b = prior_rate * s, s
    print(f"  strength {s:>3} snaps -> posterior {100 * (a + 3) / (b + 15):.2f} per 100 snaps")
'''

LESSON = Lesson(
    id="l1-11-gamma-poisson", level=1, order=11,
    title="Gamma-Poisson: Shrinking Counts Over Snaps",
    subtitle="The same prior-plus-data idea as L1.1, for events over exposure instead of yes/no outcomes",
    minutes=20,
    concept_ids=["gamma-poisson"],
    simple="""
L1.1 had a simple recipe: **prior belief + new data = posterior**, weighted by how much of each you have. It worked for **yes/no outcomes** (a pass is complete or not).

Many football numbers aren't yes/no. They are **counts over exposure**: tackles over snaps played, catches over targets, pressures over pass-rush snaps. There is no fixed number of 'trials', and you can't hit 100%. A safety could make 0, 1, 2 or more tackles, and the more snaps he plays, the more chances he gets.

The recipe is **exactly the same**. Only the pieces change: instead of 'pretend completions and incompletions', the prior is 'pretend **tackles** over pretend **snaps**'. A 15-snap player's rate is pulled hard toward the position average. A 700-snap player's rate barely moves.

*All example numbers below are **Simulated**.*
""",
    problem="A raw rate from 15 snaps can be wildly off. Three tackles in 15 snaps is 20 per 100, three times the position average, but it is mostly luck. A 120-in-700 player has a rate you can trust far more, and a plain ranking can't tell them apart.",
    why="Shrinkage puts every player on a fair scale: small-snap players are pulled toward the average, big-snap players keep their own rate, and each estimate comes with an honest uncertainty range.",
    when_useful="""
- Counts of events over a measured exposure: snaps, games, targets, minutes
- Players with very different amounts of playing time
- You need a probability ('how likely is he above average?') or a forecast of next week's count
""",
    when_not="""
- Yes/no outcomes with a fixed number of tries (completions per attempt): use Beta-Binomial (L1.1)
- Events that arrive in bursts (a few huge games, many quiet ones): plain Poisson is too tidy, and the intervals will be too narrow
- Very rare events with a prior strength you picked blindly (see the warning below)
""",
    compare="""
**Beta-Binomial (L1.1):** n trials, each yes/no; the rate is a probability, capped at 100%. **Gamma-Poisson (this lesson):** events over exposure; the rate is any positive number, per snap, per game or per 100 snaps.

| | Beta-Binomial | Gamma-Poisson |
|---|---|---|
| What you count | Successes out of n trials | Events over an amount of exposure |
| Fixed number of trials? | Yes (n attempts) | No: the count has no ceiling |
| What the rate is | A probability between 0 and 1 | Any positive number (events per snap) |
| Can it exceed 100%? | Never | Yes: 20 tackles per 100 snaps is fine |
| Football examples | Completions per attempt, field goals made, third downs converted | Tackles per 100 snaps, catches per target-game, pressures per rush snap |
| Prior is | Beta(α, β): pretend successes and failures | Gamma(a, b): pretend events over pretend exposure |
| Update | α+k, β+n−k | a+x, b+n |

**Quick test: could the count in one unit be 0, 1, 2, 3 or more, with no ceiling? Then Gamma-Poisson.** If each try is a yes or a no, Beta-Binomial.

**Where each is the wrong choice:**
- *Beta-Binomial is wrong for tackles per snap.* A snap isn't a yes/no trial, and a defender's total over a game or season has no ceiling. Forcing it into 'tackles out of snaps' makes the rate look like a probability when it isn't.
- *Gamma-Poisson is wrong for completion percentage.* Each attempt is a yes/no, so the rate can never exceed 100%. A Poisson model would happily predict 'more completions than attempts'.
""",
    sports="""
**Fictional scenario (Simulated numbers).** Two safeties, same position, same prior.

- **Prior:** safeties make about **6.5 tackles per 100 snaps**, and we treat that as worth about **150 snaps** of evidence. In Gamma terms: a = 6.5/100 × 150 = **9.75 pretend tackles**, b = **150 pretend snaps**.

| | Tackles | Snaps | Raw rate (per 100) | Weight on prior | Posterior mean (per 100) | 80% interval | Chance above average |
|---|---|---|---|---|---|---|---|
| Safety A | 3 | 15 | 20.0 | 91% | **7.7** | 5.1 to 10.6 | 70% |
| Safety B | 120 | 700 | 17.1 | 18% | **15.3** | 13.6 to 17.0 | about 100% |

- Safety A looks like a star on a 15-snap sample (20.0). The posterior pulls him to **7.7**: slightly above average, and very uncertain.
- Safety B barely moves (17.1 to 15.3). With 700 snaps the data carries most of the weight.
- Next 100 snaps: A is expected to make about **7.7** tackles, B about **15.3**.
- **Prior strength matters.** For Safety A, a prior worth 30 snaps gives 11.0, 150 gives 7.7 and 600 gives 6.8. Choosing the strength is a real modeling decision: say it out loud.
""",
    takeaway="Gamma-Poisson is L1.1's recipe for counts over exposure: add events to a, add snaps to b, and the posterior mean is a weighted average of the prior rate and the raw rate. Thin exposure is pulled to the average; heavy exposure keeps its own rate.",
    visual=visual,
    visual_caption="Drag the sliders. With few snaps the blue posterior hugs the grey prior; with many it moves to the amber data. The 'weight on prior' tile is k/(k+n). Try 3 tackles in 15 snaps, then 120 in 700. All numbers are Simulated.",
    math=[
        MathBlock(
            "Step 1: the model",
            r"\lambda \sim \text{Gamma}(a,\, b), \qquad x \mid \lambda \sim \text{Poisson}(\lambda\, n)",
            """
- $\\lambda$ (lambda): the unknown **true rate**: tackles per snap. This is what we want to learn
- $\\text{Gamma}(a, b)$: the **prior** on the rate. $a$ = pretend events (tackles), $b$ = pretend exposure (snaps)
- $x$: the tackles we actually observed
- $n$: the snaps we actually observed (the exposure)
- $\\text{Poisson}(\\lambda n)$: the count we expect, $\\lambda n$ tackles on average, with random scatter around it
""",
            "Poisson assumes events arrive independently at a steady rate. Real tackles cluster in games, so treat the intervals as a best case.",
        ),
        MathBlock(
            "Step 2: the update (conjugacy again)",
            r"\lambda \mid x \;\sim\; \text{Gamma}(a + x,\; b + n)",
            """
- Add the observed tackles to $a$ and the observed snaps to $b$
- The posterior is a Gamma again, so updating is just **addition**, as with Beta-Binomial in L1.1
- SciPy's `stats.gamma` takes `scale = 1/b`, so build it as `stats.gamma(a, scale=1/b)`
""",
        ),
        MathBlock(
            "Step 3: the posterior mean is a weighted average",
            r"E[\lambda \mid x] = \frac{a+x}{b+n} = \frac{b}{b+n}\cdot\frac{a}{b} \;+\; \frac{n}{b+n}\cdot\frac{x}{n}",
            """
- $a/b$: the **prior mean** (the position-average rate). **Prior strength = $b$**, in pseudo-snaps
- $x/n$: the **raw rate** from the data
- Weight on the prior = $\\frac{b}{b+n}$. This is the same $k/(k+n)$ idea as L1.1, with $k = b$
- Weight on the data = $\\frac{n}{b+n}$. It rises toward 1 as snaps accumulate
""",
            "Safety A: weight on prior = 150/(150+15) = 91%. Safety B: 150/(150+700) = 18%.",
        ),
    ],
    code=CODE,
    code_notes="Change the strength list in the last loop to (10, 150, 1500) and rerun. Watch how far Safety A moves toward his raw 20 per 100 when the prior is weak, and how little it moves when the prior is heavy.",
    walkthrough=[
        ("from scipy import stats", "Load SciPy's statistics module. It knows the Gamma and Poisson distributions."),
        ("prior_rate, prior_strength = 0.065, 150", "Our belief before the data: 6.5 tackles per 100 snaps (0.065 per snap), trusted as much as 150 snaps of evidence."),
        ("a0, b0 = prior_rate * prior_strength, prior_strength", "Turn the belief into Gamma parameters: a0 = 9.75 pretend tackles, b0 = 150 pretend snaps. Their ratio a0/b0 is the prior rate."),
        ("safeties = {...}", "The new evidence for two fictional players: (tackles, snaps). Simulated."),
        ("a1, b1 = a0 + x, b0 + n", "The update: add real tackles to a, add real snaps to b."),
        ("post = stats.gamma(a1, scale=1 / b1)", "Build the updated belief as a full distribution. SciPy wants scale = 1/b, not b."),
        ("lo, hi = post.ppf([0.10, 0.90]) * 100", "Find the values with 10% and 90% of the belief below them: an 80% credible interval. Multiply by 100 to read it per 100 snaps."),
        ("draws = post.rvs(20000, random_state=rng)", "Simulate 20,000 possible true rates from the posterior."),
        ("p_above = (draws > prior_rate).mean()", "The share of draws above the position average is the probability the player's true rate is above average."),
        ("next100 = rng.poisson(draws * 100)", "Forecast the next 100 snaps in two steps: draw a rate, then draw a Poisson count at that rate. This carries both the rate uncertainty and the count's own randomness."),
        ("print(f'{name}: raw ... prior weight {b0 / (b0 + n):.0%}')", "Show the raw rate, the posterior mean and the weight on the prior, b/(b+n)."),
        ("for s in (30, 150, 600):", "Replay Safety A with a weak, medium and heavy prior. This is the experiment to change yourself."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it. This step is generic.**

- **Limited-snap players:** for a limited-snap safety or a rotation defensive back, replace the raw per-snap rate (tackles, pressures, targets allowed) with the **shrunk rate**. A 20-per-100 rate from 15 snaps and a 17-per-100 rate from 700 snaps stop looking equal.
- **Report the interval.** An 80% range tells a coach whether a part-time player is a real signal or a small-sample blip.
- **State your assumptions:** snaps per game for the pool of players you compare is **assumed** in this lesson (for example, about 55 defensive snaps per game), not measured. Replace it with real counts before using any number for a decision. The prior rate and prior strength are also assumptions and should be written down before you look at the players.
- **Warning for rare events:** a fixed prior strength like 150 snaps is **too weak for rare events** such as interceptions. Say a position averages 0.5 interceptions per 100 snaps and a player has 2 in 100 snaps (Simulated). A prior worth 150 snaps gives 1.1 per 100, more than double the average, with only 60% of the weight on the prior. A prior worth 1,500 snaps gives 0.59. Real differences between players are small for rare events and most of the spread is luck, so the prior should be much stronger. How to set that strength from data is a later topic (partial pooling, L5.4). For now, treat the strength as a labelled assumption and test how sensitive the answer is to it.
""",
    quiz=[
        Question(
            "q1", "mc", "A stat can be 0, 1, 2, 3 or more in one unit, with no ceiling, and exposure varies. Which model fits?",
            "Counts over exposure with no ceiling are Gamma-Poisson. Yes/no trials with a fixed n are Beta-Binomial.",
            concept="gamma-poisson",
            options=["Gamma-Poisson", "Beta-Binomial", "Either, they are interchangeable", "Neither; use a raw average"],
            notes=["Correct: that is the quick test.", "That is for yes/no outcomes out of a fixed number of tries.",
                   "They differ in what the rate means and whether it has a ceiling.", "A raw average ignores how many snaps each player had."],
            answer=0),
        Question(
            "q2", "numeric",
            "Prior Gamma(a = 8, b = 100). A player makes 5 tackles in 40 snaps. What is the posterior mean, in tackles per 100 snaps? (2 decimals)",
            "(8 + 5) / (100 + 40) = 13/140 ≈ 0.0929 per snap, or 9.29 per 100 snaps.",
            concept="gamma-poisson", category="Application", answer=9.29, tolerance=0.05),
        Question(
            "q3", "mc",
            "Safety A has 3 tackles in 15 snaps; Safety B has 120 in 700. Same prior worth 150 snaps. Which statement is right?",
            "A's weight on the prior is 150/165 = 91%; B's is 150/850 = 18%. So A moves far toward the average; B barely moves.",
            concept="gamma-poisson", category="Application",
            options=["A is pulled much more toward the position average than B", "B is pulled more because he has more data",
                     "Both are pulled by the same amount because the prior is the same",
                     "Neither is pulled; the raw rates are used"],
            notes=["Correct: less exposure, more shrinkage.", "More data gives the data more weight, not the prior.",
                   "The prior is the same, but the weight on it depends on the snaps.", "The whole point is that raw rates are replaced."],
            answer=0),
        Question(
            "q4", "short", "Why not use Beta-Binomial for tackles per snap?",
            "A snap is not a yes/no trial and tackle counts have no ceiling; Beta-Binomial treats the rate as a probability capped at 100%.",
            concept="gamma-poisson", category="Application",
            keywords=[["trial", "yes", "no", "binary", "ceiling", "cap", "100", "probability", "fixed"],
                      ["count", "exposure", "snaps", "poisson", "gamma", "no limit", "any number", "more than"]],
            model_answer="Beta-Binomial is for yes/no trials with a fixed number of tries, so the rate is a probability that can't exceed 100%. Tackles are counts over exposure with no ceiling, so Gamma-Poisson is the right model."),
        Question(
            "q5", "mc",
            "For a very rare event like interceptions, why can a fixed prior strength of 150 snaps be a problem?",
            "It is too weak: the spread between players is mostly luck, so the prior should be much stronger or a lucky short sample looks like a star.",
            concept="gamma-poisson", category="Application",
            options=["It is too weak, so a lucky short sample is not pulled back enough", "It is too strong, so real stars are hidden",
                     "It makes the Gamma distribution invalid", "It only matters for Beta-Binomial"],
            notes=["Correct: this is why prior strength must be examined, not assumed.",
                   "For rare events the risk runs the other way.", "The Gamma is still valid; the strength is the issue.",
                   "Prior strength matters in both models."],
            answer=0),
    ],
)
