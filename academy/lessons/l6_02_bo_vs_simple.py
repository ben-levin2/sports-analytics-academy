import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import optimize as O, theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner="Running 10 Bayesian-optimization and random-search races…")
def _race():
    f = O.tuning_objective()
    return O.contest(f, O.TUNING_BOUNDS, budget=15, seeds=10)


def visual():
    if st.button("Run the tuning race", key="l62_run"):
        st.session_state["l62_res"] = _race()
    res = st.session_state.get("l62_res")
    if not res:
        return
    bo, rs, grid = res
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    n = np.arange(1, bo.shape[1] + 1)
    ax.plot(n, bo.mean(0), color=T.BLUE, lw=2.4, label="Bayesian optimization (avg of 10 runs)")
    ax.plot(n, rs.mean(0), color=T.AMBER, lw=2.4, label="Random search (avg of 10 runs)")
    ax.axhline(grid, color=T.MUTED, ls="--", lw=1.2, label="3×3 grid (9 tries)")
    ax.set_xlabel("Tries used"), ax.set_ylabel("Best error found so far"), ax.legend(fontsize=7)
    show_fig(fig)


CODE = '''import numpy as np
from academy.data import make_dl_market
from academy.lineup import bayes_opt as bo_lineup, exact, greedy, random_search as rs_lineup, total
from academy.optimize import TUNING_BOUNDS, contest, tuning_objective

# --- Race 1: tuning two settings of Level 1 shrinkage (prior strength, position pooling) ---
f = tuning_objective()
bo, rs, grid = contest(f, TUNING_BOUNDS, budget=15, seeds=8)
print("Tuning: best error found (avg over 8 runs)")
for k in (5, 10, 15):
    print(f"  after {k:>2} tries:  BO {bo[:, k - 1].mean():.3f}   random {rs[:, k - 1].mean():.3f}")
print(f"  3x3 grid (9 tries): {grid:.3f}  (lucky: the best setting sits exactly on a grid point)")

# --- Race 2: the $40M D-line build (6 players, >= 2 EDGE and >= 2 DT, cap <= $40M) ---
market = make_dl_market()                                        # SIMULATED
pool = market[market["season"] == 2025].reset_index(drop=True)
best = total(pool, exact(pool))                                  # exact solver: the true best lineup
print(f"\\nLineup score (sum of projected value): exact best {best:.2f}")
print(f"  greedy value-per-dollar:        {total(pool, greedy(pool)):.2f}  ({total(pool, greedy(pool)) / best:.0%})")
r = np.mean([rs_lineup(pool, 30, seed=s)[-1] for s in range(3)])
b = np.mean([bo_lineup(pool, 30, seed=s)[-1] for s in range(3)])
print(f"  random search, 30 tries:        {r:.2f}  ({r / best:.0%})")
print(f"  Bayesian optimization, 30 tries: {b:.2f}  ({b / best:.0%})")
'''

LESSON = Lesson(
    id="l6-02-bo-vs-simple", level=6, order=2,
    title="Bayesian Optimization, Part 2: Does It Beat Simple Search? Plus the $40M Lineup",
    subtitle="Objective functions, constraints, trade-offs, and choosing the right optimizer",
    minutes=25,
    concept_ids=["bayesian-optimization"],
    simple="""
Every optimization problem has three parts:

- **Objective:** what you're trying to make as good as possible (lowest error, highest total value).
- **Constraints:** the rules (a $40M cap, at least 2 EDGE and 2 DT, exactly 6 players).
- **Decision:** what you control (the settings, or which players).

Then you pick a **search method**, and the right one depends on the problem:

| Problem | Best tool |
|---|---|
| Few continuous settings, each try expensive | **Bayesian optimization** |
| Cheap tries | Random search or a fine grid |
| Simple score with clear rules (a cap and roster slots) | An **exact solver** (integer programming) |

This lesson races Bayesian optimization against simple alternatives on two problems, and reports who actually wins.
""",
    problem="It's easy to reach for the most sophisticated optimizer. The question is whether it beats simple search on *your* problem, under the same budget.",
    why="Choosing the right tool saves time and avoids answers that look optimal but aren't.",
    when_useful="""
- BO: tuning a few settings with slow validation
- Exact solvers: lineups and rosters when the score is a sum of player values
- Random search: a strong, simple baseline for any search problem
""",
    when_not="""
- BO for lineups with a simple additive score: an exact solver finds the true best instantly
- Any optimizer fed bad projections: it confidently picks the most overrated players (the optimizer's curse, Project P6A)
""",
    compare="""
**Exact solver vs Bayesian optimization:** if the score can be written as a simple sum with linear rules, an integer program gives the provably best answer in milliseconds. BO only approximates.

**Greedy vs everything:** greedy (best value per dollar first) is instant and explainable, but can leave cap space unused or pass up great-but-pricey players.

**Trade-offs:** a risk penalty (value − λ × uncertainty) changes the answer. There's no single "optimal" lineup without deciding how much risk you'll accept.
""",
    sports="""
**Fictional scenario (simulated).**

**Race 1: tuning two settings** of the Level 1 model, 15 tries each, averaged over many runs. BO reaches the best error (about **7.00**) within ~10 tries; random search is still around **7.18** after 15. A 3×3 grid also hits 7.00, but only because the best setting happens to sit exactly on a grid point.

**Race 2: the $40M D-line build** (6 players, ≥2 EDGE, ≥2 DT). As a share of the true best lineup score:
- **Exact solver:** 100%, instantly.
- **Bayesian optimization (40 tries):** about **90%**.
- **Greedy:** about **86%**.
- **Random search (40 tries):** about **79%**.

With a risk penalty added, **greedy actually beats BO** (about 91% vs 86%).

**Verdict:** BO earns its keep for expensive tuning. For lineups, use the exact solver, and keep greedy as the simple, explainable baseline.
""",
    takeaway="Match the optimizer to the problem: BO for expensive tuning, exact solvers for cap-and-roster problems, and always race against random search and greedy.",
    visual=visual,
    visual_caption="Press the button to race BO against random search on the 2-setting tuning problem. The lines show the best error found after each number of tries.",
    math=[
        MathBlock(
            "Step 1: the lineup problem, written down",
            r"\max_{x \in \{0,1\}^n} \sum_i x_i (v_i - \lambda s_i) \quad \text{s.t.}\quad \sum_i x_i c_i \le 40,\;\; \sum_i x_i = 6,\;\; \sum_{i \in \text{EDGE}} x_i \ge 2,\;\; \sum_{i \in \text{DT}} x_i \ge 2",
            """
- $x_i$: 1 if player $i$ is chosen, 0 if not (the **decision**)
- $v_i$: projected value; $s_i$: its uncertainty; $\\lambda$: how much you penalize risk (the **objective**)
- $c_i$: cap hit in $M; the conditions after "s.t." ("such that") are the **constraints**
""",
        ),
        MathBlock(
            "Step 2: why an exact solver works here",
            r"\text{objective and constraints are linear in } x \;\Rightarrow\; \text{mixed-integer linear program (MILP)}",
            """
- Linear: each player adds a fixed amount to the score and to the cap
- MILP solvers (here `scipy.optimize.milp`) search cleverly and prove the answer is the best possible
- Add something non-linear (chemistry between players) and you may need search methods like BO
""",
        ),
        MathBlock(
            "Step 3: comparing searches fairly",
            r"\text{regret}(k) = f^{\star} - \max_{j \le k} f(x_j)",
            """
- $f^{\\star}$: the true best score (known here thanks to the exact solver)
- $\\max_{j \\le k} f(x_j)$: the best found in the first $k$ tries
- Compare methods at the **same** number of tries $k$, averaged over several random starts
""",
        ),
    ],
    code=CODE,
    code_notes="Add `risk=1.0` to the greedy, exact and search calls (e.g. `greedy(pool, 1.0)`, `total(pool, x, 1.0)`). Does the ranking of methods change?",
    walkthrough=[
        ("f = tuning_objective()", "Race 1's objective: Level 1 shrinkage error, as a function of two settings."),
        ("contest(f, TUNING_BOUNDS, budget=15, seeds=8)", "Run BO and random search 8 times each, 15 tries per run, and record the best-so-far after each try."),
        ("bo[:, k - 1].mean()", "Average best error after k tries, across the 8 runs."),
        ("pool = market[market['season'] == 2025]", "Race 2: this season's 40 DL candidates, with projections and cap hits."),
        ("exact(pool)", "The MILP solver: the provably best legal lineup."),
        ("greedy(pool)", "Simple baseline: best value per dollar first, never breaking the rules."),
        ("rs_lineup(pool, 30, seed=s)", "Random search: 30 random legal lineups, keep the best."),
        ("bo_lineup(pool, 30, seed=s)", "Bayesian optimization over lineups: a GP learns which players help, 30 tries."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Cap-constrained builds** (like your $40M D-line case) are best solved exactly; keep greedy as the explainable baseline for staff.
- **Garbage in, garbage out:** an optimizer is only as good as its value projections. Those projections must pass their own baseline test first, under your rule.
- **Tuning:** use BO for slow-to-validate settings, on training seasons only.
""",
    quiz=[
        Question(
            "q1", "mc", "In the $40M lineup problem, the $40M cap is:",
            "A constraint: a rule every allowed lineup must satisfy.",
            options=["A constraint", "The objective", "The decision", "A baseline"],
            notes=["Correct.", "The objective is total (risk-adjusted) value.", "The decision is which players.", "Not a baseline."],
            answer=0),
        Question(
            "q2", "mc", "Why does an exact solver beat Bayesian optimization on this lineup problem?",
            "The score and rules are linear, so integer programming can find and prove the best answer quickly.",
            options=["The score and rules are linear, so it can find the provable best quickly",
                     "BO is broken", "Exact solvers guess randomly", "The cap is too high"],
            notes=["Correct.", "BO works; it's just the wrong tool here.", "They don't guess.", "Unrelated."],
            answer=0),
        Question(
            "q3", "numeric", "Best possible score 25.84; BO found 23.18 after 40 tries. What's the regret?",
            "25.84 − 23.18 = 2.66.", category="Application", answer=2.66, tolerance=0.005),
        Question(
            "q4", "short", "With a risk penalty, greedy beat BO. What should you conclude about BO for lineup building?",
            "It isn't reliably better than a simple greedy pick there; use the exact solver, and keep greedy as the baseline.",
            category="Application",
            keywords=[["not", "isn't", "doesn't", "no"],
                      ["reliabl", "better", "earn", "worth", "complex", "exact", "greedy", "solver", "simple"]],
            model_answer="BO isn't reliably better than a simple greedy pick for lineups. Use the exact solver for the answer, and keep greedy as the simple, explainable baseline."),
        Question(
            "q5", "mc", "A grid search found the best tuning setting with only 9 tries. Why shouldn't you conclude grids are always enough?",
            "It was lucky: the best setting happened to sit exactly on a grid point.",
            options=["It was lucky: the best setting sat exactly on a grid point", "Grids are always optimal",
                     "9 tries is a lot", "BO was broken"],
            notes=["Correct.", "Only if the optimum is on the grid.", "Not the point.", "BO also found it."],
            answer=0),
    ],
)
