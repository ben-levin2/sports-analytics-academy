"""Eight projects, two per level where needed. Every project has two tracks:

  Learning Lab         - exploratory, teaches a method, may use simulated data, limitations stated.
  Production Candidate - real data, defined baseline, predefined metric, out-of-sample validation,
                         must beat the baseline, checked for leakage and calibration.

Nothing is ever labelled "production-ready" because it runs or looks interesting.
To add a project: append a Project to PROJECTS (and a dataset Spec in academy/datasets.py if needed).
"""
from __future__ import annotations

from .models import Project, Track

STANDARD_GATES = [
    "Uses appropriate real data that you have rights to use",
    "Baseline defined before modeling",
    "Evaluation metric predefined (written down before seeing results)",
    "Out-of-sample validation that respects time (train on earlier seasons, test on later)",
    "Demonstrates improvement over the baseline on the held-out data",
    "Checked for leakage (no future information in features or priors)",
    "Checked for calibration where the output is a probability or interval",
]

PROJECTS = [
    # ------------------------------------------------------------------ LEVEL 1 (fully built)
    Project(
        id="p1-bayesian-player-evaluation", level=1, slot="", status="built",
        title="Bayesian Player Evaluation",
        objective="Turn raw player rates into honest estimates by shrinking each player toward his population (or position) average by the right amount, and show how uncertainty falls as observations accumulate.",
        concept_ids=["bayesian-inference", "prior-distributions", "posterior-distributions", "likelihood",
                     "marginal-likelihood", "stochastic-processes", "time-based-cv", "calibration-brier"],
        dataset_id="rate_events",
        psfal="""
**The academy is separate from PSFAL.** It never reads, changes or deploys anything in PSFAL. Results here are practice, not inputs to your site.

- **Model 8 (Recruit Development Forecast, live and frozen):** it uses recruiting rating plus early production. That is the same core idea as this project: when a player's production is thin, lean on the prior (rating or position baseline), and trust his own data more as it piles up. Model 8's +0.087 AUC gain came in exactly that case: thin production where rating adds information. That's where this project's "weight on his data" is low.
- **Models 9 and 9b (failed):** "same as last season" scored AUC 0.713 for a top-third season next year, and richer inputs did not beat it. That's why this project's Production Candidate track must beat the player's raw past rate, not just look smarter. Expect shrinkage to help most for low-volume players and very little for high-volume ones.
- **Model 7 (being redesigned, current candidate a draft-outlook model):** not treated as a fixed target here. Level 5 teaches partial pooling, which a draft-outlook model could use, but nothing here assumes Model 7's design.
- **Your rule, built in:** a method goes on your real site only if it beats a simple baseline on a test set fixed in advance. Here the gates are fixed in code before any data is evaluated, and simulated data never passes. Even a passing verdict here would still need PSFAL's own pre-set test before it goes anywhere near your site.
""",
        learning=Track(
            summary="Explore shrinkage interactively: pick a dataset and a player, then see the raw stat, the population average, the prior, the likelihood and the posterior on one chart.",
            steps=["Choose a dataset (QB completion %, LB tackle share, or pressure rate by position).",
                   "Fit the prior from all players (empirical Bayes, lesson L1.5).",
                   "Pick a low-volume and a high-volume player and compare how far each is shrunk.",
                   "Switch to position-specific priors and see who moves.",
                   "Read the uncertainty curve: how many opportunities until the interval is narrow?",
                   "Compare the raw leaderboard with the posterior leaderboard."],
            limitations=["Simulated data: players and numbers are fictional.",
                         "Assumes each opportunity is independent with a fixed rate within a season.",
                         "Ignores opponent, scheme and teammates (e.g., tackle share depends on the rest of the defense).",
                         "A prior fitted on all seasons uses information from the future relative to early seasons. Fine for learning, not for prediction (see the Production Candidate track)."],
        ),
        production=Track(
            summary="Test whether shrunk estimates predict next-season rates better than the baselines, using only past seasons.",
            steps=["Upload real player-season data on the Data Sources page and mark it as real.",
                   "Run the rolling-origin evaluation (train on seasons < T, test on T, for every T).",
                   "Read every gate. A single failure means Learning Lab only.",
                   "If all gates pass, review the per-season table before any use. Passing is necessary, not sufficient."],
            baseline="(1) the player's raw past rate; (2) the population or position mean. The model must beat the better of the two.",
            metric="Opportunity-weighted mean squared error of the predicted next-season rate.",
            validation="Rolling origin by season: prior and player history use only seasons before the test season.",
            gates=STANDARD_GATES,
        ),
        questions=["How many opportunities until a player's own data outweighs the prior (weight on data = 50%)?",
                   "Does a position-specific prior beat one shared prior out of sample? By how much?",
                   "Which players move most between the raw and posterior leaderboards, and why?",
                   "Is the 80% predictive interval honest? If coverage is low, what is the model missing?"],
    ),

    # ------------------------------------------------------------------ LEVEL 2
    Project(
        id="p2-archetypes", level=2, slot="", title="College Football Player Archetype Discovery",
        objective="Discover player types from statistical profiles, visualize them, and compare them with conventional position labels.",
        concept_ids=["latent-spaces", "k-means", "autoencoders", "vaes"], dataset_id="player_profiles",
        psfal="Archetypes can describe roster composition (how many run-stopping vs pass-rushing linemen a program develops) and give a vocabulary for recruiting fits. Clusters are **descriptive discoveries**: they say who looks similar, not who will succeed.",
        learning=Track(
            summary="Cluster simulated edge defenders, choose k, project to 2-D, and cross-tabulate clusters against listed positions.",
            steps=["Select features and justify each one.", "Standardize every feature.",
                   "Run K-means for k = 2..6 and compare silhouette scores.", "Profile each cluster (feature means).",
                   "Reduce to 2-D with PCA and plot the clusters.", "Cross-tabulate clusters vs listed position.",
                   "Check stability across random seeds."],
            code='''import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from academy.data import make_edge_defenders

edges = make_edge_defenders()                       # SIMULATED (hidden true_archetype exists only here)
features = ["pressure_rate", "run_stop_rate", "tackles_per_game", "coverage_snap_pct", "weight_lbs"]
Z = StandardScaler().fit_transform(edges[features])

scores = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=0).fit_predict(Z)) for k in range(2, 7)}
k = max(scores, key=scores.get)
print("Silhouette by k:", {kk: round(v, 3) for kk, v in scores.items()}, "-> k =", k)

edges["cluster"] = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
print("\\nCluster profiles:\\n", edges.groupby("cluster")[features].mean().round(2).to_string())
print("\\nClusters vs LISTED position (what a roster sheet says):")
print(pd.crosstab(edges["cluster"], edges["listed_position"]))
print("\\nStability across seeds (ARI):",
      round(adjusted_rand_score(edges["cluster"], KMeans(k, n_init=10, random_state=9).fit_predict(Z)), 3))

pc = PCA(2).fit_transform(Z)
fig, ax = plt.subplots()
for c in sorted(edges["cluster"].unique()):
    m = edges["cluster"] == c
    ax.scatter(pc[m, 0], pc[m, 1], s=22, label=f"cluster {c}")
ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.legend(); ax.set_title("Edge archetypes in 2-D (simulated)")
''',
            limitations=["150 simulated players with cleanly separated types; real profiles overlap far more.",
                         "Thousands of real player observations are needed before clusters are stable.",
                         "Clusters are descriptive. They do not predict success without separate validation."],
        ),
        production=Track(
            summary="An archetype model is only a production candidate if the archetypes ADD predictive value to a downstream question.",
            steps=["Fix a downstream outcome (e.g., next-season production) and its metric first.",
                   "Fit clusters on earlier seasons only.", "Compare a baseline model with and without archetype features on later seasons.",
                   "Check cluster stability over seeds and seasons."],
            baseline="Downstream model using position label only.",
            metric="Out-of-sample error on the downstream outcome (fixed in advance).",
            validation="Clusters and models fit on earlier seasons, evaluated on later seasons.",
            gates=STANDARD_GATES + ["Clusters stable across random seeds and seasons (ARI reported)"],
        ),
        questions=["Do discovered archetypes match listed positions? Where do they disagree, and is the disagreement useful?",
                   "Which features drive the grouping? What happens if you drop weight?",
                   "When is a continuous spectrum of roles a better description than discrete clusters?"],
    ),

    # ------------------------------------------------------------------ LEVEL 3A
    Project(
        id="p3a-time-to-contribution", level=3, slot="A", title="Time Until Player Contribution",
        objective="Estimate how long recruits take to become contributors (a fixed snap or production threshold), correctly handling players we never saw reach it.",
        concept_ids=["survival-analysis", "cox-ph", "calibration-brier", "time-based-cv"], dataset_id="roster_careers",
        psfal="Roster data covering about 67 teams over 10 years is the natural input. Time-to-contribution by rating band, position and program is a development question PSFAL cares about directly. Players still on roster at the end of the data, or who left, are **censored**, not failures.",
        learning=Track(
            summary="Build a Kaplan-Meier curve by hand on simulated roster careers and compare rating tiers.",
            steps=["Define contribution with one consistent threshold (e.g., 300+ snaps in a season).",
                   "Record seasons observed and whether the event happened (censoring flag).",
                   "Compute a Kaplan-Meier curve; compare with the naive 'drop censored players' approach.",
                   "Compare rating tiers.", "Repeat with 'left the roster' as the event (an alternative outcome)."],
            code='''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from academy.data import make_roster_careers

df = make_roster_careers()            # SIMULATED: ~67 teams x 10 classes; contribution = 300+ snaps
print(f"{len(df)} recruits, {df['contributed'].mean():.0%} observed contributing, rest censored")

def kaplan_meier(t, event):
    """Share still NOT contributing after each season, counting censored players correctly."""
    times = np.arange(1, t.max() + 1)
    at_risk, surv, out = len(t), 1.0, []
    for s in times:
        d = ((t == s) & (event == 1)).sum()          # contributed in season s
        c = ((t == s) & (event == 0)).sum()          # censored in season s
        surv *= 1 - d / at_risk if at_risk else 1
        out.append(surv)
        at_risk -= d + c
    return times, np.array(out)

fig, ax = plt.subplots()
df["tier"] = pd.cut(df["recruit_rating"], [0, 0.84, 0.89, 1], labels=["< .84", ".84-.89", ".89+"])
for tier, g in df.groupby("tier", observed=True):
    t, s = kaplan_meier(g["seasons_observed"].to_numpy(), g["contributed"].to_numpy())
    ax.step(t, 1 - s, where="post", label=f"rating {tier}")
    print(f"rating {tier}: {1 - s[min(2, len(s) - 1)]:.0%} contributing by season 3 (Kaplan-Meier)")

naive = df[df["contributed"] == 1]["seasons_observed"].mean()
print(f"\\nNaive mean time among contributors only: {naive:.2f} seasons (biased: ignores censoring)")
ax.set_xlabel("Seasons since enrolling"); ax.set_ylabel("Share who have contributed")
ax.set_title("Time to first contribution (simulated)"); ax.legend()
''',
            limitations=["Simulated careers; the contribution threshold is built in, not estimated.",
                         "Recent classes are heavily censored; their curves end early.",
                         "Transfers in/out and redshirts need explicit rules in real data."],
        ),
        production=Track(
            summary="A contribution-timing model qualifies only if it beats a rating-tier baseline on later recruiting classes, with calibrated probabilities.",
            steps=["Fix the threshold and the metric before looking at outcomes.",
                   "Train on early classes, test on later classes (respecting censoring).",
                   "Compare a Cox model with the Kaplan-Meier-by-tier baseline.",
                   "Check calibration of P(contributes by year 2) with a reliability diagram and Brier score."],
            baseline="Kaplan-Meier curves by rating tier.",
            metric="Brier score for P(contributes by season 2), and concordance (C-index).",
            validation="Train on classes 2016-2020, test on later classes; never use outcomes after the training cut-off.",
            gates=STANDARD_GATES,
        ),
        questions=["Why is dropping censored players biased? In which direction?",
                   "Does the gap between rating tiers shrink after year 3?",
                   "How should a player who transferred out before contributing be treated, as censored or as a competing outcome?"],
    ),

    # ------------------------------------------------------------------ LEVEL 3B
    Project(
        id="p3b-geographic-recruiting", level=3, slot="B", title="Geographic Recruiting Analysis",
        objective="Build a smooth map of player production from hometown coordinates with Gaussian processes (Kriging), with uncertainty, without mistaking population for talent.",
        concept_ids=["gp-regression", "kernel-functions", "rbf-kernels", "matern-kernels", "kriging"], dataset_id="hometowns",
        psfal="Hometown latitude and longitude in roster data allow a map of *average production per recruit* by region. The trap: regions with more people produce more players, so a cluster of productive players is not by itself a recruiting advantage.",
        learning=Track(
            summary="Fit GP regressions with RBF and Matérn kernels to simulated hometowns, compare them, and separate density from quality.",
            steps=["Plot hometowns: notice they follow population.", "Fit a GP with an RBF kernel plus noise.",
                   "Fit a GP with a Matérn kernel; compare fitted length scales and log marginal likelihood (L1.5!).",
                   "Predict production on a grid with standard deviations.",
                   "Show that player COUNTS track density, while AVERAGE production barely varies."],
            code='''import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
from academy.data import make_hometown_production

df = make_hometown_production()                       # SIMULATED hometowns, not census data
sample = df.sample(800, random_state=0)               # GPs scale as n^3: start small
X = sample[["lon", "lat"]].to_numpy()
y = sample["production"].to_numpy() - sample["production"].mean()

# length scales in degrees; bounds stop the fit from collapsing to pure noise
kernels = {"RBF": C(10.0) * RBF(length_scale=8.0, length_scale_bounds=(2, 60)) + WhiteKernel(100.0),
           "Matern 1.5": C(10.0) * Matern(length_scale=8.0, length_scale_bounds=(2, 60), nu=1.5) + WhiteKernel(100.0)}
fits = {}
for name, k in kernels.items():
    gp = GaussianProcessRegressor(kernel=k, normalize_y=False, random_state=0).fit(X, y)
    fits[name] = gp
    print(f"{name:<10} log marginal likelihood {gp.log_marginal_likelihood_value_:.1f}   kernel: {gp.kernel_}")

# Population vs quality: counts follow density, average production barely moves
df["dense"] = df["local_density"] > df["local_density"].median()
print("\\nPlayers from dense areas:", df["dense"].mean().round(2),
      "| mean production dense vs sparse:",
      df.groupby("dense")["production"].mean().round(1).to_dict())

gx, gy = np.meshgrid(np.linspace(-123, -70, 60), np.linspace(25, 49, 35))
mu, sd = fits["Matern 1.5"].predict(np.c_[gx.ravel(), gy.ravel()], return_std=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
a = axes[0].contourf(gx, gy, mu.reshape(gx.shape), 14, cmap="coolwarm"); plt.colorbar(a, ax=axes[0])
axes[0].scatter(X[:, 0], X[:, 1], s=3, c="k", alpha=0.4); axes[0].set_title("Predicted production vs average")
b = axes[1].contourf(gx, gy, sd.reshape(gx.shape), 14, cmap="viridis"); plt.colorbar(b, ax=axes[1])
axes[1].set_title("Uncertainty (SD): high where no players")
''',
            limitations=["Simulated metros and densities; real work needs census population as a control.",
                         "The true spatial effect is deliberately small. Most variation is individual noise.",
                         "800-point subsample; full data needs sparse or approximate GPs.",
                         "Production depends on the college program too, which is not modeled here."],
        ),
        production=Track(
            summary="A geographic model qualifies only if location adds out-of-sample predictive value beyond rating and program.",
            steps=["Fix the outcome and metric.", "Baseline: rating + program, no location.",
                   "Add the GP spatial term; evaluate on later recruiting classes.",
                   "Check that interval coverage holds in sparse regions."],
            baseline="Regression on rating and program, no geography.",
            metric="Out-of-sample MAE of production; interval coverage.",
            validation="Fit on earlier classes, test on later classes; spatial blocks held out to detect overfitting to dense areas.",
            gates=STANDARD_GATES + ["Controls for population density and sample size per region"],
        ),
        questions=["Is a region with many productive players producing better players, or just more players?",
                   "When does the Matérn kernel fit better than RBF, and what does the length scale mean in miles?",
                   "Where is the map most uncertain, and how should that change recruiting decisions?"],
    ),

    # ------------------------------------------------------------------ LEVEL 4
    Project(
        id="p4-game-by-game", level=4, slot="", title="Game-by-Game Player Performance",
        objective="Look for hidden hot and cold states and test simple forecasts of next-game performance, without overclaiming from short sequences.",
        concept_ids=["hmms", "lstms", "seq2seq", "conformal-prediction"], dataset_id="game_logs",
        psfal="Game-level data only covers parts of some seasons. These are **learning exercises** whenever the sample is too short. 12 games per player is far too few for an LSTM.",
        learning=Track(
            summary="Run an HMM forward filter for hot/cold form and compare forecasts with walk-forward validation.",
            steps=["Plot several players' game sequences.", "Run a two-state HMM filter with stated parameters.",
                   "Compare season-mean and recency-weighted forecasts, walk-forward.",
                   "Wrap the best forecast in a split-conformal interval and check coverage."],
            code='''import numpy as np
import pandas as pd
from scipy import stats
from academy.data import make_weekly_performance

perf = make_weekly_performance()                      # SIMULATED; 12 games per player (a SHORT sequence)

# --- HMM forward filter: P(hot | games so far) for one player, parameters fixed for teaching ---
P = np.array([[0.8, 0.2], [0.3, 0.7]])               # hot/cold transitions (from L1.6)
means, sd = np.array([72.0, 60.0]), 8.0               # typical grade when hot / cold
g = perf[perf["player_id"] == "W001"].sort_values("week")["grade"].to_numpy()
belief = np.array([0.5, 0.5])
for wk, x in enumerate(g, start=1):
    belief = belief @ P                                # predict: state may switch
    belief *= stats.norm.pdf(x, means, sd)             # update: how well each state explains the grade
    belief /= belief.sum()
    print(f"week {wk:>2}: grade {x:5.1f}  P(hot) = {belief[0]:.2f}")

# --- Walk-forward forecasts: use only earlier weeks ---
errs = {"season mean": [], "EWMA 0.3": []}
for _, d in perf.groupby("player_id"):
    y = d.sort_values("week")["grade"].to_numpy()
    for t in range(3, len(y)):
        errs["season mean"].append(y[t] - y[:t].mean())
        errs["EWMA 0.3"].append(y[t] - pd.Series(y[:t]).ewm(alpha=0.3).mean().iloc[-1])
for k, e in errs.items():
    print(f"{k:<12} RMSE {np.sqrt(np.mean(np.square(e))):.2f}")

# --- Split conformal: 80% interval from calibration residuals ---
e = np.abs(np.array(errs["EWMA 0.3"]))
half = len(e) // 2
q = np.quantile(e[:half], 0.80 * (1 + 1 / half))
print(f"\\n80% conformal half-width: +/-{q:.1f} grade points; coverage on the other half: {(e[half:] <= q).mean():.0%}")
print("LEARNING EXERCISE: 12 games per player cannot support strong predictive claims.")
''',
            limitations=["Simulated, 12 games per player: far too short for LSTMs or reliable HMM fitting.",
                         "HMM parameters are fixed by hand here; fitting them needs EM (Level 5).",
                         "Real game logs have missing games, opponent effects and injuries."],
        ),
        production=Track(
            summary="Only consider this a candidate with full-season game logs across many seasons, and a forecast that beats the season mean walk-forward.",
            steps=["Collect complete game logs; document coverage gaps.", "Fix the forecast target and metric.",
                   "Walk-forward validation across seasons.", "Conformal interval coverage checked on later seasons."],
            baseline="Season-to-date mean.",
            metric="Walk-forward RMSE of next-game performance; conformal interval coverage.",
            validation="Walk-forward within season, and train-on-earlier-seasons across seasons.",
            gates=STANDARD_GATES + ["Sample sufficiency documented (games per player, seasons covered)"],
        ),
        questions=["Is the best recency weight better than luck alone would produce? (Compare with L1.6.)",
                   "What would an HMM add over an EWMA, and what would an LSTM add over an HMM?",
                   "How many games per player would you need before trusting a sequence model?"],
    ),

    # ------------------------------------------------------------------ LEVEL 5
    Project(
        id="p5-hierarchical-development", level=5, slot="", title="Hierarchical Player Development Model",
        objective="Extend the Level 1 shrinkage project to partial pooling across positions, teams, experience levels and populations.",
        concept_ids=["hierarchical-bayes", "mcmc", "em", "mixed-effects"], dataset_id="rate_events",
        psfal="**Separate from PSFAL; never changes it.** Model 7 is being redesigned (the current candidate is a draft-outlook model), so this project does not assume its design. It teaches partial pooling: players share information with their position group, and groups share information with the population. A draft-outlook model, or a Model 8-style forecast, could use that idea, but only after beating a simple baseline on a test set fixed in advance.",
        learning=Track(
            summary="Compare no pooling, complete pooling and partial pooling (by position) on next-season prediction, then outline the full PyMC model.",
            steps=["No pooling: each player's raw rate.", "Complete pooling: one population mean for everyone.",
                   "Partial pooling: position-specific priors fitted from data (empirical Bayes).",
                   "Compare all three with rolling-origin validation.",
                   "Read the PyMC sketch: what MCMC adds (uncertainty in the priors themselves) and how to check convergence (R-hat, effective sample size)."],
            code='''from academy.data import make_pressure_by_position
from academy import shrinkage as S

df = make_pressure_by_position()                       # SIMULATED pressure rates for EDGE, DT, LB
pooled = S.evaluate_next_season(df, group=None)        # one prior for everyone
by_pos = S.evaluate_next_season(df, group="position")  # partial pooling within position

print("Weighted MSE of next-season pressure rate (lower is better)")
print(f"  no pooling (raw past rate):       {by_pos['overall']['raw']:.6f}")
print(f"  complete pooling (position mean): {by_pos['overall']['population']:.6f}")
print(f"  partial pooling, one prior:       {pooled['overall']['model']:.6f}")
print(f"  partial pooling, by position:     {by_pos['overall']['model']:.6f}")

# Full hierarchical Bayesian version (needs `pip install pymc`; not run here):
PYMC_SKETCH = """
import pymc as pm
with pm.Model() as m:
    mu_pop  = pm.Normal("mu_pop", -2.5, 1)                 # population mean (logit scale)
    sd_pos  = pm.HalfNormal("sd_pos", 0.5)
    mu_pos  = pm.Normal("mu_pos", mu_pop, sd_pos, shape=n_positions)
    sd_play = pm.HalfNormal("sd_play", 0.5)
    z       = pm.Normal("z", mu_pos[pos_idx], sd_play, shape=n_players)
    pm.Binomial("y", n=opportunities, p=pm.math.invlogit(z[player_idx]), observed=successes)
    idata = pm.sample(1000, tune=1000, chains=4)           # MCMC; then check R-hat < 1.01
"""
print(PYMC_SKETCH)
''',
            limitations=["Empirical Bayes plugs in point estimates of the priors and so understates uncertainty; full MCMC fixes that.",
                         "Team and experience levels are not yet in the learning code; they are the next extension.",
                         "Simulated data where the position structure is known to exist."],
        ),
        production=Track(
            summary="A hierarchical model qualifies if it beats the Level 1 position-prior model out of sample, converges, and produces calibrated intervals.",
            steps=["Pre-register the comparison against the Level 1 model.", "Fit on earlier seasons with MCMC.",
                   "Check convergence: R-hat < 1.01, adequate effective sample size, no divergences.",
                   "Evaluate on later seasons; check interval coverage."],
            baseline="Level 1 position-prior shrinkage model.",
            metric="Opportunity-weighted next-season MSE; 80% interval coverage.",
            validation="Rolling origin by season, identical to Level 1.",
            gates=STANDARD_GATES + ["MCMC convergence diagnostics pass (R-hat, ESS, divergences)"],
        ),
        questions=["Which players benefit most from position-level pooling, and why?",
                   "How is a mixed-effects model with a random player effect related to this hierarchical model?",
                   "When does adding team-level pooling help, and when does it just add noise?"],
    ),

    # ------------------------------------------------------------------ LEVEL 6A
    Project(
        id="p6a-lineup-optimization", level=6, slot="A", title="Constrained Lineup Optimization",
        objective="Choose a defensive-line group that maximizes expected value under a cap, and understand the trade-offs between value, cost and risk.",
        concept_ids=["bayesian-optimization"], dataset_id="lineup_candidates",
        psfal="The same structure as the cap-constrained defensive line project: an objective (value), constraints (cap, roster slots by position) and trade-offs (risk vs expected value). Bayesian optimization belongs one level up: tuning the settings of the models that produce the value estimates.",
        learning=Track(
            summary="Brute-force the best 2 EDGE + 2 DT group under a cap, then trace how the answer changes with the cap and with risk aversion.",
            steps=["Write the objective function.", "Write the constraints (cap, positions).",
                   "Search every feasible lineup.", "Trace the value-vs-cap trade-off curve.",
                   "Add a risk penalty (value minus lambda x SD) and see the choice change."],
            code='''import itertools
import numpy as np
import matplotlib.pyplot as plt
from academy.data import make_dl_free_agents

c = make_dl_free_agents()                              # SIMULATED candidates
cap_hit, value, sd = c["cap_hit_m"].to_numpy(), c["value"].to_numpy(), c["value_sd"].to_numpy()
edge = np.flatnonzero(c["position"] == "EDGE"); dt = np.flatnonzero(c["position"] == "DT")

# Every feasible shape: 2 EDGE + 2 DT (positional constraint built into the search space)
lineups = np.array([e + d for e in itertools.combinations(edge, 2) for d in itertools.combinations(dt, 2)])
total_cap = cap_hit[lineups].sum(axis=1)
total_value = value[lineups].sum(axis=1)
total_sd = np.sqrt((sd[lineups] ** 2).sum(axis=1))
print(f"{len(lineups)} possible lineups")

def best_lineup(cap, risk=0.0):
    score = total_value - risk * total_sd                  # objective (risk-adjusted value)
    score = np.where(total_cap <= cap, score, -np.inf)     # cap constraint
    i = int(np.argmax(score))
    return (c.iloc[lineups[i]], score[i]) if np.isfinite(score[i]) else (None, None)

lineup, score = best_lineup(cap=30)
print("Best 2 EDGE + 2 DT under $30M:\\n", lineup[["player_id", "position", "cap_hit_m", "value", "value_sd"]].to_string(index=False))
print(f"Total cap {lineup['cap_hit_m'].sum():.1f}M, expected value {lineup['value'].sum():.2f}")

for lam in (1.0, 2.0, 3.0):
    safe, _ = best_lineup(cap=30, risk=lam)
    print(f"risk penalty lambda = {lam}: {list(safe['player_id'])}, expected value {safe['value'].sum():.2f}")

caps = np.arange(14, 46, 2)
vals = []
for cap in caps:
    best, _ = best_lineup(cap)
    vals.append(np.nan if best is None else best["value"].sum())
fig, ax = plt.subplots()
ax.plot(caps, vals, marker="o"); ax.set_xlabel("Cap budget ($M)"); ax.set_ylabel("Best expected value")
ax.set_title("Diminishing returns to cap space (simulated)")
''',
            limitations=["Values and costs are simulated; real value estimates carry model uncertainty.",
                         "Brute force works for small pools only; larger problems need integer programming.",
                         "Ignores scheme fit and interactions between players."],
        ),
        production=Track(
            summary="An optimizer is only as good as its value estimates. It qualifies only if lineups it chooses outperform a simple heuristic in back-tests.",
            steps=["Validate the value model first (it must pass its own gates).", "Back-test chosen lineups on later seasons.",
                   "Compare with a heuristic (e.g., best value per dollar).", "Stress-test with value uncertainty."],
            baseline="Greedy value-per-dollar heuristic.",
            metric="Realized value of chosen lineups in later seasons.",
            validation="Choose using information available before the season; score on that season.",
            gates=STANDARD_GATES + ["Input value model has itself passed Production Candidate gates"],
        ),
        questions=["Where does the value-vs-cap curve flatten, and what does that imply?",
                   "How much expected value do you give up to reduce risk?",
                   "Which model settings would you tune with Bayesian optimization, and what is the objective?"],
    ),

    # ------------------------------------------------------------------ LEVEL 6B
    Project(
        id="p6b-integrated-research", level=6, slot="B", title="Integrated Research Project",
        objective="Combine methods from earlier levels into one complete, honest football analytics study.",
        concept_ids=["time-based-cv", "calibration-brier", "conformal-prediction", "hierarchical-bayes"],
        dataset_id="rate_events",
        psfal="This is the template for any PSFAL study: a registered question, a baseline, a chronological holdout, and conclusions that state uncertainty and limitations.",
        learning=Track(
            summary="Walk the full research template on simulated data, end to end, using the Level 1 engine as the 'advanced model'.",
            steps=["Research question (written first).", "Data preparation.", "Baseline model.", "Advanced model.",
                   "Validation (time-based).", "Uncertainty analysis (interval coverage).", "Written conclusions."],
            code='''from academy.data import make_qb_completions
from academy import shrinkage as S

QUESTION = "Do shrunk completion rates predict next-season completion % better than raw past rates?"
print("Research question:", QUESTION)

df = make_qb_completions()                             # SIMULATED; swap for real data via Data Sources
df = df[df["opportunities"] >= 1]                      # data preparation
print(f"Data: {df['player_id'].nunique()} QBs, {df['season'].nunique()} seasons, {len(df)} player-seasons")

res = S.evaluate_next_season(df)                       # baseline + advanced model + time-based validation
o = res["overall"]
print(f"Baseline (raw past rate) MSE: {o['raw']:.5f}")
print(f"Baseline (league mean)   MSE: {o['population']:.5f}")
print(f"Advanced (shrinkage)     MSE: {o['model']:.5f}  ({res['gain']:+.1%} vs best baseline)")
print(f"Uncertainty: 80% predictive intervals covered {res['coverage']:.1%} of next-season outcomes")

print("\\nConclusion (draft): on simulated data, shrinkage beat both baselines in",
      f"{res['wins']} of {res['n_seasons']} test seasons. This is a LEARNING result:",
      "it says nothing about real players until repeated on real data with all gates passing.")
''',
            limitations=["Simulated data; the conclusion is about the method, not about football."],
        ),
        production=Track(
            summary="A full study is a production candidate only when every standard gate passes on real data and the write-up states limitations.",
            steps=["Register the question, metric and baseline before touching outcomes.", "Run on real data.",
                   "Report all gates, including failures.", "Write conclusions with uncertainty."],
            baseline="Defined in the registration.", metric="Defined in the registration.",
            validation="Chronological holdout, locked before analysis.",
            gates=STANDARD_GATES + ["Question, metric and baseline registered before analysis"],
        ),
        questions=["What would change your conclusion?", "Which result would you most like a skeptic to reproduce?",
                   "What is the smallest improvement that would matter for a real decision?"],
    ),
]


def by_id(pid: str):
    return next((p for p in PROJECTS if p.id == pid), None)
