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
                   "Read the reliability diagram: are the top-third probabilities honest (concept 25)?",
                   "Compare AUC with 'same as last season', the baseline that beat Models 9 and 9b.",
                   "If all gates pass, review the per-season table before any use. Passing is necessary, not sufficient."],
            baseline="(1) the player's raw past rate; (2) the population or position mean. The model must beat the better of the two.",
            metric="Primary: opportunity-weighted mean squared error of the predicted next-season rate. Also checked: calibration (error <= 0.05) and Brier skill of P(top-third next season), with AUC reported against 'same as last season'.",
            validation="Rolling origin by season (concept 26): prior, player history, top-third cutoff and qualifying volume use only seasons before the test season.",
            gates=STANDARD_GATES,
        ),
        questions=["How many opportunities until a player's own data outweighs the prior (weight on data = 50%)?",
                   "Does a position-specific prior beat one shared prior out of sample? By how much?",
                   "Which players move most between the raw and posterior leaderboards, and why?",
                   "Is the 80% predictive interval honest? If coverage is low, what is the model missing?"],
    ),

    # ------------------------------------------------------------------ LEVEL 2
    Project(
        id="p2-archetypes", level=2, slot="", status="built", title="College Football Player Archetype Discovery",
        objective="Discover player types from statistical profiles, see them on a map, compare them with listed positions, and test whether they add anything to next-season prediction.",
        concept_ids=["latent-spaces", "k-means", "autoencoders", "vaes", "time-based-cv"], dataset_id="player_profiles",
        psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Descriptive value:** archetype counts by position group give a clear language for roster composition and recruiting fits ("we sign run stoppers, we need penetrators").
- **Clusters are descriptive discoveries, not predictive models.** Models 9 and 9b showed richer inputs can fail to beat "same as last season". Archetypes are richer inputs, so the Production Candidate test checks exactly that: do they improve on *this season's production + position*?
- **Your rule:** an archetype feature goes on your site only if it beats that simple baseline on a test set fixed in advance.
""",
        learning=Track(
            summary="Pick a position group and features, standardize, choose k, see the clusters on a 2-D map, name them from their profiles, and compare them with listed positions.",
            steps=["Choose a position group and a minimum-snaps filter (low-snap stats are noise).",
                   "Select features and say why each belongs.", "Look at standardized vs raw scales.",
                   "Choose k using silhouette AND stability AND football meaning.",
                   "Read the cluster profiles and name each archetype.",
                   "Compare clusters with listed positions; note where they disagree.",
                   "Write down one thing the clusters describe and one thing they can't tell you."],
            limitations=["Simulated data: three hidden styles per position group, cleaner than real football.",
                         "Per-snap rates from low-snap seasons are mostly noise; the filter matters.",
                         "K-means assumes round, similar-sized groups; real roles may form a spectrum.",
                         "Clusters are descriptive. They say who looks alike, not who will succeed."],
        ),
        production=Track(
            summary="Test whether archetypes add predictive value: do they improve next-season production forecasts beyond this season's production + listed position?",
            steps=["Upload real profiles with production and next_production, and mark them real.",
                   "Choose k and the features BEFORE running (that's the registration).",
                   "Run the rolling-origin test; every fold refits scaler, K-means and regressions on earlier seasons.",
                   "Read every gate; one failure means Learning Lab only."],
            baseline="Ridge regression on this season's production + listed position (no archetypes). 'Same as last season' is also shown.",
            metric="Mean absolute error of next-season production (rolling origin by season).",
            validation="Rolling origin (concept 26): scaler, clusters and regressions fit on seasons before the test season only.",
            gates=STANDARD_GATES + ["Clusters stable across random restarts (ARI >= 0.80)"],
        ),
        questions=["Do discovered archetypes match listed positions? Where do they disagree, and is the disagreement useful?",
                   "Did the silhouette score choose the same k you would choose for football reasons?",
                   "If archetypes don't improve next-season prediction, what are they still good for?"],
    ),

    # ------------------------------------------------------------------ LEVEL 3A
    Project(
        id="p3a-time-to-contribution", level=3, slot="A", status="built", title="Time Until Player Contribution",
        objective="Estimate how long recruits take to reach a fixed contribution threshold, handling players we never saw reach it, and test whether a Cox model predicts 'contributes by year 2' better than a rating-tier baseline.",
        concept_ids=["survival-analysis", "cox-ph", "calibration-brier", "time-based-cv"], dataset_id="roster_careers",
        psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Your roster data** (~67 teams, ~10 years) fits this directly. Every recent class is censored, so Kaplan-Meier and Cox are the right tools.
- **Model 8 connection:** Model 8 forecasts development from rating and early production. Time-to-contribution is a related outcome that keeps the recent, incomplete classes instead of dropping them.
- **Your rule:** the Production Candidate tab predicts each new class at enrollment, using only what was known before it arrived, and must beat a rating-tier baseline with calibrated probabilities.
""",
        learning=Track(
            summary="Draw Kaplan-Meier curves by group, see what goes wrong when censoring is ignored, fit a Cox model, check the proportional-hazards assumption, and switch to 'left the roster' as the outcome.",
            steps=["State the contribution threshold you're using (here: 300+ snaps in a season).",
                   "Compare Kaplan-Meier with the two naive answers.", "Compare groups (rating tiers, positions).",
                   "Fit the Cox model and read each hazard ratio with its interval.",
                   "Check proportional hazards (season 1 vs seasons 2+).",
                   "Switch the outcome to 'left the roster' and compare."],
            limitations=["Simulated careers: the threshold is built in, and position has no true effect.",
                         "Transfers are treated as censoring for the contribution outcome; if benched players leave, that biases things.",
                         "Recent classes are heavily censored, so their curves end early.",
                         "Our Cox fitter uses the Breslow method for tied seasons; fine for learning, approximate with many ties."],
        ),
        production=Track(
            summary="For each recruiting class, predict P(contributes by season 2) at enrollment using only earlier classes as known at that time; score with the Brier score.",
            steps=["Upload real roster careers with recruit_rating and mark them real.",
                   "Run the evaluation; training data is cut off at the season before each class enrolls.",
                   "Read every gate, including calibration.", "Compare AUC too, but the registered metric is the Brier score."],
            baseline="Kaplan-Meier within rating tiers (thirds of the training ratings).",
            metric="Brier score of P(contributes by season 2), per class.",
            validation="Rolling origin by recruiting class, with training outcomes truncated at the season before enrollment (concept 26).",
            gates=STANDARD_GATES,
        ),
        questions=["How much faster do top-third recruits break through, and is the effect steady over seasons?",
                   "How different is the 'left the roster' curve, and should departures be modeled separately?",
                   "Cox beats the tier baseline by only ~2% here. Is that worth the extra complexity for a staff?"],
    ),

    # ------------------------------------------------------------------ LEVEL 3B
    Project(
        id="p3b-geographic-recruiting", level=3, slot="B", status="built", title="Geographic Recruiting Analysis",
        objective="Map average recruit production by hometown with a Gaussian process (Kriging), with uncertainty, without mistaking population for talent, and test whether location adds predictive value.",
        concept_ids=["gp-regression", "kernel-functions", "rbf-kernels", "matern-kernels", "kriging"], dataset_id="hometowns",
        psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Hometown coordinates** in roster data support this map. Use production *per recruit*, adjusted for rating and population, never counts.
- **A real recruiting advantage** means recruits from a region outperform after accounting for rating, population and sample size. The uncertainty map shows where you can't tell.
- **Your rule:** a geographic term goes on your site only if it beats a rating + density baseline on later classes, with honest intervals.
""",
        learning=Track(
            summary="See where recruits come from vs how they perform, fit RBF and Matérn Kriging maps with uncertainty, read the variogram, and check regions for sample size.",
            steps=["Compare the count map with the average-production map.",
                   "Fit RBF and Matérn kernels; compare length scales and marginal likelihood.",
                   "Read the uncertainty map: where can't you tell?",
                   "Read the variogram: how quickly do players stop resembling their neighbours?",
                   "Check the regional table: players, average, standard error, players per density."],
            limitations=["Simulated hometowns and densities, not census data.",
                         "The true regional effect is deliberately small; most variation is individual.",
                         "Distances are in degrees, so east-west distances are slightly stretched at northern latitudes.",
                         "Maps use at most 600 players per fit, because Gaussian processes slow down sharply with size."],
        ),
        production=Track(
            summary="For each recruiting class, does a GP on hometown location improve production predictions beyond rating + population density?",
            steps=["Upload real hometowns with class_year, recruit_rating and local_density; mark them real.",
                   "Run the rolling-origin test (training on earlier classes only).",
                   "Read every gate, including density control and interval calibration."],
            baseline="Ridge regression on recruit_rating + local_density, with no location.",
            metric="Mean absolute error of production per recruiting class.",
            validation="Rolling origin by recruiting class (concept 26); 80% interval coverage checked (concept 25).",
            gates=STANDARD_GATES + ["Controls for population density and sample size per region"],
        ),
        questions=["Is a region producing better players, or just more players?",
                   "Does the Matérn kernel fit better than RBF? What does the length scale mean in miles?",
                   "Location adds only ~2% here. Would that change a single recruiting decision?"],
    ),

    # ------------------------------------------------------------------ LEVEL 4
    Project(
        id="p4-game-by-game", level=4, slot="", status="built", title="Game-by-Game Player Performance",
        objective="Look for hidden hot and cold form, forecast the next game from earlier games only, put honest conformal ranges on the forecasts, and test whether any of it beats the season-to-date average.",
        concept_ids=["hmms", "lstms", "seq2seq", "conformal-prediction", "time-based-cv"], dataset_id="game_logs",
        psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Your game-level data covers only parts of some seasons.** This project checks sample sufficiency first, and anything below the bar is labelled a learning exercise. No strong predictive claims.
- **Same lesson as Models 9 and 9b:** "same as last season" was hard to beat across seasons. Within a season, the season-to-date average plays that role, and the forecasts here must beat it on later seasons.
- **Your rule:** a game-level forecast or hot/cold label goes on your site only if it beats that baseline on a test set fixed in advance, with calibrated ranges.
""",
        learning=Track(
            summary="Check whether the data is sufficient, find hot/cold states with an HMM, compare 1–3 states, watch the live P(hot) for a player, compare forecasts, and (optionally) train a real LSTM.",
            steps=["Read the sample-sufficiency table. Is this a learning exercise only?",
                   "Fit HMMs with 1, 2 and 3 states (on games vs season-to-date average) and compare BIC.",
                   "Follow one player-season: live P(hot), the Viterbi path, and the hidden truth (simulated only).",
                   "Compare forecasters: season-to-date average, last game, EWMA, HMM.",
                   "Read the conformal ranges and their coverage.",
                   "Optional: train a real LSTM if PyTorch is installed."],
            limitations=["Simulated logs with hot/cold form built in; real form may not persist like this.",
                         "About 10 observed games per player-season: far too short for LSTMs; even HMM states are hard to pin down.",
                         "Missing games are skipped, so a 'next game' may be two weeks later.",
                         "Opponent strength, injuries and snaps are ignored."],
        ),
        production=Track(
            summary="Walk-forward within each test season; EWMA weight, HMM and conformal calibration fitted on earlier seasons only.",
            steps=["Upload real game logs and mark them real.", "Check the sample-sufficiency gate first.",
                   "Run the evaluation and read every gate.", "Report the best method's gain honestly, even if it's tiny."],
            baseline="Season-to-date average of the player's earlier games.",
            metric="Mean absolute error of the next game's grade.",
            validation="Rolling origin by season (concept 26) + walk-forward within season; conformal ranges calibrated on the previous season (concept 28).",
            gates=STANDARD_GATES + ["Sample sufficiency documented (games per player-season, seasons covered)"],
        ),
        questions=["Is there a hot hand beyond luck? How big is it, in grade points?",
                   "Does any forecaster beat the season-to-date average by enough to matter?",
                   "How many games per player-season would you need before trusting a sequence model?"],
    ),

    # ------------------------------------------------------------------ LEVEL 5
    Project(
        id="p5-hierarchical-development", level=5, slot="", status="built", title="Hierarchical Player Development Model",
        objective="Extend Level 1 shrinkage to partial pooling across positions, teams and experience levels, fit it by MCMC, and test whether the extra complexity beats Level 1 shrinkage and 'same as last season' on later seasons.",
        concept_ids=["hierarchical-bayes", "mcmc", "em", "mixed-effects", "time-based-cv"], dataset_id="rate_events",
        psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Model 7:** being redesigned (current candidate: draft outlook), so this project doesn't assume its design. It tests the general idea behind it: players borrowing strength from their position, team and experience level.
- **Model 8:** its gain came where production was thin. That's exactly where partial pooling changes estimates most.
- **Your rule, applied strictly:** the hierarchical model must beat **both** the Level 1 shrinkage model and "same as last season" on later seasons, converge (R-hat), and give calibrated ranges. Otherwise the extra complexity hasn't earned its place.
""",
        learning=Track(
            summary="Fit the hierarchical model with the NumPy MCMC sampler, check convergence, read the position, team and experience effects, compare with a statsmodels mixed model, and see partial pooling at work.",
            steps=["Pick a dataset: one WITH team/experience effects and one WITHOUT (the Level 1 data).",
                   "Run the sampler; read R-hat and ESS before anything else.",
                   "Read the effects: experience steps, team spread, player spread.",
                   "Compare with the mixed-effects (statsmodels) fit: same structure, different fitting.",
                   "See which players move most under pooling, and why.",
                   "Optional: read the same model written in PyMC."],
            limitations=["Simulated data; the 'with effects' dataset has team and experience effects built in, so a win there is expected.",
                         "Rates are modelled on the log-odds scale with a normal approximation for luck; very low counts are approximate.",
                         "Teaching sampler: 3 chains of 900 rounds. Production work would use PyMC/Stan and longer runs.",
                         "Players are assumed to stay with one team; transfers would need a richer model."],
        ),
        production=Track(
            summary="Season by season: fit all three approaches on earlier seasons only, then predict each returning player's next-season rate.",
            steps=["Upload real player-season rate data (with team and position) and mark it real.",
                   "Run the evaluation (about 15 seconds).",
                   "Read every gate, especially 'Beats BOTH baselines' and 'MCMC converged'.",
                   "If it doesn't beat Level 1, the honest answer is: use Level 1."],
            baseline="Two baselines: 'same as last season' (player's last raw rate) and the Level 1 position-shrinkage model. The model must beat the better one.",
            metric="Opportunity-weighted MSE of next-season rate (same as Level 1).",
            validation="Rolling origin by season (concept 26) over the last 6 seasons; R-hat checked in every fold (concept 22); 80% predictive coverage (concept 25).",
            gates=STANDARD_GATES + ["MCMC convergence diagnostics pass (R-hat, ESS)"],
        ),
        questions=["How much does the hierarchical model gain over Level 1, and where does it come from (team, experience)?",
                   "On data without team or experience effects, it loses. What does that say about adding structure?",
                   "How large is the team spread compared with the player spread? What would that mean for scheme vs talent?"],
    ),

    # ------------------------------------------------------------------ LEVEL 6A
    Project(
        id="p6a-lineup-optimization", level=6, slot="A", status="built", title="Constrained Lineup Optimization ($40M D-line)",
        objective="Choose 6 defensive linemen (at least 2 EDGE and 2 DT) under a $40M cap to maximize projected value, compare Bayesian optimization with random search, greedy and an exact solver, then back-test the choices against what actually happened.",
        concept_ids=["bayesian-optimization", "time-based-cv"], dataset_id="lineup_candidates",
        psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Same structure as your $40M D-line case:** an objective (total value), constraints (cap, roster slots) and trade-offs (risk vs expected value).
- **An optimizer amplifies its inputs:** it hunts for the most attractive projections, and those are the most likely to be overstated (the optimizer's curse). The back-test measures that directly.
- **Your rule:** an optimized lineup is only as good as its value model, which must beat a simple baseline on a test set fixed in advance. That gate can't be automated here, so it always shows as open.
""",
        learning=Track(
            summary="Set the cap rules and the risk penalty, compare the exact solver, greedy, random search and Bayesian optimization on one season's market, and trace the value-vs-cap trade-off.",
            steps=["Write down the objective, the constraints and the decision.", "Find the exact best lineup (MILP).",
                   "Compare greedy, random search and Bayesian optimization under the same budget.",
                   "Add a risk penalty and see the lineup and the method ranking change.",
                   "Trace how the best value changes with the cap (diminishing returns)."],
            limitations=["Simulated market: projections and cap hits are fictional.",
                         "Value is assumed additive: no scheme fit or chemistry between players.",
                         "Bayesian optimization is run on a problem with an exact solution, on purpose, to show when it's the wrong tool."],
        ),
        production=Track(
            summary="Each season: choose from pre-season projections only, then score the lineup by realized value. Compare with greedy and random legal lineups.",
            steps=["Upload real candidate pools with season, projected_value and realized_value; mark them real.",
                   "Run the back-test.", "Read every gate, including the optimizer's-curse check.",
                   "Remember: the projections must pass their own baseline test first."],
            baseline="Greedy value-per-dollar pick (plus the median random legal lineup for reference).",
            metric="Total realized value of the chosen 6-man group per season.",
            validation="Season-by-season back-test, choosing only from pre-season projections (concept 26).",
            gates=STANDARD_GATES + ["Input value model has itself passed Production Candidate gates"],
        ),
        questions=["How much does the exact lineup beat greedy, and how much of that survives in realized value?",
                   "How big is the optimizer's curse for the chosen players?",
                   "Where does the value-vs-cap curve flatten, and what would you do with the spare cap?"],
    ),

    # ------------------------------------------------------------------ LEVEL 6B
    Project(
        id="p6b-integrated-research", level=6, slot="B", status="built", title="Integrated Research Project",
        objective="Tie every level together: run each level's honest test on the current data and say plainly which methods beat a simple baseline and which don't; then register and run one complete study.",
        concept_ids=["time-based-cv", "calibration-brier", "conformal-prediction", "hierarchical-bayes", "bayesian-optimization"],
        dataset_id="rate_events",
        psfal="""
**The academy is separate from PSFAL and never changes it.**

- **This is the PSFAL workflow in miniature:** a registered question, a defined baseline, a chronological holdout, every gate reported (failures included), and conclusions that state uncertainty.
- **The scoreboard is your Model 8 / Models 9–9b lesson applied everywhere:** some methods earn their place, many don't, and the only way to know is to test them against a simple baseline in advance.
""",
        learning=Track(
            summary="Run the scoreboard across all six levels, read the plain-English verdicts, and walk the research template using one worked example.",
            steps=["Run the scoreboard and read which methods beat their baseline.",
                   "For each method that failed, write one sentence on why (noise, too little data, structure not present).",
                   "Pick one study and walk the template: question, data prep, baseline, advanced model, validation, uncertainty, conclusions.",
                   "Note which results are simulated (all of them, until you upload data)."],
            limitations=["On simulated data, every 'win' reflects how the simulation was built; it says nothing about real football.",
                         "Each study uses its project's default settings; tuning them could change small margins.",
                         "Running every test takes about a minute."],
        ),
        production=Track(
            summary="Register the question, metric, baseline and primary study BEFORE running anything; the study's own gates then decide the verdict.",
            steps=["Write and save the registration (it's timestamped).", "Upload real data for the primary study and mark it real.",
                   "Run the primary study's evaluation.", "Report every gate, including failures, in the conclusions."],
            baseline="Defined in the registration (each study's built-in simple baseline).",
            metric="Defined in the registration (each study's predefined metric).",
            validation="Chronological holdout, as each study implements it.",
            gates=STANDARD_GATES + ["Question, metric and baseline registered before analysis"],
        ),
        questions=["Which methods earned their complexity, and what did those have in common?",
                   "Which failures were about data (too little, too noisy) rather than the method?",
                   "If you could collect one more kind of data, which failed study would it most likely rescue?"],
    ),
]


def by_id(pid: str):
    return next((p for p in PROJECTS if p.id == pid), None)
