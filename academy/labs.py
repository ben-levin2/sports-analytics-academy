"""Python Lab exercises. All data is SIMULATED (see academy/data.py).

Each exercise has a starter (with TODOs and self-checks) and a full solution.
To add one: append an Exercise to EXERCISES.
"""
from __future__ import annotations

from .models import Exercise


def check(name, cond) -> None:
    """Used inside starter code: prints a pass/fail line. `cond` is a zero-argument function."""
    try:
        ok = bool(cond())
    except Exception:
        ok = False
    print(("✅ " if ok else "❌ ") + name + ("" if ok else "  (not there yet)"))


EXERCISES = [
    Exercise(
        id="lab-01-pandas-linebackers", level="Beginner", title="Explore linebacker production with pandas",
        goal="Load simulated linebacker seasons, compute tackles per game, and see how production grows with experience.",
        concepts=["pandas", "groupby", "derived columns"],
        starter='''import numpy as np
import pandas as pd
from academy.labs import check
from academy.data import make_linebackers

lbs = make_linebackers()          # SIMULATED data (fictional players)
print(lbs.head(), "\\n")

# Step 1: add a column "tpg" = tackles per game
lbs["tpg"] = None                 # TODO: replace None

# Step 2: average tpg for each season (a Series indexed by season)
avg_by_season = None              # TODO: use groupby

print(avg_by_season)

# ---- self-check (don't edit) ----
check("tpg column is tackles / games", lambda: np.allclose(lbs["tpg"].astype(float), lbs["tackles"] / lbs["games"]))
check("average rises with experience", lambda: avg_by_season.is_monotonic_increasing)
''',
        solution='''import numpy as np
import pandas as pd
from academy.labs import check
from academy.data import make_linebackers

lbs = make_linebackers()
lbs["tpg"] = lbs["tackles"] / lbs["games"]
avg_by_season = lbs.groupby("season")["tpg"].mean()
print(avg_by_season)

check("tpg column is tackles / games", lambda: np.allclose(lbs["tpg"].astype(float), lbs["tackles"] / lbs["games"]))
check("average rises with experience", lambda: avg_by_season.is_monotonic_increasing)
''',
        hint="`df['a'] / df['b']` divides columns row by row. `df.groupby('col')['x'].mean()` averages x within each group.",
    ),
    Exercise(
        id="lab-02-numpy-luck", level="Beginner", title="How much of a QB's game is luck? (NumPy)",
        goal="Simulate thousands of games for a QB whose true completion rate never changes, and see how much the box score still swings.",
        concepts=["NumPy random", "binomial distribution", "simulation"],
        starter='''import numpy as np
import matplotlib.pyplot as plt
from academy.labs import check

rng = np.random.default_rng(42)
true_rate, attempts = 0.62, 30            # a QB who is exactly 62% every game

# TODO: simulate 10,000 games -> completions per game (hint: rng.binomial)
completions = None
# TODO: completion percentage per game
pct = None
# TODO: share of games with a completion % of 80% or more
share_80 = None

fig, ax = plt.subplots()
if pct is not None:
    ax.hist(pct, bins=20)
ax.set_xlabel("Completion % in a game"); ax.set_title("Same QB, 10,000 games")

# ---- self-check ----
check("10,000 simulated games", lambda: len(pct) == 10_000)
check("mean near 62%", lambda: abs(pct.mean() - 0.62) < 0.01)
check("share of 80%+ games between 1% and 8%", lambda: 0.01 < share_80 < 0.08)
if share_80 is not None:
    print(f"A true 62% QB posts 80%+ in {share_80:.1%} of games purely by chance.")
''',
        solution='''import numpy as np
import matplotlib.pyplot as plt
from academy.labs import check

rng = np.random.default_rng(42)
true_rate, attempts = 0.62, 30

completions = rng.binomial(attempts, true_rate, size=10_000)
pct = completions / attempts
share_80 = (pct >= 0.80).mean()

fig, ax = plt.subplots()
ax.hist(pct, bins=20)
ax.set_xlabel("Completion % in a game"); ax.set_title("Same QB, 10,000 games")

check("10,000 simulated games", lambda: len(pct) == 10_000)
check("mean near 62%", lambda: abs(pct.mean() - 0.62) < 0.01)
check("share of 80%+ games between 1% and 8%", lambda: 0.01 < share_80 < 0.08)
print(f"A true 62% QB posts 80%+ in {share_80:.1%} of games purely by chance.")
''',
        hint="`rng.binomial(n, p, size=...)` draws the number of successes in n tries. Divide by n for a percentage. `(pct >= 0.8).mean()` is the share of True values.",
    ),
    Exercise(
        id="lab-03-matplotlib-development", level="Beginner", title="Plot player development curves (Matplotlib)",
        goal="Draw production-by-season lines for the five highest-rated recruits.",
        concepts=["Matplotlib", "filtering", "line plots"],
        starter='''import matplotlib.pyplot as plt
from academy.labs import check
from academy.data import make_development

dev = make_development()                  # SIMULATED
top5 = (dev.groupby("player_id")["recruit_rating"].first()
           .sort_values(ascending=False).head(5).index)

fig, ax = plt.subplots()
for pid in top5:
    g = dev[dev["player_id"] == pid].sort_values("season")
    # TODO: draw this player's production vs season with ax.plot(..., marker="o", label=pid)
    pass

ax.set_xlabel("Season"); ax.set_ylabel("Production index (0-100)")
ax.set_title("Top-5 recruits: development curves (simulated)")
ax.legend()

# ---- self-check ----
check("five lines drawn", lambda: len(ax.lines) == 5)
check("axes are labelled", lambda: bool(ax.get_xlabel()) and bool(ax.get_ylabel()))
''',
        solution='''import matplotlib.pyplot as plt
from academy.labs import check
from academy.data import make_development

dev = make_development()
top5 = (dev.groupby("player_id")["recruit_rating"].first()
           .sort_values(ascending=False).head(5).index)

fig, ax = plt.subplots()
for pid in top5:
    g = dev[dev["player_id"] == pid].sort_values("season")
    ax.plot(g["season"], g["production"], marker="o", label=pid)

ax.set_xlabel("Season"); ax.set_ylabel("Production index (0-100)")
ax.set_title("Top-5 recruits: development curves (simulated)")
ax.legend()

check("five lines drawn", lambda: len(ax.lines) == 5)
check("axes are labelled", lambda: bool(ax.get_xlabel()) and bool(ax.get_ylabel()))
''',
        hint="`ax.plot(x, y, marker='o', label='name')`. Do you see the curves flatten? Not every player peaks at the same time.",
    ),
    Exercise(
        id="lab-04-shrinkage", level="Intermediate", title="Shrink small-sample quarterbacks toward the league mean",
        goal="Compare raw completion % to Bayesian-shrunk estimates and test which is closer to each QB's hidden true rate.",
        concepts=["shrinkage", "Beta-Binomial", "error metrics"],
        starter='''import numpy as np
from academy.labs import check
from academy.data import make_quarterbacks

qbs = make_quarterbacks()     # SIMULATED. Includes the hidden true_rate (impossible with real data!)
league_mean = qbs["completions"].sum() / qbs["attempts"].sum()
strength = 150                # prior = "league average, worth 150 attempts". Try 20 and 1000.

raw = qbs["completions"] / qbs["attempts"]

# TODO: shrunk = (completions + league_mean * strength) / (attempts + strength)
shrunk = None

mse_raw = ((raw - qbs["true_rate"]) ** 2).mean()
print(f"MSE raw:    {mse_raw:.5f}")
if shrunk is not None:
    mse_shrunk = ((shrunk - qbs["true_rate"]) ** 2).mean()
    print(f"MSE shrunk: {mse_shrunk:.5f}")

# ---- self-check ----
check("shrunk estimates computed", lambda: len(shrunk) == len(qbs))
check("shrinkage beats the raw rate", lambda: mse_shrunk < mse_raw)
''',
        solution='''import numpy as np
from academy.labs import check
from academy.data import make_quarterbacks

qbs = make_quarterbacks()
league_mean = qbs["completions"].sum() / qbs["attempts"].sum()
strength = 150

raw = qbs["completions"] / qbs["attempts"]
shrunk = (qbs["completions"] + league_mean * strength) / (qbs["attempts"] + strength)

mse_raw = ((raw - qbs["true_rate"]) ** 2).mean()
mse_shrunk = ((shrunk - qbs["true_rate"]) ** 2).mean()
print(f"MSE raw:    {mse_raw:.5f}")
print(f"MSE shrunk: {mse_shrunk:.5f}")

check("shrunk estimates computed", lambda: len(shrunk) == len(qbs))
check("shrinkage beats the raw rate", lambda: mse_shrunk < mse_raw)
''',
        hint="The posterior mean of a Beta-Binomial is (k + α) / (n + α + β), with α = league_mean × strength and α + β = strength. Then try strength = 1000: does more shrinkage always help?",
    ),
    Exercise(
        id="lab-05-sklearn-regression", level="Intermediate", title="Do better recruits produce more? (scikit-learn regression)",
        goal="Fit a simple linear regression of year-3 tackles per game on recruit rating, then judge how much it really explains.",
        concepts=["LinearRegression", "cross-validation", "R²", "simple models"],
        starter='''import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from academy.labs import check
from academy.data import make_linebackers

lbs = make_linebackers()                       # SIMULATED
y3 = lbs[lbs["season"] == 3].copy()
y3["tpg"] = y3["tackles"] / y3["games"]

X = y3[["recruit_rating"]].values
y = y3["tpg"].values

model = LinearRegression()
# TODO: fit the model on X, y
cv_r2 = None                                   # TODO: cross_val_score(model, X, y, cv=5, scoring="r2")

print("Slope (extra tackles/game per +1.00 rating):", model.coef_[0] if hasattr(model, "coef_") else "not fit yet")
print("Cross-validated R^2:", None if cv_r2 is None else cv_r2.mean().round(3))

# ---- self-check ----
check("model is fitted with a positive slope", lambda: model.coef_[0] > 0)
check("cross-validated R^2 computed (5 folds)", lambda: len(cv_r2) == 5)
''',
        solution='''import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from academy.labs import check
from academy.data import make_linebackers

lbs = make_linebackers()
y3 = lbs[lbs["season"] == 3].copy()
y3["tpg"] = y3["tackles"] / y3["games"]

X = y3[["recruit_rating"]].values
y = y3["tpg"].values

model = LinearRegression().fit(X, y)
cv_r2 = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")

print("Slope (extra tackles/game per +1.00 rating):", round(model.coef_[0], 2))
print("Cross-validated R^2:", cv_r2.mean().round(3))
print("Takeaway: recruit rating helps a little but explains little of year-3 production. Do not expect miracles from one feature.")

check("model is fitted with a positive slope", lambda: model.coef_[0] > 0)
check("cross-validated R^2 computed (5 folds)", lambda: len(cv_r2) == 5)
''',
        hint="`model.fit(X, y)` then `cross_val_score(LinearRegression(), X, y, cv=5, scoring='r2')`. A low or even negative R² is a real finding, not a bug.",
    ),
    Exercise(
        id="lab-06-sklearn-kmeans", level="Advanced", title="Discover edge-defender archetypes (K-means)",
        goal="Cluster simulated edge defenders without labels, pick k with the silhouette score, and compare against the hidden truth.",
        concepts=["K-means", "standardization", "silhouette score"],
        starter='''import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from academy.labs import check
from academy.data import make_edge_defenders

edges = make_edge_defenders()                    # SIMULATED; true_archetype is hidden truth
features = ["pressure_rate", "run_stop_rate", "tackles_per_game", "coverage_snap_pct", "weight_lbs"]

# TODO: standardize the features so weight (~250) doesn't swamp pressure rate (~0.1)
scaled = None

scores = {}
for k in range(2, 7):
    # TODO: fit KMeans(n_clusters=k, n_init=10, random_state=0) and store silhouette_score in scores[k]
    pass

best_k = max(scores, key=scores.get) if scores else None
print("Silhouette by k:", {k: round(v, 3) for k, v in scores.items()})
print("Best k:", best_k)

# ---- self-check ----
check("data standardized (mean ~0)", lambda: abs(scaled.mean()) < 1e-6)
check("scores computed for k = 2..6", lambda: len(scores) == 5)
check("best k is 3", lambda: best_k == 3)
''',
        solution='''import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from academy.labs import check
from academy.data import make_edge_defenders

edges = make_edge_defenders()
features = ["pressure_rate", "run_stop_rate", "tackles_per_game", "coverage_snap_pct", "weight_lbs"]

scaled = StandardScaler().fit_transform(edges[features])

scores = {}
for k in range(2, 7):
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(scaled)
    scores[k] = silhouette_score(scaled, labels)

best_k = max(scores, key=scores.get)
print("Silhouette by k:", {k: round(v, 3) for k, v in scores.items()})
print("Best k:", best_k)

edges["cluster"] = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit_predict(scaled)
print(pd.crosstab(edges["cluster"], edges["true_archetype"]))
print("Clusters line up with the hidden archetypes. Real data won't be this clean.")

check("data standardized (mean ~0)", lambda: abs(scaled.mean()) < 1e-6)
check("scores computed for k = 2..6", lambda: len(scores) == 5)
check("best k is 3", lambda: best_k == 3)
''',
        hint="`StandardScaler().fit_transform(df[features])`. Inside the loop: `labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(scaled)` then `silhouette_score(scaled, labels)`.",
    ),
    Exercise(
        id="lab-07-logistic-retention", level="Advanced", title="Predict transfers with logistic regression",
        goal="Fit a transfer-risk model, evaluate it on held-out players with AUC, and read the coefficients.",
        concepts=["LogisticRegression", "train/test split", "ROC AUC", "coefficients"],
        starter='''import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from academy.labs import check
from academy.data import make_retention

df = make_retention()                      # SIMULATED
features = ["recruit_rating", "snaps_share", "in_state", "coach_change"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["transferred"], test_size=0.3, random_state=0, stratify=df["transferred"])

model = LogisticRegression(max_iter=1000)
# TODO: fit on the training data
auc = None                                 # TODO: roc_auc_score(y_test, predicted probabilities for class 1)

print("Test AUC:", None if auc is None else round(auc, 3), "(0.5 = coin flip)")
if hasattr(model, "coef_"):
    print(pd.Series(model.coef_[0], index=features).round(2))

# ---- self-check ----
check("test AUC above 0.65", lambda: auc > 0.65)
check("playing time lowers transfer risk (negative coefficient)", lambda: model.coef_[0][1] < 0)
''',
        solution='''import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from academy.labs import check
from academy.data import make_retention

df = make_retention()
features = ["recruit_rating", "snaps_share", "in_state", "coach_change"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["transferred"], test_size=0.3, random_state=0, stratify=df["transferred"])

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print("Test AUC:", round(auc, 3), "(0.5 = coin flip)")
print(pd.Series(model.coef_[0], index=features).round(2))

check("test AUC above 0.65", lambda: auc > 0.65)
check("playing time lowers transfer risk (negative coefficient)", lambda: model.coef_[0][1] < 0)
''',
        hint="`model.fit(X_train, y_train)`, then `model.predict_proba(X_test)[:, 1]` is the probability of transferring. Pass that to `roc_auc_score`.",
    ),
    Exercise(
        id="lab-08-calibration-brier", level="Intermediate", title="Calibration and the Brier score",
        goal="Check whether predicted transfer probabilities match reality: build a reliability table and compute the Brier score against a base-rate baseline.",
        concepts=["calibration", "Brier score", "reliability diagram"], concept_ids=["calibration-brier"],
        starter='''import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from academy.labs import check
from academy.data import make_retention

df = make_retention()                      # SIMULATED
X, y = df[["recruit_rating", "snaps_share", "in_state", "coach_change"]], df["transferred"]
p = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=5, method="predict_proba")[:, 1]

# TODO 1: Brier score = mean of (p - y)^2
brier = None
# TODO 2: baseline Brier: predict the base rate y.mean() for everyone
brier_base = None

# TODO 3: reliability table: bucket p into 5 quantile bins (pd.qcut), compare mean p vs mean y per bin
table = None

print("Brier:", brier, " baseline:", brier_base)
print(table)

# ---- self-check ----
check("Brier computed", lambda: abs(brier - np.mean((p - y) ** 2)) < 1e-12)
check("model beats the base-rate baseline", lambda: brier < brier_base)
check("reliability table has 5 rows", lambda: len(table) == 5)
''',
        solution='''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from academy.labs import check
from academy.data import make_retention

df = make_retention()
X, y = df[["recruit_rating", "snaps_share", "in_state", "coach_change"]], df["transferred"]
p = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=5, method="predict_proba")[:, 1]

brier = np.mean((p - y) ** 2)
brier_base = np.mean((y.mean() - y) ** 2)
table = (pd.DataFrame({"p": p, "y": y}).assign(bin=lambda d: pd.qcut(d["p"], 5))
           .groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")))
print(f"Brier: {brier:.4f}   baseline: {brier_base:.4f}   skill: {1 - brier / brier_base:.1%}")
print(table.round(3))

fig, ax = plt.subplots()
ax.plot([0, 0.5], [0, 0.5], "--", label="perfect calibration")
ax.plot(table["predicted"], table["actual"], "o-", label="model")
ax.set_xlabel("Predicted transfer probability"); ax.set_ylabel("Actual transfer rate"); ax.legend()
print("Calibration asks: do 30% predictions happen ~30% of the time? Discrimination (AUC) asks: are transfers ranked above stayers? A model can be good at one and bad at the other.")

check("Brier computed", lambda: abs(brier - np.mean((p - y) ** 2)) < 1e-12)
check("model beats the base-rate baseline", lambda: brier < brier_base)
check("reliability table has 5 rows", lambda: len(table) == 5)
''',
        hint="`np.mean((p - y) ** 2)`. For the table: `pd.DataFrame({'p': p, 'y': y}).assign(bin=lambda d: pd.qcut(d['p'], 5)).groupby('bin', observed=True).mean()`.",
    ),
    Exercise(
        id="lab-09-time-based-cv", level="Intermediate", title="Time-based cross-validation vs random splits",
        goal="See how a random train/test split leaks the future, and build a rolling-origin validation that trains on earlier seasons only.",
        concepts=["temporal leakage", "rolling-origin validation"], concept_ids=["time-based-cv"],
        starter='''import numpy as np
from academy.labs import check
from academy.data import make_qb_completions

df = make_qb_completions()               # SIMULATED player-seasons, 2016-2025
df["rate"] = df["successes"] / df["opportunities"]
df = df[df["opportunities"] >= 50]

# A LEAKY feature: the player's average rate over ALL his seasons (including future ones!)
df["career_rate"] = df.groupby("player_id")["rate"].transform("mean")

# TODO: rolling origin. For each test season T from 2019 to 2025:
#   train = seasons < T, test = season T
#   prediction for a test row = that player's mean rate in TRAIN seasons only (skip players with no history)
#   store the squared errors in honest_errors
honest_errors = []

leaky_mse = np.mean((df["career_rate"] - df["rate"]) ** 2)
honest_mse = np.mean(honest_errors) if honest_errors else None
print("Leaky MSE:", round(leaky_mse, 5), " honest MSE:", honest_mse)

# ---- self-check ----
check("honest errors computed", lambda: len(honest_errors) > 100)
check("the leaky approach looks better than it really is", lambda: leaky_mse < honest_mse)
''',
        solution='''import numpy as np
from academy.labs import check
from academy.data import make_qb_completions

df = make_qb_completions()
df["rate"] = df["successes"] / df["opportunities"]
df = df[df["opportunities"] >= 50]
df["career_rate"] = df.groupby("player_id")["rate"].transform("mean")

honest_errors = []
for T in range(2019, 2026):
    train, test = df[df["season"] < T], df[df["season"] == T]
    history = train.groupby("player_id")["rate"].mean()
    t = test[test["player_id"].isin(history.index)]
    pred = t["player_id"].map(history)
    honest_errors += list((pred - t["rate"]) ** 2)
    print(f"test {T}: {len(t):>3} players, MSE {np.mean((pred - t['rate']) ** 2):.5f}")

leaky_mse = np.mean((df["career_rate"] - df["rate"]) ** 2)
honest_mse = np.mean(honest_errors)
print(f"\\nLeaky MSE {leaky_mse:.5f} vs honest MSE {honest_mse:.5f}: the leaky number is a fantasy.")

check("honest errors computed", lambda: len(honest_errors) > 100)
check("the leaky approach looks better than it really is", lambda: leaky_mse < honest_mse)
''',
        hint="Loop `for T in range(2019, 2026)`, split with `df['season'] < T` and `== T`, build `history = train.groupby('player_id')['rate'].mean()`, then `test['player_id'].map(history)`.",
    ),
    Exercise(
        id="lab-10-conformal", level="Advanced", title="Split conformal prediction intervals",
        goal="Wrap any point prediction of season-4 production in an 80% interval with a coverage guarantee, and check the coverage.",
        concepts=["conformal prediction", "coverage", "prediction intervals"], concept_ids=["conformal-prediction"],
        starter='''import numpy as np
from sklearn.linear_model import Ridge
from academy.labs import check
from academy.data import make_development

dev = make_development(n_players=400)    # SIMULATED
wide = dev.pivot(index="player_id", columns="season", values="production")
X, y = wide[[1, 2, 3]].to_numpy(), wide[4].to_numpy()

rng = np.random.default_rng(0)
idx = rng.permutation(len(y))
train, calib, test = idx[:200], idx[200:300], idx[300:]

model = Ridge(alpha=1.0).fit(X[train], y[train])

# TODO 1: absolute residuals on the CALIBRATION set
scores = None
# TODO 2: q = the ceil((n+1)*0.8)/n quantile of the scores (n = len(calib))
q = None
# TODO 3: coverage on the TEST set: share of |y - prediction| <= q
coverage = None

print("Half-width:", q, " coverage:", coverage)

# ---- self-check ----
check("calibration scores computed", lambda: len(scores) == 100)
check("coverage is close to 80%", lambda: 0.70 <= coverage <= 0.92)
''',
        solution='''import numpy as np
from sklearn.linear_model import Ridge
from academy.labs import check
from academy.data import make_development

dev = make_development(n_players=400)
wide = dev.pivot(index="player_id", columns="season", values="production")
X, y = wide[[1, 2, 3]].to_numpy(), wide[4].to_numpy()

rng = np.random.default_rng(0)
idx = rng.permutation(len(y))
train, calib, test = idx[:200], idx[200:300], idx[300:]

model = Ridge(alpha=1.0).fit(X[train], y[train])

scores = np.abs(y[calib] - model.predict(X[calib]))
n = len(calib)
q = np.quantile(scores, min(1.0, np.ceil((n + 1) * 0.8) / n))
coverage = np.mean(np.abs(y[test] - model.predict(X[test])) <= q)
print(f"80% interval: prediction +/- {q:.2f}   test coverage: {coverage:.0%}")
print("The guarantee assumes calibration and test players are exchangeable (e.g., same era). "
      "Across seasons with rule changes, coverage can break. Re-check it, never assume it.")

check("calibration scores computed", lambda: len(scores) == 100)
check("coverage is close to 80%", lambda: 0.70 <= coverage <= 0.92)
''',
        hint="`scores = np.abs(y[calib] - model.predict(X[calib]))`, then `np.quantile(scores, np.ceil((n + 1) * 0.8) / n)`.",
    ),
    Exercise(
        id="lab-11-pca-latent-space", level="Beginner", title="Build a latent space with PCA",
        goal="Compress seven linebacker stats into two hidden dimensions and find out which stats define each one.",
        concepts=["PCA", "standardization", "loadings"], concept_ids=["latent-spaces"],
        starter='''import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from academy.labs import check
from academy.data import make_defensive_profiles, PROFILE_FEATURES

df = make_defensive_profiles()                      # SIMULATED
lb = df[(df["position_group"] == "LB") & (df["snaps"] >= 150)]

# TODO 1: standardize lb[PROFILE_FEATURES]
Z = None
# TODO 2: fit PCA with 2 components on Z
pca = None
# TODO 3: the stat with the biggest (absolute) loading on dimension 1
top_stat = None

print("Top stat on dimension 1:", top_stat)

# ---- self-check ----
check("standardized (mean ~0)", lambda: abs(Z.mean()) < 1e-9)
check("2 components kept", lambda: pca.n_components_ == 2)
check("top stat found", lambda: top_stat in PROFILE_FEATURES)
''',
        solution='''import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from academy.labs import check
from academy.data import make_defensive_profiles, PROFILE_FEATURES

df = make_defensive_profiles()
lb = df[(df["position_group"] == "LB") & (df["snaps"] >= 150)]

Z = StandardScaler().fit_transform(lb[PROFILE_FEATURES])
pca = PCA(n_components=2).fit(Z)
load = pd.Series(pca.components_[0], index=PROFILE_FEATURES)
top_stat = load.abs().idxmax()

print("Variation kept:", pca.explained_variance_ratio_.round(2))
print("Dimension 1 loadings:\\n", load.round(2).sort_values().to_string())
print("Top stat on dimension 1:", top_stat)

check("standardized (mean ~0)", lambda: abs(Z.mean()) < 1e-9)
check("2 components kept", lambda: pca.n_components_ == 2)
check("top stat found", lambda: top_stat in PROFILE_FEATURES)
''',
        hint="`StandardScaler().fit_transform(...)`, `PCA(n_components=2).fit(Z)`, then `pd.Series(pca.components_[0], index=PROFILE_FEATURES).abs().idxmax()`.",
    ),
    Exercise(
        id="lab-12-autoencoder-bottleneck", level="Advanced", title="How small can the code be? (autoencoder)",
        goal="Train autoencoders with codes of size 1, 2 and 3 and compare their rebuild error with PCA at the same size.",
        concepts=["autoencoder", "bottleneck", "reconstruction error"], concept_ids=["autoencoders"],
        starter='''import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from academy.labs import check
from academy.data import make_defensive_profiles, PROFILE_FEATURES

df = make_defensive_profiles()                      # SIMULATED
db = df[(df["position_group"] == "DB") & (df["snaps"] >= 150)]
Z = StandardScaler().fit_transform(db[PROFILE_FEATURES])

results = {}                                        # code size -> (autoencoder error, PCA error)
for size in (1, 2, 3):
    # TODO: train MLPRegressor(hidden_layer_sizes=(5, size, 5), activation="tanh", max_iter=3000, random_state=0) on Z -> Z
    # TODO: compute both rebuild errors as mean squared error and store them in results[size]
    pass

for size, (ae, pc) in results.items():
    print(f"code size {size}: autoencoder {ae:.3f}  PCA {pc:.3f}")

# ---- self-check ----
check("three sizes tried", lambda: len(results) == 3)
check("bigger codes rebuild better (PCA)", lambda: results[1][1] > results[2][1] > results[3][1])
''',
        solution='''import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from academy.labs import check
from academy.data import make_defensive_profiles, PROFILE_FEATURES

df = make_defensive_profiles()
db = df[(df["position_group"] == "DB") & (df["snaps"] >= 150)]
Z = StandardScaler().fit_transform(db[PROFILE_FEATURES])

results = {}
for size in (1, 2, 3):
    ae = MLPRegressor(hidden_layer_sizes=(5, size, 5), activation="tanh", max_iter=3000, random_state=0).fit(Z, Z)
    pca = PCA(size).fit(Z)
    results[size] = (np.mean((ae.predict(Z) - Z) ** 2),
                     np.mean((pca.inverse_transform(pca.transform(Z)) - Z) ** 2))

for size, (ae, pc) in results.items():
    print(f"code size {size}: autoencoder {ae:.3f}  PCA {pc:.3f}")
print("Bigger codes rebuild better. Does the autoencoder ever clearly beat PCA on 7 stats?")

check("three sizes tried", lambda: len(results) == 3)
check("bigger codes rebuild better (PCA)", lambda: results[1][1] > results[2][1] > results[3][1])
''',
        hint="Inside the loop: `ae = MLPRegressor(...).fit(Z, Z)`, `pca = PCA(size).fit(Z)`, then mean squared differences between `ae.predict(Z)` and Z, and between `pca.inverse_transform(pca.transform(Z))` and Z.",
    ),
]

