import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig
from ._l2 import FEATURES, group_profiles


def visual():
    c1, c2 = st.columns([1, 2])
    with c1:
        group = st.radio("Position group", ["DL", "LB", "DB"], horizontal=True, key="l21_group")
        n_dims = st.slider("Hidden dimensions to keep", 1, 7, 2, key="l21_dims")
    d, Z = group_profiles(group)
    pca = PCA().fit(Z)
    coords = pca.transform(Z)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [3, 2]})
    for pos, c in zip(sorted(d["position"].unique()), T.CYCLE):
        m = (d["position"] == pos).to_numpy()
        axes[0].scatter(coords[m, 0], coords[m, 1], s=9, alpha=0.7, color=c, label=f"listed {pos}")
    axes[0].set_xlabel("Hidden dimension 1"), axes[0].set_ylabel("Hidden dimension 2"), axes[0].legend(fontsize=7)
    ev = pca.explained_variance_ratio_
    axes[1].bar(range(1, 8), ev, color=[T.BLUE if i < n_dims else T.GRID for i in range(7)])
    axes[1].set_xlabel("Dimension"), axes[1].set_ylabel("Share of variation"), axes[1].set_xticks(range(1, 8))
    with c2:
        show_fig(fig)
    st.caption(f"Keeping {n_dims} of 7 dimensions keeps {ev[:n_dims].sum():.0%} of the variation between {group} players.")


CODE = '''import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from academy.data import make_defensive_profiles, PROFILE_FEATURES

df = make_defensive_profiles()                          # SIMULATED player-seasons
dl = df[(df["position_group"] == "DL") & (df["snaps"] >= 150)]
X = dl[PROFILE_FEATURES]                                # 7 stats per player

Z = StandardScaler().fit_transform(X)                   # put every stat on the same scale
pca = PCA(n_components=2).fit(Z)                        # find the 2 most informative directions
coords = pca.transform(Z)                               # each player's position in the latent space

print("Share of variation kept by each hidden dimension:", pca.explained_variance_ratio_.round(2))
print(f"Total kept with 2 of 7 dimensions: {pca.explained_variance_ratio_.sum():.0%}\\n")

loadings = pd.DataFrame(pca.components_.T, index=PROFILE_FEATURES, columns=["dim 1", "dim 2"])
print("What each hidden dimension is made of (loadings):")
print(loadings.round(2).to_string())

first = dl.iloc[0]
print(f"\\nPlayer {first['player_id']} sits at ({coords[0, 0]:.2f}, {coords[0, 1]:.2f}) in the latent space")
'''

LESSON = Lesson(
    id="l2-01-latent-spaces", level=2, order=1,
    title="Latent Spaces: Describing a Player With a Few Hidden Numbers",
    subtitle="Compressing many stats into a small map of player styles",
    minutes=20,
    concept_ids=["latent-spaces"],
    simple="""
A defensive lineman's profile might have seven stats: pressure rate, run-stop rate, tackle rate, coverage share, pass breakups, missed tackles and weight. Seven numbers are hard to picture.

But those stats are not independent. Players who get lots of pressure also tend to be lighter and make fewer run stops. Underneath the seven stats there may be just **two or three hidden ideas**, such as "pass-rush vs run-stopping style" and "how active he is overall".

A **latent space** is a small set of hidden dimensions that captures those ideas. ("Latent" just means *hidden*.) Every player becomes a point on a simple map, and players with similar styles land close together.

The easiest way to build one is **PCA** (principal component analysis). It finds the directions in which players differ most.
""",
    problem="Many stats overlap and are too many to picture or compare. A latent space turns a messy table into a small map where 'similar players' becomes something you can see and measure.",
    why="It shows which stats move together, makes player-comparison (comps) easy, removes duplicated information before clustering, and is the foundation for K-means, autoencoders and VAEs in this level.",
    when_useful="""
- Many correlated stats about the same player
- Finding comparable players ("who plays like him?")
- Preparing data for clustering, or plotting a whole position group at once
""",
    when_not="""
- Only 2–3 stats: just plot them directly
- You need each number to keep its football meaning for coaches. Hidden dimensions are blends of stats and need interpreting
- Low-snap players: their stats are mostly noise, and noise ends up in the map too. Filter by volume first
""",
    compare="""
**Latent space vs picking a few stats by hand:** hand-picking keeps meaning but throws away information and ignores overlap. A latent space uses all the stats but needs interpretation.

**PCA vs autoencoders (L2.3):** PCA can only find straight-line combinations of stats. Autoencoders can bend. PCA is simpler, exact and usually good enough with a few stats and a few thousand players.

**Latent space vs clusters (L2.2):** a latent space is a continuous map. Clusters cut that map into groups.
""",
    sports="""
**Fictional scenario (simulated).** About 1,000 defensive-line seasons with 150+ snaps and seven stats each.

PCA finds that one hidden dimension keeps about **43%** of all the variation. It mostly contrasts **pressure rate and lower weight** with **run stops and higher weight**: a "pass rusher ↔ run stopper" axis. The second dimension (about 15%) mostly picks up coverage snaps against missed tackles.

Two dimensions keep almost **60%** of the variation. That's enough to draw every DL on one chart and see EDGE and DT players overlap in the middle, because the listed position is a label, not a style.
""",
    takeaway="A latent space is a small map of hidden dimensions where similar players land close together.",
    visual=visual,
    visual_caption="Each dot is a player-season placed by its first two hidden dimensions. Color shows the listed position. Notice the overlap: style and roster label are not the same thing.",
    math=[
        MathBlock(
            "Step 1: put every stat on the same scale",
            r"z_{ij} = \frac{x_{ij} - \bar x_j}{s_j}",
            """
- $x_{ij}$: stat $j$ for player $i$ (e.g., his pressure rate)
- $\\bar x_j$: the average of stat $j$ over all players
- $s_j$: the standard deviation of stat $j$ (its typical spread)
- $z_{ij}$: the standardized stat: "how many spreads above or below average"
""",
            "Without this, weight (around 280) would swamp pressure rate (around 0.08) just because its numbers are bigger.",
        ),
        MathBlock(
            "Step 2: each hidden dimension is a weighted blend of stats",
            r"t_{ik} = \sum_{j=1}^{p} w_{jk}\, z_{ij}",
            """
- $t_{ik}$: player $i$'s coordinate on hidden dimension $k$
- $w_{jk}$: the **loading**: how much stat $j$ counts toward dimension $k$ (a big positive or negative value means it matters)
- $p$: number of stats (7 here)
- PCA picks the weights so dimension 1 spreads players out as much as possible, dimension 2 as much as possible among what's left, and so on
""",
        ),
        MathBlock(
            "Step 3: how much each dimension explains",
            r"\text{share}_k = \frac{\lambda_k}{\sum_{m=1}^{p}\lambda_m}",
            """
- $\\lambda_k$: the variance (spread) of players along dimension $k$
- $\\text{share}_k$: the fraction of all variation that dimension keeps. These are the bars in the visual
""",
        ),
    ],
    code=CODE,
    code_notes="Change \"DL\" to \"DB\" and rerun. Which stats now define the first hidden dimension?",
    walkthrough=[
        ("dl = df[(df['position_group'] == 'DL') & (df['snaps'] >= 150)]", "Keep defensive linemen with at least 150 snaps, so the stats aren't mostly noise."),
        ("X = dl[PROFILE_FEATURES]", "The seven stats for each player-season."),
        ("Z = StandardScaler().fit_transform(X)", "Standardize: every stat becomes 'spreads from average', so no stat dominates just because of its units."),
        ("pca = PCA(n_components=2).fit(Z)", "Find the two directions in which players differ most."),
        ("coords = pca.transform(Z)", "Place every player on the 2-D map. Each player is now just two numbers."),
        ("pca.explained_variance_ratio_", "How much of the original variation each hidden dimension keeps."),
        ("pd.DataFrame(pca.components_.T, ...)", "The loadings table: which stats build each hidden dimension. This is how you give a dimension a football name."),
        ("coords[0, 0], coords[0, 1]", "One player's location on the map. Nearby players have similar styles."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Comps:** a latent map of position-group profiles is a quick way to find similar players for scouting notes. It describes style; it does not predict success.
- **Model 8 lesson:** Model 8 added value where production was thin. Latent coordinates from low-snap seasons are mostly noise, so set a minimum snap count before mapping anyone.
- **Your rule:** a latent feature would only go on your site if adding it beats a simple baseline on a test set fixed in advance. The Level 2 project runs exactly that test.
""",
    quiz=[
        Question(
            "q1", "mc", "In plain words, a latent space is:",
            "A small set of hidden dimensions that summarizes many stats, where similar players sit close together.",
            options=["A small set of hidden dimensions that summarizes many stats",
                     "A list of the players with the most snaps", "A type of football formation",
                     "A table of raw stats sorted by position"],
            notes=["Correct.", "That's a filter, not a latent space.", "Not a football term here.", "That's just the original data."],
            answer=0),
        Question(
            "q2", "mc", "Why standardize stats before PCA?",
            "Otherwise stats with large numbers (like weight) dominate just because of their units.",
            options=["So stats with big units, like weight, don't dominate", "To remove low-snap players",
                     "To make every player average", "PCA requires whole numbers"],
            notes=["Correct.", "That's a separate filtering step.", "Players keep their differences; only the units change.", "PCA works on any numbers."],
            answer=0),
        Question(
            "q3", "numeric", "Dimension 1 explains 34% of variation and dimension 2 explains 19%. What share do the first two keep together? (decimal or %)",
            "0.34 + 0.19 = 0.53, so 53%.", category="Application", answer=0.53, tolerance=0.005),
        Question(
            "q4", "short", "A dimension has big positive loadings on pressure rate and big negative loadings on weight and run-stop rate. What would you name it, in football terms?",
            "It contrasts pass-rush style with heavy run-stopping style.",
            category="Application",
            keywords=[["pass", "rush", "pressure", "edge"], ["run", "stop", "heavy", "weight", "interior", "stuff"]],
            model_answer="A 'pass rusher vs run stopper' axis: high values are lighter, pressure-heavy players; low values are heavier run stoppers."),
        Question(
            "q5", "mc", "You map players with only 40 snaps alongside 600-snap starters. What's the main risk?",
            "Low-snap stats are noisy, so their positions on the map are largely luck (compare L1.6).",
            category="Application",
            options=["Low-snap players land in random places because their stats are mostly noise",
                     "PCA can't handle different snap counts", "Starters will be removed automatically",
                     "No risk; more players is always better"],
            notes=["Correct. Filter by volume or weight by snaps.", "It runs fine; the problem is what it shows.",
                   "Nothing is removed automatically.", "More noisy players can blur the map."],
            answer=0),
    ],
)
