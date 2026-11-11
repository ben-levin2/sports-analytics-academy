import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig
from ._l2 import group_profiles


def visual():
    d, Z = group_profiles("DL")
    c1, c2 = st.columns([1, 2])
    with c1:
        k = st.slider("Number of clusters (k)", 2, 6, 3, key="l22_k")
    km = KMeans(k, n_init=10, random_state=0).fit(Z)
    coords = PCA(2).fit(Z)
    xy = coords.transform(Z)
    centres = coords.transform(km.cluster_centers_)
    fig, ax = plt.subplots()
    for c in range(k):
        m = km.labels_ == c
        ax.scatter(xy[m, 0], xy[m, 1], s=9, alpha=0.65, color=T.CYCLE[c % 6], label=f"cluster {c}")
    ax.scatter(centres[:, 0], centres[:, 1], s=160, marker="X", color=T.TEXT, edgecolor=T.BG, label="centres")
    ax.set_xlabel("Hidden dimension 1"), ax.set_ylabel("Hidden dimension 2"), ax.legend(fontsize=7)
    with c2:
        show_fig(fig)
    m1, m2 = st.columns(2)
    m1.metric("Silhouette score", f"{silhouette_score(Z, km.labels_):.3f}", help="Higher = tighter, better-separated clusters")
    m2.metric("Within-cluster spread (inertia)", f"{km.inertia_:,.0f}", help="Always falls as k rises")


CODE = '''import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from academy.data import make_defensive_profiles, PROFILE_FEATURES

df = make_defensive_profiles()                                  # SIMULATED
dl = df[(df["position_group"] == "DL") & (df["snaps"] >= 150)].reset_index(drop=True)
Z = StandardScaler().fit_transform(dl[PROFILE_FEATURES])

print(" k   inertia   silhouette   match to hidden truth (ARI)")
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
    sil = silhouette_score(Z, km.labels_)
    ari = adjusted_rand_score(dl["true_archetype"], km.labels_)   # only possible with simulated data!
    print(f"{k:>2}  {km.inertia_:>8.0f}   {sil:>8.3f}       {ari:.2f}")

km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Z)
dl["cluster"] = km.labels_
profile = (pd.DataFrame(Z, columns=PROFILE_FEATURES).groupby(dl["cluster"]).mean())
print("\\nCluster profiles with k = 3 (in standard deviations from average):")
print(profile.round(1).to_string())
print("\\nClusters vs listed position:")
print(pd.crosstab(dl["cluster"], dl["position"]))
'''

LESSON = Lesson(
    id="l2-02-kmeans", level=2, order=2,
    title="K-Means Clustering: Sorting Players Into Archetypes",
    subtitle="How the algorithm works, how to choose k, and what clusters can't tell you",
    minutes=25,
    concept_ids=["k-means"],
    simple="""
K-means sorts players into **k groups** of similar profiles. You choose k, the number of groups. The algorithm does the rest:

1. Drop k "centres" somewhere on the map.
2. Send each player to his **nearest** centre.
3. Move each centre to the **middle** of the players assigned to it.
4. Repeat steps 2–3 until nothing changes.

The result: each player gets a group number. Reading each group's average stats lets *you* give it a football name, like "edge rusher" or "run stopper".

The algorithm never sees position labels or outcomes. It only knows which profiles look alike.
""",
    problem="Positions on a roster sheet hide very different playing styles. K-means finds groups of players who actually play alike, using the data rather than the label.",
    why="It's simple, fast and easy to explain, which makes it a good first look at structure in thousands of player profiles.",
    when_useful="""
- Describing roster composition ("how many run stoppers do we have?")
- Finding style-based comps and recruiting fits
- A first, simple look before trying fancier methods
""",
    when_not="""
- When players form a smooth spectrum rather than real groups. K-means will still cut it into k pieces
- Odd-shaped or very different-sized groups: K-means likes round, similar-sized clusters
- As a prediction of success: clusters describe; they don't forecast unless tested (the Level 2 project does that test)
""",
    compare="""
**K-means vs listed positions:** positions are labels a staff assigns; clusters come from how players actually played. They often disagree, and the disagreements can be the interesting part.

**K-means vs a latent space (L2.1):** a latent space is a continuous map; K-means draws boundaries on it.

**K-means vs Gaussian mixtures (EM, Level 5):** K-means gives each player exactly one group. A mixture model gives probabilities ("70% edge rusher, 30% penetrator") and allows oval-shaped groups.
""",
    sports="""
**Fictional scenario (simulated).** About 1,000 DL seasons, secretly generated from **three** styles: edge rusher, interior run stopper and interior penetrator.

- The **silhouette score**, a common way to pick k, prefers **k = 2**. The penetrators sit between the other two styles, so two clean groups score better than three overlapping ones.
- With **k = 3**, the clusters match the hidden styles better (a higher ARI) but look less separated.
- Clusters only partly line up with the **listed** EDGE/DT label.

**The lesson:** a statistic can suggest k, but football meaning, stability and usefulness should decide it. And with real data there is no hidden truth to check against.
""",
    takeaway="K-means groups similar profiles. You choose k, name the groups, and then test whether they're useful. Groups are descriptions, not predictions.",
    visual=visual,
    visual_caption="Move k. The white X marks are the cluster centres. Inertia always falls as k rises, so it can't choose k alone. The silhouette score peaks where groups are tightest.",
    math=[
        MathBlock(
            "Step 1: the goal (what K-means tries to make small)",
            r"\text{Inertia} = \sum_{i=1}^{N} \bigl\lVert \mathbf{z}_i - \boldsymbol{\mu}_{c(i)} \bigr\rVert^2",
            """
- $N$: number of players
- $\\mathbf{z}_i$: player $i$'s standardized stat profile (a list of 7 numbers)
- $c(i)$: the cluster player $i$ is assigned to
- $\\boldsymbol{\\mu}_{c(i)}$: the centre (average profile) of that cluster
- $\\lVert \\cdot \\rVert^2$: squared distance, the sum of squared differences across the 7 stats
""",
            "Smaller inertia = players sit closer to their cluster centres.",
        ),
        MathBlock(
            "Step 2: the two repeating moves",
            r"c(i) = \arg\min_{c}\, \lVert \mathbf{z}_i - \boldsymbol{\mu}_c \rVert^2, \qquad \boldsymbol{\mu}_c = \frac{1}{n_c}\sum_{i:\,c(i)=c} \mathbf{z}_i",
            """
- Left: **assign** each player to his nearest centre ($\\arg\\min$ = "the one that makes this smallest")
- Right: **update** each centre to the average of its $n_c$ players
- Each round can only lower the inertia, so the process settles down
""",
            "It can settle in a so-so solution, so we restart from several random starts (n_init=10) and keep the best.",
        ),
        MathBlock(
            "Step 3: the silhouette score for choosing k",
            r"s_i = \frac{b_i - a_i}{\max(a_i,\, b_i)}",
            """
- $a_i$: player $i$'s average distance to players in **his own** cluster
- $b_i$: his average distance to players in the **nearest other** cluster
- $s_i$ near 1: clearly in the right group. Near 0: on a border. Below 0: probably in the wrong group
- The silhouette score is the average of $s_i$ over all players
""",
        ),
    ],
    code=CODE,
    code_notes="The ARI column compares clusters with the hidden truth. That's only possible because the data is simulated. Watch which k has the best silhouette, and which has the best ARI.",
    walkthrough=[
        ("Z = StandardScaler().fit_transform(dl[PROFILE_FEATURES])", "Standardize first (L2.1). K-means uses distances, so units matter."),
        ("for k in range(2, 7):", "Try several numbers of clusters and compare them."),
        ("KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)", "Run K-means 10 times from different random starts and keep the best. random_state makes it repeatable."),
        ("km.inertia_", "Total squared distance from players to their centres. It always drops as k grows, so it can't choose k by itself."),
        ("silhouette_score(Z, km.labels_)", "How clearly separated the clusters are, from -1 to 1. Higher is cleaner."),
        ("adjusted_rand_score(dl['true_archetype'], km.labels_)", "Agreement with the hidden true styles. 1 = perfect, 0 = no better than chance. Real data never has this column."),
        ("pd.DataFrame(Z, ...).groupby(dl['cluster']).mean()", "Each cluster's average profile in standard deviations. Read it to name the cluster."),
        ("pd.crosstab(dl['cluster'], dl['position'])", "How clusters line up with listed positions. Disagreements show where the label hides different styles."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Descriptive use:** archetype counts by position group are a clear way to describe a roster or a recruiting class.
- **Not a predictor by default:** your Models 9 and 9b showed richer inputs can fail to beat "same as last season". Archetypes are richer inputs too. The Level 2 project tests them against exactly that kind of baseline.
- **Your rule:** an archetype feature goes on your site only if it beats a simple baseline on a test set fixed in advance.
""",
    quiz=[
        Question(
            "q1", "mc", "In each round, K-means does two things. Which?",
            "Assign each player to the nearest centre, then move each centre to the average of its players.",
            options=["Assign players to the nearest centre, then move centres to their group's average",
                     "Sort players by production, then split them evenly",
                     "Ask a scout to label players, then average them",
                     "Remove outliers, then pick k"],
            notes=["Correct.", "K-means never looks at production.", "K-means uses no labels.", "Neither is part of the algorithm."],
            answer=0),
        Question(
            "q2", "mc", "Inertia keeps falling as you raise k from 2 to 10. Does that mean k = 10 is best?",
            "Inertia always falls with more clusters (in the extreme, one cluster per player gives zero). Use silhouette, stability and football meaning.",
            options=["No; inertia always falls as k grows", "Yes; lower inertia is always better",
                     "Yes, as long as k is even", "No; inertia should rise with k"],
            notes=["Correct.", "Then one cluster per player would win.", "Even or odd doesn't matter.", "It falls, not rises."],
            answer=0),
        Question(
            "q3", "numeric", "A player's average distance to his own cluster is a = 2.0, and to the nearest other cluster b = 3.0. What is his silhouette value?",
            "(3.0 − 2.0) / max(2.0, 3.0) = 1/3 ≈ 0.333.", category="Application", answer=0.333, tolerance=0.005),
        Question(
            "q4", "short", "Your clusters split listed DTs into two groups. What would you check before telling a staff 'we have two kinds of DT'?",
            "Read the cluster profiles to name them, check stability across random starts and seasons, and check snap counts.",
            category="Application",
            keywords=[["profile", "average", "stats", "name", "interpret", "meaning"],
                      ["stab", "seed", "random", "season", "repeat", "consistent", "robust", "snap", "sample"]],
            model_answer="Read each cluster's average profile to see what really separates them, then check that the split is stable across random starts and seasons, and isn't driven by low-snap noise."),
        Question(
            "q5", "mc", "Clusters describe player styles well. Can you now say a player's archetype predicts his next season?",
            "Description is not prediction. You need a test against a simple baseline on later seasons, which is what the Level 2 project does.",
            category="Application",
            options=["Not yet; it must beat a simple baseline on a test set first",
                     "Yes; good clusters always predict", "Yes, if the silhouette is above 0.5",
                     "No; clusters can never be used in prediction"],
            notes=["Correct, and it's your rule for PSFAL.", "Many good descriptions predict nothing.",
                   "Silhouette measures separation, not prediction.", "They can be used, once they pass a fair test."],
            answer=0),
    ],
)
