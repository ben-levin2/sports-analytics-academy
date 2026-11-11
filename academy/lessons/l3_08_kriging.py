import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import data, spatial as SP, theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner="Fitting the map…")
def _map():
    df = data.make_hometown_production()
    gp = SP.fit_gp(df[["lon", "lat"]], df["production"] - df["production"].mean(), "Matern")
    gx, gy, G = SP.grid(df)
    mu, sd = SP.predict_latent(gp, G)
    return df, gx, gy, mu.reshape(gx.shape), sd.reshape(gx.shape)


def visual():
    df, gx, gy, mu, sd = _map()
    view = st.radio("Map", ["Where players come from (counts)", "Kriged average production", "Uncertainty"],
                    horizontal=True, key="l38_view")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    if view.startswith("Where"):
        hb = ax.hexbin(df["lon"], df["lat"], gridsize=28, cmap="academy_seq", mincnt=1)
        plt.colorbar(hb, ax=ax, label="players")
    elif view.startswith("Kriged"):
        cf = ax.contourf(gx, gy, mu, 14, cmap="academy_div", vmin=-6, vmax=6)
        plt.colorbar(cf, ax=ax, label="production vs average")
    else:
        cf = ax.contourf(gx, gy, sd, 14, cmap="academy_seq")
        plt.colorbar(cf, ax=ax, label="SD of the regional average")
    ax.set_xlabel("Longitude"), ax.set_ylabel("Latitude"), ax.grid(False)
    show_fig(fig)
    st.caption("Counts follow population. The kriged average moves only a few points around the national mean, "
               "and the uncertainty about that average is highest where few players come from.")


CODE = '''import numpy as np
from academy.data import make_hometown_production
from academy.spatial import empirical_variogram, fit_gp, predict_latent, regional_table

df = make_hometown_production()                     # SIMULATED hometowns (not census data)

# 1. The variogram: how different are two players' outcomes, by how far apart their hometowns are?
vg = empirical_variogram(df)
print("distance(deg)  semivariance  pairs")
for _, r in vg.iloc[::2].iterrows():
    print(f"{r['distance']:>10.1f}  {r['semivariance']:>12.1f}  {int(r['pairs']):>6}")
print("Rising slowly with distance = only a weak regional pattern; mostly individual noise.\\n")

# 2. Kriging = a Gaussian process on the map: predict at places, with uncertainty
gp = fit_gp(df[["lon", "lat"]], df["production"] - df["production"].mean(), "Matern")
print("Fitted kernel:", gp.kernel_)
places = {"Dense metro (Houston-ish)": (-95.4, 29.8), "Sparse area (Montana-ish)": (-110.0, 47.0)}
for name, (lon, lat) in places.items():
    m, s = predict_latent(gp, [[lon, lat]])          # uncertainty about the REGIONAL average
    print(f"{name:<27} {m[0]:+.1f} vs average, SD {s[0]:.1f}")

# 3. Concentration is not quality: the busiest regions supply many players, but are they better?
regions = regional_table(df)
busiest = regions.head(5)
share = busiest["players"].sum() / len(df)
avg_busy = (busiest["mean_production"] * busiest["players"]).sum() / busiest["players"].sum()
rest = regions.iloc[5:]
avg_rest = (rest["mean_production"] * rest["players"]).sum() / rest["players"].sum()
print(f"\\nThe 5 busiest of {len(regions)} regions supply {share:.0%} of recruits.")
print(f"Average production: busiest regions {avg_busy:.1f} vs everywhere else {avg_rest:.1f}")
'''

LESSON = Lesson(
    id="l3-08-kriging", level=3, order=8,
    title="Kriging: Mapping Recruit Production Without Fooling Yourself",
    subtitle="Gaussian processes on a map, variograms, and the population trap",
    minutes=25,
    concept_ids=["kriging"],
    simple="""
**Kriging** is a Gaussian process on a **map**. It was invented by mining engineers to guess how much gold was underground between drill holes. Here the "drill holes" are recruits' hometowns, and the "gold" is how productive those recruits became in college.

It does three things:
1. Uses the kernel (L3.6) to decide that **nearby hometowns are more alike** than distant ones.
2. Predicts a **smooth surface** of average production across the whole map, even where no recruit lives.
3. Gives an **uncertainty map**: confident where many recruits come from, unsure where few do.

**The trap.** Places with more people produce more players, so a map of *where good players come from* mostly shows *where people live*. That is **not** a recruiting advantage. The question to ask is whether the *average* recruit from a region produces more, after accounting for population and sample size.
""",
    problem="Maps of player counts confuse population with talent. Kriging maps the average outcome with honest uncertainty, so you can see whether a real regional effect exists.",
    why="It answers 'is there a geographic pattern in production per recruit?' with uncertainty, instead of eyeballing dots on a map.",
    when_useful="""
- Hometown latitude/longitude with an outcome (production, retention)
- Checking whether a region's recruits outperform after accounting for rating and population
- Showing where you simply don't have enough data (the uncertainty map)
""",
    when_not="""
- Counting where players come from: that's population, not quality
- Very noisy outcomes with weak regional effects: the map will be nearly flat (as here), and that's the honest answer
- Ignoring other causes: the college program, coaching and rating all matter more than geography
""",
    compare="""
**Kriging vs Gaussian process regression:** they are the same method. 'Kriging' is the mapping community's name; the inputs are just longitude and latitude.

**Kriging vs regional averages:** box-by-box averages jump at borders and are noisy in small boxes. Kriging smooths across borders and shows the uncertainty.

**Variogram vs kernel:** geostatisticians describe similarity with a variogram (how different points are at each distance). It's the kernel turned upside down.
""",
    sports="""
**Fictional scenario (simulated).** 3,000 recruit hometowns clustered around metro areas, with a production index.

- **Counts:** the 5 busiest of 44 regions supply about **35%** of all recruits. A count map lights up the big metros.
- **Average production:** about **54.8** in those regions vs **54.5** everywhere else, essentially the same. Population explains *how many*, not *how good*.
- **Kriged map:** moves only a few points around the national average. The variogram rises only slowly with distance, which means a weak regional pattern buried in individual noise.
- **Level 3 project test:** adding the map to a baseline of rating + density improves next-class prediction by only about **2%**. It barely passes, and only on simulated data.
""",
    takeaway="Kriging maps the average outcome by location, with uncertainty. Always separate 'more players' from 'better players'.",
    visual=visual,
    visual_caption="Switch maps. The count map shows where people live. The kriged map is the actual question: is the average recruit from here better? Its answer is 'only slightly, in places'.",
    math=[
        MathBlock(
            "Step 1: the variogram (similarity, the geostatistics way)",
            r"\gamma(h) = \tfrac12\, \mathbb{E}\bigl[(Z(s) - Z(s + h))^2\bigr]",
            """
- $s$: a location (a hometown); $h$: a distance
- $Z(s)$: the outcome at that location (production)
- $\\gamma(h)$: half the average squared difference between players whose hometowns are $h$ apart
- Rising $\\gamma(h)$: nearby players are more alike than distant ones
""",
        ),
        MathBlock(
            "Step 2: variogram and kernel are two views of one thing",
            r"\gamma(h) = \sigma^2 - k(h)",
            """
- $\\sigma^2$: total variance (signal + noise)
- $k(h)$: the kernel (covariance) at distance $h$
- High similarity (big $k$) means a small difference (small $\\gamma$)
""",
        ),
        MathBlock(
            "Step 3: the Kriging prediction is the GP prediction",
            r"\hat Z(s_*) = \mu + \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} (\mathbf{z} - \mu),\qquad \sigma_*^2 = k(0) - \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1}\mathbf{k}_*",
            """
- $s_*$: a place on the map to predict; $\\mu$: the overall mean
- $\\mathbf{k}_*$: kernel values between $s_*$ and each observed hometown
- $K$: kernel values among observed hometowns; $\\sigma_n^2$: individual noise
- Exactly the formulas from L3.5, with locations instead of seasons
""",
        ),
    ],
    code=CODE,
    code_notes="Add the regional table from the project: `from academy.spatial import regional_table; print(regional_table(df).head())`. Compare 'players' with 'mean_production'.",
    walkthrough=[
        ("vg = empirical_variogram(df)", "Pair up players and record how different their production is at each distance between hometowns."),
        ("vg.iloc[::2]", "Print every other distance bin, to keep the output short."),
        ("fit_gp(df[['lon', 'lat']], df['production'] - mean, 'Matern')", "Kriging: a GP with a Matérn kernel on longitude and latitude, fitted to production above or below average."),
        ("gp.kernel_", "The fitted settings: signal height, length scale (in degrees) and noise. A small signal next to a big noise = a weak map."),
        ("predict_latent(gp, [[lon, lat]])", "Predict the average production at a place, and how unsure we are about that regional average (not about one individual player)."),
        ("regions = regional_table(df)", "Group hometowns into 5-degree boxes: players, average production and its standard error."),
        ("busiest['players'].sum() / len(df)", "How concentrated recruiting is: the share of all recruits from the 5 busiest boxes."),
        ("avg_busy ... vs avg_rest", "The question that matters: is the AVERAGE recruit from the busy regions better? (Here: barely different.)"),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Roster hometown coordinates** support exactly this analysis. Do it with production per recruit, adjusted for rating and population, not with counts.
- **Recruiting decisions:** a region is only a real advantage if its recruits outperform after accounting for rating, population and sample size. The uncertainty map shows where you can't tell.
- **Your rule:** a geographic term goes on your site only if it beats a rating + density baseline on later recruiting classes. The Level 3 project runs that test.
""",
    quiz=[
        Question(
            "q1", "mc", "In plain words, Kriging is:",
            "A Gaussian process applied to map locations, predicting a smooth surface with uncertainty.",
            options=["A Gaussian process on a map, with uncertainty", "A way to count players by state",
                     "A clustering of hometowns", "A recruiting ranking system"],
            notes=["Correct.", "Counting isn't Kriging.", "That's K-means.", "It's a statistical method."],
            answer=0),
        Question(
            "q2", "mc", "A map shows many productive players from big metro areas. What's the most likely explanation?",
            "More people live there, so more players of every kind come from there. It says nothing about the average recruit.",
            category="Application",
            options=["More people live there, so more players of every kind come from there",
                     "Metro recruits are better", "The map is wrong", "Kriging always shows metros"],
            notes=["Correct.", "Only if the average recruit outperforms, after accounting for rating and population.",
                   "The map can be right and still misleading.", "It shows averages, not counts."],
            answer=0),
        Question(
            "q3", "numeric", "Total variance σ² = 110 and the kernel at distance h is k(h) = 4. What is the variogram γ(h)?",
            "γ(h) = σ² − k(h) = 110 − 4 = 106.", category="Application", answer=106, tolerance=0.01),
        Question(
            "q4", "short", "Where on a Kriging map is uncertainty highest, and why?",
            "Where few recruits come from, because there's little nearby data.",
            keywords=[["few", "sparse", "no data", "little data", "far", "rural", "empty"],
                      ["data", "players", "recruits", "observations", "nearby", "information"]],
            model_answer="Where few recruits come from (sparse areas), because there is little nearby data to learn from, so the prediction falls back toward the average with a wide band."),
        Question(
            "q5", "mc", "The variogram rises only slowly with distance. What does that suggest?",
            "Nearby and distant players are about equally different: a weak regional pattern, mostly individual noise.",
            category="Application",
            options=["A weak regional pattern; most variation is individual", "A very strong regional pattern",
                     "The data is wrong", "Distance doesn't exist"],
            notes=["Correct.", "A strong pattern would rise steeply.", "It's a legitimate finding.", "It does."],
            answer=0),
    ],
)
