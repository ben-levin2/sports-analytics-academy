import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import FancyBboxPatch

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from .. import ui
from ..ui import show_fig
from ._l2 import group_profiles


def _fit(bottleneck: int):
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPRegressor
    _, Z = group_profiles("DL")
    ae = MLPRegressor(hidden_layer_sizes=(5, bottleneck, 5), activation="tanh", max_iter=3000,
                      random_state=0, alpha=1e-4).fit(Z, Z)
    pca = PCA(bottleneck).fit(Z)
    err_ae = float(np.mean((ae.predict(Z) - Z) ** 2))
    err_pca = float(np.mean((pca.inverse_transform(pca.transform(Z)) - Z) ** 2))
    return err_ae, err_pca


@st.cache_data(show_spinner="Training small autoencoders…")
def _errors():
    return {b: _fit(b) for b in (1, 2, 3)}


def visual():
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    layers = [("7 stats in", 7), ("5", 5), ("code", 2), ("5", 5), ("7 stats out", 7)]
    for i, (name, n) in enumerate(layers):
        h = 0.18 * n
        color = T.AMBER if name == "code" else T.BLUE
        ax.add_patch(FancyBboxPatch((i * 1.6, -h / 2), 0.6, h, boxstyle="round,pad=0.04", color=color, alpha=0.8))
        ax.text(i * 1.6 + 0.3, -h / 2 - 0.25, name, ha="center", va="top", color=T.TEXT, fontsize=9)
    ax.text(1.9, 0.95, "encoder", ha="center", color=T.MUTED), ax.text(5.1, 0.95, "decoder", ha="center", color=T.MUTED)
    ax.set_xlim(-0.3, 7.3), ax.set_ylim(-1.3, 1.2), ax.axis("off")
    show_fig(fig)
    if st.button("Compare reconstruction error: autoencoder vs PCA", key="l23_run"):
        st.session_state["l23_errs"] = _errors()
    errs = st.session_state.get("l23_errs")
    if errs:
        ui.table(pd.DataFrame({"code size": list(errs), "autoencoder error": [round(v[0], 3) for v in errs.values()],
                      "PCA error": [round(v[1], 3) for v in errs.values()]}))
        st.caption("Error = average squared gap between the original standardized stats and the rebuilt ones. Lower is better.")


CODE = '''import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from academy.data import make_defensive_profiles, PROFILE_FEATURES

df = make_defensive_profiles()                               # SIMULATED
dl = df[(df["position_group"] == "DL") & (df["snaps"] >= 150)]
Z = StandardScaler().fit_transform(dl[PROFILE_FEATURES])     # 7 standardized stats

# An autoencoder: 7 -> 5 -> 2 (the code) -> 5 -> 7, trained to reproduce its own input
ae = MLPRegressor(hidden_layer_sizes=(5, 2, 5), activation="tanh",
                  max_iter=3000, random_state=0).fit(Z, Z)

def encode(model, X):
    """Run only the encoder half: input -> layer 1 -> the 2-number code."""
    h = X
    for W, b in zip(model.coefs_[:2], model.intercepts_[:2]):
        h = np.tanh(h @ W + b)
    return h

codes = encode(ae, Z)
print("First 3 players' 2-number codes:\\n", codes[:3].round(2))

err_ae = np.mean((ae.predict(Z) - Z) ** 2)
pca = PCA(2).fit(Z)
err_pca = np.mean((pca.inverse_transform(pca.transform(Z)) - Z) ** 2)
print(f"\\nReconstruction error, 2-number code:  autoencoder {err_ae:.3f}   PCA {err_pca:.3f}")
print("With 7 stats and ~1,000 players, a straight-line method (PCA) is hard to beat.")
'''

LESSON = Lesson(
    id="l2-03-autoencoders", level=2, order=3,
    title="Autoencoders: A Neural Network That Learns to Summarize Players",
    subtitle="Squeeze, rebuild, and keep the squeezed version",
    minutes=25,
    concept_ids=["autoencoders"],
    simple="""
Imagine describing a player to a friend using **only two numbers**, and the friend has to rebuild all seven of his stats from those two numbers. If the rebuilt stats are close to the real ones, your two numbers captured what matters.

An **autoencoder** is a small neural network that learns to play that game by itself:

- The **encoder** squeezes the 7 stats down to a short **code** (here, 2 numbers).
- The **decoder** tries to rebuild the 7 stats from the code.
- Training adjusts both halves until the rebuilt stats are as close to the originals as possible.

Afterwards you throw away the decoder and keep the codes. They form a latent space (L2.1), like PCA's, but one that can **bend** to follow curved patterns.
""",
    problem="PCA can only find straight-line combinations of stats. When stats relate in curved ways (common with tracking data and many features), an autoencoder can find a better compact summary.",
    why="It learns flexible, compact player summaries from many inputs without any labels, and the same idea powers the VAE in L2.4.",
    when_useful="""
- Dozens or hundreds of inputs (tracking-data features, play-by-play profiles)
- Curved relationships between stats that PCA can't follow
- Lots of data: thousands of players or more
""",
    when_not="""
- A handful of stats and about a thousand players (like here): PCA is simpler and usually just as good
- When you must explain each dimension: autoencoder codes are hard to interpret
- Small data: the network can memorize players instead of learning general patterns
""",
    compare="""
**Autoencoder vs PCA:** PCA is exact, fast, repeatable and easy to read. An autoencoder is flexible but slower, changes with the random start, and is hard to interpret. A useful fact: an autoencoder with straight-line (linear) layers learns the same summary as PCA.

**Autoencoder vs VAE (L2.4):** an autoencoder gives each player one exact point. A VAE gives a fuzzy region and organizes the whole space, so you can generate new realistic profiles.
""",
    sports="""
**Fictional scenario (simulated).** The same ~1,000 DL seasons with 7 stats each. We train a small autoencoder, 7 → 5 → **2** → 5 → 7, and compare it with PCA using 2 dimensions.

Both rebuild the stats with a similar amount of error. Here the autoencoder is about **the same as PCA, not better**. With only seven fairly straight-line stats, there is no curve for it to exploit.

That's an honest and common result: the fancier method only earns its keep with richer data.
""",
    takeaway="An autoencoder learns a compact code by practising squeezing and rebuilding. It beats PCA only when the data has curves PCA can't follow.",
    visual=visual,
    visual_caption="Stats flow left to right. The amber box is the code: the 2-number summary kept after training. Press the button to compare rebuild error with PCA for codes of size 1, 2 and 3.",
    math=[
        MathBlock(
            "Step 1: the encoder squeezes",
            r"\mathbf{h} = \tanh(W_1 \mathbf{z} + \mathbf{b}_1)",
            """
- $\\mathbf{z}$: a player's 7 standardized stats
- $W_1$: a table of weights the network learns, saying how much each stat feeds each unit
- $\\mathbf{b}_1$: learned offsets (biases)
- $\\tanh$: a gentle S-shaped bend. This is what lets the network capture curves
- $\\mathbf{h}$: the result, passed to the next layer. Repeating this down to 2 numbers gives the code
""",
        ),
        MathBlock(
            "Step 2: the decoder rebuilds",
            r"\hat{\mathbf{z}} = g(\text{code})",
            """
- $g$: the decoder: more layers of the same kind, going from 2 numbers back up to 7
- $\\hat{\\mathbf{z}}$: the rebuilt stats ("z-hat")
""",
        ),
        MathBlock(
            "Step 3: training makes the rebuild error small",
            r"L = \frac{1}{N}\sum_{i=1}^{N} \lVert \mathbf{z}_i - \hat{\mathbf{z}}_i \rVert^2",
            """
- $L$: the loss, the average squared gap between real and rebuilt stats
- Training nudges every weight a little at a time in the direction that lowers $L$ (gradient descent), many times over
""",
            "If tanh is replaced by no bend at all (linear layers), the best code spans the same space as PCA.",
        ),
    ],
    code=CODE,
    code_notes="Change hidden_layer_sizes to (5, 1, 5) and (5, 3, 5). How does the gap between autoencoder and PCA change?",
    walkthrough=[
        ("Z = StandardScaler().fit_transform(dl[PROFILE_FEATURES])", "Standardize the 7 stats. Neural networks train badly on mixed units."),
        ("MLPRegressor(hidden_layer_sizes=(5, 2, 5), activation='tanh', ...)", "A small neural network: 7 inputs, then layers of 5, 2 and 5 units, then 7 outputs. The 2 in the middle is the code."),
        (".fit(Z, Z)", "The trick: input and target are the SAME data. The network learns to reproduce what it's given."),
        ("def encode(model, X):", "scikit-learn doesn't hand you the middle layer, so we run the first half of the network ourselves."),
        ("for W, b in zip(model.coefs_[:2], model.intercepts_[:2]):", "Use only the first two weight tables: input to 5, then 5 to 2 (the encoder)."),
        ("h = np.tanh(h @ W + b)", "One layer: multiply by the weights, add the bias, apply the S-shaped bend."),
        ("np.mean((ae.predict(Z) - Z) ** 2)", "Rebuild error: how far the network's output is from the real stats."),
        ("pca.inverse_transform(pca.transform(Z))", "PCA's version: squeeze to 2 numbers, then rebuild, for a fair comparison."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **When it could matter:** with play-level or tracking-style data (dozens of inputs per player), an autoencoder could summarize style better than PCA.
- **When it doesn't:** with season box-score rates, start with PCA. It's exact, explainable and usually as good.
- **Your rule:** a learned code goes on your site only if it beats a simple baseline on a test set fixed in advance, and it should beat the PCA version too, or the extra complexity isn't earned.
""",
    quiz=[
        Question(
            "q1", "mc", "What is an autoencoder trained to do?",
            "Reproduce its own input after squeezing it through a small code.",
            options=["Rebuild its own input after squeezing it through a small code",
                     "Predict next season's production", "Sort players into positions",
                     "Rank players by recruiting rating"],
            notes=["Correct.", "It never sees outcomes.", "It uses no labels.", "It doesn't rank by anything."],
            answer=0),
        Question(
            "q2", "mc", "After training, which part do you usually keep to describe players?",
            "The encoder's code is the compact summary (the latent space).",
            options=["The code from the encoder", "Only the decoder", "The rebuild error", "The original stats"],
            notes=["Correct.", "The decoder rebuilds stats; the code is the summary.", "Error grades training, but it's not a description.",
                   "You had those already."],
            answer=0),
        Question(
            "q3", "numeric", "One player's standardized stats are (1, 0, -1) and the rebuild is (0.8, 0.1, -0.7). What is his average squared rebuild error?",
            "((0.2)² + (0.1)² + (0.3)²) / 3 = (0.04 + 0.01 + 0.09) / 3 ≈ 0.0467.",
            category="Application", answer=0.0467, tolerance=0.001),
        Question(
            "q4", "short", "With 7 season stats for ~1,000 players, the autoencoder doesn't beat PCA. Why is that not surprising?",
            "Few, mostly straight-line stats and modest data leave nothing curved for the network to exploit; PCA already captures it.",
            category="Application",
            keywords=[["few", "small", "7", "seven", "not much", "simple", "little"],
                      ["linear", "straight", "curve", "pca", "nothing to", "already"]],
            model_answer="There are only a few stats with mostly straight-line relationships and not much data, so there are no curves for the network to use. PCA already captures what matters."),
        Question(
            "q5", "mc", "An autoencoder's layers have no bend (linear activations). What does it learn?",
            "A linear autoencoder learns the same subspace as PCA.",
            options=["Essentially the same summary as PCA", "Something better than PCA",
                     "Nothing useful at all", "A clustering into k groups"],
            notes=["Correct.", "Without bends it can't do better than PCA.", "It learns PCA's summary, which is useful.", "It doesn't make groups."],
            answer=0),
    ],
)
