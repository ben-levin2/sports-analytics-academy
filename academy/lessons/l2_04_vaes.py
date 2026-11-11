import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Ellipse

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig
from ._l2 import FEATURES, group_profiles


def linear_vae(Z, q=2):
    """Closed-form linear-Gaussian latent model (probabilistic PCA): what a linear VAE learns."""
    S = np.cov(Z, rowvar=False)
    vals, vecs = np.linalg.eigh(S)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    sigma2 = vals[q:].mean()
    W = vecs[:, :q] * np.sqrt(vals[:q] - sigma2)
    M = W.T @ W + sigma2 * np.eye(q)
    Minv = np.linalg.inv(M)
    return W, sigma2, Minv @ W.T, sigma2 * Minv        # decoder W, noise, encoder matrix, posterior cov


def visual():
    d, Z = group_profiles("DL")
    W, s2, E, C = linear_vae(Z)
    mu = Z @ E.T
    c1, c2 = st.columns([1, 2])
    with c1:
        n_gen = st.slider("Generate synthetic players", 0, 300, 100, 50, key="l24_gen")
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots()
    ax.scatter(mu[:, 0], mu[:, 1], s=7, color=T.BLUE, alpha=0.35, label="real players (encoded means)")
    vals, vecs = np.linalg.eigh(C)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    for i in rng.choice(len(mu), 6, replace=False):
        ax.add_patch(Ellipse(mu[i], 2 * 2 * np.sqrt(vals[1]), 2 * 2 * np.sqrt(vals[0]), angle=angle,
                             fill=False, color=T.AMBER, lw=1.6))
    if n_gen:
        zs = rng.normal(size=(n_gen, 2))
        ax.scatter(zs[:, 0], zs[:, 1], s=14, marker="+", color=T.GREEN, label="sampled from the prior N(0, 1)")
    ax.set_xlabel("Latent dimension 1"), ax.set_ylabel("Latent dimension 2"), ax.legend(fontsize=7, loc="upper right")
    with c2:
        show_fig(fig)
    st.caption("Amber ovals: the encoder's uncertainty region for six players (about 95%). "
               "Green crosses: new latent points drawn from the prior. Decoding them gives synthetic profiles.")


CODE = '''import numpy as np
from sklearn.preprocessing import StandardScaler
from academy.data import make_defensive_profiles, PROFILE_FEATURES

df = make_defensive_profiles()                               # SIMULATED
dl = df[(df["position_group"] == "DL") & (df["snaps"] >= 150)]
scaler = StandardScaler().fit(dl[PROFILE_FEATURES])
Z = scaler.transform(dl[PROFILE_FEATURES])
q = 2                                                        # size of the latent space

# A LINEAR VAE has an exact answer (probabilistic PCA), so no neural-network training is needed
S = np.cov(Z, rowvar=False)
vals, vecs = np.linalg.eigh(S)
vals, vecs = vals[::-1], vecs[:, ::-1]                       # biggest directions first
sigma2 = vals[q:].mean()                                     # leftover variation = noise
W = vecs[:, :q] * np.sqrt(vals[:q] - sigma2)                 # decoder: latent -> stats

# Encoder: for each player, a MEAN position and an UNCERTAINTY (covariance) in the latent space
M = W.T @ W + sigma2 * np.eye(q)
post_mean = Z @ (np.linalg.inv(M) @ W.T).T
post_cov = sigma2 * np.linalg.inv(M)
print("Player 1 latent mean:", post_mean[0].round(2), " latent SDs:", np.sqrt(np.diag(post_cov)).round(2))

# KL term: how far each player's latent region is from the standard prior N(0, I)
kl = 0.5 * (np.trace(post_cov) + (post_mean ** 2).sum(1) - q - np.log(np.linalg.det(post_cov)))
print(f"Average KL to the prior: {kl.mean():.2f}")

# Generate 5 synthetic players: sample latent points from the prior and decode them
rng = np.random.default_rng(1)
z_new = rng.normal(size=(5, q))
x_new = z_new @ W.T + rng.normal(0, np.sqrt(sigma2), size=(5, len(PROFILE_FEATURES)))
synthetic = scaler.inverse_transform(x_new)
print("\\nSynthetic DL profiles (SIMULATED from the model, not real players):")
print("Impossible values (negative rates) in the synthetic set:", int((synthetic[:, :6] < 0).sum()))
for row in synthetic:
    print("  " + ", ".join(f"{f}={v:.3f}" for f, v in zip(PROFILE_FEATURES[:3], row[:3])),
          f"weight={row[-1]:.0f}")
'''

LESSON = Lesson(
    id="l2-04-vaes", level=2, order=4,
    title="Variational Autoencoders: Latent Spaces With Uncertainty",
    subtitle="Fuzzy codes, a tidy map, and generating new profiles",
    minutes=30,
    concept_ids=["vaes"],
    simple="""
An autoencoder (L2.3) gives each player **one exact point** in the latent space. A **variational autoencoder (VAE)** changes two things:

1. **Each player gets a fuzzy region, not a point.** The encoder says "this player is around here, give or take this much". A low-snap player with noisy stats *should* get a bigger region.
2. **The whole map is kept tidy.** Training gently pulls every player's region toward one standard bell-shaped cloud centred at zero (the **prior**). That leaves no big empty gaps.

Because the map is tidy, you can pick **any** point near the middle, run it through the decoder, and get a **realistic new profile**. That's why VAEs are called *generative* models.

Full VAEs use neural networks and a deep-learning library (PyTorch), which this academy doesn't install. We use the **linear** version, which has an exact formula and is what a linear VAE learns. The ideas are identical.
""",
    problem="A plain autoencoder's map can have holes and gives no sense of how sure it is about each player. A VAE adds uncertainty for each player and a smooth, well-organized map you can sample from.",
    why="You get a latent position with error bars, smooth blends between player types, and the ability to simulate realistic synthetic profiles, for example to test a pipeline without real data.",
    when_useful="""
- You want uncertainty in each player's latent position
- Generating realistic synthetic profiles (clearly labelled as synthetic)
- Large, rich data where a smooth latent space helps later modelling
""",
    when_not="""
- Small data and few stats: the linear version (or PCA) is all the data supports
- When you just need clusters or a 2-D picture: PCA and K-means are simpler
- Never present generated profiles as real players
- Stats with hard limits (rates between 0 and 1): the linear version can generate impossible values
""",
    compare="""
**VAE vs autoencoder:** an autoencoder gives a point and only cares about rebuild error. A VAE gives a region and balances rebuild error against staying close to the prior.

**VAE vs PCA:** a linear VAE equals probabilistic PCA, which is PCA plus a noise model and uncertainty. A neural VAE can bend like an autoencoder.

**VAE vs Bayesian inference (Level 1):** it's the same recipe: a prior, a likelihood (the decoder) and an approximate posterior (the encoder). That's why its training objective is called the evidence lower bound, or ELBO, linking back to the evidence in L1.5.
""",
    sports="""
**Fictional scenario (simulated).** The same ~1,000 DL seasons, mapped into a 2-D latent space with the linear VAE.

- Each player gets a **mean position** plus a **standard deviation** in each direction: his uncertainty oval.
- Sampling 5 points from the prior and decoding them produces **5 synthetic DL profiles**. Most look like real DL seasons, but one comes out with a **negative pressure rate**, and the code counts 3 impossible values in total. The linear model doesn't know rates can't go below zero. A neural VAE with suitable outputs can fix that. Either way, these are clearly not real players.
- The KL number measures how far each player's region sits from the standard cloud. Training keeps it modest so the map stays tidy.
""",
    takeaway="A VAE is an autoencoder with Bayesian thinking: each player gets a fuzzy region, the map stays tidy, and you can generate realistic new profiles.",
    visual=visual,
    visual_caption="Blue dots: real players' latent means. Amber ovals: uncertainty regions. Green crosses: samples from the prior, which fall where real players live because the map is organized around that prior.",
    math=[
        MathBlock(
            "Step 1: the decoder says how stats come from a latent point",
            r"\mathbf{z}_{\text{stats}} = W\mathbf{t} + \boldsymbol{\varepsilon},\qquad \mathbf{t} \sim \mathcal{N}(\mathbf{0}, I),\quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I)",
            """
- $\\mathbf{t}$: a point in the latent space (2 numbers here)
- $\\mathcal{N}(\\mathbf{0}, I)$: the **prior**, a standard bell-shaped cloud centred at zero
- $W$: the decoder: how each latent dimension turns into the 7 stats
- $\\boldsymbol{\\varepsilon}$: random noise with spread $\\sigma$: what the latent space doesn't explain
""",
            "In a neural VAE, Wt is replaced by a neural network. Everything else stays the same.",
        ),
        MathBlock(
            "Step 2: the encoder gives each player a region",
            r"q(\mathbf{t} \mid \mathbf{z}_i) = \mathcal{N}\bigl(\boldsymbol{\mu}_i,\; \Sigma\bigr)",
            """
- $q(\\cdot)$: the encoder's guess at where player $i$ sits in the latent space
- $\\boldsymbol{\\mu}_i$: the centre of his region (his best-guess position)
- $\\Sigma$: the size and shape of the region (uncertainty). In the linear version every player shares the same $\\Sigma$; in a neural VAE it can differ by player
""",
        ),
        MathBlock(
            "Step 3: what training maximizes (the ELBO)",
            r"\text{ELBO} = \underbrace{\mathbb{E}_{q}\bigl[\log p(\mathbf{z}_i \mid \mathbf{t})\bigr]}_{\text{rebuild well}} \;-\; \underbrace{\text{KL}\bigl(q(\mathbf{t}\mid \mathbf{z}_i)\,\|\,\mathcal{N}(\mathbf{0}, I)\bigr)}_{\text{stay near the prior}}",
            """
- First term: how well the decoder rebuilds the player's stats from points in his region (higher is better)
- $\\text{KL}$: Kullback–Leibler divergence, a measure of how different two distributions are. Here, how far his region is from the prior
- Maximizing the ELBO balances "rebuild accurately" against "keep the map tidy"
- ELBO = evidence lower bound: it's always below the log of the evidence from L1.5
""",
        ),
    ],
    code=CODE,
    code_notes="Change q from 2 to 3. The noise sigma2 shrinks, because more dimensions explain more, and each player's latent SDs change. The synthetic profiles are SIMULATED from the model; never present them as real.",
    walkthrough=[
        ("S = np.cov(Z, rowvar=False)", "How the 7 standardized stats vary and move together across players."),
        ("vals, vecs = np.linalg.eigh(S)", "Break that into directions (vecs) and their sizes (vals), the same ingredients as PCA."),
        ("sigma2 = vals[q:].mean()", "The variation not captured by the q kept directions is treated as noise."),
        ("W = vecs[:, :q] * np.sqrt(vals[:q] - sigma2)", "The decoder: how each latent dimension maps to the 7 stats."),
        ("post_mean = Z @ (np.linalg.inv(M) @ W.T).T", "Encoder centre: each player's best-guess position in the latent space."),
        ("post_cov = sigma2 * np.linalg.inv(M)", "Encoder uncertainty: the size and shape of each player's region."),
        ("kl = 0.5 * (np.trace(post_cov) + ... )", "The KL formula for two bell curves: how far each player's region is from the standard prior."),
        ("z_new = rng.normal(size=(5, q))", "Pick 5 random points from the prior cloud..."),
        ("x_new = z_new @ W.T + rng.normal(...)", "...and decode them (plus noise) into 7 standardized stats."),
        ("scaler.inverse_transform(x_new)", "Undo the standardization to get stats in normal units: synthetic, clearly labelled profiles."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Synthetic data for testing:** a VAE (or its linear version) can generate realistic fake profiles to test a pipeline before touching real data. Always label them simulated, as this academy does.
- **Uncertainty in comps:** a player's latent region shows how sure the map is about his style. That matters for low-snap players, the thin-production case where Model 8 found its edge.
- **Your rule:** VAE features would only reach your site after beating a simple baseline, and the PCA version, on a test set fixed in advance.
""",
    quiz=[
        Question(
            "q1", "mc", "How does a VAE's encoder differ from a plain autoencoder's?",
            "It outputs a region (mean + uncertainty), not a single point.",
            options=["It gives a region (mean and uncertainty) instead of a single point",
                     "It uses position labels", "It predicts next season", "It has no decoder"],
            notes=["Correct.", "Neither uses labels.", "Neither predicts outcomes.", "It has a decoder, used to rebuild and generate."],
            answer=0),
        Question(
            "q2", "mc", "Why can you generate realistic new profiles from a VAE but not reliably from a plain autoencoder?",
            "The KL term organizes the latent space around a known prior, so points drawn from that prior decode into realistic profiles.",
            options=["Its map is organized around a known prior, so sampled points land where real players live",
                     "It memorizes every player", "It has more layers", "It uses real player names"],
            notes=["Correct.", "Memorizing would make generation worse, not better.", "Depth isn't the reason.", "It uses no names."],
            answer=0),
        Question(
            "q3", "numeric", "Latent dimension 1 for a player has posterior SD 0.4. Roughly how wide is his 95% region along that dimension (±2 SD, total width)?",
            "±2 × 0.4 = ±0.8, total width 1.6.", category="Application", answer=1.6, tolerance=0.01),
        Question(
            "q4", "short", "The ELBO has two parts. Name them and say what each one encourages.",
            "A rebuild term (reconstruct the stats well) and a KL term (stay close to the prior, keeping the map tidy).",
            keywords=[["rebuild", "reconstruct", "likelihood", "accurate", "fit"],
                      ["kl", "prior", "tidy", "close", "regular", "organized", "divergence"]],
            model_answer="A rebuild (reconstruction) term that rewards reproducing the player's stats, and a KL term that keeps each player's region close to the prior so the map stays tidy."),
        Question(
            "q5", "mc", "You generate 500 synthetic DL profiles with a VAE. What is an acceptable use?",
            "Synthetic profiles are fine for testing code and methods, if clearly labelled. Never present them as real players or as evidence.",
            category="Application",
            options=["Testing your pipeline, clearly labelled as simulated",
                     "Adding them to a real recruiting board", "Reporting them as real stats",
                     "Using them as the test set for your model"],
            notes=["Correct.", "They aren't real people.", "That would be fabricating statistics.",
                   "A test set must be real outcomes the model hasn't seen."],
            answer=0),
    ],
)
