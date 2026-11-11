import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


def matern(r, length, nu):
    s = np.abs(r) / length
    if nu == 0.5:
        return np.exp(-s)
    if nu == 1.5:
        return (1 + np.sqrt(3) * s) * np.exp(-np.sqrt(3) * s)
    return (1 + np.sqrt(5) * s + 5 * s ** 2 / 3) * np.exp(-np.sqrt(5) * s)


def rbf(r, length):
    return np.exp(-0.5 * (r / length) ** 2)


def visual():
    c1, c2 = st.columns([1, 2])
    with c1:
        length = st.slider("Length scale", 0.5, 5.0, 2.0, 0.25, key="l36_len")
        st.caption("Same length scale for every kernel.")
    r = np.linspace(0, 10, 300)
    xs = np.linspace(0, 20, 300)
    D = xs[:, None] - xs[None, :]
    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    specs = [("RBF", lambda d: rbf(d, length), T.BLUE), ("Matérn 2.5", lambda d: matern(d, length, 2.5), T.GREEN),
             ("Matérn 1.5", lambda d: matern(d, length, 1.5), T.AMBER), ("Matérn 0.5", lambda d: matern(d, length, 0.5), T.ROSE)]
    for name, f, c in specs:
        axes[0].plot(r, f(r), color=c, lw=2, label=name)
        draw = rng.multivariate_normal(np.zeros(len(xs)), f(D) + 1e-8 * np.eye(len(xs)))
        axes[1].plot(xs, draw, color=c, lw=1.4)
    axes[0].set_xlabel("Distance apart"), axes[0].set_ylabel("Similarity"), axes[0].legend(fontsize=7)
    axes[1].set_xlabel("x (e.g. seasons or miles)"), axes[1].set_title("One random curve per kernel", fontsize=9)
    with c2:
        show_fig(fig)


CODE = '''import numpy as np

def rbf(r, length):
    """RBF: similarity falls off as a smooth bell curve of distance."""
    return np.exp(-0.5 * (r / length) ** 2)

def matern(r, length, nu):
    """Matern: similarity with a 'roughness' knob nu (0.5 rough, 1.5 medium, 2.5 fairly smooth)."""
    s = np.abs(r) / length
    if nu == 0.5:
        return np.exp(-s)
    if nu == 1.5:
        return (1 + np.sqrt(3) * s) * np.exp(-np.sqrt(3) * s)
    return (1 + np.sqrt(5) * s + 5 * s ** 2 / 3) * np.exp(-np.sqrt(5) * s)

length = 2.0
print("distance   RBF   Matern2.5  Matern1.5  Matern0.5")
for r in (0.5, 1, 2, 4, 6):
    print(f"{r:>6}   {rbf(r, length):.2f}     {matern(r, length, 2.5):.2f}       "
          f"{matern(r, length, 1.5):.2f}       {matern(r, length, 0.5):.2f}")

# Draw 20 random curves from each kernel and measure how sharply they bend
x = np.linspace(0, 20, 400)
D = x[:, None] - x[None, :]
rng = np.random.default_rng(0)
print("\\nAverage bend between neighbouring points, over 20 random curves (bigger = rougher):")
for name, K in [("RBF", rbf(D, length)), ("Matern 2.5", matern(D, length, 2.5)),
                ("Matern 1.5", matern(D, length, 1.5)), ("Matern 0.5", matern(D, length, 0.5))]:
    f = rng.multivariate_normal(np.zeros(len(x)), K + 1e-8 * np.eye(len(x)), size=20)
    print(f"  {name:<11} {np.mean(np.abs(np.diff(f, n=2, axis=1))):.5f}")
'''

LESSON = Lesson(
    id="l3-06-kernels", level=3, order=6,
    title="Gaussian Processes, Part 2: Kernels, the Rule for 'Similar'",
    subtitle="Kernel functions, RBF and Matérn",
    minutes=20,
    concept_ids=["kernel-functions", "rbf-kernels", "matern-kernels"],
    simple="""
A Gaussian process needs one rule: **how similar should two points be, given how far apart they are?** That rule is the **kernel**.

- Two seasons in a row, or two hometowns 20 miles apart: probably similar.
- Two seasons four years apart, or towns 800 miles apart: barely related.

Two settings shape every kernel:
- **Length scale:** how far apart points can be and still count as similar.
- **Smoothness:** whether the curve glides gently or is allowed to wiggle and jump.

The two most common kernels:
- **RBF** (radial basis function): extremely smooth curves, like a pencil line that never has a corner.
- **Matérn:** has an extra **roughness** dial ν ("nu"). ν = 0.5 is jagged, ν = 1.5 is moderately smooth, ν = 2.5 is close to RBF. Real-world data is often a little rougher than RBF assumes, so Matérn 1.5 or 2.5 is a common default.
""",
    problem="A GP is only as good as its notion of similarity. The kernel decides how far information spreads and how smooth the predictions can be.",
    why="Choosing the kernel is choosing your assumptions about the world: how quickly production changes across seasons, or how far a regional pattern reaches across a map.",
    when_useful="""
- Any GP or Kriging model (L3.5, L3.7, L3.8)
- Encoding football knowledge: 'development changes gradually', 'nearby hometowns are similar'
- Comparing kernels by their marginal likelihood (L1.5) or by validation
""",
    when_not="""
- Treating kernel choice as a magic fix: with noisy data, kernels often give near-identical predictions (as in the Level 3 map)
- Using a smooth RBF for data with sudden jumps (injury, scheme change): it will blur them
- Choosing the length scale by eye: let the data fit it (L3.7), then sanity-check it
""",
    compare="""
**RBF vs Matérn:** RBF assumes extreme smoothness. Matérn lets you dial down smoothness and is often more realistic. With ν → ∞, Matérn becomes RBF.

**Kernels vs distance-based averaging (nearest neighbours):** both use 'closer counts more', but a kernel also gives a proper uncertainty band through the GP.

**Kernels in K-means and SVMs:** the same word means a similarity function there too.
""",
    sports="""
**Fictional scenario (simulated).** For recruit hometowns, the Level 3 map project fits both kernels by marginal likelihood:

- **RBF:** length scale about 13 degrees (very roughly 800–900 miles)
- **Matérn 1.5:** length scale about 13 degrees

Their fits are almost identical. With individual production this noisy, the data can't tell smooth from rough. That's an honest and common finding: **the kernel matters less than the noise level and the sample size.**

Where kernels do matter: development curves with sudden changes, or dense data with sharp regional edges.
""",
    takeaway="The kernel says what counts as similar. Its length scale says how far information spreads, and its smoothness says how wiggly the curve can be.",
    visual=visual,
    visual_caption="Left: similarity vs distance for each kernel. Right: one random curve drawn from each. RBF curves glide, while Matérn 0.5 curves are jagged. Move the length scale to stretch or squeeze both.",
    math=[
        MathBlock(
            "Step 1: a kernel turns distance into similarity",
            r"k(x, x') = \text{similarity of } f(x) \text{ and } f(x'),\qquad r = |x - x'|",
            """
- $x, x'$: two inputs (two seasons, or two hometowns)
- $r$: the distance between them
- $k$: near its maximum when $r = 0$, shrinking toward 0 as $r$ grows
""",
        ),
        MathBlock(
            "Step 2: the RBF kernel",
            r"k_{\text{RBF}}(r) = \sigma_f^2 \exp\!\left(-\frac{r^2}{2\ell^2}\right)",
            """
- $\\sigma_f^2$: signal height: how far curves stray from the average
- $\\ell$: **length scale**: about how far apart points can be and still be strongly related
- The squared distance makes similarity fall off as a smooth bell, which gives very smooth curves
""",
        ),
        MathBlock(
            "Step 3: the Matérn kernel (ν = 1.5 shown)",
            r"k_{\nu=1.5}(r) = \sigma_f^2 \left(1 + \frac{\sqrt{3}\, r}{\ell}\right) \exp\!\left(-\frac{\sqrt{3}\, r}{\ell}\right)",
            """
- $\\nu$ (nu): the roughness dial. 0.5 = jagged, 1.5 = once-smooth, 2.5 = twice-smooth, ∞ = RBF
- $\\ell, \\sigma_f^2$: same meaning as for RBF
- The plain (not squared) distance in the exponent lets similarity drop faster at short range, which gives rougher curves
""",
        ),
    ],
    code=CODE,
    code_notes="Change `length = 2.0` to 5.0. Every kernel keeps points similar over longer distances, and every random curve calms down.",
    walkthrough=[
        ("def rbf(r, length):", "The RBF kernel: similarity is a bell curve of distance."),
        ("np.exp(-0.5 * (r / length) ** 2)", "Squared distance over the length scale. At r = length, similarity is about 0.61."),
        ("def matern(r, length, nu):", "The Matérn kernel with its roughness dial nu."),
        ("if nu == 0.5: return np.exp(-s)", "The roughest common Matérn: similarity drops straight away, giving jagged curves."),
        ("for r in (0.5, 1, 2, 4, 6):", "Print each kernel's similarity at a few distances, to compare how fast they fade."),
        ("D = x[:, None] - x[None, :]", "All pairwise distances between 400 points, so we can build a full similarity table."),
        ("rng.multivariate_normal(..., size=20)", "Draw 20 random curves that obey this kernel's similarity rules (samples from the GP prior). One curve alone can mislead."),
        ("np.diff(f, n=2, axis=1)", "How much the slope changes from point to point: the bend. Smooth curves barely bend; jagged ones bend constantly."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Geography:** the fitted length scale of a hometown map is a useful number in itself. It says how far a regional pattern reaches.
- **Development:** a short length scale over seasons means production is hard to forecast from earlier seasons. That's consistent with "same as last season" being tough to beat (Models 9 and 9b).
- **Your rule:** don't pick a kernel because its map looks nicer. Choose by marginal likelihood or validation, and keep the method exploratory until it beats a simple baseline on a fixed test set.
""",
    quiz=[
        Question(
            "q1", "mc", "In plain words, what does a kernel decide in a Gaussian process?",
            "How similar two points are expected to be, based on how far apart they are.",
            concept="kernel-functions",
            options=["How similar two points should be, given their distance", "How many clusters to make",
                     "Which players to remove", "The final prediction"],
            notes=["Correct.", "That's K-means.", "It removes nothing.", "The prediction comes from the kernel plus the data."],
            answer=0),
        Question(
            "q2", "mc", "A larger length scale means:",
            "Points further apart still count as similar, so curves are smoother and information spreads further.",
            concept="kernel-functions",
            options=["Information spreads further and curves are smoother", "Curves become more jagged",
                     "Only the nearest point matters", "The GP ignores the data"],
            notes=["Correct.", "That's a smaller length scale or a lower nu.", "That's a very small length scale.", "It still uses the data."],
            answer=0),
        Question(
            "q3", "numeric", "RBF kernel with length scale 2 (and signal height 1). What is the similarity at distance r = 2? (e^-0.5)",
            "exp(−0.5 × (2/2)²) = e^−0.5 ≈ 0.607.", category="Application", concept="rbf-kernels",
            answer=0.607, tolerance=0.005),
        Question(
            "q4", "mc", "Random curves from an RBF kernel are:",
            "RBF assumes extreme smoothness: no corners at all.",
            concept="rbf-kernels",
            options=["Very smooth, with no corners", "Jagged like a stock chart", "Always straight lines", "Always flat"],
            notes=["Correct.", "That's Matérn 0.5.", "They curve.", "They wander around the mean."],
            answer=0),
        Question(
            "q5", "mc", "What does the Matérn ν ('nu') setting control?",
            "Roughness: 0.5 is jagged, 1.5 moderately smooth, 2.5 smoother; as ν grows it approaches RBF.",
            concept="matern-kernels",
            options=["How rough or smooth the curves are", "The number of seasons", "The noise level", "The mean curve"],
            notes=["Correct.", "Not related.", "Noise is a separate term.", "The mean is set separately."],
            answer=0),
        Question(
            "q6", "short", "RBF and Matérn give almost identical fits on the hometown map. What does that tell you?",
            "The data is too noisy or sparse to distinguish smooth from rough patterns; kernel choice isn't what matters here.",
            category="Application", concept="matern-kernels",
            keywords=[["noise", "noisy", "can't tell", "cannot tell", "not enough", "sparse", "little signal", "weak"],
                      ["doesn't matter", "not matter", "same", "either", "choice", "kernel", "little difference"]],
            model_answer="The data is too noisy to tell smooth from rough patterns, so kernel choice barely matters here. Noise and sample size matter more."),
    ],
)
