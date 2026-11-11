"""Visual identity: dark research-lab palette, matplotlib style, CSS."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

BG = "#0B1220"
PANEL = "#111B2E"
GRID = "#243049"
TEXT = "#E6EDF7"
MUTED = "#8FA3BF"
BLUE = "#4C9AFF"
CYAN = "#38D5E6"
AMBER = "#F5B94E"
ROSE = "#FF6B8A"
GREEN = "#4ADE80"
VIOLET = "#B794F6"
CYCLE = [BLUE, AMBER, CYAN, ROSE, GREEN, VIOLET]


def apply_mpl_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
        "figure.figsize": (7, 3.6), "figure.dpi": 110, "font.size": 10,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
        "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.6, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
        "axes.prop_cycle": cycler(color=CYCLE),
        "legend.facecolor": PANEL, "legend.edgecolor": GRID, "legend.labelcolor": TEXT,
    })


CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:wght@500;600&display=swap');
.stApp { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 3.6rem; max-width: 1180px; }
section[data-testid="stSidebar"] { border-right: 1px solid #1C2940; }

.aa-label { font-family: 'IBM Plex Mono', monospace; font-size: .68rem; letter-spacing: .16em;
            text-transform: uppercase; color: #8FA3BF; }
.aa-hero { border: 1px solid #1F2E49; border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
           background: linear-gradient(135deg, #0F1A30 0%, #12203B 60%, #0F2A4A 100%);
           position: relative; overflow: hidden; }
.aa-hero:after { content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .5;
           background-image: linear-gradient(#1c2c4a 1px, transparent 1px), linear-gradient(90deg, #1c2c4a 1px, transparent 1px);
           background-size: 32px 32px; mask-image: linear-gradient(120deg, transparent 40%, #000 100%);
           -webkit-mask-image: linear-gradient(120deg, transparent 40%, #000 100%); }
.aa-hero h1 { margin: .15rem 0 .2rem 0; font-size: 2rem; position: relative; z-index: 1; }
.aa-hero p { margin: 0; color: #9FB3CF; position: relative; z-index: 1; }
.aa-hero .aa-label { position: relative; z-index: 1; color: #5FA8FF; }

.aa-tile { background: #111B2E; border: 1px solid #1F2E49; border-radius: 10px; padding: .85rem 1rem; height: 100%; }
.aa-tile .v { font-size: 1.7rem; font-weight: 600; color: #E6EDF7; line-height: 1.15; margin-top: .25rem; }
.aa-tile .s { font-size: .78rem; color: #8FA3BF; margin-top: .1rem; }

.aa-bar { height: 6px; background: #1B2842; border-radius: 3px; overflow: hidden; margin: .35rem 0 .15rem; }
.aa-bar > div { height: 100%; background: linear-gradient(90deg, #2E6FDB, #4C9AFF); border-radius: 3px; }
.aa-bar.big { height: 10px; border-radius: 5px; }

.aa-pill { display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: .68rem; letter-spacing: .06em;
           padding: .12rem .55rem; border-radius: 99px; border: 1px solid #2A3C5E; color: #9FB3CF; margin-right: .3rem; }
.aa-pill.blue { color: #7DB8FF; border-color: #2D5C9E; background: #10233F; }
.aa-pill.green { color: #6EE7A0; border-color: #1F6B45; background: #0E2A21; }
.aa-pill.amber { color: #F5B94E; border-color: #7A5A1E; background: #2A2111; }
.aa-pill.rose { color: #FF8FA8; border-color: #7A2E40; background: #2A1219; }
.aa-pill.gray { color: #8FA3BF; }

.aa-sec { display: flex; align-items: baseline; gap: .7rem; margin: 2rem 0 .6rem; padding-top: .9rem;
          border-top: 1px solid #1C2940; }
.aa-sec .n { font-family: 'IBM Plex Mono', monospace; color: #4C9AFF; font-size: .8rem; }
.aa-sec .t { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.35rem; font-weight: 600; color: #E6EDF7; }

.aa-sim { border-left: 3px solid #F5B94E; background: #1D1A12; color: #E7D3A3; font-size: .85rem;
          padding: .5rem .8rem; border-radius: 0 6px 6px 0; margin: .5rem 0 1rem; }
.aa-takeaway { border-left: 3px solid #4C9AFF; background: #0F2038; padding: .6rem .9rem; border-radius: 0 6px 6px 0; margin-top: .8rem; }
.aa-row { display: flex; justify-content: space-between; gap: 1rem; padding: .35rem 0; border-bottom: 1px solid #172338; font-size: .88rem; }
.aa-row .t { color: #8FA3BF; font-family: 'IBM Plex Mono', monospace; font-size: .75rem; white-space: nowrap; }
.aa-brand { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem; font-weight: 600; line-height: 1.15; }
.aa-brand span { color: #4C9AFF; }
.aa-map { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: .55rem; margin: .4rem 0 1rem; }
@media (max-width: 1100px) { .aa-map { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 640px) { .aa-map { grid-template-columns: 1fr; } }
.aa-lv { background: #0F1829; border: 1px solid #1F2E49; border-radius: 10px; padding: .6rem .55rem .7rem; position: relative; }
.aa-lv.cur { border-color: #2D5C9E; box-shadow: 0 0 0 1px #2D5C9E inset; }
.aa-lv h4 { font-family: 'IBM Plex Mono', monospace; font-size: .68rem; letter-spacing: .12em; color: #5FA8FF;
            margin: 0 0 .1rem; font-weight: 500; }
.aa-lv .lt { font-size: .78rem; color: #C9D6EA; line-height: 1.25; margin-bottom: .45rem; min-height: 2.2em; }
.aa-node { font-size: .72rem; line-height: 1.2; padding: .28rem .4rem; border-radius: 6px; margin-bottom: .25rem;
           border: 1px solid #263754; color: #9FB3CF; display: flex; gap: .35rem; }
.aa-node b { font-family: 'IBM Plex Mono', monospace; font-weight: 500; color: #6F86A6; min-width: 1.5em; }
.aa-node.planned { border-style: dashed; opacity: .6; }
.aa-node.new { border-color: #34507A; color: #C9D6EA; }
.aa-node.learning { border-color: #7A5A1E; background: #241D10; color: #F5D49A; }
.aa-node.proficient { border-color: #2D5C9E; background: #10233F; color: #A9CDFF; }
.aa-node.mastered { border-color: #1F6B45; background: #0E2A21; color: #8EF0B5; }
.aa-node.mastered b, .aa-node.proficient b, .aa-node.learning b { color: inherit; }
.aa-proj { font-family: 'IBM Plex Mono', monospace; font-size: .64rem; color: #8FA3BF; border-top: 1px dashed #263754;
           margin-top: .4rem; padding-top: .35rem; line-height: 1.5; }
.aa-legend { font-size: .72rem; color: #8FA3BF; display: flex; gap: .9rem; flex-wrap: wrap; }
.aa-legend span:before { content: ""; display: inline-block; width: .7em; height: .7em; border-radius: 2px;
                         margin-right: .3em; vertical-align: -1px; border: 1px solid var(--c); background: var(--b); }
.aa-track { border-radius: 10px; padding: .7rem .9rem; margin-bottom: .6rem; border: 1px solid; }
.aa-track.lab { border-color: #2D5C9E; background: #0E1D33; }
.aa-track.prod { border-color: #7A5A1E; background: #1D1810; }
.aa-gate { display: flex; gap: .6rem; padding: .4rem 0; border-bottom: 1px solid #172338; font-size: .88rem; }
.aa-gate .d { color: #8FA3BF; font-size: .8rem; }
.aa-walk { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 3fr); gap: .25rem .9rem; font-size: .88rem; }
.aa-walk code { font-family: 'IBM Plex Mono', monospace; font-size: .76rem; background: #0F1829; color: #A9CDFF;
                padding: .3rem .45rem; border-radius: 5px; display: block; white-space: pre-wrap; word-break: break-word; }
.aa-walk div { padding: .15rem 0 .45rem; border-bottom: 1px solid #172338; }
div[data-testid="stMetric"] { background: #111B2E; border: 1px solid #1F2E49; border-radius: 10px; padding: .6rem .9rem; }
"""


def css_block() -> str:
    return f"<style>{CSS}</style>"
