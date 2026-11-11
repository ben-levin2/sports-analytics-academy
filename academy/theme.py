"""Design system: one palette, one font pairing, one chart style. Every page and chart uses this file.

Palette: one PRIMARY (blue), one ACCENT (amber), and neutrals. Status colors (pass/fail) appear only
in small status pills. Legacy names (CYAN, ROSE, GREEN, VIOLET) map onto tints of the same palette so
older chart code stays on-brand without edits.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# ---------------------------------------------------------------- palette
BG = "#0B1220"            # page
PANEL = "#111A2B"         # cards, charts
BORDER = "#22304A"
GRID = "#1E2A3F"
TEXT = "#E7ECF4"
MUTED = "#93A3BA"
FAINT = "#6B7C95"

PRIMARY = "#5B9BF5"
PRIMARY_LIGHT = "#A9C8F8"
PRIMARY_DARK = "#2F64B5"
ACCENT = "#E9B04F"
ACCENT_DARK = "#C47F2C"
NEUTRAL_LIGHT = "#C5CFDD"

PASS = "#6CC59A"           # status pills only
FAIL = "#E3826F"

# legacy names used across chart code -> palette
BLUE, AMBER = PRIMARY, ACCENT
CYAN, VIOLET = PRIMARY_LIGHT, PRIMARY_DARK
ROSE, GREEN = ACCENT_DARK, NEUTRAL_LIGHT
CYCLE = [PRIMARY, ACCENT, NEUTRAL_LIGHT, PRIMARY_LIGHT, ACCENT_DARK, FAINT]

CMAP_SEQ = LinearSegmentedColormap.from_list("academy_seq", ["#16233A", PRIMARY_DARK, PRIMARY, PRIMARY_LIGHT])
CMAP_DIV = LinearSegmentedColormap.from_list("academy_div", [PRIMARY_DARK, PRIMARY, "#26324A", ACCENT, ACCENT_DARK])
for _c in (CMAP_SEQ, CMAP_DIV):
    try:
        matplotlib.colormaps.register(_c)
    except ValueError:
        pass


def apply_mpl_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
        "figure.figsize": (7, 3.4), "figure.dpi": 120, "font.size": 10, "font.family": "sans-serif",
        "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.labelsize": 9.5, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT, "axes.edgecolor": BORDER,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 1.0,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 5, "ytick.major.pad": 5,
        "text.color": TEXT, "axes.prop_cycle": cycler(color=CYCLE), "lines.linewidth": 2.0,
        "lines.markersize": 5, "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": TEXT,
        "image.cmap": "academy_seq", "patch.edgecolor": PANEL,
    })


CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root { --s1: .25rem; --s2: .5rem; --s3: .75rem; --s4: 1rem; --s5: 1.5rem; --s6: 2rem;
        --bg: #0B1220; --panel: #111A2B; --border: #22304A; --text: #E7ECF4; --muted: #93A3BA; --faint: #6B7C95;
        --primary: #5B9BF5; --primary-soft: #14243F; --accent: #E9B04F; --accent-soft: #2A2214;
        --pass: #6CC59A; --fail: #E3826F; --radius: 10px; }
.stApp, .stApp p, .stApp li, .stApp label, .stMarkdown { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
.stApp code, .stApp pre, .stCode { font-family: 'IBM Plex Mono', ui-monospace, monospace !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
        font-weight: 600 !important; letter-spacing: -0.01em; color: var(--text); }
.stApp h1 { font-size: 1.75rem !important; } .stApp h2 { font-size: 1.4rem !important; }
.stApp h3 { font-size: 1.15rem !important; } .stApp h4 { font-size: 1.0rem !important; }
.block-container { padding-top: 3.2rem; padding-bottom: 4rem; max-width: 1120px; }
.stMarkdown p, .stMarkdown li { line-height: 1.6; }
section[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] .stRadio label p { font-size: .92rem; }
div[data-testid="stExpander"] details { border-color: var(--border) !important; border-radius: var(--radius); }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: var(--radius); }
.stButton button, .stDownloadButton button, .stFormSubmitButton button { border-radius: 8px; font-weight: 500; }

/* labels, page header */
.aa-label { font-size: .7rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--muted);
            margin: var(--s4) 0 var(--s2); }
.aa-hero { padding: 0 0 var(--s4); margin-bottom: var(--s5); border-bottom: 1px solid var(--border); }
.aa-hero .aa-label { color: var(--primary); margin: 0 0 var(--s1); }
.aa-hero h1 { margin: 0 0 var(--s1) 0; padding: 0; }
.aa-hero p { margin: 0; color: var(--muted); font-size: .95rem; max-width: 46rem; }

/* tiles & progress */
.aa-tile { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: var(--s3) var(--s4); height: 100%; }
.aa-tile .v { font-size: 1.55rem; font-weight: 600; color: var(--text); line-height: 1.2; margin-top: var(--s1); }
.aa-tile .s { font-size: .78rem; color: var(--muted); margin-top: 2px; }
.aa-tile .aa-label { margin: 0; }
.aa-bar { height: 6px; background: #1A263B; border-radius: 3px; overflow: hidden; margin: var(--s2) 0 var(--s1); }
.aa-bar > div { height: 100%; background: var(--primary); border-radius: 3px; }
.aa-bar.big { height: 8px; }

/* pills */
.aa-pill { display: inline-block; font-size: .72rem; font-weight: 500; padding: .1rem .5rem; border-radius: 99px;
           border: 1px solid var(--border); color: var(--muted); margin: 0 .3rem .3rem 0; white-space: nowrap; }
.aa-pill.blue { color: #BBD4FA; border-color: #2D4E80; background: var(--primary-soft); }
.aa-pill.green { color: var(--pass); border-color: #2B5A44; background: #102419; }
.aa-pill.amber { color: var(--accent); border-color: #6A5124; background: var(--accent-soft); }
.aa-pill.rose { color: var(--fail); border-color: #6A3229; background: #2A1512; }
.aa-pill.gray { color: var(--muted); }

/* lesson steps */
.aa-steps { display: flex; flex-wrap: wrap; gap: .35rem; margin: 0 0 var(--s5); }
.aa-steps a { font-size: .74rem; color: var(--muted); text-decoration: none; border: 1px solid var(--border);
              border-radius: 99px; padding: .15rem .55rem; }
.aa-steps a:hover { color: var(--text); border-color: var(--primary); }
.aa-sec { display: flex; align-items: baseline; gap: .6rem; margin: var(--s6) 0 var(--s3); padding-top: var(--s4);
          border-top: 1px solid var(--border); scroll-margin-top: 4rem; }
.aa-sec .n { font-size: .72rem; font-weight: 600; letter-spacing: .08em; color: var(--primary); text-transform: uppercase; }
.aa-sec .t { font-size: 1.2rem; font-weight: 600; color: var(--text); }

/* callouts */
.aa-sim { border-left: 3px solid var(--accent); background: var(--accent-soft); color: #E8D6AE; font-size: .85rem;
          padding: var(--s2) var(--s3); border-radius: 0 8px 8px 0; margin: var(--s2) 0 var(--s4); }
.aa-takeaway { border: 1px solid #2D4E80; background: var(--primary-soft); padding: var(--s3) var(--s4); border-radius: var(--radius);
               margin: var(--s4) 0; color: var(--text); }
.aa-takeaway .aa-label { color: var(--primary); margin: 0 0 var(--s1); }
.aa-row { display: flex; justify-content: space-between; gap: var(--s4); padding: .45rem 0; border-bottom: 1px solid #172338; font-size: .88rem; }
.aa-row .t { color: var(--muted); font-size: .76rem; white-space: nowrap; }
.aa-brand { font-size: 1.1rem; font-weight: 600; line-height: 1.2; color: var(--text); }
.aa-brand span { color: var(--primary); }

/* curriculum map */
.aa-map { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: var(--s2); margin: var(--s2) 0 var(--s3); }
@media (max-width: 1100px) { .aa-map { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
.aa-lv { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: var(--s3); }
.aa-lv.cur { border-color: var(--primary); }
.aa-lv h4 { font-size: .68rem !important; letter-spacing: .08em; color: var(--primary) !important; margin: 0 !important; font-weight: 600; }
.aa-lv .lt { font-size: .8rem; color: var(--text); line-height: 1.3; margin: 2px 0 var(--s2); min-height: 2.1em; }
.aa-node { font-size: .74rem; line-height: 1.25; padding: .28rem .4rem; border-radius: 6px; margin-bottom: .25rem;
           border: 1px solid var(--border); color: var(--muted); display: flex; gap: .35rem; }
.aa-node b { font-weight: 600; min-width: 1.4em; color: var(--faint); }
.aa-node.planned { border-style: dashed; opacity: .55; }
.aa-node.new { color: var(--text); }
.aa-node.learning { border-color: #6A5124; background: var(--accent-soft); color: #F2D59B; }
.aa-node.proficient { border-color: #2D4E80; background: var(--primary-soft); color: #BBD4FA; }
.aa-node.mastered { border-color: var(--primary); background: #1B3A66; color: #FFFFFF; }
.aa-node.mastered b, .aa-node.proficient b, .aa-node.learning b { color: inherit; }
.aa-proj { font-size: .68rem; color: var(--muted); border-top: 1px dashed var(--border); margin-top: var(--s2); padding-top: var(--s2); line-height: 1.5; }
@media (max-width: 640px) {
  .aa-map { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .aa-node { display: inline-flex; margin-right: .2rem; } .aa-node span { display: none; }
  .aa-proj { display: none; } .aa-lv .lt { min-height: 0; }
  .block-container { padding-top: 2.6rem; }
}
.aa-legend { font-size: .74rem; color: var(--muted); display: flex; gap: .9rem; flex-wrap: wrap; margin-bottom: var(--s3); }
.aa-legend span:before { content: ""; display: inline-block; width: .7em; height: .7em; border-radius: 2px;
                         margin-right: .3em; vertical-align: -1px; border: 1px solid var(--c); background: var(--b); }

/* project tracks & gates */
.aa-track { border-radius: var(--radius); padding: var(--s3) var(--s4); margin-bottom: var(--s3); border: 1px solid var(--border); background: var(--panel); }
.aa-track .aa-label { margin: 0 0 var(--s1); }
.aa-track.lab .aa-label { color: var(--primary); }
.aa-track.prod .aa-label { color: var(--accent); }
.aa-verdict { border-radius: var(--radius); padding: var(--s3) var(--s4); margin: var(--s2) 0 var(--s3); border: 1px solid; font-weight: 600; }
.aa-verdict.ok { border-color: #2B5A44; background: #102419; color: var(--pass); }
.aa-verdict.no { border-color: #6A5124; background: var(--accent-soft); color: var(--accent); }
.aa-gate { display: grid; grid-template-columns: 3.4rem 1fr; gap: .6rem; padding: .5rem 0; border-bottom: 1px solid #172338; font-size: .88rem; }
.aa-gate .st { font-size: .66rem; font-weight: 600; letter-spacing: .06em; text-align: center; border-radius: 99px; padding: .12rem 0; height: fit-content; margin-top: 2px; }
.aa-gate .st.pass { color: var(--pass); background: #102419; border: 1px solid #2B5A44; }
.aa-gate .st.fail { color: var(--fail); background: #2A1512; border: 1px solid #6A3229; }
.aa-gate .d { color: var(--muted); font-size: .8rem; }

/* code walkthrough */
.aa-walk { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 3fr); gap: .25rem .9rem; font-size: .88rem; }
@media (max-width: 640px) { .aa-walk { grid-template-columns: 1fr; } .aa-walk div { margin-bottom: .4rem; } }
.aa-walk code { font-family: 'IBM Plex Mono', monospace; font-size: .76rem; background: #0E1626; color: #BBD4FA;
                padding: .3rem .45rem; border-radius: 6px; display: block; white-space: pre-wrap; word-break: break-word; border: 1px solid var(--border); }
.aa-walk div { padding: .15rem 0 .5rem; border-bottom: 1px solid #172338; color: var(--text); }
div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: .6rem .9rem; }
div[data-testid="stMetricLabel"] p { font-size: .78rem !important; color: var(--muted) !important; }
div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
"""


def css_block() -> str:
    return f"<style>{CSS}</style>"
