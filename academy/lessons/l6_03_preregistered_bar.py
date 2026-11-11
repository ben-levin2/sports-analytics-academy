import numpy as np
import pandas as pd
import streamlit as st

from ..models import Lesson, MathBlock, Question


@st.cache_data(show_spinner="Running 20 challengers on a simulated season…")
def _season():
    """Same simulation as the code below: 19 same-skill challengers + 1 truly better, cluster bootstrap draws."""
    from scipy.special import expit
    from ..smalldata import fast_auc, fit_logistic, predict_logistic
    from ..validation import ece
    rng = np.random.default_rng(34)

    def season():
        g = np.repeat(np.arange(12), 50)
        x = rng.normal(size=g.size)
        return g, x, (rng.random(g.size) < expit(-2.0 + 1.2 * x)).astype(int)
    g_tr, x_tr, y_tr = season()
    g_te, x_te, y_te = season()

    def model(sd, seed):
        r = np.random.default_rng(seed)
        s_tr, s_te = x_tr + r.normal(scale=sd, size=x_tr.size), x_te + r.normal(scale=sd, size=x_te.size)
        X = lambda s: np.column_stack([np.ones_like(s), s])
        return predict_logistic(fit_logistic(X(s_tr), y_tr), X(s_te))
    base = model(0.8, 1)
    cands = {f"Same-skill {k + 1}": model(0.8, 100 + k) for k in range(19)}
    cands["Truly better"] = model(0.3, 2)
    groups = [np.where(g_te == k)[0] for k in range(12)]
    idx = [np.concatenate([groups[i] for i in r]) for r in np.random.default_rng(0).integers(0, 12, (500, 12))]
    out = {}
    for k, p in cands.items():
        draws = np.array([fast_auc(p[i], y_te[i]) - fast_auc(base[i], y_te[i]) for i in idx])
        out[k] = {"gain": fast_auc(p, y_te) - fast_auc(base, y_te), "draws": draws, "ece": ece(p, y_te)}
    return out


def _passes(r, bar, k):
    alpha = 0.05 / k if bar["split"] else 0.05
    low = float(np.quantile(r["draws"], alpha))
    ok = r["gain"] >= bar["gain"] and (low > 0 or not bar["boot"]) and r["ece"] <= bar["ece"]
    return ok, low


def visual():
    locked = st.session_state.get("l63_bar")
    v = locked or {"k": 20, "gain": 0.02, "boot": True, "split": True, "ece": 0.05}
    with st.form("l63_form", border=True):
        st.markdown("**1 · Write your bar.** Results stay hidden until you lock it.")
        c1, c2 = st.columns(2)
        k = c1.slider("How many challengers will you test?", 1, 20, v["k"], disabled=bool(locked))
        gain = c2.select_slider("Minimum AUC gain", [0.0, 0.01, 0.02, 0.03, 0.05], value=v["gain"], disabled=bool(locked))
        c3, c4 = st.columns(2)
        boot = c3.checkbox("Bootstrap lower bound must stay above 0", v["boot"], disabled=bool(locked))
        split = c3.checkbox("Split that check across all challengers tested", v["split"], disabled=bool(locked))
        ece = c4.select_slider("Maximum calibration error", [0.03, 0.05, 0.08, 0.2], value=v["ece"], disabled=bool(locked))
        if st.form_submit_button("Lock my bar", disabled=bool(locked)):
            st.session_state["l63_bar"] = {"k": k, "gain": gain, "boot": boot, "split": split, "ece": ece}
            st.rerun()
    if not locked:
        return
    if st.button("Unlock and start over", key="l63_reset"):
        st.session_state.pop("l63_bar", None)
        st.rerun()
    res = _season()
    names = [n for n in res if n != "Truly better"][: locked["k"] - 1] + ["Truly better"]
    rows = []
    for n in names:
        ok, low = _passes(res[n], locked, locked["k"])
        rows.append({"Challenger": n, "AUC gain": f"{res[n]['gain']:+.3f}", "Bootstrap low": f"{low:+.3f}",
                     "Calibration error": f"{res[n]['ece']:.3f}", "Passes your bar": "PASS" if ok else "fail"})
    st.markdown("**2 · Results under the bar you locked.**")
    from ..ui import table
    table(pd.DataFrame(rows), height=min(38 * len(rows) + 40, 420))
    weak = [n for n in names if n != "Truly better"]
    fake = sum(_passes(res[n], locked, locked["k"])[0] for n in weak)
    st.markdown(f"Same-skill models that passed: **{fake} of {len(weak)}**. "
                f"The truly better model: **{'passed' if _passes(res['Truly better'], locked, locked['k'])[0] else 'failed'}**.")
    st.markdown("**3 · The mistake: moving the bar after looking.**")
    if weak:
        best = max(weak, key=lambda n: res[n]["gain"])
        moved = {"gain": min(locked["gain"], round(res[best]["gain"], 3)), "boot": False, "split": False,
                 "ece": max(locked["ece"], round(res[best]["ece"], 3) + 0.001)}
        n_moved = sum(_passes(res[n], moved, locked["k"])[0] for n in weak)
        st.markdown(f"The luckiest same-skill model is **{best}** (gain {res[best]['gain']:+.3f}). Lower the gain bar to "
                    f"{moved['gain']:+.3f} and drop the bootstrap check, and **{n_moved} of {len(weak)}** same-skill "
                    "models 'pass', though none adds anything over the baseline.")


CODE = '''import numpy as np
from scipy.special import expit
from academy.smalldata import fast_auc, fit_logistic, predict_logistic
from academy.validation import ece

rng = np.random.default_rng(34)
def season(n_groups=12, per=50):
    """SIMULATED: 12 groups (e.g. classes) of 50 players; x is true talent, y = 1 if the outcome happened."""
    g = np.repeat(np.arange(n_groups), per)
    x = rng.normal(size=g.size)
    y = (rng.random(g.size) < expit(-2.0 + 1.2 * x)).astype(int)
    return g, x, y

g_tr, x_tr, y_tr = season()          # past data, used only to turn scores into probabilities
g_te, x_te, y_te = season()          # the test

def model(noise_sd, seed):
    """A model = talent seen through noise. Calibrated into probabilities on PAST data only."""
    r = np.random.default_rng(seed)
    s_tr, s_te = x_tr + r.normal(scale=noise_sd, size=x_tr.size), x_te + r.normal(scale=noise_sd, size=x_te.size)
    X = lambda s: np.column_stack([np.ones_like(s), s])
    return predict_logistic(fit_logistic(X(s_tr), y_tr), X(s_te))

base = model(0.8, 1)
K = 20                                                     # challengers tested
cands = {f"same-skill {k}": model(0.8, 100 + k) for k in range(K - 1)}   # no better than the baseline
cands["truly better"] = model(0.3, 2)                      # sees talent more clearly

groups = [np.where(g_te == k)[0] for k in np.unique(g_te)]
boot_idx = [np.concatenate([groups[i] for i in r]) for r in np.random.default_rng(0).integers(0, 12, (500, 12))]
def judge(p):
    gain = fast_auc(p, y_te) - fast_auc(base, y_te)
    draws = [fast_auc(p[i], y_te[i]) - fast_auc(base[i], y_te[i]) for i in boot_idx]   # cluster bootstrap
    return gain, np.array(draws), ece(p, y_te)
res = {k: judge(p) for k, p in cands.items()}

BAR = {"min_gain": 0.02, "max_ece": 0.05}                  # written down BEFORE any result is seen
for label, alpha in [("Bar written first, split across 20 tests", 0.05 / K), ("Bar written first, no split", 0.05)]:
    ok = [k for k, (gain, d, cal) in res.items()
          if gain >= BAR["min_gain"] and np.quantile(d, alpha) > 0 and cal <= BAR["max_ece"]]
    print(f"{label}: passes = {ok}")

# The mistake: look first, then set the bar to fit the best-looking same-skill model
weak = {k: v for k, v in res.items() if k != "truly better"}
best = max(weak, key=lambda k: weak[k][0])
gain, d, cal = weak[best]
print(f"Luckiest same-skill model: {best}: gain {gain:+.3f}, bootstrap 5% low {np.quantile(d, 0.05):+.3f}")
print(f"Same-skill models with gain >= +0.02: {sum(v[0] >= 0.02 for v in weak.values())} of {len(weak)}")

# Why more tests need a stricter bar: with a gain-only bar, each same-skill model has some chance of a lucky pass
p1 = np.mean([v[0] >= 0.02 for v in weak.values()])
for k in (1, 5, 20):
    print(f"Testing {k:>2} same-skill models with a gain-only bar: chance at least one 'passes' = {1 - (1 - p1) ** k:.0%}")
'''

LESSON = Lesson(
    id="l6-03-preregistered-bar", level=6, order=3,
    title="Write the Bar Before You Look",
    subtitle="Pre-registered pass bars, and why testing more models needs a stricter bar",
    minutes=20,
    concept_ids=["preregistered-bar"],
    simple="""
Before you see any results, write down exactly what "passing" means:
- **Minimum gain:** e.g. at least +0.02 AUC over the strongest simple baseline (L3.11).
- **Uncertainty:** the gain's bootstrap range must stay above zero, not just the single number.
- **Calibration:** probabilities must be honest, e.g. calibration error at most 0.05 (L3.1).

Then look. If you set or adjust the bar *after* seeing results, you can make almost anything pass. You'll pick the number the model happened to hit.

And if you test **many** challengers, some will look good by luck. The fix is a **stricter** uncertainty check, split across everything you tested.
""",
    problem="A bar chosen after seeing results measures your choice, not the model. Testing many models and reporting the best one turns luck into a 'win'.",
    why="A written bar turns 'this looks promising' into a yes/no decision that can't be argued into a pass afterward. It's what makes a result believable to someone who didn't watch you work.",
    when_useful="""
- Any time a model might be used for a real decision
- Whenever you compare more than one challenger
- When the test set is small and noisy (one season, one program)
""",
    when_not="""
- Early exploration, where you're learning what to try. Just don't call those results evidence
- Don't make the bar so strict nothing could ever pass; size it for the data you'll have
""",
    compare="""
**Eyeballing results:** fast, unreliable. **Bar written after looking:** looks rigorous, isn't. **Bar written first, one model:** honest. **Bar written first, many models, split uncertainty check (Bonferroni):** honest when you try several things. **Frozen forward test (L3.12):** the strongest version, with the bar inside the hash.
""",
    sports="""
**A common mistake, simulated.** A fictional season: 12 groups of 50 players. Twenty challengers are compared with a baseline:
- **19 "same-skill" models:** they see talent exactly as clearly as the baseline, just with different noise. They add nothing.
- **1 "truly better" model:** it sees talent more clearly.

**Bar written first** (gain ≥ +0.02, bootstrap lower bound > 0, calibration ≤ 0.05): only the **truly better** model passes, whether or not the bootstrap check is split across the 20 tests.

**Bar moved after looking:** the luckiest same-skill model shows a gain of **+0.028**, but its bootstrap 5% low is **−0.026**. Drop the bootstrap check after seeing that, and it "passes".

**Why more tests need a stricter bar.** With a gain-only bar, **2 of 19** same-skill models clear +0.02 by luck (about 11% each). Test 5 such models and the chance at least one "passes" is about **43%**; test 20 and it's about **89%**.
""",
    takeaway="Write the bar first. Make it include uncertainty and calibration, and make it stricter the more models you test.",
    visual=visual,
    visual_caption="Set your bar and lock it before any results appear. Then see who passes, and how far the bar would have to move for same-skill models to 'pass'.",
    math=[
        MathBlock(
            "Step 1: the bar, written first",
            r"\text{pass} \iff \Delta \ge \delta_{\min} \;\;\text{and}\;\; L_{\alpha} > 0 \;\;\text{and}\;\; \text{ECE} \le e_{\max}",
            """
- $\\Delta$: the AUC gain over the strongest simple baseline
- $\\delta_{\\min}$: the minimum gain worth acting on (e.g. 0.02)
- $L_{\\alpha}$: the $\\alpha$ quantile of the bootstrap gains (resampling whole groups), a lower bound on the gain
- $\\text{ECE}$: calibration error; $e_{\\max}$ its limit
""",
        ),
        MathBlock(
            "Step 2: many tests, more lucky passes",
            r"P(\text{at least one false pass among } k) = 1 - (1 - q)^k",
            """
- $q$: the chance one no-better model passes by luck
- $k$: how many challengers you test
- With $q = 0.11$: $k = 5$ gives 0.43, $k = 20$ gives 0.89
""",
        ),
        MathBlock(
            "Step 3: split the check (Bonferroni)",
            r"\alpha_{\text{each}} = \frac{\alpha}{k}",
            """
- $\\alpha$: the overall false-pass rate you accept (e.g. 0.05)
- Each challenger's bootstrap check uses $\\alpha/k$, a lower quantile, so a stricter bound
- Then the chance that *any* no-better model passes stays at about $\\alpha$ or less
""",
            "The P3C project tests 3 challengers, so each uses 0.05 / 3 ≈ 1.67%.",
        ),
    ],
    code=CODE,
    code_notes="Change the 19 same-skill models to noise 0.7 (slightly better than the baseline). Do any pass now, and does splitting the check matter?",
    walkthrough=[
        ("def season(n_groups=12, per=50):", "A fictional season: 12 groups (like classes) of 50 players, with a hidden talent x."),
        ("def model(noise_sd, seed):", "A 'model' sees talent through noise, then is calibrated into probabilities on past data only."),
        ("cands = {f'same-skill {k}': model(0.8, 100 + k) ...}", "19 challengers exactly as good as the baseline: any 'gain' they show is luck."),
        ("cands['truly better'] = model(0.3, 2)", "One challenger that really sees talent more clearly."),
        ("boot_idx = [...]", "Cluster bootstrap: resample whole groups, the same resamples for every model."),
        ("BAR = {'min_gain': 0.02, 'max_ece': 0.05}", "The bar, written before any result is computed."),
        ("np.quantile(d, alpha) > 0", "The uncertainty check. With alpha = 0.05/20 it's split across all 20 tests."),
        ("best = max(weak, key=lambda k: weak[k][0])", "The mistake: find the best-looking model first, then decide what the bar should be."),
        ("1 - (1 - p1) ** k", "The chance at least one of k no-better models clears a gain-only bar by luck."),
    ],
    psfal="""
**The academy is separate from PSFAL; this lesson uses simulated data only.**

The general habit: write the bar (minimum gain over the strongest simple baseline, an uncertainty check that stays above zero, a calibration limit) before scoring, and tighten the uncertainty check when you test several challengers. Every Production Candidate tab in the academy has its gates fixed in code before any data is scored; P3C splits its bootstrap check across its 3 challengers.
""",
    quiz=[
        Question(
            "q1", "mc", "Why must the pass bar be written before you see results?",
            "Otherwise you can choose a bar that the model happened to clear, so the bar measures your choice, not the model.",
            concept="preregistered-bar",
            options=["So it can't be fitted to the results", "So it's easier to pass", "Because metrics change over time", "To save computation"],
            notes=["Correct.", "It usually makes passing harder, honestly.", "The metrics don't change.", "Unrelated."],
            answer=0),
        Question(
            "q2", "numeric", "Each no-better model has a 10% chance of passing by luck. You test 10. Chance at least one passes? (2 decimals)",
            "1 − 0.9^10 ≈ 0.65.",
            concept="preregistered-bar", category="Application", answer=0.65, tolerance=0.01),
        Question(
            "q3", "numeric", "You accept a 5% overall false-pass rate and test 4 challengers. With a Bonferroni split, what level does each challenger's bootstrap check use? (as a decimal)",
            "0.05 / 4 = 0.0125.",
            concept="preregistered-bar", category="Application", answer=0.0125, tolerance=0.0005),
        Question(
            "q4", "mc", "A model's gain is +0.03 but its bootstrap lower bound is −0.02. Under a written bar requiring the bound above 0, what's the verdict?",
            "It fails: the gain could easily be zero or negative.",
            concept="preregistered-bar", category="Application",
            options=["Fail", "Pass, the gain is above 0.02", "Pass if you drop the bootstrap check", "Retest until it passes"],
            notes=["Correct.", "The single number isn't enough under this bar.", "That's moving the bar after looking.", "That's another way of moving the bar."],
            answer=0),
        Question(
            "q5", "short", "Name the three parts of a good pre-registered pass bar.",
            "A minimum gain over the strongest baseline, an uncertainty range that must stay above zero, and a calibration limit.",
            concept="preregistered-bar",
            keywords=[["gain", "improve", "minimum"], ["uncertain", "bootstrap", "interval", "range", "above zero"],
                      ["calibrat"]],
            model_answer="A minimum gain over the strongest simple baseline, an uncertainty check (e.g. the bootstrap lower bound) that must stay above zero, and a calibration limit, all written before seeing results."),
    ],
)
