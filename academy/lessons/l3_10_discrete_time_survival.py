import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap

from .. import theme as T
from ..models import Lesson, MathBlock, Question
from ..ui import show_fig


@st.cache_data(show_spinner=False)
def _data():
    from ..data import make_draft_classes
    return make_draft_classes(per_class=60, data_end=2035)


def visual():
    from ..smalldata import person_period
    later = _data()
    as_of = st.slider("You are standing just after the draft of", 2014, 2026, 2022, key="l310_asof")
    classes = list(range(2008, as_of))
    grid = np.zeros((len(classes), 3), int)
    for i, c in enumerate(classes):
        for j, k in enumerate((3, 4, 5)):
            grid[i, j] = 1 if c + k <= as_of else 0           # 1 = this draft has happened
    fig, ax = plt.subplots(figsize=(4.2, 0.22 * len(classes) + 1.0))
    ax.imshow(grid, cmap=ListedColormap([T.PANEL, T.PRIMARY]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1, 2], ["Year 3", "Year 4", "Year 5"])
    ax.set_yticks(range(len(classes)), [str(c) for c in classes], fontsize=7)
    ax.grid(False), ax.set_title("Blue = draft already happened", fontsize=9)
    now = later.copy()
    now.loc[now["draft_year"] > as_of, ["drafted", "draft_year"]] = [0, np.nan]
    upto = now[now["class_year"] < as_of]
    done = now[now["class_year"] + 5 <= as_of]
    pp = person_period(upto, as_of)
    truth = later[later["class_year"] < as_of]["drafted"].mean()
    c1, c2 = st.columns([1, 1.3])
    with c1:
        show_fig(fig)
    with c2:
        st.metric("True share drafted within 5 years", f"{truth:.3f}")
        st.metric("Finished classes only", f"{done['drafted'].mean():.3f}" if len(done) else "no data",
                  f"{len(done)} recruits", delta_color="off")
        st.metric("Unfinished counted as 'never'", f"{upto['drafted'].mean():.3f}", f"{len(upto)} recruits", delta_color="off")
        if len(pp):
            h = pp.groupby("period")["event"].mean().reindex([3, 4, 5]).fillna(0)
            st.metric("Discrete-time survival", f"{1 - np.prod(1 - h):.3f}", f"{pp['player_id'].nunique()} recruits",
                      delta_color="off")


CODE = '''import numpy as np
from academy.data import make_draft_classes
from academy.smalldata import Design, person_period, fit_survival

now = make_draft_classes(per_class=60, data_end=2022)      # SIMULATED: what you know after the 2022 draft
later = make_draft_classes(per_class=60, data_end=2035)    # same players, years later (for checking only)
truth = later[later["class_year"] <= 2021]["drafted"].mean()
print(f"True share drafted within 5 years (classes 2008-2021): {truth:.3f}")

# 1. Drop unfinished classes: only 2008-2017 have had all three draft chances by 2022
done = now[now["class_year"] + 5 <= 2022]
print(f"1. Finished classes only:   {done['drafted'].mean():.3f}  from {len(done)} recruits")

# 2. Treat 'not drafted yet' as 'never drafted' (the classic mistake)
upto = now[now["class_year"] <= 2021]
print(f"2. Unfinished = never:      {upto['drafted'].mean():.3f}  from {len(upto)} recruits (biased low)")

# 3. Person-period rows: one row per recruit per draft he was eligible for, up to 2022
pp = person_period(upto, as_of=2022)
hazard = pp.groupby("period")["event"].mean()                  # share drafted in that year, among those still waiting
p_within_5 = 1 - np.prod(1 - hazard)
print(f"3. Discrete-time survival:  {p_within_5:.3f}  from {pp['player_id'].nunique()} recruits, {len(pp)} person-period rows")
print("   Hazard by draft year (3, 4, 5 years after enrolling):", hazard.round(3).tolist())
print("   Recruits contributing to each year:", pp.groupby("period").size().tolist())

# 4. Add inputs: the same idea as a logistic model on person-period rows
d = Design(pp.drop_duplicates("player_id"))
m = fit_survival(d, pp)
print(f"   Rating effect on the yearly draft hazard: odds x{np.exp(m['beta'][3]):.2f} per +1 SD")
'''

LESSON = Lesson(
    id="l3-10-discrete-time-survival", level=3, order=10,
    title="Discrete-Time Survival: Unfinished Outcomes Are Still Information",
    subtitle="Person-period rows, censoring, and yearly hazards",
    minutes=25,
    concept_ids=["discrete-time-survival"],
    simple="""
Some outcomes take years to finish. A recruit who enrolled in 2019 had his first draft chance in 2022, but two more are still to come. If you ask "was he drafted?" today, "no" might just mean "not yet."

You have three choices:
1. **Drop** every class that hasn't finished. Honest, but you throw away your most recent (and most relevant) data.
2. **Pretend** "not yet" means "never." That makes the draft look rarer than it is.
3. **Use what you know:** "he made it through the 2022 draft without being picked." That's real information.

**Discrete-time survival** does option 3. It splits each player into one row per year he was at risk (a **person-period** row), asks "was he drafted *this* year, given he wasn't before?", and stops his rows when his story is cut off (**censoring**).
""",
    problem="Recent classes are always unfinished. Dropping them wastes data; counting them as failures biases every estimate downward.",
    why="It uses every year you actually observed, nothing more, nothing less. And it's just logistic regression on a reshaped table, so everything from L1.7 and L1.8 (penalties, priors) carries over.",
    when_useful="""
- Outcomes that happen in yearly (or weekly) steps: draft years, first starts, transfers, contribution seasons
- Recent cohorts that haven't finished yet
- Small data, where losing your newest classes really hurts
""",
    when_not="""
- When time is truly continuous and exact (minutes, days): Kaplan-Meier and Cox (L3.3–L3.4) are the natural tools
- When the window always closes before you need the prediction: plain logistic on finished classes is simpler
- When players leave for reasons tied to the outcome (e.g. benched players transfer) and you treat that as plain censoring: estimates can be biased
""",
    compare="""
**Logistic on finished classes:** simple and unbiased, but throws away recent classes. **Logistic with 'not yet' = 'no':** biased low. **Kaplan-Meier (L3.3):** survival curve without inputs, same censoring idea. **Cox (L3.4):** continuous time, a shared baseline hazard, effects as hazard ratios. **Discrete-time survival:** yearly steps, just a logistic regression on person-period rows, so priors and penalties are easy to add.
""",
    sports="""
**Fictional scenario (simulated).** A program signs 60 recruits a class. It's spring 2022, just after the draft. Players can be drafted 3, 4 or 5 years after enrolling.

What share of recruits are drafted within 5 years? The truth (from the simulated future): **0.105**.

| Method | Estimate | Recruits used |
|---|---|---|
| Finished classes only (2008–2017) | 0.098 | 600 |
| Unfinished counted as "never" | 0.085 | 840 |
| **Discrete-time survival** | **0.104** | 720 (1,888 person-period rows) |

- Counting "not yet" as "never" is **too low**: it misses players still to be drafted.
- Survival uses 120 more recruits than the finished-only approach (2018–2019 classes, partially observed) and lands closest to the truth.
- The yearly hazards are 0.058, 0.034, 0.015: most picks come in year 3, fewer in years 4 and 5.

Adding inputs works exactly like logistic regression: here one SD of rating multiplies the yearly draft odds by about **2.8**.
""",
    takeaway="'Not yet' is not 'no'. Person-period rows let unfinished players count for exactly the years you saw.",
    visual=visual,
    visual_caption="Each row is a class; blue cells are drafts that have already happened. Move the year: recent classes have only some blue cells. Finished-only methods ignore those rows; survival uses every blue cell.",
    math=[
        MathBlock(
            "Step 1: the yearly hazard",
            r"h_{ik} = P(\text{drafted in year } k \mid \text{not drafted before year } k), \qquad k \in \{3, 4, 5\}",
            """
- $h_{ik}$: the **hazard**: recruit $i$'s chance of being drafted in year $k$ after enrolling, *if he's still available*
- $k$: years since enrolling (3 = first draft he's eligible for)
- The condition matters: a player drafted in year 3 is no longer "at risk" in year 4
""",
        ),
        MathBlock(
            "Step 2: person-period rows and censoring",
            r"\text{rows for recruit } i:\; k = 3, \dots, \min(K_i, C_i), \qquad e_{ik} = \mathbb{1}[\text{drafted in year } k]",
            """
- $K_i$: the year he was drafted (if he was)
- $C_i$: the last year we've seen (the most recent draft for his class). If $C_i$ comes first, he is **censored**: his rows simply stop
- $e_{ik}$: 1 in the year he was drafted, 0 in every other row
- $\\mathbb{1}[\\cdot]$: the indicator, 1 if true, 0 if not
""",
            "A 2019 recruit seen through the 2022 draft gets one row (year 3) with e = 0. He tells us 'not drafted in year 3', and nothing more.",
        ),
        MathBlock(
            "Step 3: a logistic model for the hazard, and back to 'within 5 years'",
            r"\log\frac{h_{ik}}{1-h_{ik}} = \alpha_k + x_i^\top\beta, \qquad P(\text{drafted within 5 years}) = 1 - \prod_{k=3}^{5} (1 - h_{ik})",
            """
- $\\alpha_k$: a separate baseline for each draft year (year 3 is busier than year 5)
- $x_i^\\top\\beta$: inputs like rating and year-1 percentile, with the same priors as L1.8
- $1 - h_{ik}$: chance he gets through year $k$ undrafted; multiply across years to get "never drafted", subtract from 1
""",
        ),
    ],
    code=CODE,
    code_notes="Change data_end=2022 to 2019 in the first line. More classes are unfinished, the 'never' method gets worse, and the gap between methods grows.",
    walkthrough=[
        ("now = make_draft_classes(per_class=60, data_end=2022)", "What you'd know just after the 2022 draft. Later draft picks are not in this table yet."),
        ("later = make_draft_classes(per_class=60, data_end=2035)", "The same simulated players years later, used only to check the answers."),
        ("done = now[now['class_year'] + 5 <= 2022]", "Method 1: keep only classes whose last draft chance (year 5) has happened."),
        ("upto['drafted'].mean()", "Method 2: count every 'not drafted yet' as 'never'. Biased low."),
        ("pp = person_period(upto, as_of=2022)", "Method 3: one row per recruit per draft he was eligible for and that has happened, stopping after he's drafted."),
        ("pp.groupby('period')['event'].mean()", "Yearly hazards: of the players still available in that year, the share who were drafted."),
        ("1 - np.prod(1 - hazard)", "Chance of being drafted within 5 years: one minus the chance of surviving all three drafts undrafted."),
        ("fit_survival(d, pp)", "The same idea with inputs: a Bayesian logistic regression on person-period rows, one baseline per draft year."),
    ],
    psfal="""
**The academy is separate from PSFAL and never changes it.**

- **Every recent class in your roster data is unfinished.** Any PSFAL outcome that takes several seasons (contribution, draft, transfer) has this problem; discrete-time survival is the logistic-friendly way to keep those classes.
- **Model 8** forecasts development from early data. If its target ever needs several seasons to resolve, person-period rows would let the newest classes inform training instead of being dropped.
- **Project P3C** tests whether that extra information actually improves draft predictions on later classes, against a plain logistic baseline, before anything is believed.
""",
    quiz=[
        Question(
            "q1", "mc", "A 2019 recruit hasn't been drafted as of the 2022 draft. What does a person-period table record for him?",
            "One row for year 3 with 'not drafted'. Years 4 and 5 haven't happened, so he's censored after year 3.",
            concept="discrete-time-survival", category="Application",
            options=["One row (year 3), not drafted", "Three rows, all not drafted", "Nothing: he's unfinished", "One row marked 'never drafted'"],
            notes=["Correct.", "Years 4 and 5 haven't happened.", "That's the drop-unfinished approach.", "That's the biased 'not yet = never' approach."],
            answer=0),
        Question(
            "q2", "mc", "Why does counting 'not drafted yet' as 'never drafted' bias the draft rate?",
            "Some of those players will be drafted later, so the rate comes out too low.",
            concept="discrete-time-survival",
            options=["It counts future picks as failures, so it's too low", "It double-counts drafted players",
                     "It makes the rate too high", "It has no effect with enough data"],
            notes=["Correct: 0.085 vs a true 0.105.", "Drafted players appear once.", "The bias is downward.", "More unfinished data makes it worse, not better."],
            answer=0),
        Question(
            "q3", "numeric", "Yearly hazards are 0.10, 0.05 and 0.02 for years 3, 4 and 5. What is P(drafted within 5 years)? (3 decimals)",
            "1 − (0.90 × 0.95 × 0.98) = 1 − 0.8379 = 0.162.",
            concept="discrete-time-survival", category="Application", answer=0.162, tolerance=0.002),
        Question(
            "q4", "mc", "What does 'censored' mean here?",
            "The player's story was cut off by the end of the data before the outcome could happen.",
            concept="discrete-time-survival",
            options=["We stopped observing him before the outcome could finish", "He was drafted", "He was removed for bad data", "He transferred to another position"],
            notes=["Correct.", "Drafted is the event, not censoring.", "Censoring isn't a data error.", "Unrelated."],
            answer=0),
        Question(
            "q5", "short", "How is a discrete-time survival model related to logistic regression?",
            "It is a logistic regression fitted on person-period rows, with a separate baseline for each period.",
            concept="discrete-time-survival",
            keywords=[["logistic", "logit"], ["person", "period", "row", "year", "reshap"]],
            model_answer="It's just a logistic regression on person-period rows (one row per player per year at risk), with a separate intercept for each year, so priors and penalties carry over directly."),
    ],
)
