import streamlit as st

from ..models import Lesson, MathBlock, Question


def visual():
    from .. import forward as F
    from ..ui import table
    seed = st.number_input("Simulated season (random seed)", 1, 999, 36, key="l312_seed")
    if st.button("1 · Predict the 2022 class and freeze", key="l312_freeze"):
        preds = F.simulate_forecast(seed=int(seed))
        st.session_state["l312_frozen"] = F.freeze(preds, F.make_plan(2022, written="spring 2023"))
        st.session_state["l312_seedused"] = int(seed)
    fz = st.session_state.get("l312_frozen")
    if not fz:
        st.caption("Nothing frozen yet. Outcomes stay hidden until you freeze.")
        return
    preds, plan = F.unpack(fz)
    st.code(f"SHA-256: {fz['hash']}\nPredictions: {len(preds)} recruits, mean forecast {preds['p'].mean():.3f}\n"
            f"Plan: {plan['rule']}", language="text")
    tamper = st.checkbox("Quietly change one prediction after freezing", key="l312_tamper")
    when = st.select_slider("2 · Score after the draft of", [2025, 2026, 2027], value=2026, key="l312_when")
    use = dict(fz, payload=fz["payload"].replace('",0.', '",0.9', 1)) if tamper else fz
    res = F.score(use, F.simulate_outcomes(seed=st.session_state["l312_seedused"], observed_through=when), when)
    if not res["ok"]:
        st.error(res["reason"])
        return
    c = st.columns(3)
    c[0].metric("Window complete?", "Yes" if res["complete"] else "No", f"ends {plan['window_end']}", delta_color="off")
    c[1].metric("AUC (scored)", f"{res['auc']:.3f}")
    if "ece" in res:
        c[2].metric("Calibration error (scored)", f"{res['ece']:.3f}")
    else:
        c[2].metric("Calibration", "not scored", f"would read {res['ece_if_misused']:.3f} if misused", delta_color="off")
    st.caption(f"Drafted so far: {res['observed_rate']:.3f}. Forecast of 'ever drafted in 5 years': {res['mean_forecast']:.3f}.")


CODE = '''from academy.forward import make_plan, freeze, verify, score, simulate_forecast, simulate_outcomes

# 1. Spring 2023: predict the 2022 class (SIMULATED, 120 recruits) from classes finished by then
preds = simulate_forecast(per_class=120, class_year=2022, as_of=2023)
plan = make_plan(class_year=2022, written="2023-05-01")            # the scoring plan, written NOW
frozen = freeze(preds, plan)
print("Frozen", len(preds), "predictions. SHA-256:", frozen["hash"][:16], "...")
print("Mean forecast of 'drafted within 5 years':", round(preds["p"].mean(), 3))

# 2. Someone 'fixes' one prediction after the fact. The hash catches it.
edited = dict(frozen, payload=frozen["payload"].replace(f'{preds["p"].iloc[0]}', "0.99", 1))
print("Original verifies:", verify(frozen), "| edited copy verifies:", verify(edited))

# 3. Score in 2026: only draft years 3 and 4 have happened (the window ends in 2027)
early = score(frozen, simulate_outcomes(class_year=2022, observed_through=2026), observed_through=2026)
print(f"2026: window complete? {early['complete']}. Scored: {early['metrics']}. AUC {early['auc']:.3f}")
print(f"      drafted so far {early['observed_rate']:.3f} vs forecast {early['mean_forecast']:.3f}"
      f" -> calibration error if misused: {early['ece_if_misused']:.3f} (not a real score)")

# 4. Score in 2027, after the last draft in the window: now calibration is fair game
final = score(frozen, simulate_outcomes(class_year=2022, observed_through=2027), observed_through=2027)
print(f"2027: window complete? {final['complete']}. Scored: {final['metrics']}. "
      f"AUC {final['auc']:.3f}, calibration error {final['ece']:.3f}, drafted {final['observed_rate']:.3f}")
'''

LESSON = Lesson(
    id="l3-12-forward-tests", level=3, order=12,
    title="Forward Tests: Freeze Predictions Before the Outcomes Exist",
    subtitle="Out-of-time testing, hashing, a scoring plan written in advance, and the window-mismatch trap",
    minutes=20,
    concept_ids=["forward-tests"],
    simple="""
A **backtest** (L3.2) hides past seasons and pretends not to know them. It's useful, but you always *could* have peeked, tuned or re-run until it looked good.

A **forward test** removes that doubt. You make predictions **before the outcomes exist**, lock them, and score them later. Three pieces make it trustworthy:
1. **Freeze** the predictions and the **scoring plan** together.
2. **Hash** them: a SHA-256 fingerprint. Change one digit and the fingerprint changes completely.
3. **Score later**, exactly as the plan says, after checking the fingerprint still matches.

One trap to plan for: if the model predicts "**ever** drafted (within 5 years)" but you score after only 2 of those years, the outcomes are incomplete. Ranking (AUC) is still fair; calibration is not.
""",
    problem="Backtests can be quietly re-run and re-tuned until they pass. And scoring a long-horizon forecast against a short window makes a good model look badly calibrated.",
    why="A frozen, hashed forecast with a plan written in advance is the most convincing evidence a model can give: nobody, including you, could have adjusted it after seeing the answers.",
    when_useful="""
- The final check before trusting any model for real use
- Any model whose outcomes arrive later (next season, next draft)
- When different people need to trust the result without trusting each other
""",
    when_not="""
- It is slow: you wait for outcomes. Use backtests to develop, a forward test to confirm
- A single forward season is a small sample; plan the bar for that (L6.3)
- A hash proves nothing was changed, not that the plan was sensible. Write the plan carefully first
""",
    compare="""
**Random split:** leaks the future (L3.2). **Rolling-origin backtest:** honest about time, but repeatable and tunable. **Frozen forward test:** predictions locked before outcomes exist; can't be re-run. **Scoring plan:** the metric, the bar and the timing, written before the outcomes; it travels inside the hash.
""",
    sports="""
**A common mistake, simulated.** Spring 2023: a model trained on finished classes predicts which of 120 fictional 2022 recruits will be drafted within 5 years (by 2027). The predictions and the plan are frozen together; mean forecast **0.125**.

- **Tampering:** change one prediction after freezing and the hash no longer matches. The score is refused.
- **Scoring in 2026** (two of the three drafts in the window): AUC **0.924**. Only 0.067 have been drafted so far, against a forecast of 0.125, so calibration *would* read 0.058 and fail a 0.05 bar. That's the trap: the forecast isn't wrong, the window is incomplete. The plan says AUC only, so calibration isn't scored.
- **Scoring in 2027** (window complete): AUC **0.947**, calibration error **0.034**, drafted **0.100**. Now calibration is fair, and it's fine.
""",
    takeaway="Freeze predictions and the scoring plan together, fingerprint them, and score only what the outcome window can fairly judge.",
    visual=visual,
    visual_caption="Freeze first, then choose when to score. Before 2027 only ranking is scored. Tick 'quietly change one prediction' to see the hash reject the edit.",
    math=[
        MathBlock(
            "Step 1: the fingerprint",
            r"h = \text{SHA256}\big(\text{canonical}(\text{predictions},\ \text{plan})\big)",
            """
- $\\text{canonical}(\\cdot)$: one exact text: rows sorted by player, probabilities rounded to 6 decimals, plan included
- $\\text{SHA256}$: a standard fingerprint (64 hex characters). Any change to the text gives a completely different $h$
- You can publish or save $h$ before outcomes arrive; later, anyone can recompute it from the frozen text
""",
        ),
        MathBlock(
            "Step 2: what the forecast means",
            r"\hat p_i = P(\text{drafted by } c + 5), \qquad \bar y(t) = \frac{1}{n}\sum_i \mathbb{1}[\text{drafted by } t]",
            """
- $\\hat p_i$: the frozen forecast for recruit $i$ from class $c$: drafted at any point in the 5-year window
- $\\bar y(t)$: the share drafted by the scoring time $t$
- If $t < c + 5$, $\\bar y(t)$ is still growing, so comparing it with $\\hat p_i$ makes the forecast look too high
""",
        ),
        MathBlock(
            "Step 3: the scoring plan",
            r"\text{score}(t) = \begin{cases} \text{AUC only} & t < c + 5 \\ \text{AUC and calibration} & t \ge c + 5 \end{cases}",
            """
- Ranking still works early: players who will be drafted later are mostly the ones ranked high, and those drafted early are too
- Calibration needs the full window: same question, same time span as the forecast
""",
            "Written before outcomes exist, and inside the hash, so the timing rule itself can't be changed later.",
        ),
    ],
    code=CODE,
    code_notes="Score in 2025 too (only one draft in the window). How far does the 'misused' calibration error drift?",
    walkthrough=[
        ("preds = simulate_forecast(per_class=120, class_year=2022, as_of=2023)", "Spring 2023: fit on classes finished by then and predict the 2022 class. Outcomes don't exist yet."),
        ("plan = make_plan(class_year=2022, written='2023-05-01')", "The scoring plan: target, window end, and which metrics count if scored early."),
        ("frozen = freeze(preds, plan)", "Turn predictions + plan into one canonical text and take its SHA-256 fingerprint."),
        ("verify(edited)", "Change one prediction and the recomputed fingerprint no longer matches the saved one."),
        ("score(frozen, simulate_outcomes(..., observed_through=2026), observed_through=2026)", "Score in 2026: the window isn't complete, so the plan allows AUC only."),
        ("early['ece_if_misused']", "Printed only to show the trap: what calibration would look like if you ignored the plan."),
        ("score(..., observed_through=2027)", "After the last draft in the window: AUC and calibration are both fair."),
    ],
    psfal="""
**The academy is separate from PSFAL; this lesson uses simulated data only.**

The general habit: when a model's outcomes arrive later, freeze its predictions together with the scoring plan, keep the hash somewhere you can't edit, and score only what the outcome window can fairly judge. The P3C project's Production Candidate view has a Forward test tab that does this with simulated data.
""",
    quiz=[
        Question(
            "q1", "mc", "What does the hash prove in a forward test?",
            "That the frozen predictions and plan were not changed after freezing.",
            concept="forward-tests",
            options=["Nothing in the frozen predictions or plan changed", "The model is accurate", "The plan is sensible", "The outcomes are correct"],
            notes=["Correct.", "It says nothing about accuracy.", "A bad plan hashes just as well.", "Outcomes aren't in the hash."],
            answer=0),
        Question(
            "q2", "mc", "A model predicts 'drafted within 5 years'. You score it after 2 of the window's 3 drafts. Which metric is fair?",
            "Ranking (AUC) is still fair; calibration compares a 5-year forecast with partial outcomes and would look too high.",
            concept="forward-tests", category="Application",
            options=["AUC only", "Calibration only", "Both", "Neither"],
            notes=["Correct.", "That's exactly the mismatch trap.", "Calibration must wait for the full window.", "Ranking is still informative."],
            answer=0),
        Question(
            "q3", "mc", "Why must the scoring plan be inside the hash?",
            "So the metric, bar and timing can't be changed after outcomes are seen.",
            concept="forward-tests",
            options=["So the plan can't be changed after seeing outcomes", "To make the file smaller", "Because SHA-256 requires it", "So predictions are more accurate"],
            notes=["Correct.", "Unrelated.", "SHA-256 hashes any text.", "Hashing doesn't change predictions."],
            answer=0),
        Question(
            "q4", "numeric", "Forecast mean 0.12 for 'drafted within 5 years'. After 2 of 3 drafts, 0.07 have been drafted. What's the gap between forecast and observed? (2 decimals)",
            "0.12 − 0.07 = 0.05, and most of it is the window not being finished, not a bad forecast.",
            concept="forward-tests", category="Application", answer=0.05, tolerance=0.001),
        Question(
            "q5", "short", "How is a frozen forward test more convincing than a rolling-origin backtest?",
            "Predictions are locked before outcomes exist, so they can't have been re-run or tuned after seeing results.",
            concept="forward-tests",
            keywords=[["before", "lock", "frozen", "freeze", "advance"],
                      ["outcome", "result", "tune", "re-run", "rerun", "change", "peek"]],
            model_answer="Because the predictions are locked (and hashed) before the outcomes exist, nobody could re-run or tune the model after seeing the results. A backtest can always be repeated until it looks good."),
    ],
)
