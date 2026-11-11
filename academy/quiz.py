"""Quiz grading. Pure functions: grade(question, response) -> (fraction, feedback)."""
from __future__ import annotations

from .models import Question


def _parse_number(text):
    if text is None:
        return None
    s = str(text).strip().replace(",", "")
    pct = s.endswith("%")
    s = s.rstrip("%").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return v / 100 if pct else v


def grade(q: Question, response):
    """Return (fraction in [0,1], markdown feedback).

    mc:      response is the selected option text (or None)
    numeric: response is a string; "66%" and "0.66" both work
    short:   response is free text, graded by keyword groups (all groups hit = full credit)
    """
    if q.kind == "mc":
        if response is None or response not in q.options:
            return 0.0, f"**No answer selected.** Correct answer: **{q.options[q.answer]}**. {q.explanation}"
        idx = q.options.index(response)
        if idx == q.answer:
            return 1.0, f"**Correct.** {q.notes[idx]} {q.explanation}"
        return 0.0, (f"**Not quite.** {q.notes[idx]}  \n"
                     f"Correct answer: **{q.options[q.answer]}**. {q.notes[q.answer]} {q.explanation}")

    if q.kind == "numeric":
        v = _parse_number(response)
        if v is None:
            return 0.0, f"**Enter a number.** Expected about **{q.answer}**. {q.explanation}"
        if q.answer <= 1 and 1 < v <= 100:     # typed 66.1 instead of 0.661
            v /= 100
        if abs(v - q.answer) <= q.tolerance:
            return 1.0, f"**Correct.** {q.explanation}"
        return 0.0, f"**Not quite.** You answered {v:g}; the answer is about **{q.answer:g}**. {q.explanation}"

    if q.kind == "short":
        text = (response or "").lower()
        if not text.strip():
            return 0.0, f"**No answer written.** Model answer: {q.model_answer}"
        hits = sum(any(w in text for w in group) for group in q.keywords)
        frac = hits / len(q.keywords) if q.keywords else 0.0
        verdict = ("**Covered the key ideas.**" if frac == 1
                   else "**Partly there.**" if frac > 0 else "**Missed the key ideas.**")
        return frac, (f"{verdict} ({hits}/{len(q.keywords)} key ideas detected by keyword matching, "
                      f"so compare with the model answer yourself.)  \nModel answer: {q.model_answer}")

    raise ValueError(f"Unknown question kind: {q.kind}")


def score_quiz(questions, responses: dict):
    """responses: {question_id: response}. Returns (score 0..1, points, per_q dict, feedback dict)."""
    per_q, feedback = {}, {}
    for q in questions:
        frac, fb = grade(q, responses.get(q.id))
        per_q[q.id] = round(frac, 3)
        feedback[q.id] = fb
    points = sum(per_q.values())
    return (points / len(questions) if questions else 0.0), points, per_q, feedback
