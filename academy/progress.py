"""Progress tracking and persistence. Pure Python (no Streamlit) so it is easy to test.

Design: we store only raw facts (quiz attempts, opened timestamps, checked steps).
"Completed" and "mastered" are always *derived* from quiz scores, so they can never
drift out of sync, and merely opening a lesson can never mark it mastered.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timedelta, timezone

from . import config


# ---------------------------------------------------------------- time helpers
def _tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(config.TIMEZONE)
    except Exception:  # tzdata missing -> fall back to UTC
        return timezone.utc


def now() -> datetime:
    return datetime.now(_tz())


def today() -> date:
    return now().date()


def iso() -> str:
    return now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- state shape
def default_state() -> dict:
    return {
        "schema": config.SCHEMA_VERSION,
        "created": iso(),
        "lessons": {},       # lesson_id -> {"opened", "attempts": [...], "best"}
        "labs": {},          # exercise_id -> {"done": bool, "ts": iso}
        "projects": {},      # project_id -> {"steps": [idx...], "notes": str}
        "activity": [],      # [{"ts","kind","text"}] newest last, capped
        "active_days": [],   # ISO dates with meaningful activity (drives streak)
        "meta": {"last_export": None, "last_saved": None, "changes_since_export": 0},
    }


def normalize(raw) -> dict:
    """Validate/repair an arbitrary dict into a well-formed state. Raises ValueError."""
    if not isinstance(raw, dict) or "lessons" not in raw:
        raise ValueError("This does not look like an Academy progress file.")
    st = default_state()
    st["created"] = str(raw.get("created", st["created"]))
    for key in ("lessons", "labs", "projects"):
        v = raw.get(key, {})
        st[key] = v if isinstance(v, dict) else {}
    act = raw.get("activity", [])
    st["activity"] = [a for a in act if isinstance(a, dict) and "ts" in a and "text" in a][-200:]
    days = raw.get("active_days", [])
    st["active_days"] = sorted({d for d in days if _valid_date(d)})
    meta = raw.get("meta", {})
    if isinstance(meta, dict):
        for k in st["meta"]:
            if k in meta:
                st["meta"][k] = meta[k]
    for lid in list(st["lessons"]):
        rec = st["lessons"][lid]
        if not isinstance(rec, dict):
            del st["lessons"][lid]
            continue
        attempts = []
        for a in rec.get("attempts", []):
            try:
                attempts.append({
                    "ts": str(a["ts"]), "score": float(a["score"]),
                    "points": float(a.get("points", 0)), "total": int(a.get("total", 0)),
                    "per_q": a.get("per_q", {}) if isinstance(a.get("per_q", {}), dict) else {},
                })
            except (KeyError, TypeError, ValueError):
                continue
        st["lessons"][lid] = {
            "opened": rec.get("opened"),
            "attempts": attempts,
            "best": max((a["score"] for a in attempts), default=0.0),
        }
    return st


def _valid_date(d) -> bool:
    try:
        date.fromisoformat(str(d))
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------- disk I/O
def load():
    """Return (state, info). Never raises."""
    path = config.PROGRESS_FILE
    if not path.exists():
        return default_state(), {"source": "new", "message": "No saved progress file found; starting fresh."}
    try:
        state = normalize(json.loads(path.read_text(encoding="utf-8")))
        return state, {"source": "file", "message": "Progress loaded from disk."}
    except Exception as exc:  # corrupt file: keep it aside, never crash the app
        try:
            path.replace(path.with_name("progress.corrupt.json"))
        except OSError:
            pass
        return default_state(), {
            "source": "error",
            "message": f"Saved progress could not be read ({exc}). Starting fresh; the old file was kept as progress.corrupt.json.",
        }


def save(state: dict) -> bool:
    """Atomic write (temp file + replace). Returns False if the disk is read-only."""
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        state["meta"]["last_saved"] = iso()
        fd, tmp = tempfile.mkstemp(dir=config.DATA_DIR, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, config.PROGRESS_FILE)
        return True
    except OSError:
        return False


def _commit(state: dict, counts: bool = True) -> None:
    if counts:
        state["meta"]["changes_since_export"] = int(state["meta"].get("changes_since_export") or 0) + 1
    save(state)


# ---------------------------------------------------------------- activity
def log(state: dict, kind: str, text: str) -> None:
    state["activity"].append({"ts": iso(), "kind": kind, "text": text})
    del state["activity"][:-200]


def touch_day(state: dict) -> None:
    d = today().isoformat()
    if d not in state["active_days"]:
        state["active_days"].append(d)


def streak(state: dict) -> int:
    """Consecutive active days ending today (or yesterday, so the streak isn't lost mid-day)."""
    days = {date.fromisoformat(d) for d in state["active_days"] if _valid_date(d)}
    cur = today()
    if cur not in days:
        cur -= timedelta(days=1)
    n = 0
    while cur in days:
        n += 1
        cur -= timedelta(days=1)
    return n


# ---------------------------------------------------------------- lessons
def lesson_rec(state: dict, lid: str) -> dict:
    return state["lessons"].setdefault(lid, {"opened": None, "attempts": [], "best": 0.0})


def mark_opened(state: dict, lid: str, title: str) -> None:
    rec = lesson_rec(state, lid)
    if not rec["opened"]:
        rec["opened"] = iso()
        log(state, "open", f"Opened {title}")
        _commit(state, counts=False)


def record_attempt(state: dict, lid: str, title: str, score: float, points: float, total: int, per_q: dict) -> None:
    rec = lesson_rec(state, lid)
    rec["attempts"].append({"ts": iso(), "score": round(score, 4), "points": round(points, 2),
                            "total": total, "per_q": per_q})
    rec["best"] = max(a["score"] for a in rec["attempts"])
    touch_day(state)
    log(state, "quiz", f"Quiz on {title}: {score:.0%}")
    _commit(state)


def best_score(state: dict, lid: str) -> float:
    return float(state["lessons"].get(lid, {}).get("best", 0.0))


def lesson_status(state: dict, lid: str) -> str:
    """new | opened | review | completed | mastered (derived from quiz scores only)."""
    rec = state["lessons"].get(lid)
    if not rec:
        return "new"
    if rec["best"] >= config.MASTERY_THRESHOLD:
        return "mastered"
    if rec["best"] >= config.COMPLETE_THRESHOLD:
        return "completed"
    if rec["attempts"]:
        return "review"
    return "opened"


def is_completed(state: dict, lid: str) -> bool:
    return lesson_status(state, lid) in ("completed", "mastered")


def is_mastered(state: dict, lid: str) -> bool:
    return lesson_status(state, lid) == "mastered"


# ---------------------------------------------------------------- labs
def set_lab_done(state: dict, ex_id: str, title: str, done: bool) -> None:
    state["labs"][ex_id] = {"done": bool(done), "ts": iso()}
    if done:
        touch_day(state)
        log(state, "lab", f"Completed lab: {title}")
    _commit(state)


def lab_done(state: dict, ex_id: str) -> bool:
    return bool(state["labs"].get(ex_id, {}).get("done"))


# ---------------------------------------------------------------- projects
def project_rec(state: dict, pid: str) -> dict:
    rec = state["projects"].setdefault(pid, {"steps": [], "notes": ""})
    rec.setdefault("steps", [])
    rec.setdefault("notes", "")
    return rec


def set_project_step(state: dict, pid: str, title: str, idx, done: bool, label: str = "") -> None:
    """idx is an int (old format) or a string key like "lab-2" / "prod-0"."""
    rec = project_rec(state, pid)
    steps = set(rec["steps"])
    (steps.add if done else steps.discard)(idx)
    rec["steps"] = sorted(steps, key=str)
    if done:
        touch_day(state)
        log(state, "project", f"{title}: {label or f'step {idx}'}")
    _commit(state)


def log_project_run(state: dict, pid: str, title: str, verdict: str) -> None:
    """Record a Production Candidate evaluation run (the verdict, not the data)."""
    rec = project_rec(state, pid)
    rec.setdefault("runs", []).append({"ts": iso(), "verdict": verdict})
    rec["runs"] = rec["runs"][-20:]
    touch_day(state)
    log(state, "project", f"{title}: evaluation run ({verdict})")
    _commit(state)


def set_registration(state: dict, pid: str, reg: dict) -> None:
    """Save a study registration (question, metric, baseline, primary study) with a timestamp."""
    rec = project_rec(state, pid)
    rec["registration"] = {**reg, "ts": iso()}
    log(state, "project", f"Registered study: {reg.get('question', '')[:60]}")
    _commit(state)


def set_project_notes(state: dict, pid: str, notes: str) -> None:
    project_rec(state, pid)["notes"] = notes
    _commit(state)


# ---------------------------------------------------------------- export / import
def to_json(state: dict) -> str:
    return json.dumps(state, indent=2)


def from_json(text: str) -> dict:
    try:
        return normalize(json.loads(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Not valid JSON: {exc}") from exc


def mark_exported(state: dict) -> None:
    state["meta"]["last_export"] = iso()
    state["meta"]["changes_since_export"] = 0
    save(state)


def merge(a: dict, b: dict) -> dict:
    """Union of two states: keeps every attempt, every checked step, every active day."""
    out = normalize(a)
    other = normalize(b)
    for lid, rec in other["lessons"].items():
        mine = out["lessons"].setdefault(lid, {"opened": rec["opened"], "attempts": [], "best": 0.0})
        seen = {x["ts"] for x in mine["attempts"]}
        mine["attempts"] += [x for x in rec["attempts"] if x["ts"] not in seen]
        mine["attempts"].sort(key=lambda x: x["ts"])
        mine["best"] = max((x["score"] for x in mine["attempts"]), default=0.0)
        opened = [o for o in (mine["opened"], rec["opened"]) if o]
        mine["opened"] = min(opened) if opened else None
    for eid, rec in other["labs"].items():
        if rec.get("done") or eid not in out["labs"]:
            out["labs"][eid] = rec if rec.get("done") else out["labs"].get(eid, rec)
    for pid, rec in other["projects"].items():
        mine = out["projects"].setdefault(pid, {"steps": [], "notes": ""})
        mine["steps"] = sorted(set(mine.get("steps", [])) | set(rec.get("steps", [])), key=str)
        runs = {(r["ts"], r["verdict"]) for r in mine.get("runs", [])}
        mine["runs"] = sorted(mine.get("runs", []) + [r for r in rec.get("runs", [])
                                                      if (r["ts"], r["verdict"]) not in runs],
                              key=lambda r: r["ts"])[-20:]
        if not mine["notes"]:
            mine["notes"] = rec.get("notes", "")
        if "registration" in rec and "registration" not in mine:
            mine["registration"] = rec["registration"]
    seen = {(x["ts"], x["text"]) for x in out["activity"]}
    out["activity"] += [x for x in other["activity"] if (x["ts"], x["text"]) not in seen]
    out["activity"] = sorted(out["activity"], key=lambda x: x["ts"])[-200:]
    out["active_days"] = sorted(set(out["active_days"]) | set(other["active_days"]))
    return out
