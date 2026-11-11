"""Dataset registry: what each project needs, and where the data comes from.

Priority for every dataset:  1) a CSV you uploaded on the Data Sources page
                             2) a file path in datasets.toml (or env var ACADEMY_DATASET_<ID>)
                             3) SIMULATED data from academy/data.py (always labelled)

Swapping simulated data for real data never requires code changes: provide a CSV with the
required columns (the upload page lets you map your column names onto them).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from . import config, data, progress

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None


@dataclass
class Spec:
    id: str
    name: str
    description: str
    required: dict                  # column -> meaning
    optional: dict = field(default_factory=dict)
    simulated: dict = field(default_factory=dict)   # variant name -> generator
    used_by: list = field(default_factory=list)
    check: Callable | None = None   # extra validation: df -> list[str]
    numeric: list = field(default_factory=list)
    features: bool = False          # True = user picks extra numeric feature columns on upload


def _check_rates(df):
    errs = []
    if (df["opportunities"] <= 0).any():
        errs.append("Some rows have opportunities <= 0. Drop them or fix them.")
    if (df["successes"] < 0).any() or (df["successes"] > df["opportunities"]).any():
        errs.append("successes must be between 0 and opportunities on every row.")
    if df.duplicated(["player_id", "season"]).any():
        errs.append("Duplicate player_id + season rows. Aggregate to one row per player-season first.")
    return errs


def _check_hometowns(df):
    errs = []
    if not df["lat"].between(-90, 90).all() or not df["lon"].between(-180, 180).all():
        errs.append("lat must be between -90 and 90, and lon between -180 and 180.")
    return errs


def _check_roster(df):
    errs = []
    if (df["seasons_observed"] < 1).any():
        errs.append("seasons_observed must be 1 or more.")
    if not df["contributed"].isin([0, 1]).all():
        errs.append("contributed must be 0 or 1.")
    return errs


def _check_profiles(df):
    from .archetypes import feature_candidates
    errs = []
    if len(feature_candidates(df)) < 3:
        errs.append("Needs at least three numeric feature columns besides id/season/position/snaps/production.")
    if df.duplicated(["player_id", "season"]).any():
        errs.append("Duplicate player_id + season rows. Use one row per player-season.")
    return errs


SPECS = {s.id: s for s in [
    Spec("rate_events", "Player rate stats by season",
         "One row per player-season with a count of successes out of opportunities "
         "(completions/attempts, tackles/team tackles, pressures/pass-rush snaps).",
         {"player_id": "unique player key", "season": "season year (integer)",
          "successes": "count of successes", "opportunities": "count of chances"},
         {"team": "team key", "position": "position label (enables position-specific priors)"},
         {"QB completion %": data.make_qb_completions, "LB tackle share": data.make_lb_tackle_share,
          "Pressure rate by position": data.make_pressure_by_position},
         ["P1", "P5"], _check_rates, ["season", "successes", "opportunities"]),
    Spec("player_profiles", "Player statistical profiles",
         "One row per player-season with three or more numeric per-snap or per-game stats. "
         "Add production and next_production to unlock the Production Candidate test.",
         {"player_id": "unique player key", "season": "season year (integer)", "position": "listed position label"},
         {"position_group": "e.g. DL / LB / DB (lets you cluster within a group)", "snaps": "playing volume (for a minimum-snaps filter)",
          "production": "this season's production measure", "next_production": "the same player's production the following season",
          "<feature columns>": "three or more numeric stats, chosen when you upload"},
         {"Defensive profiles (~3,600 player-seasons)": data.make_defensive_profiles,
          "Edge defenders (150, 3 hidden types)": data.make_edge_defenders},
         ["P2"], _check_profiles, ["season"], features=True),
    Spec("roster_careers", "Roster careers (time to contribution)",
         "One row per recruit: how many seasons we observed him, and whether he reached the contribution threshold.",
         {"player_id": "unique player key", "team": "team key", "class_year": "recruiting class",
          "seasons_observed": "seasons until contribution, departure or end of data",
          "contributed": "1 = reached threshold in the last observed season, 0 = censored"},
         {"recruit_rating": "composite rating", "position": "position",
          "left_before_contributing": "1 = left the roster before contributing (alternative outcome)"},
         {"~67 teams x 10 classes": data.make_roster_careers}, ["P3A"], _check_roster,
         numeric=["class_year", "seasons_observed", "contributed"]),
    Spec("hometowns", "Recruit hometowns",
         "One row per player with hometown coordinates and a production outcome.",
         {"player_id": "unique player key", "lat": "hometown latitude", "lon": "hometown longitude",
          "production": "college production measure"},
         {"local_density": "population-density proxy (needed to separate talent from population)",
          "class_year": "recruiting class (needed for time-based validation)", "recruit_rating": "composite rating"},
         {"Hometowns (simulated metros)": data.make_hometown_production}, ["P3B"], _check_hometowns,
         numeric=["lat", "lon", "production"]),
    Spec("game_logs", "Game-by-game performance",
         "One row per player-game, in order.",
         {"player_id": "unique player key", "week": "game order within the season", "grade": "game performance measure"},
         {"season": "season year"},
         {"Weekly grades (60 players x 12 weeks)": data.make_weekly_performance}, ["P4"],
         numeric=["week", "grade"]),
    Spec("lineup_candidates", "Lineup candidates with costs",
         "One row per candidate player with a cost and an estimated value.",
         {"player_id": "unique player key", "position": "position group", "cap_hit_m": "cost in $M", "value": "estimated value"},
         {"value_sd": "uncertainty of the value estimate"},
         {"Defensive-line candidates": data.make_dl_free_agents}, ["P6A"],
         numeric=["cap_hit_m", "value"]),
]}


# ---------------------------------------------------------------- storage of uploads
def _upload_dir():
    return config.DATA_DIR / "uploads"


def _paths(spec_id):
    d = _upload_dir()
    return d / f"{spec_id}.csv", d / f"{spec_id}.meta.json"


def configured_path(spec_id: str):
    env = os.environ.get(f"ACADEMY_DATASET_{spec_id.upper()}")
    if env:
        return env
    f = config.ROOT / "datasets.toml"
    if tomllib and f.exists():
        try:
            return tomllib.loads(f.read_text()).get("paths", {}).get(spec_id)
        except Exception:
            return None
    return None


def validate(spec_id: str, df: pd.DataFrame) -> list:
    spec = SPECS[spec_id]
    errs = [f"Missing required column: {c}" for c in spec.required if c not in df.columns]
    if errs:
        return errs
    for c in spec.numeric:
        if not pd.api.types.is_numeric_dtype(df[c]):
            errs.append(f"Column {c} must be numeric.")
    if df.empty:
        errs.append("The file has no rows.")
    if not errs and spec.check:
        errs += spec.check(df)
    return errs


def save_upload(spec_id: str, df: pd.DataFrame, label: str, real: bool) -> list:
    errs = validate(spec_id, df)
    if errs:
        return errs
    csv, meta = _paths(spec_id)
    try:
        csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False)
        meta.write_text(json.dumps({"label": label, "real": bool(real), "rows": len(df), "ts": progress.iso()}))
    except OSError as exc:
        return [f"Could not save: {exc}"]
    return []


def remove_upload(spec_id: str) -> None:
    for p in _paths(spec_id):
        p.unlink(missing_ok=True)


def load(spec_id: str, variant: str | None = None):
    """Return (df, info). info: source ('upload'|'path'|'simulated'), label, real (bool), error."""
    spec = SPECS[spec_id]
    csv, meta = _paths(spec_id)
    if csv.exists():
        try:
            df = pd.read_csv(csv)
            m = json.loads(meta.read_text()) if meta.exists() else {}
            if not validate(spec_id, df):
                return df, {"source": "upload", "label": m.get("label", "uploaded CSV"),
                            "real": bool(m.get("real")), "error": None}
        except Exception:
            pass
    path = configured_path(spec_id)
    if path:
        try:
            df = pd.read_csv(path)
            errs = validate(spec_id, df)
            if not errs:
                return df, {"source": "path", "label": path, "real": True, "error": None}
            err = f"{path}: " + "; ".join(errs)
        except Exception as exc:
            err = f"{path}: {exc}"
    else:
        err = None
    name = variant if variant in spec.simulated else next(iter(spec.simulated))
    df = spec.simulated[name]()
    return df, {"source": "simulated", "label": name, "real": False, "error": err}


def status(spec_id: str) -> dict:
    csv, meta = _paths(spec_id)
    if csv.exists():
        m = json.loads(meta.read_text()) if meta.exists() else {}
        return {"source": "upload", **m}
    if configured_path(spec_id):
        return {"source": "path", "label": configured_path(spec_id)}
    return {"source": "simulated"}
