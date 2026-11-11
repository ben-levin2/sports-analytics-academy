"""SIMULATED college-football data generators.

Every function returns a pandas DataFrame of FICTIONAL players with `df.attrs["simulated"] = True`.
Some frames include a hidden "truth" column (e.g. `true_rate`) that you can never see in real
data. They exist so you can check whether a method recovers the truth. Real data never gives you that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIM_NOTE = "SIMULATED: fictional players generated with NumPy. Not real football data."


def _tag(df: pd.DataFrame) -> pd.DataFrame:
    df.attrs["simulated"] = True
    return df


def make_linebackers(n_players: int = 120, seasons: int = 3, seed: int = 7) -> pd.DataFrame:
    """Linebacker production by season. Columns: player_id, season, recruit_rating, games,
    tackles, tfl, sacks. Better recruits tend to produce more, and everyone improves with experience."""
    rng = np.random.default_rng(seed)
    rating = np.clip(rng.normal(0.87, 0.04, n_players), 0.75, 0.99)
    talent = 0.4 * (rating - rating.mean()) / rating.std() + 0.9 * rng.normal(size=n_players)
    rows = []
    for i in range(n_players):
        for s in range(1, seasons + 1):
            rate = max(0.5, 3.0 + 0.9 * (s - 1) + 0.8 * talent[i])      # tackles per game
            games = int(rng.integers(8, 13))
            rows.append((f"LB{i + 1:03d}", s, round(float(rating[i]), 3), games,
                         int(rng.poisson(rate * games)),
                         int(rng.poisson(0.12 * rate * games)),
                         int(rng.poisson(0.05 * rate * games))))
    return _tag(pd.DataFrame(rows, columns=["player_id", "season", "recruit_rating", "games",
                                            "tackles", "tfl", "sacks"]))


def make_quarterbacks(n: int = 60, seed: int = 11) -> pd.DataFrame:
    """Season passing totals. `true_rate` is the hidden real ability (only exists in simulation)."""
    rng = np.random.default_rng(seed)
    true_rate = np.clip(rng.normal(0.62, 0.04, n), 0.50, 0.75)
    attempts = rng.integers(20, 450, n)
    completions = rng.binomial(attempts, true_rate)
    return _tag(pd.DataFrame({"qb_id": [f"QB{i + 1:02d}" for i in range(n)], "attempts": attempts,
                              "completions": completions, "true_rate": true_rate.round(4)}))


def make_development(n_players: int = 80, seasons: int = 5, seed: int = 5) -> pd.DataFrame:
    """Production index (0-100) by season with player-specific growth that flattens at a peak."""
    rng = np.random.default_rng(seed)
    rating = np.clip(rng.normal(0.87, 0.04, n_players), 0.75, 0.99)
    rows = []
    for i in range(n_players):
        base = 45 + 120 * (rating[i] - 0.87) + rng.normal(0, 6)
        growth = rng.normal(7, 2.5)
        peak = rng.uniform(3.5, 6.0)
        for s in range(1, seasons + 1):
            t = s - 1
            prod = base + growth * t - (growth / (2 * (peak - 1))) * t ** 2 + rng.normal(0, 3)
            rows.append((f"P{i + 1:03d}", round(float(rating[i]), 3), s, round(float(prod), 2)))
    return _tag(pd.DataFrame(rows, columns=["player_id", "recruit_rating", "season", "production"]))


def make_retention(n: int = 600, seed: int = 3) -> pd.DataFrame:
    """One row per player: did they transfer out after year 1? (`transferred` = 1)."""
    rng = np.random.default_rng(seed)
    rating = np.clip(rng.normal(0.86, 0.05, n), 0.70, 0.99)
    snaps_share = rng.beta(2, 3, n)
    in_state = rng.binomial(1, 0.45, n)
    coach_change = rng.binomial(1, 0.25, n)
    logit = -1.3 + 3.0 * (0.35 - snaps_share) + 4.0 * (rating - 0.86) - 0.6 * in_state + 0.7 * coach_change
    transferred = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return _tag(pd.DataFrame({
        "player_id": [f"R{i + 1:04d}" for i in range(n)], "recruit_rating": rating.round(3),
        "snaps_share": snaps_share.round(3), "in_state": in_state, "coach_change": coach_change,
        "transferred": transferred}))


def make_edge_defenders(n: int = 150, seed: int = 21) -> pd.DataFrame:
    """Edge defenders from three hidden archetypes. `true_archetype` is the hidden truth."""
    rng = np.random.default_rng(seed)
    protos = {  # pressure_rate, run_stop_rate, tackles_per_game, coverage_snap_pct, weight_lbs
        "pass_rusher": (0.14, 0.05, 2.5, 0.02, 255),
        "run_stuffer": (0.06, 0.11, 4.2, 0.02, 280),
        "hybrid": (0.09, 0.07, 3.5, 0.15, 245),
    }
    sd = np.array([0.02, 0.015, 0.5, 0.02, 8])
    names = list(protos)
    kinds = rng.choice(names, n)
    X = np.array([rng.normal(protos[k], sd) for k in kinds])
    X = np.clip(X, 0, None)
    df = pd.DataFrame(X, columns=["pressure_rate", "run_stop_rate", "tackles_per_game",
                                  "coverage_snap_pct", "weight_lbs"]).round(3)
    df.insert(0, "player_id", [f"E{i + 1:03d}" for i in range(n)])
    df["true_archetype"] = kinds
    # Conventional label, imperfectly related to the hidden archetype (like real rosters)
    label_map = {"pass_rusher": ["EDGE"] * 8 + ["DT"] + ["LB"], "run_stuffer": ["DT"] * 8 + ["EDGE"] * 2,
                 "hybrid": ["LB"] * 6 + ["EDGE"] * 4}
    df["listed_position"] = [rng.choice(label_map[k]) for k in kinds]
    df["season"] = 2025
    df["position"] = df["listed_position"]
    return _tag(df)


def make_weekly_performance(n_players: int = 60, weeks: int = 12, seed: int = 9) -> pd.DataFrame:
    """Weekly game grades (0-100) = player baseline + slowly drifting 'form' + game noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_players):
        baseline = rng.normal(65, 6)
        form = 0.0
        for w in range(1, weeks + 1):
            form = 0.8 * form + rng.normal(0, 4)
            rows.append((f"W{i + 1:03d}", w, round(float(baseline + form + rng.normal(0, 6)), 2)))
    return _tag(pd.DataFrame(rows, columns=["player_id", "week", "grade"]))


# =====================================================================================
# Rate-event datasets for the Level 1 project. Common schema (also what a real CSV needs):
#   player_id, season, team, position, successes, opportunities
# `true_rate` exists only in simulation and is never used by the evaluation code.
# =====================================================================================
def _rate_seasons(rng, n_players, teams, seasons, positions, pos_mean, spread, opp_sampler, drift=0.01):
    rows = []
    first = seasons[0]
    for i in range(n_players):
        pos = positions[i % len(positions)]
        mu = pos_mean[pos]
        # true ability on the logit scale, persistent across a player's career
        z = np.log(mu / (1 - mu)) + rng.normal(0, spread)
        start = int(rng.integers(first, seasons[-1] + 1))
        length = int(rng.integers(1, 5))
        team = f"T{int(rng.integers(1, teams + 1)):02d}"
        for s in range(start, min(start + length, seasons[-1] + 1)):
            z += rng.normal(0, drift * 4)
            p = 1 / (1 + np.exp(-z))
            n = int(opp_sampler(rng))
            rows.append((f"S{i + 1:04d}", s, team, pos, int(rng.binomial(n, p)), n, round(float(p), 4)))
    return pd.DataFrame(rows, columns=["player_id", "season", "team", "position",
                                       "successes", "opportunities", "true_rate"])


def make_qb_completions(seed: int = 101) -> pd.DataFrame:
    """QB completions (successes) on pass attempts (opportunities). ~67 teams, 2016-2025."""
    rng = np.random.default_rng(seed)
    df = _rate_seasons(rng, 700, 67, list(range(2016, 2026)), ["QB"], {"QB": 0.61}, 0.22,
                       lambda r: min(550, max(5, int(r.lognormal(4.6, 1.0)))) if r.random() < 0.85 else int(r.integers(1, 15)))
    df.attrs["labels"] = ("completions", "pass attempts", "completion rate")
    return _tag(df)


def make_lb_tackle_share(seed: int = 202) -> pd.DataFrame:
    """Linebacker tackles (successes) out of team tackles while he played (opportunities)."""
    rng = np.random.default_rng(seed)
    df = _rate_seasons(rng, 1200, 67, list(range(2016, 2026)), ["LB"], {"LB": 0.09}, 0.45,
                       lambda r: int(r.integers(1, 13)) * int(r.integers(45, 70)))
    df.attrs["labels"] = ("tackles", "team tackles while active", "tackle share")
    return _tag(df)


def make_pressure_by_position(seed: int = 303) -> pd.DataFrame:
    """Pressures (successes) on pass-rush snaps (opportunities) for EDGE, DT and LB.
    Positions have genuinely different averages, so 'shrink toward the position mean' matters."""
    rng = np.random.default_rng(seed)
    df = _rate_seasons(rng, 1500, 67, list(range(2016, 2026)), ["EDGE", "DT", "LB"],
                       {"EDGE": 0.11, "DT": 0.065, "LB": 0.05}, 0.35,
                       lambda r: min(600, max(3, int(r.lognormal(4.8, 0.9)))))
    df.attrs["labels"] = ("pressures", "pass-rush snaps", "pressure rate")
    return _tag(df)


def make_roster_careers(seed: int = 404, last_season: int = 2025) -> pd.DataFrame:
    """One row per recruit: ~67 teams x 10 classes. Time (in seasons) until first season with
    >= 300 snaps ('contribution'), or censoring when he left or the data window ended."""
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0
    for t in range(1, 68):
        for cls in range(2016, last_season + 1):
            for _ in range(int(rng.integers(14, 24))):
                pid += 1
                rating = float(np.clip(rng.normal(0.86, 0.04), 0.75, 0.999))
                pos = rng.choice(["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"])
                hazard = 1 / (1 + np.exp(-(-1.6 + 18 * (rating - 0.86))))      # per-season chance
                leave_h = 0.12 + 0.10 * (rating < 0.84)
                window = last_season - cls + 1
                event, time = 0, None
                for yr in range(1, min(window, 5) + 1):
                    if rng.random() < hazard * (1 + 0.25 * (yr - 1)):
                        event, time = 1, yr
                        break
                    if rng.random() < leave_h:
                        time = yr
                        break
                left = 0 if event else (1 if time is not None else 0)
                if time is None:
                    time = min(window, 5)
                rows.append((f"R{pid:05d}", f"T{t:02d}", int(cls), pos, round(rating, 3), int(time), int(event), int(left)))
    df = pd.DataFrame(rows, columns=["player_id", "team", "class_year", "position", "recruit_rating",
                                     "seasons_observed", "contributed", "left_before_contributing"])
    return _tag(df)


_METROS = [  # (lat, lon, weight): fictional-scale population centres, NOT real counts
    (40.7, -74.0, 9), (34.0, -118.2, 8), (41.9, -87.6, 6), (29.8, -95.4, 7), (33.7, -84.4, 6),
    (32.8, -96.8, 7), (25.8, -80.2, 6), (39.9, -75.2, 5), (33.5, -112.1, 4), (30.3, -97.7, 4),
    (38.6, -90.2, 3), (40.4, -80.0, 3), (39.1, -84.5, 3), (36.2, -86.8, 3), (30.0, -90.1, 4),
    (35.2, -80.8, 3), (39.7, -105.0, 2), (47.6, -122.3, 2), (40.0, -83.0, 4), (32.4, -86.3, 3),
]


def make_hometown_production(n: int = 3000, seed: int = 505) -> pd.DataFrame:
    """Recruit hometowns (lat/lon) and a college production index (0-100).
    Players cluster where people live; the TRUE spatial effect on production is small.
    `local_density` is a stand-in for population density (simulated, not census data)."""
    rng = np.random.default_rng(seed)
    w = np.array([m[2] for m in _METROS], float)
    idx = rng.choice(len(_METROS), n, p=w / w.sum())
    lat = np.array([_METROS[i][0] for i in idx]) + rng.normal(0, 1.2, n)
    lon = np.array([_METROS[i][1] for i in idx]) + rng.normal(0, 1.5, n)
    density = np.array([_METROS[i][2] for i in idx]) * np.exp(-((rng.normal(0, 1, n)) ** 2) / 2)
    spatial = 5.0 * np.sin(np.radians(lat - 25) * 8) + 3.0 * np.cos(np.radians(lon + 95) * 6)   # modest true effect
    rating = np.clip(rng.normal(0.86, 0.04, n), 0.75, 0.999)
    production = 50 + spatial + 80 * (rating - 0.86) + rng.normal(0, 10, n)
    df = pd.DataFrame({"player_id": [f"H{i + 1:05d}" for i in range(n)], "lat": lat.round(3),
                       "lon": lon.round(3), "local_density": density.round(2),
                       "class_year": rng.integers(2016, 2026, n), "recruit_rating": rating.round(3),
                       "production": production.round(1), "true_spatial_effect": spatial.round(2)})
    return _tag(df)


def make_dl_free_agents(n: int = 24, seed: int = 606) -> pd.DataFrame:
    """Defensive-line candidates with a cap hit ($M) and an estimated value (wins above replacement x 10)."""
    rng = np.random.default_rng(seed)
    pos = rng.choice(["EDGE", "DT"], n, p=[0.55, 0.45])
    value = np.round(np.clip(rng.gamma(2.2, 1.6, n), 0.3, None), 2)
    cap = np.round(np.clip(1.2 + 2.1 * value + rng.normal(0, 2.0, n), 0.9, 30), 2)
    sd = np.round(0.25 * value + rng.uniform(0.2, 1.0, n), 2)
    return _tag(pd.DataFrame({"player_id": [f"DL{i + 1:02d}" for i in range(n)], "position": pos,
                              "cap_hit_m": cap, "value": value, "value_sd": sd}))


# =====================================================================================
# Defensive player profiles for Level 2 (thousands of player-seasons, hidden archetypes)
# =====================================================================================
PROFILE_FEATURES = ["pressure_rate", "run_stop_rate", "tackle_rate", "coverage_share",
                    "pbu_rate", "missed_tackle_rate", "weight_lbs"]

_ARCHETYPES = {   # group: {archetype: (feature means in PROFILE_FEATURES order, listed-position mix, dev effect)}
    "DL": {"Edge rusher":          ((0.13, 0.05, 0.05, 0.03, 0.002, 0.13, 255), {"EDGE": 0.85, "DT": 0.15}, 3.0),
           "Interior run stopper": ((0.05, 0.10, 0.07, 0.00, 0.001, 0.10, 305), {"DT": 0.85, "EDGE": 0.15}, -2.0),
           "Interior penetrator":  ((0.09, 0.08, 0.06, 0.01, 0.002, 0.14, 285), {"DT": 0.75, "EDGE": 0.25}, 1.5)},
    "LB": {"Run thumper":          ((0.04, 0.09, 0.12, 0.30, 0.004, 0.11, 240), {"LB": 1.0}, -1.0),
           "Coverage LB":          ((0.03, 0.05, 0.09, 0.60, 0.012, 0.09, 225), {"LB": 0.8, "S": 0.2}, 2.0),
           "Blitzer":              ((0.10, 0.07, 0.10, 0.25, 0.004, 0.14, 232), {"LB": 0.75, "EDGE": 0.25}, 1.0)},
    "DB": {"Man corner":           ((0.00, 0.02, 0.05, 0.92, 0.020, 0.12, 188), {"CB": 0.95, "S": 0.05}, 1.0),
           "Ball-hawk safety":     ((0.01, 0.03, 0.06, 0.85, 0.016, 0.10, 200), {"S": 0.8, "CB": 0.2}, 2.5),
           "Box safety":           ((0.02, 0.06, 0.09, 0.55, 0.008, 0.13, 210), {"S": 0.9, "LB": 0.1}, -1.5)},
}
_SD = np.array([0.020, 0.015, 0.015, 0.06, 0.004, 0.025, 9.0])


def make_defensive_profiles(n_players: int = 1300, seed: int = 707) -> pd.DataFrame:
    """One row per player-season (2019-2025, ~4,000 rows). Per-snap rate features get noisier
    when snaps are low. `production` is a 0-100 index; `next_production` is the same player's
    production the following season (missing in his last season). `true_archetype` is hidden truth."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_players):
        group = rng.choice(["DL", "LB", "DB"], p=[0.33, 0.27, 0.40])
        arch = rng.choice(list(_ARCHETYPES[group]))
        mean, mix, dev = _ARCHETYPES[group][arch]
        pos = rng.choice(list(mix), p=list(mix.values()))
        talent = rng.normal(0, 1)
        start = int(rng.integers(2019, 2025))
        length = int(rng.integers(2, 5))
        player_mean = np.array(mean) + rng.normal(0, 0.5, len(mean)) * _SD    # this player's own style
        prod = 50 + 8 * talent + rng.normal(0, 4)
        for s in range(start, min(start + length, 2026)):
            snaps = int(np.clip(rng.lognormal(5.6, 0.6), 40, 900))
            noise = rng.normal(0, 1, len(mean)) * _SD * np.sqrt(250 / snaps) * np.r_[np.ones(6), 0.3]
            feats = np.clip(player_mean + noise, 0, None)
            feats[3] = min(feats[3], 1.0)                     # coverage share is a share
            rows.append([f"D{i + 1:04d}", s, group, pos, snaps, *feats.round(4), round(float(prod), 1), arch])
            prod = 0.65 * prod + 0.35 * 50 + dev + 3 * talent + rng.normal(0, 4.5)
    df = pd.DataFrame(rows, columns=["player_id", "season", "position_group", "position", "snaps",
                                     *PROFILE_FEATURES, "production", "true_archetype"])
    df = df.sort_values(["player_id", "season"])
    nxt = df.groupby("player_id")["production"].shift(-1)
    gap = df.groupby("player_id")["season"].shift(-1) - df["season"]
    df["next_production"] = nxt.where(gap == 1)
    cols = [c for c in df.columns if c != "true_archetype"] + ["true_archetype"]
    return _tag(df[cols].reset_index(drop=True))


# =====================================================================================
# Game logs for Level 4: hidden hot/cold form + partial season coverage
# =====================================================================================
def make_game_logs(n_players: int = 160, seasons=range(2019, 2026), weeks: int = 12,
                   coverage: float = 0.8, seed: int = 808) -> pd.DataFrame:
    """One row per player-game: a 0-100 game grade. Each player has his own level, plus a hidden
    HOT or COLD form that persists from game to game (a Markov chain, lesson L1.6). Only ~80% of
    games are present, mimicking logs that cover portions of seasons. `true_state` is hidden truth."""
    rng = np.random.default_rng(seed)
    P = np.array([[0.80, 0.20], [0.15, 0.85]])        # rows: from HOT, from COLD
    offset = np.array([6.0, -4.0])
    rows = []
    for i in range(n_players):
        level = rng.normal(62, 6)
        first = int(rng.integers(min(seasons), max(seasons) + 1))
        for s in range(first, min(first + int(rng.integers(2, 5)), max(seasons) + 1)):
            state = int(rng.random() < 0.6)            # start of season: 40% hot, 60% cold
            level_s = level + rng.normal(0, 2)
            for w in range(1, weeks + 1):
                if w > 1:
                    state = int(rng.random() < P[state, 1])
                if rng.random() < coverage:
                    g = level_s + offset[state] + rng.normal(0, 7)
                    rows.append((f"G{i + 1:03d}", s, w, round(float(np.clip(g, 0, 100)), 1),
                                 "hot" if state == 0 else "cold"))
    df = pd.DataFrame(rows, columns=["player_id", "season", "week", "grade", "true_state"])
    return _tag(df)


def make_pressure_hierarchical(n_players: int = 1500, seed: int = 909) -> pd.DataFrame:
    """Pressure rates (rate-events schema) with the structure a hierarchical model is built for:
    position averages, TEAM effects (scheme/coaching), growth with EXPERIENCE, and player talent.
    Effects are modest on purpose; most variation is still player talent and luck."""
    rng = np.random.default_rng(seed)
    pos_mean = {"EDGE": 0.11, "DT": 0.065, "LB": 0.05}
    team_eff = rng.normal(0, 0.15, 67)                       # logit scale
    rows = []
    for i in range(n_players):
        pos = ["EDGE", "DT", "LB"][i % 3]
        team = int(rng.integers(0, 67))
        talent = rng.normal(0, 0.30)
        start = int(rng.integers(2016, 2026))
        for exp, s in enumerate(range(start, min(start + int(rng.integers(1, 5)), 2026))):
            mu = pos_mean[pos]
            z = np.log(mu / (1 - mu)) + team_eff[team] + 0.12 * min(exp, 3) + talent + rng.normal(0, 0.08)
            p = 1 / (1 + np.exp(-z))
            n = min(600, max(3, int(rng.lognormal(4.8, 0.9))))
            rows.append((f"H{i + 1:04d}", s, f"T{team + 1:02d}", pos, int(rng.binomial(n, p)), n, round(float(p), 4)))
    df = pd.DataFrame(rows, columns=["player_id", "season", "team", "position", "successes", "opportunities", "true_rate"])
    df.attrs["labels"] = ("pressures", "pass-rush snaps", "pressure rate")
    return _tag(df)
