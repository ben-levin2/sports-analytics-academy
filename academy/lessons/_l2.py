"""Shared data for Level 2 lesson visuals (leading underscore: not a lesson)."""
from functools import lru_cache

from .. import data

FEATURES = data.PROFILE_FEATURES


@lru_cache(maxsize=4)
def group_profiles(group: str = "DL", min_snaps: int = 150):
    df = data.make_defensive_profiles()
    d = df[(df["position_group"] == group) & (df["snaps"] >= min_snaps)].reset_index(drop=True)
    z = (d[FEATURES] - d[FEATURES].mean()) / d[FEATURES].std()
    return d, z.to_numpy()
