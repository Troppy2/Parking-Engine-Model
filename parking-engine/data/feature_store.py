# Feature store — the anti-skew keystone.
# Assembles a flat feature vector from user preferences, spot attributes, and
# live traffic context for each (user, spot, context) triplet fed to the ranker.
#
# ONE function, called by BOTH worlds: training (looped over many rows) and
# serving (once, one user x N spots). Same code path means train/serve skew is
# structurally impossible rather than merely unlikely.
#
# Must end in `return df[FEATURE_COLUMNS]` — that reindex enforces column order
# and raises loudly on a missing/renamed feature instead of silently shifting
# the matrix. LightGBM sees a matrix, not names.
import pandas as pd
from datetime import datetime, timedelta

from schema import FEATURE_COLUMNS, SPOT_TYPES


# Campus calendar rules. Belongs in a config module alongside API keys /
# Redis URL — pinned here only because we don't have config yet.
#
#   Fall semester:    always begins on the 1st Tuesday of September.
#   Spring semester:  always begins the Monday after MLK Day (3rd Monday of Jan).
#
# We pick whichever semester window contains `now`. Outside both windows
# (summer break) we anchor to the most recent fall start so semester_week
# still returns a sensible 1..16 instead of going negative.
# A semester is ~17 weeks per the synthetic generator's range
# (sem_week drawn from [1, 17] and is_finals = sem_week >= 15).
SEMESTER_WEEKS = 17


def _semester_start(today: datetime) -> datetime:
    """Return the start date of the semester that contains `today`.

    Pick the candidate window (fall / spring of current year, fall of
    previous year) whose [start, start + 16 weeks) covers today. Outside
    any window — i.e. summer break — fall back to the previous fall so
    week counting still returns a positive integer.
    """
    fall_this_year = _first_tuesday_of_september(today.year)
    spring_this_year = _week_after_mlk(today.year)
    fall_last_year = _first_tuesday_of_september(today.year - 1)

    candidates = [fall_this_year, spring_this_year, fall_last_year]
    window = timedelta(days=SEMESTER_WEEKS * 7 - 1)
    for start in candidates:
        if start <= today <= start + window:
            return start
    # Summer break - none of the windows cover today - Anchors to last fall.
    return fall_last_year


def _first_tuesday_of_september(year: int) -> datetime:
    """Sep 1 of `year`, rolled forward to the first Tuesday on/after it."""
    d = datetime(year, 9, 1)
    return datetime(year, 9, 1 + ((1 - d.weekday()) % 7))


def _week_after_mlk(year: int) -> datetime:
    """3rd Monday of January + 7 days = the Monday of the following week."""
    jan1 = datetime(year, 1, 1)
    first_monday = jan1 + timedelta(days=(0 - jan1.weekday()) % 7)
    third_monday = first_monday + timedelta(weeks=2)
    return third_monday + timedelta(days=7)


def _temporal_features(context: dict) -> dict:
    """Return the four temporal features from a historical or serving context."""
    return {
        "hour_of_day": context["hour_of_day"],
        "day_of_week": context["day_of_week"],
        "semester_week": context["semester_week"],
        "is_finals_week": int(context["is_finals_week"]),
    }


def build_feature_matrix(user: dict, spots: list[dict],
                         traffic_by_spot: dict, context: dict) -> pd.DataFrame:
    """Assemble one row per spot, columns in FEATURE_COLUMNS order.

    Called identically by training and serving — that shared path IS the
    anti-skew guarantee.

    context supplies the temporal features. Serving passes values derived from
    datetime.now(); training passes the historical values off the parquet row.
    This function must never read the clock itself.
    """
    rows = []
    for spot in spots:
        traffic = traffic_by_spot[spot["id"]]

        preferred_spot_type = user["preferred_spot_type"]
        if isinstance(preferred_spot_type, str):
            preferred_spot_type = SPOT_TYPES[preferred_spot_type]

        spot_type = spot["spot_type"]
        if isinstance(spot_type, str):
            spot_type = SPOT_TYPES[spot_type]

        row = {
            "cost_sensitivity": user["cost_sensitivity"],
            "max_walk_minutes": user["max_walk_minutes"],
            "preferred_spot_type": preferred_spot_type,
            "base_cost": spot["base_cost"],
            "walk_time_minutes": spot["walk_time_minutes"],
            "spot_type": spot_type,
            "is_verified": int(spot["is_verified"]),
            "historical_congestion": spot["historical_congestion"],
            "live_congestion": traffic["live_congestion"],
            "traffic_delay_minutes": traffic["traffic_delay_minutes"],
            **_temporal_features(context),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[FEATURE_COLUMNS]


def temporal_features_now() -> dict:
    """Serving-side helper: today's temporal features from the clock.

    Deliberately NOT called inside build_feature_matrix — see the module
    docstring. Training passes historical values instead.
    """
    now = datetime.now()
    semester_start = _semester_start(now)
    window_end = semester_start + timedelta(days=SEMESTER_WEEKS * 7 - 1)
    in_semester = semester_start <= now <= window_end
    raw_week = ((now - semester_start).days // 7) + 1
    
    semester_week = max(1, min(raw_week, SEMESTER_WEEKS)) if in_semester else 1
    return {
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
        "semester_week": semester_week,
        "is_finals_week": 1 if semester_week >= 15 else 0,
    }