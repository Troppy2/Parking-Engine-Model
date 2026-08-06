"""Single source of truth for the model's feature contract.

Every module that touches features imports FEATURE_COLUMNS from here.
Column ORDER matters — LightGBM sees a matrix, not names, at predict time.
"""

# The exact order the model expects. Never reorder without retraining.
FEATURE_COLUMNS = [
    # --- user ---
    "cost_sensitivity",
    "max_walk_minutes",
    "preferred_spot_type",      # int-encoded: 0=ramp, 1=surface, 2=street
    # --- spot ---
    "base_cost",
    "walk_time_minutes",
    "spot_type",                # same encoding as preferred_spot_type
    "is_verified",              # 0/1
    "historical_congestion",    # 0.0–1.0
    # --- traffic context (from TomTom) ---
    "live_congestion",          # 0.0–1.0, live flow-tile congestion around spot
    "traffic_delay_minutes",    # est. added minutes vs. free-flow
    # --- temporal ---
    "hour_of_day",
    "day_of_week",
    "semester_week",
    "is_finals_week",           # 0/1
]

# Which columns LightGBM should treat as categorical (not ordinal numbers).
CATEGORICAL_FEATURES = ["preferred_spot_type", "spot_type"]

# Stable integer encodings so training and serving agree.
SPOT_TYPES = {"ramp": 0, "surface": 1, "street": 2}
WEATHER_CONDITIONS = {"clear": 0, "clouds": 1, "rain": 2, "snow": 3,
                      "thunderstorm": 4, "extreme": 5}

# TomTom Traffic Incident Details "iconCategory" codes -> a stable bucket name.
# Same idea as WEATHER_CONDITIONS: the model/backend only ever sees the int on
# the left; the frontend can key off the same int to pick an icon/color without
# needing to know anything about TomTom's wire format.
TRAFFIC_INCIDENT_CATEGORIES = {
    "unknown": 0, "accident": 1, "fog": 2, "dangerous_conditions": 3,
    "rain": 4, "ice": 5, "jam": 6, "lane_closed": 7, "road_closed": 8,
    "road_works": 9, "wind": 10, "flooding": 11, "detour": 12,
    "cluster": 13, "broken_down_vehicle": 14,
}

# TomTom "magnitudeOfDelay" -> our 0-3 congestion severity + a suggested
# frontend color, same pairing as the `ty` table in TomTom's dashboard example.
TRAFFIC_MAGNITUDE = {
    0: {"label": "unknown",   "severity": 0, "color": "rgba(0,0,0,0.25)"},
    1: {"label": "minor",     "severity": 1, "color": "rgba(0,153,0,0.25)"},
    2: {"label": "moderate",  "severity": 2, "color": "rgba(255,102,0,0.25)"},
    3: {"label": "major",     "severity": 3, "color": "rgba(255,0,0,0.25)"},
    4: {"label": "undefined", "severity": 2, "color": "rgba(102,204,255,0.25)"},
}

# The column the synthetic generator writes and training reads as the label.
LABEL_COLUMN = "selected"          # 1 if user chose this spot, else 0
GROUP_COLUMN = "session_id"        # rows sharing a session = one ranking query