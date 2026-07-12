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
    # --- weather context ---
    "temp_f",
    "precip_intensity",
    "wind_speed_mph",
    "weather_severity",         # 0–3
    "condition_code",           # bucket int, see WEATHER_CONDITIONS
    # --- event context ---
    "events_near_spot",
    "event_on_commute",         # 0/1
    "nearest_event_distance_mi",
    "event_max_attendance",
    # --- traffic context ---
    "live_congestion",          # 0.0–1.0, live flow-tile congestion around spot
    "traffic_delay_minutes",    # est. added minutes vs. free-flow
    # --- temporal ---
    "hour_of_day",
    "day_of_week",
    "semester_week",
    "is_finals_week",           # 0/1
]

# Which columns LightGBM should treat as categorical (not ordinal numbers).
CATEGORICAL_FEATURES = ["preferred_spot_type", "spot_type", "condition_code"]

# Stable integer encodings so training and serving agree.
SPOT_TYPES = {"ramp": 0, "surface": 1, "street": 2}
WEATHER_CONDITIONS = {"clear": 0, "clouds": 1, "rain": 2, "snow": 3,
                      "thunderstorm": 4, "extreme": 5}

# The column the synthetic generator writes and training reads as the label.
LABEL_COLUMN = "selected"          # 1 if user chose this spot, else 0
GROUP_COLUMN = "session_id"        # rows sharing a session = one ranking query