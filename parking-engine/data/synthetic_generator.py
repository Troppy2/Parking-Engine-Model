# Synthetic training data generator.
# Produces parking_history rows covering diverse scenarios (rush-hour
# congestion, finals week, clear early mornings, relaxed weekends, late-semester
# scarcity) with realistic noise, bootstrapping the ranker until real
# parking_history accumulates.
#
# Output: data/processed/training_data.parquet
#
# _true_utility() below computes the label, so it defines what the model can
# possibly learn. Every column in FEATURE_COLUMNS must appear in it — a feature
# it never reads is pure noise the model will overfit, and is ignored at serve
# time. See tests/test_generator.py::test_no_feature_is_pure_noise.

import numpy as np, pandas as pd
from pathlib import Path
from schema import FEATURE_COLUMNS, LABEL_COLUMN, GROUP_COLUMN, SPOT_TYPES
OUT = Path(__file__).resolve().parent / "processed" / "training_data.parquet"
# This helps us create noise in our data so it is not too clean where our model overfits
RNG = np.random.default_rng(42) 

def make_session(session_id: int) -> pd.DataFrame:
    # This creates a sample context from our context layers 
    ctx = _sample_context()
    user = _sample_user()
    # 5-9 candidate spots, ~8 on average
    spots = [_sample_spot() for _ in range(RNG.integers(5, 10))]

    utilities = np.array([_true_utility(user, s, ctx) for s in spots])
    # Noise must be scaled RELATIVE to how far apart the utilities in this
    # session actually are. A fixed 0.15 is nothing next to a spread of ~30,
    # so argmax became deterministic and the model just memorized the rule.
    spread = utilities.std() or 1.0
    utilities = utilities + RNG.normal(0, 0.35 * spread, len(utilities))
    chosen = int(np.argmax(utilities))
    rows = [{**user, **s, **ctx,
             LABEL_COLUMN: int(i == chosen),
             GROUP_COLUMN: session_id}
            for i, s in enumerate(spots)]
    # Reindex against the contract instead of trusting that the three sample
    # dicts happen to be typed in FEATURE_COLUMNS order. This raises loudly on a
    # missing/renamed feature rather than silently shifting the matrix — the
    # train/serve skew the schema exists to prevent.
    return pd.DataFrame(rows)[FEATURE_COLUMNS + [LABEL_COLUMN, GROUP_COLUMN]]
def _sample_user() -> dict:
    """Synthetic user preference profile. Only the fields _true_utility needs —
    no identity/PII, this is a ranking weight vector, not a person."""
    return {
        "cost_sensitivity": float(np.clip(RNG.beta(2, 2), 0, 1)),
        "max_walk_minutes": float(RNG.integers(3, 20)),
        "preferred_spot_type": int(RNG.integers(0, len(SPOT_TYPES))),
    }

def _sample_spot() -> dict:
    """Synthetic candidate spot. Drawn fresh from distributions every call —
    never seeded from real DB/campus data, which an ML would have memorized
    and the model would overfit to instead of learning the underlying signal."""
    return {
        "base_cost": float(np.clip(RNG.gamma(2.0, 3.0), 0, 30)),
        "walk_time_minutes": float(np.clip(RNG.exponential(6.0), 0.5, 25)),
        "spot_type": int(RNG.integers(0, len(SPOT_TYPES))),
        "is_verified": int(RNG.random() < 0.6),
        "historical_congestion": float(np.clip(RNG.beta(2, 3), 0, 1)),
    }

def _sample_context() -> dict:
    """Synthetic traffic context. In training we simulate the TomTom values
    directly — the real adapter (context/traffic.py) runs only at serve time."""
    hour = int(RNG.integers(6, 23))
    dow = int(RNG.integers(0, 7))
    sem_week = int(RNG.integers(1, 17))
    is_finals = int(sem_week >= 15)
    # Simulate higher congestion during rush hours.
    base_jam = 0.3 if 7 <= hour <= 9 or 16 <= hour <= 18 else 0.1
    return {
        "live_congestion": float(np.clip(RNG.normal(base_jam, 0.15), 0, 1)),
        "traffic_delay_minutes": float(np.clip(RNG.exponential(2.0), 0, 20)),
        "hour_of_day": hour,
        "day_of_week": dow,
        "semester_week": sem_week,
        "is_finals_week": is_finals,
    }

def _true_utility(user: dict, spot: dict, ctx: dict) -> float:
    """This function IS your domain knowledge. Encode the scenarios here:
       - heavy traffic around spot → penalize, harder when time-pressed (finals)
       - significant delay → penalize based on cost sensitivity
       - finals week → speed dominates, cost secondary
       - clear roads in morning → cost dominates
    The model's whole job is to *rediscover* this from data. Make it rich."""
    finals = ctx["is_finals_week"]
    weekend = ctx["day_of_week"] >= 5
    early_morning = ctx["hour_of_day"] <= 9
    u = 0.0

    # --- cost. Dominates on clear early mornings, fades under finals pressure.
    cost_weight = user["cost_sensitivity"] * (0.3 if finals else 1.0)
    if early_morning and ctx["live_congestion"] < 0.2:
        cost_weight *= 1.6          # clear roads in the morning → cost dominates
    u -= spot["base_cost"] * cost_weight

    # --- walk. Mild inside the user's stated tolerance, punishing past it.
    # This is what makes max_walk_minutes a real feature instead of noise.
    walk, tolerance = spot["walk_time_minutes"], user["max_walk_minutes"]
    u -= walk * 0.5
    u -= max(0.0, walk - tolerance) * (1.0 if weekend else 2.0)

    # --- spot type preference (makes preferred_spot_type/spot_type meaningful)
    u += 2.0 if spot["spot_type"] == user["preferred_spot_type"] else 0.0

    # --- traffic. Bites harder when the user is time-pressed.
    u -= ctx["live_congestion"] * (6.0 if finals else 3.0)
    u -= ctx["traffic_delay_minutes"] * (0.7 if finals else 0.4)
    u -= spot["historical_congestion"] * 2.0

    # --- competition for spots rises later in the semester. Two effects, and
    # the second must NOT be gated on is_verified or semester_week goes dead
    # for every unverified spot.
    scarcity = ctx["semester_week"] / 16.0
    u += spot["is_verified"] * (1.5 + 2.0 * scarcity)   # verification worth more when lots fill
    u -= spot["historical_congestion"] * 3.0 * scarcity # a busy lot is a worse bet in week 14

    # --- weekends are relaxed: less time pressure, cost/walk trade freely.
    if weekend:
        u += 1.0 - ctx["live_congestion"] * 1.5
    return u

# The main function that generates the syntehic data we have just coded up
def main(n_sessions=7000, out=OUT):
    df = pd.concat([make_session(i) for i in range(n_sessions)],
                   ignore_index=True)

    # --- contract checks at the write boundary
    assert list(df.columns) == FEATURE_COLUMNS + [LABEL_COLUMN, GROUP_COLUMN], \
        "column contract mismatch"
    assert df[GROUP_COLUMN].nunique() == n_sessions, "session count mismatch"
    assert not df.isnull().any().any(), "nulls found"
    assert (df.groupby(GROUP_COLUMN)[LABEL_COLUMN].sum() == 1).all(), \
        "every session needs exactly one selection"

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    # --- look at it
    avg_group = len(df) / n_sessions
    print(f"wrote {len(df):,} rows / {n_sessions:,} sessions -> {out}")
    print(f"avg candidates per session: {avg_group:.2f}")
    print(f"label rate: {df[LABEL_COLUMN].mean():.4f}  (expect ~{1/avg_group:.4f})")

    # Compare the CHOSEN spot against the ones passed over. Averaging every row
    # tells you nothing: spots are sampled independently of context, so the pool
    # of candidates is statistically identical in every session. Only the
    # *choice* carries the domain logic — and the choice is always within a
    # session, which is the same comparison the ranker itself makes.
    sel = df[df[LABEL_COLUMN] == 1]
    uns = df[df[LABEL_COLUMN] == 0]
    print("\nchosen vs passed over:")
    for col in ["base_cost", "walk_time_minutes", "is_verified", "historical_congestion"]:
        print(f"  {col:<24} {sel[col].mean():7.3f}  vs {uns[col].mean():7.3f}")

    # Scenario check: under finals pressure the cost weight drops, so the chosen
    # spot should be less of a bargain relative to its own session's average.
    gap = df[LABEL_COLUMN].eq(1)
    cost_gap = (df.base_cost - df.groupby(GROUP_COLUMN).base_cost.transform("mean"))[gap]
    print("\ncost premium of chosen spot vs its session "
          "(should rise during finals):")
    print(cost_gap.groupby(df.is_finals_week[gap]).mean().round(3).to_string())

if __name__ == "__main__":
    main()