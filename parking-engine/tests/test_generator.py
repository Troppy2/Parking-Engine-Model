"""Phase 2 verification.

The doc's Definition of Done: columns exactly FEATURE_COLUMNS + [selected,
session_id], every session has exactly one selected == 1, and a sanity check
confirms the scenarios are actually visible in the numbers.
"""
import sys, pathlib
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from data.synthetic_generator import (make_session, _sample_user, _sample_spot,
                                      _sample_context, _true_utility)
from schema import FEATURE_COLUMNS, LABEL_COLUMN, GROUP_COLUMN, SPOT_TYPES


@pytest.fixture(scope="module")
def corpus():
    """One corpus for every statistical assertion in this file. These are
    property checks over a distribution, so they can share a sample."""
    return pd.concat([make_session(i) for i in range(1500)], ignore_index=True)


# --- structure ------------------------------------------------------------

def test_columns_match_the_contract_in_order():
    """Order, not just membership — LightGBM sees a matrix, not names."""
    df = make_session(0)
    assert list(df.columns) == FEATURE_COLUMNS + [LABEL_COLUMN, GROUP_COLUMN]


def test_exactly_one_selection_per_session(corpus):
    per_session = corpus.groupby(GROUP_COLUMN)[LABEL_COLUMN].sum()
    assert (per_session == 1).all()


def test_session_size_in_range(corpus):
    sizes = corpus.groupby(GROUP_COLUMN).size()
    assert sizes.between(5, 9).all()


def test_label_rate_matches_average_group_size(corpus):
    avg_group = corpus.groupby(GROUP_COLUMN).size().mean()
    assert corpus[LABEL_COLUMN].mean() == pytest.approx(1 / avg_group, abs=0.02)


def test_generation_is_reproducible():
    """The seed guarantees a whole RUN reproduces, not that make_session(i) is
    a pure function of i — it draws from the module-level RNG, so position
    matters. Reload = fresh seed = the same dataset a second time."""
    import importlib, data.synthetic_generator as g

    def run():
        importlib.reload(g)
        return pd.concat([g.make_session(i) for i in range(20)], ignore_index=True)

    pd.testing.assert_frame_equal(run(), run())


# --- feature ranges -------------------------------------------------------

def test_features_are_in_declared_ranges(corpus):
    df = corpus
    assert df["live_congestion"].between(0, 1).all()
    assert df["historical_congestion"].between(0, 1).all()
    assert df["cost_sensitivity"].between(0, 1).all()
    assert df["traffic_delay_minutes"].between(0, 20).all()
    assert df["hour_of_day"].between(6, 22).all()
    assert df["day_of_week"].between(0, 6).all()
    assert df["semester_week"].between(1, 16).all()
    assert df["spot_type"].isin(SPOT_TYPES.values()).all()
    assert df["preferred_spot_type"].isin(SPOT_TYPES.values()).all()
    assert df["is_verified"].isin([0, 1]).all()


def test_is_finals_week_is_consistent_with_semester_week(corpus):
    df = corpus
    assert (df["is_finals_week"] == (df["semester_week"] >= 15).astype(int)).all()


# --- THE important one: every contract feature must carry signal -----------

def test_no_feature_is_pure_noise():
    """A feature the utility function never reads is noise the model will
    happily overfit. Perturb each feature and assert the utility moves."""
    user, spot, ctx = _sample_user(), _sample_spot(), _sample_context()
    # Pin the sampled values this test reasons about — a randomly tiny
    # walk_time would make the max_walk_minutes bump a no-op and flake.
    spot["is_verified"], spot["historical_congestion"] = 0, 0.4
    spot["walk_time_minutes"], user["max_walk_minutes"] = 12.0, 10.0
    user["preferred_spot_type"] = spot["spot_type"]     # start matched...
    base = _true_utility(user, spot, ctx)
    bumps = {
        "cost_sensitivity": (user, 0.9), "max_walk_minutes": (user, 1.0),
        # ...so bumping either side breaks the match and must move the utility
        "preferred_spot_type": (user, (spot["spot_type"] + 1) % 3),
        "base_cost": (spot, 25.0), "walk_time_minutes": (spot, 24.0),
        "spot_type": (spot, (spot["spot_type"] + 1) % 3),
        "is_verified": (spot, 1),
        "historical_congestion": (spot, 0.95),
        "live_congestion": (ctx, 0.95), "traffic_delay_minutes": (ctx, 19.0),
        "hour_of_day": (ctx, 8), "day_of_week": (ctx, 6),
        "semester_week": (ctx, 16), "is_finals_week": (ctx, 1 - ctx["is_finals_week"]),
    }
    dead = []
    for feat, (owner, new) in bumps.items():
        old = owner[feat]
        owner[feat] = new
        # hour_of_day only bites on clear roads; give it the context it needs
        saved = ctx["live_congestion"]
        if feat == "hour_of_day":
            ctx["live_congestion"] = 0.05
        if _true_utility(user, spot, ctx) == base and new != old:
            dead.append(feat)
        ctx["live_congestion"] = saved
        owner[feat] = old
    assert not dead, f"features with zero influence on the label: {dead}"


# --- scenario sanity checks (the doc's "see it in the numbers") ------------

def test_selected_spots_beat_unselected_on_the_things_users_want(corpus):
    df = corpus
    sel = df[df[LABEL_COLUMN] == 1]
    unsel = df[df[LABEL_COLUMN] == 0]
    assert sel["walk_time_minutes"].mean() < unsel["walk_time_minutes"].mean()
    assert sel["base_cost"].mean() < unsel["base_cost"].mean()
    assert sel["is_verified"].mean() > unsel["is_verified"].mean()
    assert sel["historical_congestion"].mean() < unsel["historical_congestion"].mean()


def test_finals_week_makes_users_less_cost_sensitive(corpus):
    """Encoded scenario: under time pressure cost weight drops, so the chosen
    spot's price premium relative to the alternatives should rise."""
    df = corpus.copy()
    df["cost_gap"] = df["base_cost"] - df.groupby(GROUP_COLUMN)["base_cost"].transform("mean")
    sel = df[df[LABEL_COLUMN] == 1]
    assert sel[sel.is_finals_week == 1]["cost_gap"].mean() > \
           sel[sel.is_finals_week == 0]["cost_gap"].mean()


def test_spot_type_preference_is_learnable(corpus):
    df = corpus.copy()
    df["match"] = (df["spot_type"] == df["preferred_spot_type"]).astype(int)
    assert df[df[LABEL_COLUMN] == 1]["match"].mean() > \
           df[df[LABEL_COLUMN] == 0]["match"].mean()


def test_choices_are_not_perfectly_separable(corpus):
    """With too little noise the argmax is deterministic and NDCG lies. The
    chosen spot should NOT always be the cheapest/closest one available."""
    df = corpus
    is_min_walk = df.groupby(GROUP_COLUMN)["walk_time_minutes"].transform("min") \
        == df["walk_time_minutes"]
    always_closest = df[df[LABEL_COLUMN] == 1].index.isin(df[is_min_walk].index).mean()
    assert 0.15 < always_closest < 0.95, always_closest
