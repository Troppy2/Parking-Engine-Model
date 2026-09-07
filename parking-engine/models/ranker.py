# LightGBM ranker wrapper.
# Exposes fit(), predict(), save(), and load() around a LightGBM model
# trained with the lambdarank objective.
import lightgbm as lgb
import numpy as np
from schema import FEATURE_COLUMNS, CATEGORICAL_FEATURES


class Ranker:
    def __init__(self):
        # ── TODO 1 ────────────────────────────────────────────────────────
        # Sentinel for "no model loaded yet". See Q1.
        self.model = None
        
    def guard_group_invariant(self, X, group):
        # Guard the invariant BEFORE handing data to LightGBM. A bad group list
        # is the one mistake here that produces a plausible-looking wrong answer.
        assert sum(group) == len(X), "Sum of group sizes must equal number of rows in X"
    def fit(self, X, y, group, X_val=None, y_val=None, group_val=None):
        # ── TODO 2 ────────────────────────────────────────────────────────
        # Guard the invariant BEFORE handing data to LightGBM. A bad group list
        # This seems like a whole new function to guard an invarient it should be a function outside of the fit that we call
        self.guard_group_invariant(X, group)
        
        # is the one mistake here that produces a plausible-looking wrong answer.
        # assert sum(group) == len(X), "..."
        assert len(y) == len(X), "y must have same number of rows as X"
        ...

        # ── TODO 3 ────────────────────────────────────────────────────────
        # Build the training Dataset. lgb.Dataset(X, label=..., group=...,
        # categorical_feature=..., feature_name=...)
        #
        # Why pass categorical_feature? Ints like spot_type=2 are NOT "twice
        # spot_type=1" — they're labels. Telling LightGBM lets it split on
        # set membership instead of thresholds.
        train = ...

        # ── TODO 4 ────────────────────────────────────────────────────────
        # params dict. Start from PHASE_GUIDE's and understand each one:
        #   objective="lambdarank"  the whole point — optimize ORDER
        #   metric="ndcg"           what early stopping watches
        #   ndcg_eval_at=[1,3,5]    @1 is what the user sees first
        #   learning_rate=0.05      smaller = slower, usually better
        #   num_leaves=31           model capacity; too high overfits
        #   min_data_in_leaf=50     refuse leaves built on too few rows
        #   verbose=-1              silence LightGBM's chatter
        params = {...}

        # ── TODO 5 ────────────────────────────────────────────────────────
        # If validation data was passed, build a Dataset for it with
        # reference=train (shares the binning so the two are comparable),
        # then train with early stopping via callbacks=[lgb.early_stopping(50)].
        # If not, train without valid_sets.
        #
        # SYNTAX: `[x] if cond else None` inlines that branch.
        ...

    def predict(self, X) -> np.ndarray:
        # ── TODO 6 ────────────────────────────────────────────────────────
        # Raise a clear RuntimeError if no model is loaded, then predict with
        # column order enforced. See Q2.
        ...

    def save(self, path):
        ...

    def load(self, path):
        # ── TODO 7 ────────────────────────────────────────────────────────
        # lgb.Booster(model_file=path). Return self so callers can write
        # `Ranker().load(p)` in one line — that's why engine.py can do it at
        # import time.
        ...