# Public interface for the ML parking engine.
# Accepts a user + candidate spots, fetches live context, builds features,
# and returns spots ranked best → worst for this user right now.
# Falls back to the heuristic scorer if the model is not loaded.
