# Park & Go — ML Parking Engine (V3)

A LightGBM ranker that orders candidate parking spots for **this user, right now**, using
live traffic context. Replaces the V2 static 75-point heuristic, which stays as a fallback.

> **Full design and rationale:** [`doc/PARKING_ENGINE.md`](../doc/PARKING_ENGINE.md).
> **Phase-by-phase build guide:** [`doc/PHASE_GUIDE.md`](../doc/PHASE_GUIDE.md).
> **Workbook for writing Phases 3–5 yourself:** [`doc/LEARNING_TRACK.md`](../doc/LEARNING_TRACK.md).
> This file is the operational README — what exists, how to run it. The design docs are the
> single source of truth for *why*; keep them, not this file, when the two disagree.

---

## Status

| Phase | What | State |
|---|---|---|
| 0 | `schema.py` — feature contract | ✅ done |
| 1 | `context/traffic.py` — TomTom adapter | ✅ done, 15 tests |
| 2 | `data/synthetic_generator.py` | ⚠️ generator done; `main()` + parquet still to write |
| 3 | `feature_store.py`, `ranker.py`, `training/train.py` | ⬜ stubs |
| 4 | `engine.py` — serving + fallback | ⬜ stub |
| 5 | Google Maps navigation deep link | ⬜ not started |

---

## Running it

Everything must be run **from inside `parking-engine/`** — modules do `from schema import ...`,
and the parent directory has a hyphen in its name, so it can't be a Python package.

```bash
cd parking-engine
python -c "from schema import FEATURE_COLUMNS; print(len(FEATURE_COLUMNS))"   # -> 14
python -m pytest tests/ -q                                                    # -> 27 passed
python -m data.synthetic_generator                                            # writes the parquet (Phase 2)
python -m training.train                                                      # trains the ranker (Phase 3)
```

**Use `-m`, not a file path.** `python data/synthetic_generator.py` puts `data/` on `sys.path`
instead of `parking-engine/`, so `from schema import ...` raises `ModuleNotFoundError`. `-m` puts
the current directory on the path, which is what these top-level imports need. Same applies to
`training/train.py`.

Tests need no API keys and make no network calls — TomTom and Redis are both faked.

### Environment

Copy `.env.example` to `.env` and fill in:

| Var | Notes |
|---|---|
| `TOMTOM_API_KEY` | Free tier at developer.tomtom.com. **Name must match exactly** — a typo fails soft and silently serves clear-road defaults. |
| `REDIS_URL` | Defaults to `redis://localhost:6379/0` if unset. |

Nothing calls `load_dotenv()` yet — see [`TODO.md`](../TODO.md).

---

## The feature contract

`schema.py` is the single source of truth. All 14 columns, in order, are what the model sees —
`FEATURE_COLUMNS` order is load-bearing because LightGBM receives a matrix, not names.

| Group | Features |
|---|---|
| User | `cost_sensitivity`, `max_walk_minutes`, `preferred_spot_type` |
| Spot | `base_cost`, `walk_time_minutes`, `spot_type`, `is_verified`, `historical_congestion` |
| Traffic (TomTom) | `live_congestion`, `traffic_delay_minutes` |
| Temporal | `hour_of_day`, `day_of_week`, `semester_week`, `is_finals_week` |

Label is `selected` (0/1); ranking group is `session_id`.

**One vendor.** TomTom covers congestion, delay, and weather-related hazards (fog, ice,
flooding arrive as incident categories). There is no OpenWeatherMap or events integration —
those were cut when the design consolidated to TomTom.

### Two things that will bite you

**TomTom has no `jamFactor` field.** That's HERE's API. Congestion is derived from
`currentSpeed` vs `freeFlowSpeed`; delay from `currentTravelTime - freeFlowTravelTime`.

**`iconCategory` arrives as an int**, and `TRAFFIC_INCIDENT_CATEGORIES`' values *are* TomTom's
enum. Use the wire value directly; use `TRAFFIC_INCIDENT_LABELS` for the reverse lookup.

### Temporal features & semester calendar

`semester_week` and `is_finals_week` come from a campus calendar with **two rules**:

- **Fall** always begins on the **1st Tuesday of September**.
- **Spring** always begins the **Monday after MLK Day** (3rd Monday of January).

There is **no summer-session rule** and no inter-semester bridge — winter break, spring
break, and the May–August summer window are intentionally unsupported. During those
periods `temporal_features_now()` returns a neutral fallback (`semester_week=1`,
`is_finals_week=0`) so the model is never fed an out-of-distribution value. Expect
reduced ranking quality outside the fall and spring windows; that is by design, not a
bug.

If your campus uses different rules (e.g. quarter system, summer session), edit
`_first_tuesday_of_september` and `_week_after_mlk` in `data/feature_store.py` — they
are the only two calendar functions in the project.

---

## Layout

```
parking-engine/
├── context/traffic.py          # TomTom Flow + Incidents; cached, fail-soft, never raises
├── data/
│   ├── synthetic_generator.py  # Generates training_data.parquet
│   ├── feature_store.py        # Feature vectors — shared by train AND serve (anti-skew)
│   └── processed/              # Build output (gitignored)
├── models/
│   ├── ranker.py               # LightGBM wrapper
│   └── trained/                # Model artifacts (gitignored)
├── training/train.py           # Training pipeline
├── tests/                      # No network, no API keys
├── schema.py                   # FEATURE_COLUMNS, encodings, TomTom lookup tables
└── engine.py                   # Public interface + Google Maps deep link
```

---

## Non-negotiables

These are the rules that keep the system honest. Breaking one fails *silently*.

1. **Train and serve build features with the same function** (`feature_store.py`), ending in
   `df[FEATURE_COLUMNS]`. This is the anti-skew keystone.
2. **`context/traffic.py` never raises.** A dead TomTom or a dead Redis degrades to
   clear-road defaults; it does not propagate into the request path.
3. **Split by group, never by row.** Rows from one `session_id` must not straddle train/val.
4. **Rows must be sorted by `session_id`** before computing LightGBM `group` sizes.
5. **Always beat the heuristic baseline** on the same validation sessions. "It works" is not
   a result; "it beats the heuristic by X NDCG" is.
6. **Ship the fallback and test the failure path on purpose.** A feature that can crash
   `/api/recommendations` is worse than no feature.
