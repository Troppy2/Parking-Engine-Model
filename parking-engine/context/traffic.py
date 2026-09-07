import os, json, requests, redis
from schema import (TRAFFIC_INCIDENT_CATEGORIES, TRAFFIC_INCIDENT_LABELS,
                    TRAFFIC_MAGNITUDE)

# By python convention these _ means its a private variable and can't be accessed
_redis = redis.from_url(
    os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True,   # so json.loads() gets str, not bytes
)
FLOW_CACHE_TTL = 3 * 60 # This lives for around 2 - 5 mins
INCIDENT_CACHE_TTL = 5 * 60
FAILURE_CACHE_TTL = 30  # short negative cache so an outage can't be hammered
TOMTOM_BASE = "https://api.tomtom.com"
BBOX_DELTA = 0.004          # ~400 m box around the spot, in degrees
CLOSURE_DELAY_MINUTES = 30.0  # a closed road has no travel time to measure;
                              # this is a deliberate "very bad" stand-in
REQUEST_TIMEOUT = 3

# Reuse one pooled connection instead of a fresh TCP+TLS handshake per call —
# this is on the request path, once per candidate spot.
_session = requests.Session()

# Always handed out by value (see _get_flow's fallback) — never return this dict
# directly, a caller mutating it would corrupt the default for every later request.
_DEFAULT_FLOW = {"live_congestion": 0.0, "traffic_delay_minutes": 0.0}


def get_traffic_context(spot: dict) -> dict:
    """
    This function returns live congestion features for a single spot via TomTom
    Process:
    Call Flow Segment data for speed/travel-time and traffic incidents for weather hazards
    -> returns 2 features the model consumes and a sidebar the frontend uses for map overlay
    """
    flow = _get_flow(spot)
    return {
        # model features
        "live_congestion": flow["live_congestion"],
        "traffic_delay_minutes": flow["traffic_delay_minutes"],
        # sidebar context (not model inputs — used by frontend/engine for display)
        "incidents": _get_incidents(spot),
    }


# --- shared cache/fetch plumbing -----------------------------------------

def _cell(spot: dict) -> str:
    return f"{round(spot['lat'], 3)}:{round(spot['lon'], 3)}"  # ~100 m cell


def _cache_read(key: str):
    """Reads are best-effort: a dead Redis must not raise into the request path."""
    try:
        cached = _redis.get(key)
        return json.loads(cached) if cached is not None else None
    except Exception:
        return None


def _cache_write(key: str, ttl: int, value) -> None:
    try:
        _redis.setex(key, ttl, json.dumps(value))
    except Exception:
        pass  # caching is best-effort; never fail the request over it


def _cached(key: str, ttl: int, fetch, fallback):
    """Redis-cached, fail-soft wrapper. Every external read in this module goes
    through here so the caching and the never-raise guarantee live in one place.
    On failure the fallback is cached briefly so an outage can't be hammered.
    """
    cached = _cache_read(key)
    if cached is not None:
        return cached
    try:
        value, cache_ttl = fetch(), ttl
    except Exception:
        value, cache_ttl = fallback(), FAILURE_CACHE_TTL
    _cache_write(key, cache_ttl, value)
    return value


def _get(path: str, params: dict) -> dict:
    params = {**params, "key": os.environ["TOMTOM_API_KEY"]}
    r = _session.get(f"{TOMTOM_BASE}{path}", params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


# --- flow ----------------------------------------------------------------

def _get_flow(spot: dict) -> dict:
    return _cached(f"flow:{_cell(spot)}", FLOW_CACHE_TTL,
                   lambda: _fetch_flow(spot), lambda: dict(_DEFAULT_FLOW))


def _fetch_flow(spot: dict) -> dict:
    seg = _get("/traffic/services/4/flowSegmentData/absolute/10/json",
               {"point": f"{spot['lat']},{spot['lon']}", "unit": "KMPH"})["flowSegmentData"]

    # A closed road is maximum congestion, not missing data.
    if seg.get("roadClosure"):
        return {"live_congestion": 1.0,
                "traffic_delay_minutes": CLOSURE_DELAY_MINUTES}

    # TomTom has NO jamFactor field (that's HERE's API). Congestion is derived
    # from how far current speed has fallen below free-flow: 0 = free-flow, 1 = stopped.
    free_flow = seg.get("freeFlowSpeed") or 0
    current = seg.get("currentSpeed")
    current = free_flow if current is None else current
    congestion = 0.0 if free_flow <= 0 else 1.0 - (current / free_flow)

    # Delay comes straight from TomTom's own travel times (seconds), which beats
    # inventing a catchment distance and dividing by speed.
    delay_s = seg.get("currentTravelTime", 0) - seg.get("freeFlowTravelTime", 0)

    return {
        "live_congestion": round(min(max(congestion, 0.0), 1.0), 3),
        "traffic_delay_minutes": round(max(delay_s, 0) / 60.0, 2),
    }


# --- incidents -----------------------------------------------------------

def _get_incidents(spot: dict) -> list[dict]:
    return _cached(f"incidents:{_cell(spot)}", INCIDENT_CACHE_TTL,
                   lambda: _fetch_incidents(spot), list)


def _fetch_incidents(spot: dict) -> list[dict]:
    # TomTom documents bbox as minLon,minLat,maxLon,maxLat — lon first.
    bbox = (f"{spot['lon'] - BBOX_DELTA},{spot['lat'] - BBOX_DELTA},"
            f"{spot['lon'] + BBOX_DELTA},{spot['lat'] + BBOX_DELTA}")
    raw = _get("/traffic/services/5/incidentDetails", {
        "bbox": bbox,
        "fields": "{incidents{type,geometry{type,coordinates},properties{iconCategory,magnitudeOfDelay}}}",
        "language": "en-GB",
    }).get("incidents", [])

    normalized = []
    for inc in raw:
        props = inc.get("properties", {})
        # iconCategory arrives as an INT already in our encoding — look up the
        # label, don't look up the id (that was silently making everything "unknown").
        cat_id = props.get("iconCategory", TRAFFIC_INCIDENT_CATEGORIES["unknown"])
        mag = props.get("magnitudeOfDelay", 0)
        mag_info = TRAFFIC_MAGNITUDE.get(mag, TRAFFIC_MAGNITUDE[0])
        normalized.append({
            "category_id": cat_id,
            "category_label": TRAFFIC_INCIDENT_LABELS.get(cat_id, "unknown"),
            "magnitude_id": mag,
            "severity": mag_info["severity"],
            "color": mag_info["color"],
        })
    return normalized
