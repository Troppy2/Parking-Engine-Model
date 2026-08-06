import os, json, requests, redis
from schema import TRAFFIC_INCIDENT_CATEGORIES, TRAFFIC_MAGNITUDE

# By python convention these _ means its a private variable and can't be accessed
_redis = redis.from_url(os.environ["REDIS_URL"])
FLOW_CACHE_TTL = 3 * 60 # This lives for around 2 - 5 mins
INCIDENT_CACHE_TTL = 5 * 60
TOMTOM_BASE = "https://api.tomtom.com"


def get_traffic_context(spot: dict) -> dict:
    """
    This function returns live congestion features for a single spot via TomTom
    Process:
    Call Flow Segment data for jamfator + delay and traffic incidents for weather hazards
    -> returns 2 features teh model consumes and a sidebar the frontend uses for map overlay
    """
    flow = _get_flow(spot)
    incidents = _get_incidents(spot)
    return {
        # model features
        "live_congestion": flow["live_congestion"],
        "traffic_delay_minutes": flow["traffic_delay_minutes"],
        # sidebar context (not model inputs — used by frontend/engine for display)
        "incidents": incidents,
    }
# --- flow ----------------------------------------------------------------
def _get_flow(spot: dict) -> dict:
    key = f"flow:{round(spot['lat'], 3)}:{round(spot['lon'], 3)}"  # ~100 m cell
    cached = _redis.get(key)
    if cached:
        return json.loads(cached)
    try:
        r = requests.get(
            f"{TOMTOM_BASE}/traffic/services/4/flowSegmentData/absolute/10/json",
            params={
                "point": f"{spot['lat']},{spot['lon']}",
                "unit": "KMPH",
                "key": os.environ["TOMTOM_API_KEY"],
            },
            timeout=3,
        )
        r.raise_for_status()
        seg = r.json()["flowSegmentData"]
        # jamFactor is 0–10; normalize to 0–1.
        jam = seg.get("jamFactor", 0.0)
        current_speed = seg.get("currentSpeed", seg.get("freeFlowSpeed", 1))
        free_flow = seg.get("freeFlowSpeed", current_speed) or 1
        # Estimate added minutes over a ~400 m catchment at free-flow vs. current speed.
        catchment_km = 0.4
        delay = max(0.0, (catchment_km / current_speed - catchment_km / free_flow) * 60)
        ctx = {
            "live_congestion": round(jam / 10.0, 3),
            "traffic_delay_minutes": round(delay, 2),
        }
        _redis.setex(key, FLOW_CACHE_TTL, json.dumps(ctx))
        return ctx
    except Exception:
        return _DEFAULT_FLOW

_DEFAULT_FLOW = {"live_congestion": 0.0, "traffic_delay_minutes": 0.0}
    

# --- incidents -----------------------------------------------------------

def _get_incidents(spot: dict) -> list[dict]:
    key = f"incidents:{round(spot['lat'], 3)}:{round(spot['lon'], 3)}"
    cached = _redis.get(key)
    # I am assuming same process right we do the while try and except thing
    if cached:
        return json.loads(cached)
    try:
        delta = 0.004 # keeps a bound of 400 m around our spot a little circle
        bbox = (f"{spot['lat'] - delta},{spot['lon'] - delta},"
                f"{spot['lat'] + delta},{spot['lon'] + delta}")
        r = requests.get(
            f"{TOMTOM_BASE}/traffic/services/5/incidentDetails",
            params={
                "bbox": bbox,
                "fields": "{incidents{type,geometry{type,coordinates},properties{iconCategory,magnitudeOfDelay}}}",
                "language": "en-GB",
                "key": os.environ["TOMTOM_API_KEY"],
            },
            timeout=3,
        )
        r.raise_for_status()
        raw = r.json().get("incidents", [])
        normalized = []
        for inc in raw:
            props = inc.get("properties", {})
            icon = props.get("iconCategory", "unknown")
            mag = props.get("magnitudeOfDelay", 0)
            cat_id = TRAFFIC_INCIDENT_CATEGORIES.get(icon, 0)
            mag_info = TRAFFIC_MAGNITUDE.get(mag, TRAFFIC_MAGNITUDE[0])
            normalized.append({
                "category_id": cat_id,
                "category_label": icon,
                "magnitude_id": mag,
                "severity": mag_info["severity"],
                "color": mag_info["color"],
            })
        _redis.setex(key, INCIDENT_CACHE_TTL, json.dumps(normalized))
        return normalized
    except Exception:
        return []