"""Phase 1 verification — zero real API calls.

Everything external (TomTom, Redis) is faked. The contract under test:
get_traffic_context never raises, caches, and normalizes TomTom's wire
format into the two columns the model consumes.
"""
import json, sys, os, pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("TOMTOM_API_KEY", "test-key")

import context.traffic as traffic
from schema import TRAFFIC_INCIDENT_CATEGORIES, FEATURE_COLUMNS

SPOT = {"lat": 44.9740, "lon": -93.2277}


class FakeRedis:
    """Minimal Redis stand-in. `down=True` makes every op raise, which is how
    we prove the module degrades instead of exploding."""
    def __init__(self, down=False):
        self.store, self.down = {}, down

    def get(self, k):
        if self.down:
            raise ConnectionError("Connection refused")
        return self.store.get(k)

    def setex(self, k, ttl, v):
        if self.down:
            raise ConnectionError("Connection refused")
        self.store[k] = v


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.calls = 0

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def fresh_redis(monkeypatch):
    monkeypatch.setattr(traffic, "_redis", FakeRedis())


def _flow_payload(**over):
    seg = {"currentSpeed": 25, "freeFlowSpeed": 50,
           "currentTravelTime": 120, "freeFlowTravelTime": 60,
           "roadClosure": False}
    seg.update(over)
    return {"flowSegmentData": seg}


def _patch_get(monkeypatch, flow=None, incidents=None):
    """Route TomTom URLs to canned payloads and count the calls."""
    counter = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        counter["n"] += 1
        if "flowSegmentData" in url:
            return FakeResponse(flow if flow is not None else _flow_payload())
        return FakeResponse({"incidents": incidents or []})

    monkeypatch.setattr(traffic._session, "get", fake_get)
    return counter


# --- normalization --------------------------------------------------------

def test_congestion_from_speed_ratio(monkeypatch):
    """Half of free-flow speed => 0.5 congestion. TomTom has no jamFactor field,
    so this must be derived from currentSpeed vs freeFlowSpeed."""
    _patch_get(monkeypatch)
    ctx = traffic.get_traffic_context(SPOT)
    assert ctx["live_congestion"] == 0.5
    # 120s current - 60s free flow = 60s = 1.0 min
    assert ctx["traffic_delay_minutes"] == 1.0


def test_free_flow_speed_means_no_congestion(monkeypatch):
    _patch_get(monkeypatch, flow=_flow_payload(
        currentSpeed=50, currentTravelTime=60))
    ctx = traffic.get_traffic_context(SPOT)
    assert ctx["live_congestion"] == 0.0
    assert ctx["traffic_delay_minutes"] == 0.0


def test_gridlock_is_max_congestion_not_a_crash(monkeypatch):
    """currentSpeed=0 used to raise ZeroDivisionError and fall back to
    'clear roads' — the exact opposite of the truth."""
    _patch_get(monkeypatch, flow=_flow_payload(
        currentSpeed=0, currentTravelTime=600))
    ctx = traffic.get_traffic_context(SPOT)
    assert ctx["live_congestion"] == 1.0
    assert ctx["traffic_delay_minutes"] == 9.0


def test_road_closure_is_max_congestion(monkeypatch):
    _patch_get(monkeypatch, flow=_flow_payload(roadClosure=True))
    assert traffic.get_traffic_context(SPOT)["live_congestion"] == 1.0


def test_delay_never_negative(monkeypatch):
    _patch_get(monkeypatch, flow=_flow_payload(
        currentTravelTime=40, freeFlowTravelTime=60))
    assert traffic.get_traffic_context(SPOT)["traffic_delay_minutes"] == 0.0


# --- incidents ------------------------------------------------------------

def test_incident_icon_category_is_an_int_on_the_wire(monkeypatch):
    """TomTom sends iconCategory as an int. Looking that int up in a
    string-keyed dict silently made every incident 'unknown'."""
    _patch_get(monkeypatch, incidents=[
        {"properties": {"iconCategory": TRAFFIC_INCIDENT_CATEGORIES["fog"],
                        "magnitudeOfDelay": 3}}])
    inc = traffic.get_traffic_context(SPOT)["incidents"][0]
    assert inc["category_id"] == 2
    assert inc["category_label"] == "fog"
    assert inc["severity"] == 3
    assert inc["color"] == "rgba(255,0,0,0.25)"


def test_unknown_incident_category_degrades_cleanly(monkeypatch):
    _patch_get(monkeypatch, incidents=[{"properties": {}}])
    inc = traffic.get_traffic_context(SPOT)["incidents"][0]
    assert inc["category_label"] == "unknown"


def test_incident_bbox_is_lon_lat_order(monkeypatch):
    """TomTom documents bbox as minLon,minLat,maxLon,maxLat."""
    seen = {}

    def fake_get(url, params=None, timeout=None):
        if "incidentDetails" in url:
            seen["bbox"] = params["bbox"]
            return FakeResponse({"incidents": []})
        return FakeResponse(_flow_payload())

    monkeypatch.setattr(traffic._session, "get", fake_get)
    traffic.get_traffic_context(SPOT)
    min_lon, min_lat, max_lon, max_lat = (float(x) for x in seen["bbox"].split(","))
    assert min_lon < max_lon < 0 and 0 < min_lat < max_lat


# --- caching --------------------------------------------------------------

def test_second_call_hits_cache(monkeypatch):
    counter = _patch_get(monkeypatch)
    traffic.get_traffic_context(SPOT)
    assert counter["n"] == 2          # one flow + one incidents
    traffic.get_traffic_context(SPOT)
    assert counter["n"] == 2          # served entirely from cache


def test_nearby_spots_share_a_cache_cell(monkeypatch):
    counter = _patch_get(monkeypatch)
    traffic.get_traffic_context(SPOT)
    traffic.get_traffic_context({"lat": SPOT["lat"] + 0.0001,
                                 "lon": SPOT["lon"] - 0.0001})
    assert counter["n"] == 2          # ~10 m apart => same ~100 m cell


# --- fail-soft ------------------------------------------------------------

def test_api_failure_returns_clear_road_default(monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("tomtom is down")
    monkeypatch.setattr(traffic._session, "get", boom)
    ctx = traffic.get_traffic_context(SPOT)
    assert ctx == {"live_congestion": 0.0, "traffic_delay_minutes": 0.0,
                   "incidents": []}


def test_redis_down_does_not_raise(monkeypatch):
    """The original code called _redis.get() outside the try, so a dead cache
    propagated ConnectionError straight into the request path."""
    monkeypatch.setattr(traffic, "_redis", FakeRedis(down=True))
    _patch_get(monkeypatch)
    ctx = traffic.get_traffic_context(SPOT)
    assert ctx["live_congestion"] == 0.5     # cache dead, API still served


def test_default_flow_cannot_be_mutated_by_a_caller(monkeypatch):
    def boom(*a, **k):
        raise TimeoutError
    monkeypatch.setattr(traffic._session, "get", boom)
    traffic.get_traffic_context(SPOT)["live_congestion"] = 999
    assert traffic._DEFAULT_FLOW["live_congestion"] == 0.0


def test_api_failure_is_negative_cached(monkeypatch):
    """An outage must not mean N timeouts per request forever."""
    counter = _patch_get(monkeypatch)

    def boom(*a, **k):
        counter["n"] += 1
        raise TimeoutError
    monkeypatch.setattr(traffic._session, "get", boom)
    traffic.get_traffic_context(SPOT)
    n_after_first = counter["n"]
    traffic.get_traffic_context(SPOT)
    assert counter["n"] == n_after_first     # second call served from neg-cache


# --- the contract itself --------------------------------------------------

def test_model_features_are_in_the_feature_contract(monkeypatch):
    _patch_get(monkeypatch)
    ctx = traffic.get_traffic_context(SPOT)
    for k in ("live_congestion", "traffic_delay_minutes"):
        assert k in FEATURE_COLUMNS
    assert "incidents" not in FEATURE_COLUMNS   # sidebar only, never a feature
