# OpenWeatherMap client.
# Fetches current conditions for a given lat/lon, maps OWM condition codes
# to a severity bucket (0=clear → 3=severe), and caches results in Redis
# with a 15-minute TTL.

import os, json, requests,redis
from schema import WEATHER_CONDITIONS

_redis = "" 
CACHE_TTL = 15 * 60

# Map OpenWeatherMap condition-code ranges -> our buckets -> a 0–3 severity.
"""
Open weather map ID - categorization

"""
def _bucket_and_severity(owm_id: int, precip_mm: float):
        if owm_id // 100 == 2:       return "thunderstorm", 3
        if owm_id // 100 == 6:       return "snow", 3 if precip_mm > 4 else 2
        if owm_id // 100 == 5:       return "rain", 2 if precip_mm > 4 else 1
        if owm_id in (800,):         return "clear", 0
        if 801 <= owm_id <= 804:     return "clouds", 0
        if owm_id >= 900:            return "extreme", 3
        return "clear", 0
def get_weather_context(lat: float, lon: float) -> dict:
    key = f"weather:{round(lat, 2)}:{round(lon, 2)}"   # round → one call per ~1km cell
    cached = _redis.get(key)
    if cached:
        return json.loads(cached)
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "units": "imperial",
                    "appid": os.environ["OPENWEATHER_API_KEY"]},
            timeout=3,
        )
        r.raise_for_status()
        d = r.json()
        owm_id = d["weather"][0]["id"]
        precip = d.get("rain", {}).get("1h", 0) or d.get("snow", {}).get("1h", 0)
        bucket, severity = _bucket_and_severity(owm_id, precip)
        ctx = {
            "temp_f": d["main"]["temp"],
            "precip_intensity": precip,
            "wind_speed_mph": d["wind"]["speed"],
            "weather_severity": severity,
            "condition_code": WEATHER_CONDITIONS[bucket],
        }
        _redis.setex(key, CACHE_TTL, json.dumps(ctx))
        return ctx
    except Exception:
        return _DEFAULT_WEATHER          # fail soft — clear-day defaults

_DEFAULT_WEATHER = {"temp_f": 60.0, "precip_intensity": 0.0, "wind_speed_mph": 5.0,
                    "weather_severity": 0, "condition_code": WEATHER_CONDITIONS["clear"]}