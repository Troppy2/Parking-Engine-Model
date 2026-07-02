# OpenWeatherMap client.
# Fetches current conditions for a given lat/lon, maps OWM condition codes
# to a severity bucket (0=clear → 3=severe), and caches results in Redis
# with a 15-minute TTL.
