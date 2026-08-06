# Event context extractor.
# Queries campus_events for currently active events, computes distance from
# each candidate parking spot, and checks whether a high-attendance event
# falls along the user's commute corridor.

from geopy.distance import distance as geo_distance

NEAR_RADIUS_MI = 0.5
# This gets the context layer for the ML engine
def get_events_context(spot, user_origin, active_events) -> dict:
    spot_pt = (spot["lat"])
    # 
def _crosses_commute(orign, dest, events) -> bool:
    pass