"""
Retrieval Agent — geo-aware doctor search using PostGIS.

Queries the structured database for doctors matching the specialty
near the user's location, with auto-expanding radius on zero results.
"""

from geopy.geocoders import Nominatim

from app.config import settings
from app.agents.state import AgentState
from app.db.session import get_session_factory
from app.db.queries import search_doctors_nearby


# Nominatim geocoder (free, OSM-based)
_geocoder = Nominatim(user_agent="remedy-radar-v0.1")


def _geocode_city(city: str) -> dict | None:
    """Convert a city name to lat/lng using Nominatim (free)."""
    try:
        result = _geocoder.geocode(city, timeout=10)
        if result:
            return {
                "lat": result.latitude,
                "lng": result.longitude,
                "city": city,
            }
    except Exception:
        pass
    return None


# Fallback coordinates for demo cities
CITY_COORDS = {
    "bangalore": {"lat": 12.9716, "lng": 77.5946},
    "bengaluru": {"lat": 12.9716, "lng": 77.5946},
    "san francisco": {"lat": 37.7749, "lng": -122.4194},
    "sf": {"lat": 37.7749, "lng": -122.4194},
}

RADIUS_EXPANSION_STEPS = [10_000, 15_000, 30_000, 50_000]  # meters


async def run_retrieval_agent(state: AgentState) -> AgentState:
    """
    Search for doctors matching the recommended specialty near the user.
    Auto-expands radius if no results found.
    """
    specialty = state.get("recommended_specialty", "General Practitioner")
    location = state.get("location", {})
    preferences = state.get("preferences", {})

    # Resolve location to coordinates
    lat = location.get("lat")
    lng = location.get("lng")

    if not lat or not lng:
        city = location.get("city", "")
        if city:
            # Try Nominatim first
            geo = _geocode_city(city)
            if geo:
                lat, lng = geo["lat"], geo["lng"]
            else:
                # Fallback to hardcoded demo cities
                coords = CITY_COORDS.get(city.lower().strip())
                if coords:
                    lat, lng = coords["lat"], coords["lng"]

    if not lat or not lng:
        # Default to Bangalore if no location resolved
        lat, lng = 12.9716, 77.5946
        state["location"] = {"lat": lat, "lng": lng, "city": "Bangalore (default)"}

    # Build structured filters from preferences
    filters = {}
    if preferences.get("gender"):
        filters["gender"] = preferences["gender"]
    if preferences.get("telehealth"):
        filters["telehealth"] = True
    if preferences.get("language"):
        filters["language"] = preferences["language"]
    if preferences.get("max_fee"):
        filters["max_fee"] = preferences["max_fee"]

    # Search with auto-expanding radius
    candidate_doctors = []
    radius_used = 0

    async_session_factory = get_session_factory()
    for radius_m in RADIUS_EXPANSION_STEPS:
        async with async_session_factory() as session:
            candidate_doctors = await search_doctors_nearby(
                session=session,
                lat=lat,
                lng=lng,
                specialty=specialty,
                radius_m=radius_m,
                limit=30,
                filters=filters,
            )
        radius_used = radius_m
        if candidate_doctors:
            break
            
    # Dispose the dynamically created engine to prevent resource leaks
    await async_session_factory.kw['bind'].dispose()

    state["candidate_doctors"] = candidate_doctors
    state["search_radius_used"] = radius_used / 1000  # Convert to km

    return state
