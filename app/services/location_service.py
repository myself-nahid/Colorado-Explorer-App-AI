import googlemaps
from app.core.config import settings

class LocationService:
    def __init__(self):
        self.gmaps = googlemaps.Client(key=settings.GOOGLE_MAPS_API_KEY)

    def search_places_in_colorado(self, query: str) -> list:
        """
        Searches for places using the Google Maps Places API, strongly biased to Colorado.
        """

        geocode_result = self.gmaps.geocode('Colorado, USA')
        if not geocode_result:
            return []

        location_bias = geocode_result[0]['geometry']['bounds']

        places_result = self.gmaps.places(
            query=f"{query} in Colorado",
            region='US'
        )

        return places_result.get('results', [])

location_service = LocationService()