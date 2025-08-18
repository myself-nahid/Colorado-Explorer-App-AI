import googlemaps
from app.core.config import settings
import json 

class LocationService:
    def __init__(self):
        self.gmaps = googlemaps.Client(key=settings.GOOGLE_MAPS_API_KEY)

    def search_places_in_colorado(self, query: str) -> list:
        """
        Searches for places using the Google Maps Places API, handles errors,
        and returns a simplified, clean list of results for the AI.
        """
        print(f"--- Received query for Google Maps: '{query}' ---")
        try:
            places_result = self.gmaps.places(
                query=f"{query} in Colorado",
                region='US'
            )

            print("--- Full Google Maps API Response ---")
            print(json.dumps(places_result, indent=2))
            print("------------------------------------")

            status = places_result.get('status')
            if status not in ['OK', 'ZERO_RESULTS']:
                print(f"Google Maps API returned an error status: {status}")
                return [] 

            if status == 'ZERO_RESULTS':
                print("Google Maps API returned ZERO_RESULTS for the query.")
                return []

            simplified_results = []
            for place in places_result.get('results', []):
                simplified_results.append({
                    'name': place.get('name'),
                    'address': place.get('formatted_address'),
                    'rating': place.get('rating', 'N/A'),
                    'user_ratings_total': place.get('user_ratings_total', 0)
                })
            
            print(f"--- Simplified and returning {len(simplified_results)} results. ---")
            return simplified_results

        except googlemaps.exceptions.ApiError as e:
            print(f"A Google Maps API Error occurred: {e}")
            return []
        except Exception as e:
            print(f"An unexpected exception occurred in LocationService: {e}")
            return []

location_service = LocationService()