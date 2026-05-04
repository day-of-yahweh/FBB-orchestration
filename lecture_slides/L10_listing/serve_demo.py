import requests
import json

def predict_expedition(expedition_data: dict):
    url = "http://127.0.0.1:5001/invocations"
    response = requests.post(url, json=expedition_data)
    return response.json()

everest_spring = {
    "dataframe_split": {
        "columns": ["season", "host", "total_members", "total_hired", "hired_ratio",
                    "o2_used", "o2_climb", "o2_sleep", "camps", "rope_fixed",
                    "height_m", "height_scaled", "is_8000er", "is_trekking_peak",
                    "year", "total_days", "summit_days", "standard_route",
                    "commercial_route", "solo", "small_team", "large_team"],
        "data": [[1, 1, 12, 15, 1.15, 1, 1, 1, 4, 8000, 8849, 8.849, 1, 0, 2024, 45, 35, 1, 1, 0, 0, 1]]
    }
}

solo_winter = {
    "dataframe_split": {
        "columns": ["season", "host", "total_members", "total_hired", "hired_ratio",
                    "o2_used", "o2_climb", "o2_sleep", "camps", "rope_fixed",
                    "height_m", "height_scaled", "is_8000er", "is_trekking_peak",
                    "year", "total_days", "summit_days", "standard_route",
                    "commercial_route", "solo", "small_team", "large_team"],
        "data": [[4, 1, 1, 0, 0, 0, 0, 0, 3, 2000, 8163, 8.163, 1, 0, 2024, 60, 50, 0, 0, 1, 0, 0]]
    }
}

print("Everest Spring Expedition:", predict_expedition(everest_spring))
print("Solo Winter 8000er:", predict_expedition(solo_winter))
