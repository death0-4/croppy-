import time
from pathlib import Path

import joblib
import pandas as pd
import requests

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "crop_model_lite.joblib"
FEATURE_ORDER = ["temperature", "humidity", "ph", "rainfall"]
DEFAULT_PH = 6.5

_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def pincode_to_coords(pincode):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"postalcode": pincode, "country": "India", "format": "json", "limit": 1}
    headers = {"User-Agent": "croppy-student-project"}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"])


def get_weather(lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": "2025-01-01", "end_date": "2025-12-31",
        "daily": "temperature_2m_mean,precipitation_sum",
        "hourly": "relative_humidity_2m",
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()

    temp = pd.Series(j["daily"]["temperature_2m_mean"]).mean()
    humidity = pd.Series(j["hourly"]["relative_humidity_2m"]).mean()
    rain_monthly = pd.Series(j["daily"]["precipitation_sum"]).sum() / 12

    return {"temperature": float(round(temp, 1)),
            "humidity": float(round(humidity, 1)),
            "rainfall": float(round(rain_monthly, 1))}


def get_soil_ph(lat, lon):
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {"lat": lat, "lon": lon, "property": "phh2o",
              "depth": "0-5cm", "value": "mean"}
    for attempt in range(2):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            raw = r.json()["properties"]["layers"][0]["depths"][0]["values"]["mean"]
            return None if raw is None else round(float(raw) / 10, 1)
        except Exception as e:
            print("SoilGrids problem:", type(e).__name__)
            time.sleep(15)
    return None


def recommend(pincode, manual_ph=None):
    pincode = str(pincode).strip()
    if not (pincode.isdigit() and len(pincode) == 6):
        return {"error": "Pincode must be 6 digits"}

    if manual_ph is not None and not (0 <= manual_ph <= 14):
        return {"error": "pH must be between 0 and 14"}

    coords = pincode_to_coords(pincode)
    if coords is None:
        return {"error": "Pincode not found"}
    lat, lon = coords

    weather = get_weather(lat, lon)

    ph = manual_ph if manual_ph is not None else get_soil_ph(lat, lon)
    ph_estimated = False
    if ph is None:
        ph = DEFAULT_PH
        ph_estimated = True

    row = pd.DataFrame([{
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "ph": ph,
        "rainfall": weather["rainfall"],
    }], columns=FEATURE_ORDER)

    model = load_model()
    probs = model.predict_proba(row)[0]
    top3 = probs.argsort()[::-1][:3]
    return {
        "inputs": row.iloc[0].to_dict(),
        "ph_estimated": ph_estimated,
        "top3": [(str(model.classes_[i]), float(round(probs[i] * 100, 1))) for i in top3],
    }