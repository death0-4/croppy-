# Croppy

Croppy is a crop recommendation system for farmers. A farmer enters a pincode, and Croppy automatically fetches the local climate and soil pH and suggests the three most suitable crops.

## How it works

1. **Pincode to location:** the pincode is converted to latitude/longitude using OpenStreetMap Nominatim.
2. **Climate data:** one year of past weather (2025) is fetched from the Open-Meteo API and averaged: temperature, humidity, and monthly rainfall.
3. **Soil pH:** fetched from the SoilGrids API. If no data is available, the farmer can enter pH manually (for example from a Soil Health Card), or a default of 6.5 is used and the result is marked as "estimated".
4. **Prediction:** a Random Forest model returns the top 3 crops with confidence scores.

## Model

- Dataset: Kaggle Crop Recommendation Dataset (2,200 samples, 22 crops)
- Full model (temperature, humidity, pH, rainfall, N, P, K): ~99% accuracy
- Lite model (temperature, humidity, pH, rainfall only): 95% accuracy
- The lite model is used by the pipeline because N, P, K cannot be fetched automatically.
- Weakest crop in the lite model: pomegranate (recall 0.75)

## Project structure

```
croppy/
├── data/            # dataset
├── notebooks/       # week1.ipynb (model training), week2.ipynb (pipeline)
├── models/          # saved .joblib models
├── src/
│   └── pipeline.py  # pincode -> weather + pH -> top 3 crops
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Testing

7 pytest tests with mocked APIs (no internet needed) cover:

- invalid pincode format
- pincode not found
- manual pH is used and skips SoilGrids
- missing pH falls back to 6.5
- feature column order matches training
- out-of-range pH is rejected
- top 3 output format

Run them with:

```
pytest -v
```

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Limitations

- The model only knows 22 crops, so it cannot recommend wheat, mustard, sugarcane, bajra and others.
- The dataset is very clean, so the accuracy numbers do not mean the same accuracy on real farms.
- SoilGrids often returns no value, so pH frequently falls back to 6.5.
- Confidence scores are the share of trees voting for a crop, not true probabilities.
- Weather is a yearly average, not a growing-season average.
- A pincode gives the area's center point, not the exact field.

## Planned features

- Streamlit website with pincode form and optional pH and N, P, K inputs
- Season selection (Kharif, Rabi, Zaid)
- Soil photo classifier
- Upload a Soil Health Card and read N, P, K with OCR
- Hindi/English support

## Tech stack

Python, pandas, scikit-learn, requests, pytest, Jupyter