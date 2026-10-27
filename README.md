# Croppy

Croppy is a crop recommendation system for farmers. Given climate and soil
conditions, it suggests the most suitable crops to grow.

## How it works
- Trained on the Kaggle Crop Recommendation Dataset (2,200 samples, 22 crops)
- Inputs: temperature, humidity, soil pH, rainfall (optionally N, P, K soil nutrients)
- Model: Random Forest classifier

## Results
- Full model (with N, P, K): ~99% accuracy
- Lite model (without N, P, K): 95% accuracy

Note: the dataset is very clean, so these numbers show the model learns the
data well, not that it will be this accurate on real farms.

## Planned features
- Enter a pincode to automatically fetch local temperature, humidity, rainfall and soil pH
- Show the top 3 recommended crops with confidence scores
- Web interface for farmers
- Upload soil test report to read N, P, K values

## Tech stack
Python, pandas, scikit-learn, Jupyter
