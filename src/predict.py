import pandas as pd
import numpy as np
import joblib

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned.csv")

# Select one existing property
row = df.iloc[[0]]

# --------------------------------------------------
# Route to the land model or the building model.
#
# type_land is dropped from both models' feature sets (train_model_split.py
# drops it from the land model explicitly, and it's never present in the
# building model since building = ~is_land), so it has to be read off the
# original row before selecting which model/feature file to load.
# --------------------------------------------------
is_land = row["type_land"].iloc[0] == 1

if is_land:
    model = joblib.load("models/house_price_model_land.pkl")
    features = joblib.load("models/model_features_land.pkl")
else:
    model = joblib.load("models/house_price_model_building.pkl")
    features = joblib.load("models/model_features_building.pkl")

sample = row[features]

# Predict log(price)
pred_log = model.predict(sample)[0]

# Convert back to actual price
pred_price = np.expm1(pred_log)

# Actual price
actual_price = row["price"].iloc[0]

print("HOUSE PRICE PREDICTION")
print("======================")
print(f"Model used      : {'land' if is_land else 'building'}")
print(f"Predicted Price : ₹{pred_price:,.2f}")
print(f"Actual Price    : ₹{actual_price:,.2f}")