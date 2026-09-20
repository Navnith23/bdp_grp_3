import pandas as pd
import numpy as np
import joblib

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned.csv")

# Load trained model and feature list
model = joblib.load("models/house_price_model.pkl")
features = joblib.load("models/model_features.pkl")

# Select one existing property
sample = df.iloc[[0]][features]

# Predict log(price)
pred_log = model.predict(sample)[0]

# Convert back to actual price
pred_price = np.expm1(pred_log)

# Actual price
actual_price = df.iloc[0]["price"]

print("HOUSE PRICE PREDICTION")
print("======================")
print(f"Predicted Price : ₹{pred_price:,.2f}")
print(f"Actual Price    : ₹{actual_price:,.2f}")
