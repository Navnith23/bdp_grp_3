import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Load model, feature list, and cleaned dataset
# --------------------------------------------------
model = joblib.load("models/house_price_model.pkl")
features = joblib.load("models/model_features.pkl")
df = pd.read_csv("data/processed/cleaned.csv")


# --------------------------------------------------
# Get user input
# --------------------------------------------------
print("=" * 45)
print("       HOUSE PRICE PREDICTION")
print("=" * 45)

built_up_sqft = float(input("Built-up area (sqft): "))
land_area_cents = float(input("Land area (cents): "))
bedrooms = float(input("Number of bedrooms: "))
bathrooms = float(input("Number of bathrooms: "))
parking = float(input("Parking spaces: "))
property_age = float(input("Property age (years): "))

print("\nDistrict options:")
districts = [
    "Ernakulam",
    "Idukki",
    "Kannur",
    "Kasargod",
    "Kollam",
    "Kottayam",
    "Kozhikode",
    "Malappuram",
    "Palakkad",
    "Pathanamthitta",
    "Thiruvananthapuram",
    "Thrissur",
    "Wayanad"
]

for i, district in enumerate(districts, 1):
    print(f"{i}. {district}")

district_choice = int(input("Select district number: "))

if district_choice < 1 or district_choice > len(districts):
    raise ValueError("Invalid district selection.")

district = districts[district_choice - 1]

print("\nProperty type:")
property_types = [
    "commercial",
    "house",
    "land",
    "other",
    "villa"
]

for i, ptype in enumerate(property_types, 1):
    print(f"{i}. {ptype}")

type_choice = int(input("Select property type number: "))

if type_choice < 1 or type_choice > len(property_types):
    raise ValueError("Invalid property type selection.")

property_type = property_types[type_choice - 1]


# --------------------------------------------------
# Create one row
# --------------------------------------------------
row = {feature: 0 for feature in features}


# --------------------------------------------------
# Basic numerical features
# --------------------------------------------------
row["listing_year"] = 2026

row["land_area_cents"] = land_area_cents
row["built_up_sqft"] = built_up_sqft
row["bedrooms"] = bedrooms
row["bathrooms"] = bathrooms
row["parking"] = parking
row["property_age"] = property_age


# --------------------------------------------------
# Missing-value flags
# User supplied values, so these are 0
# --------------------------------------------------
row["land_area_cents_missing"] = 0
row["property_age_missing"] = 0
row["built_up_sqft_missing"] = 0
row["bedrooms_missing"] = 0


# --------------------------------------------------
# Capped features
#
# These limits are based on the maximum values
# currently present in cleaned.csv.
# --------------------------------------------------
row["land_area_cents_capped"] = min(
    land_area_cents,
    df["land_area_cents"].max()
)

row["built_up_sqft_capped"] = min(
    built_up_sqft,
    df["built_up_sqft"].max()
)

row["bedrooms_capped"] = min(
    bedrooms,
    df["bedrooms"].max()
)

row["bathrooms_capped"] = min(
    bathrooms,
    df["bathrooms"].max()
)

row["property_age_capped"] = min(
    property_age,
    df["property_age"].max()
)

row["parking_capped"] = min(
    parking,
    df["parking"].max()
)


# --------------------------------------------------
# Number of amenities
#
# No amenities entered in this basic version.
# --------------------------------------------------
row["n_amenities"] = 0


# --------------------------------------------------
# District one-hot encoding
# --------------------------------------------------
district_column = "dist_" + district

if district_column in row:
    row[district_column] = 1


# --------------------------------------------------
# Property type one-hot encoding
# --------------------------------------------------
type_column = "type_" + property_type

if type_column in row:
    row[type_column] = 1


# --------------------------------------------------
# Locality frequency
#
# We don't have the original locality name from
# this simple input form, so use the median value
# from the cleaned dataset as a neutral fallback.
# --------------------------------------------------
row["locality_freq"] = df["locality_freq"].median()


# --------------------------------------------------
# Amenity columns
#
# No amenities selected in this basic version,
# therefore all amenity flags remain 0.
# --------------------------------------------------
amenity_columns = [
    col for col in features
    if col.startswith("amen_")
]

for col in amenity_columns:
    row[col] = 0


# --------------------------------------------------
# Make sure feature order is EXACTLY the same
# as during model training
# --------------------------------------------------
input_df = pd.DataFrame([row])

input_df = input_df[features]


# --------------------------------------------------
# Prediction
# --------------------------------------------------
predicted_log_price = model.predict(input_df)[0]

predicted_price = np.expm1(predicted_log_price)


# --------------------------------------------------
# Display result
# --------------------------------------------------
print("\n" + "=" * 45)
print("           PREDICTION RESULT")
print("=" * 45)

print(f"District      : {district}")
print(f"Property type : {property_type}")
print(f"Built-up area : {built_up_sqft:,.0f} sqft")
print(f"Land area     : {land_area_cents:,.2f} cents")
print(f"Bedrooms      : {bedrooms:.0f}")
print(f"Bathrooms     : {bathrooms:.0f}")
print(f"Parking       : {parking:.0f}")
print(f"Property age  : {property_age:.0f} years")

print("-" * 45)
print(f"Predicted Price : ₹{predicted_price:,.2f}")
print("=" * 45)
