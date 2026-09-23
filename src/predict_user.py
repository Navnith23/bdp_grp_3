import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Load cleaned dataset
#
# Model and feature list are loaded further down, once we know
# whether this is a land or building property (they use different
# feature sets - see train_model_split.py).
# --------------------------------------------------
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
    "Alappuzha",
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
    "apartment",
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
# Route to the land model or the building model
# --------------------------------------------------
is_land = property_type == "land"

if is_land:
    model = joblib.load("models/house_price_model_land.pkl")
    features = joblib.load("models/model_features_land.pkl")
else:
    model = joblib.load("models/house_price_model_building.pkl")
    features = joblib.load("models/model_features_building.pkl")

# --------------------------------------------------
# Structural zeros
#
# Matches preprocess.py: a plot of land has no building, and an
# apartment has no land area of its own.
# --------------------------------------------------
if property_type == "land":
    built_up_sqft = 0.0
    bedrooms = 0.0
    bathrooms = 0.0
    property_age = 0.0

if property_type == "apartment":
    land_area_cents = 0.0


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
# In training these are 0/1 flags (was this value beyond the 99th
# percentile seen in the data?), not the clipped value itself. The
# value fed to the model is also clipped to that same max, matching
# what cap_outliers() does in preprocess.py.
# --------------------------------------------------
capped_inputs = {
    "land_area_cents": land_area_cents,
    "built_up_sqft": built_up_sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "parking": parking,
    "property_age": property_age,
}

for col, value in capped_inputs.items():
    cap = df[col].max()
    row[col] = min(value, cap)
    row[f"{col}_capped"] = int(value > cap)


# --------------------------------------------------
# Number of amenities
#
# No amenities entered in this basic version, so use the dataset's
# typical (median) count rather than 0 - most listings have several.
# --------------------------------------------------
row["n_amenities"] = int(df["n_amenities"].median())


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

print(f"Model used    : {'land' if is_land else 'building'}")
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
print("Note: this is a rough estimate. Median error on the test set")
print("is roughly 30%, and higher for land, commercial and other.")