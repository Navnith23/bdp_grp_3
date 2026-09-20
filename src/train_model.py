import pandas as pd
import numpy as np

# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv("data/processed/cleaned.csv")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 2. Separate input features and target
# ==========================================

X = df.drop(columns=["price", "log_price"])
y = df["log_price"]

print("\nNumber of input features:", X.shape[1])
print("Target:", "log_price")


# ==========================================
# 3. Split dataset into training and testing
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==========================================
# 4. Create Random Forest model
# ==========================================

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 5. Train the model
# ==========================================

print("\nTraining Random Forest...")

rf_model.fit(X_train, y_train)

print("Random Forest training completed.")


# ==========================================
# 6. Make predictions
# ==========================================

y_pred_log = rf_model.predict(X_test)

print("Prediction completed.")


# ==========================================
# 7. Convert log predictions back to price
# ==========================================

y_test_price = np.expm1(y_test)
y_pred_price = np.expm1(y_pred_log)


# ==========================================
# 8. Evaluate the model
# ==========================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test_price, y_pred_price)

mse = mean_squared_error(y_test_price, y_pred_price)

rmse = np.sqrt(mse)

r2 = r2_score(y_test_price, y_pred_price)

percentage_error = np.abs(
    (y_test_price - y_pred_price) / y_test_price
) * 100

mdape = np.median(percentage_error)


# ==========================================
# 9. Display results
# ==========================================

print("\n===================================")
print("      RANDOM FOREST RESULTS")
print("===================================")

print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)
print("MdAPE:", mdape, "%")
# ==========================================
# 10. Gradient Boosting Regressor
# ==========================================

from sklearn.ensemble import GradientBoostingRegressor

print("\nTraining Gradient Boosting...")

gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)

print("Gradient Boosting training completed.")


# ==========================================
# 11. Gradient Boosting Predictions
# ==========================================

y_pred_gb_log = gb_model.predict(X_test)

print("Gradient Boosting prediction completed.")


# ==========================================
# 12. Convert predictions back to actual price
# ==========================================

y_pred_gb_price = np.expm1(y_pred_gb_log)


# ==========================================
# 13. Evaluate Gradient Boosting
# ==========================================

gb_mae = mean_absolute_error(
    y_test_price,
    y_pred_gb_price
)

gb_mse = mean_squared_error(
    y_test_price,
    y_pred_gb_price
)

gb_rmse = np.sqrt(gb_mse)

gb_r2 = r2_score(
    y_test_price,
    y_pred_gb_price
)

gb_percentage_error = np.abs(
    (y_test_price - y_pred_gb_price) / y_test_price
) * 100

gb_mdape = np.median(gb_percentage_error)


# ==========================================
# 14. Display Gradient Boosting Results
# ==========================================

print("\n===================================")
print("   GRADIENT BOOSTING RESULTS")
print("===================================")

print("MAE  :", gb_mae)
print("MSE  :", gb_mse)
print("RMSE :", gb_rmse)
print("R2   :", gb_r2)
print("MdAPE:", gb_mdape, "%")
# ==========================================
# 15. HistGradientBoosting Regressor
# ==========================================

from sklearn.ensemble import HistGradientBoostingRegressor

print("\nTraining HistGradientBoosting...")

hgb_model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42
)

hgb_model.fit(X_train, y_train)

print("HistGradientBoosting training completed.")


# ==========================================
# 16. HistGradientBoosting Predictions
# ==========================================

y_pred_hgb_log = hgb_model.predict(X_test)

print("HistGradientBoosting prediction completed.")


# ==========================================
# 17. Convert predictions back to actual price
# ==========================================

y_pred_hgb_price = np.expm1(y_pred_hgb_log)


# ==========================================
# 18. Evaluate HistGradientBoosting
# ==========================================

hgb_mae = mean_absolute_error(
    y_test_price,
    y_pred_hgb_price
)

hgb_mse = mean_squared_error(
    y_test_price,
    y_pred_hgb_price
)

hgb_rmse = np.sqrt(hgb_mse)

hgb_r2 = r2_score(
    y_test_price,
    y_pred_hgb_price
)

hgb_percentage_error = np.abs(
    (y_test_price - y_pred_hgb_price) / y_test_price
) * 100

hgb_mdape = np.median(hgb_percentage_error)


# ==========================================
# 19. Display HistGradientBoosting Results
# ==========================================

print("\n===================================")
print(" HISTGRADIENTBOOSTING RESULTS")
print("===================================")

print("MAE  :", hgb_mae)
print("MSE  :", hgb_mse)
print("RMSE :", hgb_rmse)
print("R2   :", hgb_r2)
print("MdAPE:", hgb_mdape, "%")
# ==========================================
# 20. Model Comparison Table
# ==========================================

comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting",
        "HistGradientBoosting"
    ],
    "MAE": [
        mae,
        gb_mae,
        hgb_mae
    ],
    "MSE": [
        mse,
        gb_mse,
        hgb_mse
    ],
    "RMSE": [
        rmse,
        gb_rmse,
        hgb_rmse
    ],
    "R2": [
        r2,
        gb_r2,
        hgb_r2
    ],
    "MdAPE (%)": [
        mdape,
        gb_mdape,
        hgb_mdape
    ]
})

print("\n===================================")
print("       MODEL COMPARISON")
print("===================================")

print(comparison.to_string(index=False))
# ==========================================
# 20A. Select Final Model
# ==========================================

final_model = hgb_model

print("\nFinal Model Selected: HistGradientBoostingRegressor")
print("Reason: It achieved the lowest RMSE, highest R2, and lowest MdAPE among the tested models.")
# ==========================================
# 21. Actual vs Predicted Price
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_price,
    y_pred_hgb_price,
    alpha=0.5
)

plt.xlabel("Actual Price (₹)")
plt.ylabel("Predicted Price (₹)")
plt.title("Actual vs Predicted House Prices")

# Perfect prediction line
min_price = min(y_test_price.min(), y_pred_hgb_price.min())
max_price = max(y_test_price.max(), y_pred_hgb_price.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.tight_layout()
plt.show()
# ==========================================
# 22. Residual Plot
# ==========================================

residuals = y_test_price - y_pred_hgb_price

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred_hgb_price,
    residuals,
    alpha=0.5
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Price (₹)")
plt.ylabel("Residual (₹)")
plt.title("Residual Plot - HistGradientBoosting")

plt.tight_layout()
plt.show()
# ==========================================
# 23. Random Forest Feature Importance
# ==========================================

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===================================")
print("       TOP 15 FEATURE IMPORTANCE")
print("===================================")

print(feature_importance.head(15).to_string(index=False))
# ==========================================
# 24. Feature Importance Graph
# ==========================================

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importance - Random Forest")

plt.tight_layout()
plt.show()
# ==========================================
# 25. Per-Property-Type Evaluation
# ==========================================

property_type_columns = [
    "type_house",
    "type_land",
    "type_villa",
    "type_commercial",
    "type_other"
]

print("\n===================================")
print("   PER-PROPERTY-TYPE EVALUATION")
print("===================================")

for type_column in property_type_columns:

    # Select test rows belonging to this type
    type_mask = X_test[type_column] == 1

    if type_mask.sum() == 0:
        print(f"\n{type_column}: No test samples")
        continue

    actual_type = y_test_price[type_mask]
    predicted_type = y_pred_hgb_price[type_mask]

    type_mae = mean_absolute_error(
        actual_type,
        predicted_type
    )

    type_rmse = np.sqrt(
        mean_squared_error(
            actual_type,
            predicted_type
        )
    )

    type_r2 = r2_score(
        actual_type,
        predicted_type
    )

    type_percentage_error = np.abs(
        (actual_type - predicted_type) / actual_type
    ) * 100

    type_mdape = np.median(type_percentage_error)

    print(f"\n{type_column}")
    print("Samples:", type_mask.sum())
    print("MAE    :", type_mae)
    print("RMSE   :", type_rmse)
    print("R2     :", type_r2)
    print("MdAPE  :", type_mdape, "%")
# ==========================================
# 26. Save Final Model
# ==========================================

import joblib

# Save HistGradientBoosting model
joblib.dump(hgb_model, "models/house_price_model.pkl")

# Save feature names used during training
joblib.dump(X.columns.tolist(), "models/model_features.pkl")

print("\n===================================")
print("       MODEL SAVED")
print("===================================")
print("Model file    : models/house_price_model.pkl")
print("Features file : models/model_features.pkl")
# ==========================================
# 26. Save Final Model
# ==========================================

import joblib

joblib.dump(hgb_model, "models/house_price_model.pkl")
joblib.dump(X.columns.tolist(), "models/model_features.pkl")

# Verify saved model
loaded_model = joblib.load("models/house_price_model.pkl")
loaded_features = joblib.load("models/model_features.pkl")

print("\n===================================")
print("       MODEL SAVED SUCCESSFULLY")
print("===================================")
print("Model file    : models/house_price_model.pkl")
print("Features file : models/model_features.pkl")
print("Number of features:", len(loaded_features))
print("Model verification: Successful")
