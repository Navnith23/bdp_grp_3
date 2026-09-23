"""
Two separate regression models instead of one shared model:

  - LAND model   : land_area_cents is the dominant feature; built_up_sqft
                    etc. don't apply (they're structurally 0 for land).
  - BUILDING model: apartment / house / villa / commercial / other, where
                    built_up_sqft is the dominant feature.

Splitting like this stops land rows from being scored by a model whose
#1 feature (built_up_sqft) is always 0 for them, and vice versa.

Run:  python src/train_model_split.py
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/cleaned.csv"
MODELS_DIR = "models"


# ----------------------------------------------------------------- split
def split_land_building(df: pd.DataFrame):
    """Row's property type comes back from the one-hot columns, since
    they were dropped with drop_first (apartment is the all-zero baseline)."""
    is_land = df["type_land"] == 1
    land = df[is_land].drop(columns=["type_land"])
    building = df[~is_land].drop(columns=["type_land"])
    return land, building


LAND_ONLY_COLS = ["built_up_sqft", "built_up_sqft_missing", "built_up_sqft_capped",
                   "bedrooms", "bedrooms_missing", "bedrooms_capped",
                   "bathrooms", "bathrooms_capped", "property_age",
                   "property_age_missing", "property_age_capped"]


def make_xy(df: pd.DataFrame, drop_building_only: bool):
    """drop_building_only=True for the land model: those columns are
    structurally 0 for every land row, so they carry no signal there."""
    cols_to_drop = ["price", "log_price"]
    if drop_building_only:
        cols_to_drop += LAND_ONLY_COLS
    X = df.drop(columns=cols_to_drop)
    y = df["log_price"]
    return X, y


# ----------------------------------------------------------------- train
def train_and_compare(X_train, y_train, X_test, y_test, label: str):
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42
        ),
    }

    y_test_price = np.expm1(y_test)
    rows = []
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_price = np.expm1(model.predict(X_test))
        preds[name] = pred_price
        ape = np.abs((y_test_price - pred_price) / y_test_price) * 100
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test_price, pred_price),
            "RMSE": np.sqrt(mean_squared_error(y_test_price, pred_price)),
            "R2": r2_score(y_test_price, pred_price),
            "MdAPE (%)": np.median(ape),
        })

    comparison = pd.DataFrame(rows)
    print(f"\n=== {label} model comparison (n_train={len(X_train)}, n_test={len(X_test)}) ===")
    print(comparison.to_string(index=False))

    best_name = comparison.sort_values("MdAPE (%)").iloc[0]["Model"]
    best_model = models[best_name]
    print(f"Selected: {best_name} (lowest MdAPE)")
    return best_model, best_name, comparison


# ----------------------------------------------------------------- plots
def plot_actual_vs_predicted(y_test_price, pred_price, label: str):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_price, pred_price, alpha=0.5)
    lo, hi = min(y_test_price.min(), pred_price.min()), max(y_test_price.max(), pred_price.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual Price (₹)")
    plt.ylabel("Predicted Price (₹)")
    plt.title(f"Actual vs Predicted - {label}")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, X_test, y_test, label: str):
    perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    fi = pd.DataFrame({"Feature": X_test.columns, "Importance": perm.importances_mean})
    fi = fi.sort_values("Importance", ascending=False).head(15)
    print(f"\nTop features - {label}")
    print(fi.to_string(index=False))

    plt.figure(figsize=(10, 7))
    plt.barh(fi["Feature"][::-1], fi["Importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top 15 Feature Importance (Permutation) - {label}")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------- main
def main():
    df = pd.read_csv(DATA_PATH)
    print("Full dataset shape:", df.shape)

    land_df, building_df = split_land_building(df)
    print(f"Land rows: {len(land_df)}   Building rows: {len(building_df)}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}

    for name, subset, drop_building_only in [("Land", land_df, True), ("Building", building_df, False)]:
        X, y = make_xy(subset, drop_building_only)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        model, best_name, comparison = train_and_compare(X_train, y_train, X_test, y_test, name)

        y_test_price = np.expm1(y_test)
        pred_price = np.expm1(model.predict(X_test))
        plot_actual_vs_predicted(y_test_price, pred_price, name)
        plot_feature_importance(model, X_test, y_test, name)

        model_file = f"{MODELS_DIR}/house_price_model_{name.lower()}.pkl"
        features_file = f"{MODELS_DIR}/model_features_{name.lower()}.pkl"
        joblib.dump(model, model_file)
        joblib.dump(X.columns.tolist(), features_file)
        print(f"Saved: {model_file}, {features_file}")

        results[name] = {"comparison": comparison, "best": best_name}

    print("\n=== Summary ===")
    for name, r in results.items():
        best_row = r["comparison"].set_index("Model").loc[r["best"]]
        print(f"{name:10} best={r['best']:22} MdAPE={best_row['MdAPE (%)']:.1f}%  R2={best_row['R2']:.3f}")


if __name__ == "__main__":
    main()
