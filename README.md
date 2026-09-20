# Kerala House Price Prediction

Predicts the sale price of a Kerala property (house, villa, apartment, land, commercial) from listings scraped from keralarealestate.com. The pipeline cleans the raw scraped JSON, trains three regression models, and saves the best one for making predictions.

## Project structure

```
kerala-house-price/
├── data/
│   ├── raw/
│   │   └── final.json            scraped listings (input)
│   └── processed/
│       └── cleaned.csv           model-ready table (generated)
├── models/
│   ├── house_price_model.pkl     trained model
│   └── model_features.pkl        feature names, in training order
├── src/
│   ├── preprocess.py             raw JSON -> cleaned.csv
│   ├── train_model.py            trains and compares models, saves the best
│   ├── predict.py                predicts one row from the dataset
│   └── predict_user.py           interactive prediction from typed input
├── docs/
│   └── TRAINING_NOTES.md         details on the data and things to watch out for
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

`scikit-learn` is pinned to 1.9.1, the version that saved the model file. A different version may load it with warnings or give slightly different results. If you retrain, the pin no longer matters.

## How to run

Run every command from the project root folder.

1. **Preprocess the data** (skip if `cleaned.csv` already exists)
   ```bash
   python src/preprocess.py data/raw/final.json data/processed/cleaned.csv
   ```
2. **Train and compare models**
   ```bash
   python src/train_model.py
   ```
   Trains Random Forest, Gradient Boosting and Histogram Gradient Boosting, prints a comparison table, shows plots, and saves the final model to `models/`.
3. **Predict**
   ```bash
   python src/predict_user.py      # asks for area, bedrooms, district, type, etc.
   python src/predict.py           # predicts the first row of the dataset
   ```

## What the preprocessing does

`preprocess.py` takes about 8,000 scraped listings down to 5,732 clean rows in nine steps:

1. Load and flatten the JSON.
2. Parse text into numbers (price, land area in cents, built-up area in sq ft, bedrooms, bathrooms, property age from construction year, district, property type from the title).
3. Drop rent-only listings, rows with no price, prices under ₹1 lakh, and duplicate URLs.
4. Mark impossible feature values as missing.
5. Drop rows with an impossible price for their own area (per-cent bounds for land, per-sq-ft bounds for buildings).
6. Drop duplicate listings.
7. Cap the upper tail of numeric features at the 99th percentile. Price is not capped.
8. Fill missing values with district and property-type medians, and add `*_missing` flags.
9. Build features: amenity flags, locality frequency, `log_price`, and one-hot encoding of district and type.

The full details and reasoning are in `docs/TRAINING_NOTES.md`.

## Model

- **Target:** `log_price` (predictions are converted back with `expm1`).
- **Features:** 86, covering area, rooms, age, listing year, amenities, locality frequency, district and property type.
- **Split:** random 80/20, `random_state=42`.
- **Saved model:** Histogram Gradient Boosting (300 iterations, learning rate 0.05).

### Results on the test set

| Metric | Value |
|---|---|
| Median absolute percentage error | about 31% |
| Predictions within ±20% of actual | about 35% |
| R² on log price | about 0.56 |
| R² on price | about 0.26 |

Accuracy varies a lot by property type: about 23% median error for apartments and houses, 26% for villas, and much higher for land (66%), commercial (51%) and other (87%). Treat predictions as rough estimates, not valuations.

The three models scored very close to each other, so the choice between them makes little difference.

## Known limitations

- The data is scraped, and some prices and areas are wrong even after cleaning. Land listings are the noisiest.
- Caps, imputation medians and locality frequencies are computed on all rows, so the train/test split in `train_model.py` leaks a little information from the test rows. Refit these on the training set only for a strict evaluation.
- `listing_year` is one of the most important features, so predictions assume the price level of the year you enter (2026 by default).
- `predict_user.py` takes no amenities or locality input; it uses typical values, so its output is only a rough estimate.
- Neither prediction script has been validated against real market data outside the scraped listings.

## Data source

Listings scraped from keralarealestate.com. Check the site's terms before reusing or redistributing the data.
