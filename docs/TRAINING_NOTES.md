# Kerala Real-Estate Price Model: Notes for Training

Read this before training. It covers what the cleaned data contains, what was done to it, and what to watch out for.

- **Source:** `final.json`, about 8,000 scraped listings from keralarealestate.com
- **Script:** `preprocess.py` (`python preprocess.py final.json cleaned.csv`)
- **Output:** `cleaned.csv`, 5,732 rows x 88 columns, all numeric, no NaNs, no duplicate rows

## 1. Target

- Predict **`log_price`** (= log1p of `price`). Raw `price` is heavily skewed (skew about 8); the log is close to symmetric (skew about -0.6).
- Price is in rupees. Range is 1 lakh to 50 crore, median about 72 lakh.
- **Do not use `price` as a feature.** It is the target in raw form. Drop it (or `log_price`) from X.

## 2. What the pipeline did (rows: 8,072 -> 5,732)

| Step | Rows dropped | Why |
|---|---|---|
| Rent-only listings | 5 | Price is rent, not a sale price |
| No price | 2,115 | No target |
| Price under Rs 1 lakh | 95 | Almost surely per-cent prices, rent or typos |
| Duplicate URL / title+price+place | 21 | Re-scrapes |
| Implausible price | 98 | Price impossible for the listing's own area (see below) |
| Duplicate listings | 5 | Identical on every model-relevant field |
| Identical after encoding | 1 | Two listings that became identical after imputation |

**Plausibility bounds** (fixed domain rules, not learned from data):
- Land: Rs 2,000 to Rs 1 crore per cent (69 rows below, 6 above)
- Apartments: Rs 750 to Rs 40,000 per sq ft (none dropped)
- Houses and villas: at least Rs 750 per sq ft, no upper bound, because the price includes land and high values are often legitimate (23 rows dropped)

Legitimately expensive houses and large plots were **kept on purpose**. Price is not capped anywhere.

## 3. Columns

- `price`, `log_price`: target (see above)
- `listing_year`: year the listing was posted (2013 to 2026). It is a real market signal; prices move over time.
- `land_area_cents`, `built_up_sqft`, `bedrooms`, `bathrooms`, `parking`, `property_age`: numeric features
- `*_missing` (4 flags: land area, built-up area, property age, bedrooms/bathrooms): 1 = value was imputed
- `*_capped` (6 flags): 1 = value was winsorised at the 99th percentile
- `n_amenities`, `amen_*` (49 columns): amenity count and flags for amenities present in at least 3% of listings
- `locality_freq`: how common the locality is (frequency encoding; 0 if locality unknown)
- `dist_*` (13) and `type_*` (5): one-hot with `drop_first=True`. **Baselines are Alappuzha (district) and apartment (type).** An apartment is all zeros in `type_*`.

Property type comes from title keywords (apartment, villa, house, land, commercial, other), not from a source field.

## 4. Imputation rules to be aware of

- **Structural zeros, not missing:** plots of land get 0 for built-up area, bedrooms, bathrooms and age; apartments get 0 land area. These have no missing flag.
- Everything else missing is filled with the district x property-type median (only if that group has at least 5 observed values), then the type median, then the global median.
- Land area is missing (imputed) for about 61% of rows and property age for about 56%. Most of these are houses and apartments where the field was never listed. **The imputed values carry little signal; the `_missing` flags matter more.**
- Parking not listed = 0.

## 5. Leakage: what still needs fixing before a real split

Steps 1 to 6 of the script are fixed-rule cleaning and are safe to run before splitting. Steps 7 to 9 **learn statistics from all rows**:

- the 99th-percentile caps
- the grouped medians used for imputation
- `locality_freq`
- the choice of which amenities are frequent enough to keep

If you split into train and test, refit these on the **training set only** and apply them to the test set. This is not done yet; the script currently fits everything on the full data, so a split made on `cleaned.csv` leaks a little. The clean approach is to restructure steps 7 to 9 into fit/transform functions.

Other points:
- Do the split **after** de-duplication (already done), so copies of one listing can't land on both sides.
- Consider a time-aware or stratified split. Listings span 2013 to 2026 and district/type mix varies a lot.

## 6. Known limits of the data

- **Prices inside the plausible range can still be wrong**, especially land (per-cent price typed as the total, or acre values typed as cents). The bounds catch only the impossible ones (about 2% of rows). Expect some irreducible noise.
- **Some built-up areas look wrong** (for example a "4 BHK" of 500 sq ft). This weakens the feature on a few houses; it does not corrupt the target.
- Type mix: house 2,572, land 1,702, apartment 619, villa 559, commercial 195, other 85. "Other" and "commercial" are small, so expect weak results there.
- District counts are uneven (Ernakulam is by far the largest; Kasargod has under 20 rows).
- Property type is inferred from titles and can be wrong on unusual titles.

## 7. Suggestions for training and evaluation

- Use `log_price` as the target and convert predictions back with `expm1`.
- Prefer tree-based models, or a robust loss (for example Huber), since a few bad prices remain.
- Report metrics **per property type**, not just overall. Land will probably be the worst.
- Report a percentage-based metric (median absolute percentage error) alongside RMSE. A few bad rows dominate squared error.
- If results look far worse than expected, check the largest residuals against the original listings before tuning; the cause is likely bad source data, mostly land rows.
- For plain linear regression, note that `bedrooms` and `bathrooms` are strongly correlated and that `type_land` correlates strongly with bedrooms and age.
- Model training has **not** been started; nothing here has been fitted or evaluated.
