# Airbnb Pricing and Booking Prediction using Temporal Modeling

## Author
**Himani Ajit Mali**

---

## Project Overview
This project analyzes Airbnb listing data across multiple U.S. cities using two temporal snapshots (December 2024 and March 2025). The goal is to understand seasonality patterns in pricing and booking behavior, construct a night-level panel dataset, and build predictive models for:

1. Nightly price prediction (regression)
2. Booking probability prediction (classification)

The project emphasizes temporal feature engineering, time-aware data splits, and model comparison between tree-based and neural network approaches.

---

## Data Description
The dataset consists of Airbnb listings and calendar data for the following cities:
- Austin
- Chicago
- Santa Cruz
- Washington DC

Each city contains two snapshots collected at different points in time. Listings data includes structural attributes (room type, property type, location, reviews), while calendar data provides daily availability and pricing.

A merged night-level panel dataset is constructed for modeling.

---

## Feature Engineering
Key engineered features include:
- Cleaned nightly price (`price_clean`)
- Booking indicator (`is_booked`)
- Temporal features: month, day of week, week of year, day of year, weekend indicator
- Structural listing attributes (room type, property type, neighborhood)
- Review and capacity-related features

---

## Exploratory and Seasonality Analysis
Seasonality analysis compares:
- Monthly average price
- Monthly booking probability
- Weekend vs weekday price and booking behavior
- Price differences by room type

Findings show that booking probability exhibits stronger seasonal patterns than prices, while room type is the dominant driver of pricing.

---

## Modeling Approach

### Temporal Splits
Data is split chronologically to avoid leakage:
- Training: January–September
- Validation: October–November
- Test: December–February

### Models
Two modeling approaches are evaluated:
- **XGBoost**
  - Regression for price prediction
  - Classification for booking probability
- **Neural Networks**
  - Fully connected feed-forward networks
  - Trained with TensorBoard logging

Large datasets are handled using sparse one-hot encoding and controlled subsampling.

---

## Model Evaluation

### Price Prediction (Regression)
- XGBoost achieves lower RMSE and MAE than the neural network.
- Tree-based models better capture nonlinear relationships in listing features.

### Booking Prediction (Classification)
- XGBoost outperforms neural networks in both AUC and accuracy.
- Booking behavior is more sensitive to temporal features than pricing.

### Feature Importance
XGBoost feature importance highlights property type, room type, neighborhood, capacity, and review metrics as the strongest drivers of price.

---

## Key Conclusions
- Booking probability shows stronger seasonality than price.
- Room type and property characteristics dominate price determination.
- XGBoost consistently outperforms neural networks for both regression and classification tasks.
- Tree-based models offer better performance, interpretability, and computational efficiency for this application.

---

## Files Included
- `Untitled57.ipynb` – Full analysis notebook
- `README.md` – Project documentation

---

## Requirements
- Python 3
- pandas, numpy
- scikit-learn
- xgboost
- tensorflow / keras
- matplotlib
- Google Colab (recommended for large-scale execution)

---

## Notes
Due to dataset size, subsampling and sparse encodings are used to ensure stable execution in cloud environments.

