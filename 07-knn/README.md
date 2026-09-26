# K-Nearest Neighbors

## Objective

Practice distance-based classification on Bank customer data.

## Dataset

Bank customer data referenced by the exercise; the CSV is not included.

## Methods

One-Hot Encoding, StandardScaler for numeric features, and comparisons across K values.

## What I Practiced

KNN is sensitive to feature scale. Fit the scaler on training data and only transform test data with that fitted scaler.

## Key Observation

This coursework script selects K by comparing observed test-set performance. That is not rigorous hyperparameter tuning; the test set should not be reused for model selection.

## Files

- `bank-knn.py`