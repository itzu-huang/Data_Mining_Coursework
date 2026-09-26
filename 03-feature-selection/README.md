# Feature Selection

## Objective

Compare classification with all available features against reduced feature sets.

## Dataset

Bank customer data referenced by the exercise; the CSV is not included.

## Methods

Chi-square scoring with SelectKBest and model-based feature importance from a Decision Tree.

## What I Practiced

Fit feature selection using training data, transform the held-out data with the selected features, and compare the resulting models.

## Key Observation

Reducing the number of features did not automatically improve testing performance in this coursework exercise.

## Files

- `feature-selection.py`