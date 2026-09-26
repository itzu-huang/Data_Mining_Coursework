# Model Validation

## Objective

Observe how tree complexity and parameter choices affect training and testing results.

## Dataset

Bank customer data referenced by the exercise; the CSV is not included.

## Methods

Holdout train/test comparison, a manual sweep of `min_samples_split`, tree leaf/depth inspection, and cross-validation.

## What I Practiced

Compare train and test accuracy, tune tree complexity, and use cross-validation to observe variation across folds.

## Key Observation

Increasing model complexity can improve training performance without guaranteeing a better held-out result; this is an observation from the coursework split, not a general law.

## Files

- `overfitting-parameter-tuning.py`