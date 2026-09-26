# Support Vector Machines

## Objective

Practice SVM classification and evaluate class-sensitive performance.

## Dataset

Bank customer data referenced by the exercise; the CSV is not included.

## Methods

StandardScaler, LinearSVC, the `C` parameter, class weighting, and weighted F1. SVC with an RBF kernel is also compared in the ensemble exercise.

## What I Practiced

Scale features for distance/margin-based models and inspect weighted F1 alongside accuracy when classes may be uneven.

## Key Observation

The exercise compares linear SVM settings; it does not establish a universal preference for one kernel or `C` value.

## Files

- `bank-svm.py`