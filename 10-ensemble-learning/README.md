# Ensemble Learning

## Objective

Compare Decision Tree, Random Forest, KNN, and SVM with Hard Voting and Soft Voting.

## Dataset

Titanic training data referenced by the exercise; the CSV and original report are not included.

## Methods

Train/test split, categorical encoding, scaling for distance-based models, and scikit-learn VotingClassifier.

## What I Practiced

Hard Voting combines predicted labels; Soft Voting combines class probabilities and requires probability-capable base estimators.

## Key Observation

Ensemble methods did not automatically outperform the strongest individual model in the final project comparison. This is a result of that coursework comparison, not a general rule.

## Files

- `titanic-voting-comparison.py`