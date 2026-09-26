# Data Mining Coursework

Statistics and Information Science · Fu Jen Catholic University

## About This Repository

Selected Data Mining exercises and an applied final project from Statistics and Information Science at Fu Jen Catholic University. Original class folders remain a local source archive and are excluded from the public coursework structure.

## Learning Progression

Data Handling & Preprocessing
↓
Decision Tree Foundations
↓
Feature Selection
↓
Validation & Overfitting
↓
Model Evaluation
↓
Class Imbalance
↓
KNN / SVM / Random Forest
↓
Ensemble Learning
↓
K-Means Clustering
↓
Final Project

## Coursework Index

- [Data Preprocessing](01-data-preprocessing/README.md)
- [Decision Tree](02-decision-tree/README.md)
- [Feature Selection](03-feature-selection/README.md)
- [Model Validation](04-model-validation/README.md)
- [Model Evaluation](05-model-evaluation/README.md)
- [Class Imbalance](06-class-imbalance/README.md)
- [KNN](07-knn/README.md)
- [SVM](08-svm/README.md)
- [Random Forest](09-random-forest/README.md)
- [Ensemble Learning](10-ensemble-learning/README.md)
- [K-Means Clustering](11-kmeans-clustering/README.md)

## Final Project

The [New Taipei City Building Permit project](final-project/new-taipei-building-permit/README.md) analyzes permit records. The raw dataset and original reports are not included. Conflicting report versions are documented without reproducing exact model accuracy.

## Key Learning Takeaways

- Accuracy alone can hide class-specific errors; confusion matrices, precision, recall, F1, and application costs provide additional context.
- Distance-based methods such as KNN require attention to feature scale.
- Fit preprocessing steps on training data, then apply the fitted transform to test data.
- Feature reduction or an ensemble does not necessarily improve testing performance in a particular coursework comparison.
- Clusters describe patterns in the analyzed data and do not establish causal relationships.

## Repository Structure

```text
01-data-preprocessing/
02-decision-tree/
03-feature-selection/
04-model-validation/
05-model-evaluation/
06-class-imbalance/
07-knn/
08-svm/
09-random-forest/
10-ensemble-learning/
11-kmeans-clustering/
final-project/new-taipei-building-permit/
README.md
```