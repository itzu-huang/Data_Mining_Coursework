# Data Preprocessing

## Objective

Build the data-cleaning and feature-representation foundation used by later classification exercises.

## Dataset

Bank missing-value data, a ProductSales exercise dataset, and an XML-format YouBike station feed. Source datasets are not copied into this repository.

## Methods

Missing-value removal and imputation with mean, median, and mode; equal-width and equal-frequency discretization; Label Encoding; scaling; and XML parsing into a DataFrame.

## What I Practiced

Use pandas and NumPy to inspect and transform CSV data, and ElementTree to parse structured XML.

## Key Observation

Preprocessing choices depend on feature type: numeric values can use mean/median or discretization, while categorical values need an appropriate category representation.

## Files

- `missing-value-discretization.py`
- `productsales-preprocessing.py`
- `xml-to-dataframe.py`

The input datasets and XML feed are intentionally not included.