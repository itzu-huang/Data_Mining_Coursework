# New Taipei City Building Permit Analysis

## Objective

Prepare building-permit records, classify whether permits are for public use, compare supervised models, and explore clusters.

## Dataset

New Taipei City Building Permit Records, published by the New Taipei City Public Works Bureau through Taiwan's Government Open Data platform: [Dataset 123639](https://data.gov.tw/dataset/123639). The report describes 14,243 original records and 13,072 cleaned records. The raw CSV is intentionally not included because it contains names, identifiers, and detailed addresses.

## Methods

Data cleaning, date and numeric transformations, category grouping, missing-value handling, encoding, Decision Tree, Random Forest, SVM, KNN, Voting, and K-Means.

## What I Practiced

Build a workflow from messy public-record fields through feature preparation, supervised comparison, and cluster profiling while keeping identifying fields out of model inputs.

## Key Observation

Random Forest showed strong testing performance among the supervised models in the final comparison. Soft Voting provided an ensemble comparison. K-Means was more useful for exploring cluster structure than as the primary classification method. These are qualitative summaries; original report versions contain conflicting metrics, so no exact accuracy is stated here and no version is designated authoritative.

## Files

- `src/`: sanitized Python copies
- `docs/README.md`: why original reports are retained locally
- `figures/README.md`: no standalone figure cleared for release
- `results/README.md`: no exact results published while versions conflict

The source scripts expect a local copy of the dataset at the original filename. Do not commit that raw file.