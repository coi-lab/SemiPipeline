# SemiPipeline

A semiconductor manufacturing machine learning project for detecting failed wafer runs from SECOM sensor data.

## Status

Active development

## Overview

SemiPipeline explores how sensor readings from a semiconductor manufacturing process can be cleaned, transformed, and modeled to predict whether a wafer run passes or fails.

The project uses the SECOM dataset, which contains hundreds of anonymized sensor measurements and a highly imbalanced pass/fail label. The main challenge is not raw accuracy: most samples pass, so a model can look accurate while missing the rare failures. The project therefore focuses on failure-class precision, recall, threshold tuning, feature selection, and leakage-aware preprocessing.

## Research Question

Can sensor data from semiconductor manufacturing runs be used to detect failed wafer runs with useful precision and recall, despite class imbalance and missing sensor values?

## Methodology

The project currently uses notebook-based experimentation:

- Load raw SECOM sensor and label files
- Convert labels from `-1` / `1` into `0` / `1`
- Remove columns with more than 50% missing values
- Remove constant or near-dead sensor columns
- Split data into train, validation, and test sets
- Fit preprocessing only on training data to reduce data leakage
- Scale features with `MinMaxScaler`
- Impute missing values with `KNNImputer`
- Select important sensors with `SelectKBest` or `SelectFromModel`
- Train XGBoost classifiers with class weighting
- Tune decision thresholds using validation-set precision/recall behavior
- Evaluate final model performance on an isolated test set
- Save a selected model pipeline as a `.pkl` asset
- Explore equivalent data cleaning logic in PySpark

## Data

Raw data is stored in:

```txt
SemiPipeline/data/raw/
|-- secom.data
`-- secom_labels.data
```

Current dataset shape after combining labels and sensor readings:

```txt
Rows: 1,567
Columns: 592
Pass labels: 1,463
Fail labels: 104
```

After removing high-missing and constant columns:

```txt
Cleaned shape: 1,567 rows x 448 columns
```

## Models

The main modeling approach uses:

- XGBoost binary classification
- Class weighting through `scale_pos_weight`
- Feature selection with `SelectKBest` and `SelectFromModel`
- Grid search using average precision / AUCPR scoring
- Threshold tuning using validation precision-recall curves
- A scikit-learn `Pipeline` for preprocessing, feature selection, and model training

The current saved model artifact is:

```txt
SemiPipeline/notebooks/secom_xgboost_production_v1.pkl
```

## Results

Early baseline models achieved high overall accuracy but failed to detect the minority failure class. For example, one default XGBoost run reached about 93% accuracy while predicting zero failures.

The later leakage-aware tuned pipeline produced the following saved test-set result:

```txt
Accuracy: 0.88
Fail-class precision: 0.24
Fail-class recall: 0.38
Fail-class F1-score: 0.29
Average Precision / AUCPR: 0.2488
```

These results show improvement over the baseline because the model begins catching failed runs, but performance is still limited by the small number of failure examples.

## Tech Stack

- Language: Python
- Libraries: pandas, NumPy, scikit-learn, XGBoost, joblib
- Distributed processing: PySpark
- Environment: Jupyter notebooks
- Data format: raw `.data` files and serialized `.pkl` model artifact

## Folder Structure

```txt
SemiPipeline/
|-- data/
|   `-- raw/
|       |-- secom.data
|       `-- secom_labels.data
|-- notebooks/
|   |-- 01_data_ingestion.ipynb
|   |-- 02_final_data_ingestion.ipynb
|   |-- 03_pyspark_pipeline.ipynb
|   `-- secom_xgboost_production_v1.pkl
`-- README.md
```

## Setup

Install the main Python dependencies:

```bash
python -m pip install pandas numpy scikit-learn xgboost joblib
```

For the PySpark notebook, install PySpark and make sure Java 17 is available:

```bash
python -m pip install pyspark
java -version
```

The PySpark notebook was tested with Java 17. If Spark hangs during startup or throws a Java gateway error, confirm that `JAVA_HOME` points to a JDK 17 installation and restart the notebook kernel.

## Usage

Open the notebooks in Jupyter or VS Code:

```txt
SemiPipeline/notebooks/01_data_ingestion.ipynb
SemiPipeline/notebooks/02_final_data_ingestion.ipynb
SemiPipeline/notebooks/03_pyspark_pipeline.ipynb
```

Recommended order:

1. Run `01_data_ingestion.ipynb` to see the exploratory cleaning and modeling process.
2. Run `02_final_data_ingestion.ipynb` for the cleaner final pipeline and saved model artifact.
3. Run `03_pyspark_pipeline.ipynb` to inspect the PySpark-based distributed cleaning experiment.

## Current Progress

Completed so far:

- Raw SECOM data ingestion
- Label conversion from pass/fail encoding
- Missing-value and constant-column analysis
- KNN imputation workflow
- XGBoost baseline modeling
- Feature selection experiments
- Validation-set threshold tuning
- Grid search using average precision scoring
- Final model serialization with `joblib`
- PySpark ingestion and distributed column audit

The PySpark notebook currently reproduces the cleaned shape of the pandas workflow:

```txt
Found 144 dead or heavily missing sensor columns.
Cleaned Distributed Shape: (1567, 448)
```

## Roadmap

- [ ] Move repeated notebook logic into reusable Python scripts
- [ ] Add a `requirements.txt` or environment file
- [ ] Add tests for preprocessing and leakage-safe pipeline behavior
- [ ] Compare additional models against XGBoost
- [ ] Tune threshold selection based on project-specific cost of false alarms vs missed failures
- [ ] Improve PySpark row alignment instead of relying on `monotonically_increasing_id()`
- [ ] Document how to load and use the saved `.pkl` model

## Limitations

- The dataset is highly imbalanced, with only 104 failure examples.
- Accuracy alone is misleading because most samples are passes.
- Current model performance is still modest for the failure class.
- Some exploratory notebook cells use older approaches that were later improved.
- The PySpark notebook is experimental and still needs safer row-index alignment.
- The saved model should be treated as a learning artifact, not a production-quality manufacturing system.

## Lessons Learned

This project highlights the importance of leakage-aware preprocessing, validation-set threshold tuning, and choosing metrics that match the actual problem. For rare failure detection, a high accuracy score can hide a model that misses every failure.

It also shows that feature selection and class weighting can help, but the main challenge remains the small number of failure examples and the tradeoff between catching more failures and creating more false alarms.

## Notes

- The raw data files are required for the notebooks to run.
- The `.pkl` file is a serialized model artifact generated from the notebook workflow.
- PySpark requires a working Java installation.
- Several notebooks contain exploratory code and saved outputs, so the final pipeline should be preferred for cleaner reproduction.
