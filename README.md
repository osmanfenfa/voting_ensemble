# voting_ensemble
Computer Science Final Year Data Mining and Machine Learning Project
Link: https://github.com/osmanfenfa/voting_ensemble

# Voting Ensemble ML Project

This project compares three classifiers on a classification dataset:
- Decision Tree
- K-Nearest Neighbors (KNN)
- Voting Ensemble (Decision Tree + KNN)

It also evaluates how outlier removal affects model performance using:
- Z-score method
- IQR method

## Project Structure

```text
voting_ensemble_ml_project/
|
|-- data/
|   |-- raw/
|   |   `-- dataset.csv
|   |
|   `-- processed/
|       |-- dataset_no_outliers_zscore.csv
|       `-- dataset_no_outliers_iqr.csv
|
|-- notebooks/
|   `-- exploration.ipynb
|
|-- src/
|   |-- data_preprocessing.py
|   |-- outlier_detection.py
|   |-- models.py
|   |-- ensemble_model.py
|   `-- evaluation.py
|
|-- results/
|   |-- model_results.txt
|   `-- accuracy_comparison.png
|
|-- main.py
|-- requirements.txt
`-- README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

If your dataset is at `data/raw/dataset.csv` and the target is the last column:

```bash
python main.py
```

Specify dataset path and target column explicitly:

```bash
python main.py --dataset-path data/raw/dataset.csv --target-column target
```

Optional parameters:
- `--zscore-threshold` (default: `3.0`)
- `--iqr-multiplier` (default: `1.5`)
- `--test-size` (default: `0.2`)
- `--random-state` (default: `42`)
- `--no-stratify` to disable stratified split

## Outputs

- `data/processed/dataset_no_outliers_zscore.csv`
- `data/processed/dataset_no_outliers_iqr.csv`
- `results/model_results.txt`
- `results/accuracy_comparison.png`

## Notes

- Outliers are detected using numeric feature columns only.
- The target column is excluded from outlier detection.
- If `--target-column` is not provided, the last column in the CSV is used.
