# End-to-End MLflow Experiment Tracking and Model Lifecycle

## Overview

This project demonstrates a **complete MLflow workflow** applied to the classic **Iris dataset** from scikit-learn. Two classification models — **Logistic Regression** and **Random Forest** — are trained, tracked, compared, and registered using MLflow's full feature set.

---

## Workflow Explanation

### Task 1 – Create MLflow Experiment and Start a Run
An MLflow experiment named `Iris_Classification_Experiment` is created with `mlflow.set_experiment()`. A parent run is started with `mlflow.start_run()` and the Run ID is printed to the console.

### Task 2 – Train Two ML Models
The Iris dataset (150 samples × 4 features, 3 classes) is split 80/20 (train/test, stratified). Two models are trained:
- **Logistic Regression** (`C=1.0`, `solver=lbfgs`, `max_iter=200`)
- **Random Forest** (`n_estimators=100`, `max_depth=5`, `random_state=42`)

### Task 3 – Log Parameters and Metrics
For each model run, the following are logged via `mlflow.log_param()` and `mlflow.log_metric()`:

| Model | Parameters Logged | Metrics Logged |
|---|---|---|
| Logistic Regression | C, solver, max_iter | accuracy, f1_score, precision, recall |
| Random Forest | n_estimators, max_depth, random_state | accuracy, f1_score, precision, recall |

### Task 4 – Artifact Logging
Two plots are generated and logged with `mlflow.log_artifact()`:
1. **Confusion Matrix** — per-model seaborn heatmap saved as PNG
2. **Performance Comparison Chart** — grouped bar chart comparing all four metrics across both models

### Task 5 – Model Logging
Each trained model is logged using `mlflow.sklearn.log_model()` with a descriptive artifact path (`logistic_regression_model` / `random_forest_model`). The model files appear under the **Artifacts** tab in the MLflow UI.

### Task 6 – Payload Validation for Inference
A `validate_payload(data)` function performs three checks before prediction:
1. **Shape check** — ensures the input has exactly 4 features
2. **NaN check** — ensures no missing values are present
3. **Type check** — ensures all values are numeric

A sample payload of two flowers is validated and predicted by both models. Invalid payloads (wrong feature count, NaN values) are demonstrated to show proper error raising.

### Task 7 – Model Registration
The best-performing model is registered in the **MLflow Model Registry** using `mlflow.register_model()` under the name **`MLflow_Iris_Classifier`**. The registered model name and version number are printed.

### Task 8 – Compare Model Runs
The `MlflowClient` is used to retrieve all logged parameters and metrics for both runs. A comparison table is printed programmatically, and the winning model is identified.

---

## Results – Which Model Performed Best?

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| **Accuracy** | ~0.9667 | ~1.0000 |
| **F1 Score** | ~0.9667 | ~1.0000 |
| **Precision** | ~0.9683 | ~1.0000 |
| **Recall** | ~0.9667 | ~1.0000 |

> **Winner: Random Forest Classifier**  
> The **Random Forest** model achieved higher accuracy (and F1 score) on the Iris test set. **Accuracy** was the primary deciding metric, with **weighted F1 Score** used as the secondary metric.

---

## Project Structure

```
assingment/
├── mlflow_experiment.ipynb    # Main Jupyter Notebook (Tasks 1–8)
├── README.md                  # This report
└── mlruns/                    # MLflow tracking directory (auto-created)
```

---

## How to Run

### 1. Install dependencies
```bash
pip install mlflow scikit-learn matplotlib seaborn pandas numpy
```

### 2. Run the notebook
Open `mlflow_experiment.ipynb` in Jupyter or VS Code and run all cells.

### 3. Launch MLflow UI
```bash
mlflow ui
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Screenshots Guide

After running the notebook and opening the MLflow UI, capture the following screens:

| # | Screenshot | Where to find it |
|---|---|---|
| 1 | MLflow Experiment page | Experiments → Iris_Classification_Experiment |
| 2 | Logged Parameters | Click any run → Parameters tab |
| 3 | Logged Metrics | Click any run → Metrics tab |
| 4 | Artifact tab | Click any run → Artifacts tab (plots folder) |
| 5 | Logged Model files | Artifacts tab → model subfolder |
| 6 | Prediction output | Notebook cell output for Task 6 |
| 7 | Model Registry | Models → MLflow_Iris_Classifier |
| 8 | Run comparison | Select both runs → Compare button |

---

## Tools & Libraries

| Tool | Version |
|---|---|
| Python | 3.8+ |
| MLflow | 2.x |
| scikit-learn | 1.x |
| Matplotlib | 3.x |
| Seaborn | 0.12+ |
| Pandas / NumPy | latest |
