# Stage 6: Model Evaluation with MLflow and Model Versioning

## Overview

Stage 6 implements a complete model evaluation pipeline with MLflow integration and model versioning. This stage:

1. **Evaluates** the trained XGBoost model on test data
2. **Calculates** comprehensive metrics (Precision, Recall, F1, Accuracy, ROC-AUC)
3. **Logs** metrics and model to MLflow with remote tracking (DagsHub)
4. **Registers** the model with versioning capabilities
5. **Generates** confusion matrix and classification reports
6. **Saves** evaluation results locally for audit trails

---

## Architecture

### Components

#### 1. **Model Evaluation Component** (`src/loan_default_prediction/components/model_evaluation.py`)

```python
class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig)
    def load_data(data_path: str) -> pd.DataFrame
    def load_model(model_path: str)
    def eval_metrics(y_true, y_pred) -> dict
    def create_confusion_matrix_artifact(y_true, y_pred) -> dict
    def log_into_mlflow()
    def save_evaluation_results()
```

**Key Methods:**
- `load_data()`: Load test dataset with validation
- `load_model()`: Load trained pickle model
- `eval_metrics()`: Calculate Precision, Recall, F1, Accuracy, ROC-AUC
- `create_confusion_matrix_artifact()`: Generate confusion matrix
- `log_into_mlflow()`: Log metrics, model, and artifacts to MLflow with versioning
- `save_evaluation_results()`: Save metrics locally as JSON

#### 2. **Stage 6 Pipeline** (`src/loan_default_prediction/pipeline/stage_06_model_evaluation.py`)

```python
class ModelEvaluationPipeline:
    def main():
        # Load configuration
        # Save evaluation results locally
        # Log to MLflow with model versioning
```

#### 3. **Configuration** (`src/loan_default_prediction/entity/config_entity.py`)

```python
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path                      # Artifacts root
    test_data_path: Path                # Test dataset path
    model_path: Path                    # Trained model path
    all_params: dict                    # Model hyperparameters
    metric_file_name: Path              # Output metrics file
    target_column: str                  # Target column name
    mlflow_uri: str                     # MLflow tracking URI
```

---

## Configuration

### `config/config.yaml`
```yaml
model_evaluation:
  root_dir: artifacts/model_evaluation
  test_data_path: artifacts/data_transformation/split_data/test.csv
  model_path: artifacts/model_training/xgboost_model.pkl
  metric_file_name: artifacts/model_evaluation/model_evaluation_metrics.json
  target_column: Default
  mlflow_uri: https://dagshub.com/ajaychaudhary8104/End_to_End_ML_project_for_Loan_Default_Prediction.mlflow
```

### `params.yaml`
```yaml
model_params:
    objective: binary:logistic
    eval_metric: auc
    use_label_encoder: false
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42
    n_jobs: -1
    early_stopping_rounds: 10
```

### DVC Pipeline (`dvc.yaml`)
```yaml
model_evaluation:
  cmd: python src/loan_default_prediction/pipeline/stage_06_model_evaluation.py
  deps:
    - src/loan_default_prediction/pipeline/stage_06_model_evaluation.py
    - src/loan_default_prediction/components/model_evaluation.py
    - artifacts/model_training/xgboost_model.pkl
    - artifacts/data_transformation/split_data/test.csv
    - config/config.yaml
    - params.yaml
  outs:
    - artifacts/model_evaluation/model_evaluation_metrics.json
```

---

## Evaluation Metrics

### Metrics Calculated

| Metric | Description |
|--------|-------------|
| **Precision** | TP / (TP + FP) - Accuracy of positive predictions |
| **Recall** | TP / (TP + FN) - Coverage of positive class |
| **F1-Score** | 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean |
| **Accuracy** | (TP + TN) / Total - Overall correctness |
| **ROC-AUC** | Area under ROC curve - Probability of correct ranking |

### Additional Artifacts

1. **Confusion Matrix**: Shows TP, TN, FP, FN breakdown
2. **Classification Report**: Per-class precision, recall, F1
3. **Model Artifacts**: Serialized model for reproducibility

---

## MLflow Integration & Model Versioning

### Features

✅ **Remote Tracking**: Logs to DagsHub MLflow backend
✅ **Metric Logging**: All evaluation metrics tracked
✅ **Model Registration**: Automatic model versioning
✅ **Artifact Storage**: Model and reports saved
✅ **Parameter Logging**: Hyperparameters tracked
✅ **Experiment Management**: Organized runs

### Model Registry

Models are registered as `XGBClassifier` with automatic versioning:
- Version 1: First evaluation run
- Version 2: Subsequent evaluation runs
- etc.

---

## Usage

### Run Stage 6 Directly

```bash
# Run individual stage
python src/loan_default_prediction/pipeline/stage_06_model_evaluation.py

# Or run entire pipeline
python main.py
```

### Run with DVC

```bash
# Run specific stage
dvc repro model_evaluation

# Run full pipeline
dvc repro
```

### Use in Notebook

```python
from src.loan_default_prediction.config.configuration import ConfigurationManager
from src.loan_default_prediction.components.model_evaluation import ModelEvaluation

# Get configuration
config = ConfigurationManager()
model_eval_config = config.get_model_evaluation_config()

# Create evaluator
evaluator = ModelEvaluation(config=model_eval_config)

# Save results locally
evaluator.save_evaluation_results()

# Log to MLflow
evaluator.log_into_mlflow()
```

---

## Output Files

### Local Artifacts

```
artifacts/model_evaluation/
├── model_evaluation_metrics.json    # Evaluation metrics
│   ├── precision
│   ├── recall
│   ├── f1
│   ├── accuracy
│   ├── roc_auc
│   ├── confusion_matrix
│   └── classification_report
```

### MLflow Remote Artifacts

```
MLflow Experiment > Run
├── Parameters
│   ├── objective
│   ├── eval_metric
│   ├── n_estimators
│   ├── max_depth
│   ├── learning_rate
│   └── ...
├── Metrics
│   ├── precision
│   ├── recall
│   ├── f1
│   ├── accuracy
│   └── roc_auc
├── Artifacts
│   ├── model/                       # Serialized model
│   ├── confusion_matrix.json
│   ├── classification_report.json
└── Registered Model
    └── XGBClassifier (versioned)
```

---

## Metrics Format

### Local JSON Output

```json
{
  "precision": 0.8542,
  "recall": 0.8231,
  "f1": 0.8383,
  "accuracy": 0.8650,
  "roc_auc": 0.9123,
  "confusion_matrix": [
    [1250, 150],
    [180, 1320]
  ],
  "classification_report": {
    "0": {
      "precision": 0.8743,
      "recall": 0.8929,
      "f1-score": 0.8835,
      "support": 1400
    },
    "1": {
      "precision": 0.8979,
      "recall": 0.8800,
      "f1-score": 0.8888,
      "support": 1500
    },
    "accuracy": 0.8865,
    "macro avg": {...},
    "weighted avg": {...}
  }
}
```

---

## Error Handling

The component includes comprehensive error handling:

```python
# Data loading errors
FileNotFoundError: If test data or model files missing

# Metric calculation errors
ValueError: If predictions/labels format invalid

# MLflow logging errors
Warnings logged if registry unavailable (falls back to local)
```

---

## Integration with Other Stages

### Input Dependencies
- **Stage 5 (Model Training)**: Provides trained model
- **Stage 4 (Data Transformation)**: Provides test dataset

### Output Usage
- **Deployment**: Uses registered model from MLflow
- **Monitoring**: Uses metrics for baseline comparison
- **Reporting**: Uses JSON metrics for dashboards

---

## Best Practices

1. **Always save locally** before MLflow logging for audit trails
2. **Version your models** using MLflow registry
3. **Track hyperparameters** for reproducibility
4. **Monitor key metrics** (Precision, Recall, F1)
5. **Review confusion matrix** for class imbalance issues
6. **Use proper train/val/test split** (already done in Stage 4)
7. **Log artifacts** for comparison across runs

---

## Troubleshooting

### Issue: MLflow connection error
**Solution**: Check credentials and internet connection
```bash
# Verify MLflow URI
python -c "import mlflow; print(mlflow.get_tracking_uri())"
```

### Issue: Model not registered
**Solution**: Ensure tracking URL is remote (not file://)
```bash
# Use DagsHub or remote MLflow server
os.environ["MLFLOW_TRACKING_URI"] = "https://..."
```

### Issue: Missing model file
**Solution**: Run Stage 5 first to train model
```bash
python src/loan_default_prediction/pipeline/stage_05_model_training.py
```

---

## Advanced Usage

### Custom Evaluation Metrics

```python
from src.loan_default_prediction.components.model_evaluation import ModelEvaluation

class CustomModelEvaluation(ModelEvaluation):
    def custom_metric(self, y_true, y_pred):
        # Your custom metric logic
        pass
```

### MLflow with Different Backends

```python
# Local file store
os.environ["MLFLOW_TRACKING_URI"] = "file:///./mlruns"

# Remote server
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

# AWS S3
os.environ["MLFLOW_TRACKING_URI"] = "s3://bucket/path"
```

---

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [DagsHub MLflow Integration](https://dagshub.com/docs/mlflow)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## Author Notes

This implementation follows MLflow best practices:
- ✅ Reproducible evaluation pipeline
- ✅ Centralized metric tracking
- ✅ Model versioning and registry
- ✅ Complete artifact logging
- ✅ Production-ready error handling
- ✅ DVC integration for pipeline orchestration
