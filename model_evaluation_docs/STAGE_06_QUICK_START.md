# Stage 6: Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure MLflow (Optional but Recommended)

If you want to log to DagsHub, update `config/config.yaml`:

```yaml
model_evaluation:
  mlflow_uri: https://dagshub.com/YOUR_USERNAME/YOUR_REPO.mlflow
```

And set environment variables (or update in the notebook):
```bash
set MLFLOW_TRACKING_USERNAME=your_username
set MLFLOW_TRACKING_PASSWORD=your_password
```

### Step 3: Run Stage 6

#### Option A: Quick Run (Python)
```bash
cd d:\End_to_End_ML_project_for_Loan_Default_Prediction
python src/loan_default_prediction/pipeline/stage_06_model_evaluation.py
```

#### Option B: Full Pipeline (Python)
```bash
python main.py
```

#### Option C: Using DVC
```bash
dvc repro model_evaluation
```

#### Option D: From Jupyter
```bash
cd research
jupyter notebook 06_model_evaluation_with_mlflow.ipynb
```

---

## 📊 View Results

### 1. Local Metrics File
```bash
cat artifacts/model_evaluation/model_evaluation_metrics.json
```

Output example:
```json
{
  "precision": 0.8542,
  "recall": 0.8231,
  "f1": 0.8383,
  "accuracy": 0.8650,
  "roc_auc": 0.9123,
  "confusion_matrix": [[1250, 150], [180, 1320]],
  "classification_report": {...}
}
```

### 2. MLflow UI (Local)
```bash
mlflow ui
# Visit http://localhost:5000
```

### 3. DagsHub Remote
Visit your DagsHub repository MLflow section to see:
- Metrics
- Parameters
- Model versions
- Artifacts

---

## 🧪 Verify Success

✅ **Look for these indicators**:

1. **Console Output**:
   ```
   >>>>>> stage Model Evaluation stage started <<<<<<
   Model loaded successfully
   Saving evaluation results locally...
   Logging to MLflow with model versioning...
   >>>>>> stage Model Evaluation stage completed <<<<<<
   ```

2. **Files Created**:
   ```
   artifacts/model_evaluation/model_evaluation_metrics.json
   ```

3. **Metrics Printed**:
   - Precision, Recall, F1, Accuracy, ROC-AUC
   - Confusion matrix displayed

4. **MLflow Remote**:
   - Run created in MLflow
   - Model registered (if using remote tracking)

---

## 🐛 Common Issues

### Issue 1: "Model file not found"
**Solution**: Run Stage 5 first
```bash
python src/loan_default_prediction/pipeline/stage_05_model_training.py
```

### Issue 2: "Test data path not found"
**Solution**: Run all stages from beginning
```bash
python main.py
```

### Issue 3: MLflow connection error
**Solution**: Check credentials or use local tracking
```bash
# Use local file store
set MLFLOW_TRACKING_URI=file:///./mlruns
```

### Issue 4: "Module not found" errors
**Solution**: Install all dependencies
```bash
pip install -r requirements.txt --upgrade
```

---

## 📈 Expected Metrics

For a well-trained XGBoost model on loan default data:

| Metric | Expected Range | Your Result |
|--------|-----------------|-------------|
| Accuracy | 0.80 - 0.95 | __________ |
| Precision | 0.75 - 0.90 | __________ |
| Recall | 0.70 - 0.85 | __________ |
| F1-Score | 0.75 - 0.90 | __________ |
| ROC-AUC | 0.85 - 0.95 | __________ |

---

## 💾 Outputs Summary

### Files Generated
- ✅ `artifacts/model_evaluation/model_evaluation_metrics.json` - Local metrics
- ✅ MLflow run with parameters, metrics, and artifacts
- ✅ Registered model version in MLflow

### Metrics Tracked
- ✅ Precision (weighted average)
- ✅ Recall (weighted average)
- ✅ F1-Score (weighted average)
- ✅ Accuracy
- ✅ ROC-AUC
- ✅ Confusion Matrix
- ✅ Classification Report

### Artifacts Stored
- ✅ Serialized model
- ✅ Confusion matrix JSON
- ✅ Classification report JSON

---

## 🔄 Rerun Stage 6

To rerun evaluation after model updates:

```bash
# Method 1: Direct
python src/loan_default_prediction/pipeline/stage_06_model_evaluation.py

# Method 2: Full pipeline
python main.py

# Method 3: DVC (removes old outputs and reruns)
dvc repro model_evaluation --force

# View all runs
mlflow ui
```

---

## 📖 Learn More

For comprehensive documentation, see:
- `STAGE_06_DOCUMENTATION.md` - Complete guide
- `STAGE_06_IMPLEMENTATION_SUMMARY.md` - Implementation details

---

## ⚡ Pro Tips

1. **Save metrics before MLflow logs**: Local JSON is always available
2. **Compare model versions**: Use MLflow UI to track improvements
3. **Monitor key metrics**: Focus on F1-Score for imbalanced data
4. **Version your experiments**: Each run gets automatic versioning
5. **Use DVC for reproducibility**: `dvc dag` shows pipeline dependencies

---

## 📞 Need Help?

1. Check logs for error messages
2. Verify configuration in `config/config.yaml`
3. Ensure all dependencies installed
4. Check model file exists: `artifacts/model_training/xgboost_model.pkl`
5. Verify test data exists: `artifacts/data_transformation/split_data/test.csv`

---

**Happy Evaluating! 🎉**
