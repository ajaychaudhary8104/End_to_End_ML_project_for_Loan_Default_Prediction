# Stage 6: Advanced Usage Guide

## 🚀 Advanced Patterns and Customizations

This guide covers advanced usage patterns for Stage 6 Model Evaluation.

---

## 1. Custom Evaluation Component

### Extend with Additional Metrics

```python
from src.loan_default_prediction.components.model_evaluation import ModelEvaluation
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score

class AdvancedModelEvaluation(ModelEvaluation):
    def advanced_metrics(self, y_true, y_pred):
        """Calculate additional evaluation metrics."""
        metrics = self.eval_metrics(y_true, y_pred)
        
        # Add custom metrics
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
        
        return metrics
    
    def log_advanced_metrics(self):
        """Log advanced metrics to MLflow."""
        test_data = self.load_data(str(self.config.test_data_path))
        model = self.load_model(str(self.config.model_path))
        
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[[self.config.target_column]].values.flatten()
        
        y_pred = model.predict(X_test)
        metrics = self.advanced_metrics(y_test, y_pred)
        
        with mlflow.start_run():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
```

### Usage
```python
config = ConfigurationManager()
eval_config = config.get_model_evaluation_config()
advanced_eval = AdvancedModelEvaluation(config=eval_config)
advanced_eval.log_advanced_metrics()
```

---

## 2. Multi-Model Comparison

### Compare Multiple Models

```python
from src.loan_default_prediction.components.model_evaluation import ModelEvaluation
import pickle

class MultiModelComparison:
    def __init__(self, config):
        self.config = config
        self.evaluator = ModelEvaluation(config=config)
    
    def compare_models(self, model_paths: dict):
        """
        Compare multiple models on test data.
        
        Args:
            model_paths: {'model_name': 'path/to/model.pkl'}
        """
        test_data = self.evaluator.load_data(str(self.config.test_data_path))
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[[self.config.target_column]].values.flatten()
        
        results = {}
        
        for model_name, model_path in model_paths.items():
            print(f"Evaluating {model_name}...")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            y_pred = model.predict(X_test)
            metrics = self.evaluator.eval_metrics(y_test, y_pred)
            results[model_name] = metrics
            
            # Log to MLflow with model identifier
            with mlflow.start_run(run_name=model_name):
                mlflow.log_params({'model_name': model_name})
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
        
        return results
    
    def display_comparison(self, results: dict):
        """Display model comparison as table."""
        import pandas as pd
        
        comparison_df = pd.DataFrame(results).T
        print("\nModel Comparison:")
        print(comparison_df)
        return comparison_df
```

### Usage
```python
config = ConfigurationManager()
eval_config = config.get_model_evaluation_config()
comparator = MultiModelComparison(config=eval_config)

models = {
    'XGBoost_v1': 'models/xgboost_v1.pkl',
    'XGBoost_v2': 'models/xgboost_v2.pkl',
    'LightGBM': 'models/lightgbm.pkl'
}

results = comparator.compare_models(models)
comparison_df = comparator.display_comparison(results)
```

---

## 3. Threshold Optimization

### Find Optimal Classification Threshold

```python
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

class ThresholdOptimizer:
    def __init__(self, config):
        self.config = config
        self.evaluator = ModelEvaluation(config=config)
    
    def optimize_threshold(self, metric='f1'):
        """Find optimal classification threshold."""
        test_data = self.evaluator.load_data(str(self.config.test_data_path))
        model = self.evaluator.load_model(str(self.config.model_path))
        
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[[self.config.target_column]].values.flatten()
        
        # Get probability predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            if metric == 'f1':
                score = f1_score(y_test, y_pred)
            elif metric == 'precision':
                score = precision_score(y_test, y_pred)
            elif metric == 'recall':
                score = recall_score(y_test, y_pred)
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        return {
            'threshold': optimal_threshold,
            'score': optimal_score,
            'metric': metric,
            'all_thresholds': list(thresholds),
            'all_scores': list(scores)
        }
    
    def plot_threshold_analysis(self, results: dict):
        """Plot threshold vs performance."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(results['all_thresholds'], results['all_scores'], 'b-o')
        plt.axvline(x=results['threshold'], color='r', linestyle='--', 
                    label=f"Optimal: {results['threshold']:.2f}")
        plt.xlabel('Classification Threshold')
        plt.ylabel(f"{results['metric'].capitalize()} Score")
        plt.title(f"Threshold Optimization - {results['metric'].upper()}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt
```

### Usage
```python
config = ConfigurationManager()
eval_config = config.get_model_evaluation_config()
optimizer = ThresholdOptimizer(config=eval_config)

results = optimizer.optimize_threshold(metric='f1')
print(f"Optimal Threshold: {results['threshold']:.2f}")
print(f"F1 Score: {results['score']:.4f}")

# Visualize
plt = optimizer.plot_threshold_analysis(results)
plt.show()
```

---

## 4. Class Imbalance Analysis

### Analyze and Handle Class Imbalance

```python
class ImbalanceAnalyzer:
    def __init__(self, config):
        self.config = config
        self.evaluator = ModelEvaluation(config=config)
    
    def analyze_imbalance(self):
        """Analyze class distribution and imbalance."""
        test_data = self.evaluator.load_data(str(self.config.test_data_path))
        y_test = test_data[self.config.target_column]
        
        class_counts = y_test.value_counts()
        class_ratios = (class_counts / len(y_test) * 100).round(2)
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        analysis = {
            'class_counts': class_counts.to_dict(),
            'class_percentages': class_ratios.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'majority_class': class_counts.idxmax(),
            'minority_class': class_counts.idxmin(),
            'total_samples': len(y_test)
        }
        
        return analysis
    
    def per_class_performance(self, y_true, y_pred):
        """Get performance metrics per class."""
        from sklearn.metrics import classification_report
        
        report = classification_report(y_true, y_pred, output_dict=True)
        
        per_class = {}
        for class_id in set(y_true):
            per_class[str(class_id)] = {
                'precision': report[str(class_id)]['precision'],
                'recall': report[str(class_id)]['recall'],
                'f1-score': report[str(class_id)]['f1-score'],
                'support': int(report[str(class_id)]['support'])
            }
        
        return per_class
```

### Usage
```python
config = ConfigurationManager()
eval_config = config.get_model_evaluation_config()
analyzer = ImbalanceAnalyzer(config=eval_config)

imbalance = analyzer.analyze_imbalance()
print(f"Class Distribution: {imbalance['class_percentages']}")
print(f"Imbalance Ratio: {imbalance['imbalance_ratio']:.2f}:1")
```

---

## 5. Cross-Validation Evaluation

### Evaluate with Cross-Validation

```python
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

class CrossValidationEvaluation:
    def __init__(self, config):
        self.config = config
        self.evaluator = ModelEvaluation(config=config)
    
    def cv_evaluation(self, n_splits=5):
        """Perform cross-validation evaluation."""
        # Load full dataset
        train_data = pd.read_csv(self.config.train_file_path)
        X = train_data.drop([self.config.target_column], axis=1)
        y = train_data[self.config.target_column]
        
        # Load model
        model = self.evaluator.load_model(str(self.config.model_path))
        
        # Define scoring
        scoring = {
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1': make_scorer(f1_score, average='weighted'),
            'accuracy': 'accuracy'
        }
        
        # Perform CV
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        
        # Summarize results
        summary = {
            'cv_splits': n_splits
        }
        
        for metric in scoring.keys():
            key = f'test_{metric}'
            summary[f'{metric}_mean'] = cv_results[key].mean()
            summary[f'{metric}_std'] = cv_results[key].std()
        
        return summary, cv_results
```

### Usage
```python
config = ConfigurationManager()
eval_config = config.get_model_evaluation_config()
cv_eval = CrossValidationEvaluation(config=eval_config)

summary, results = cv_eval.cv_evaluation(n_splits=5)
print("Cross-Validation Results:")
for metric, value in summary.items():
    print(f"  {metric}: {value:.4f}")
```

---

## 6. Automated Model Comparison Pipeline

### Full Comparison and Selection

```python
class AutomatedModelEvaluation:
    def __init__(self, config, candidate_models: dict):
        """
        Args:
            config: ModelEvaluationConfig
            candidate_models: {'model_name': model_object}
        """
        self.config = config
        self.candidate_models = candidate_models
        self.evaluator = ModelEvaluation(config=config)
    
    def evaluate_all(self):
        """Evaluate all candidate models."""
        test_data = self.evaluator.load_data(str(self.config.test_data_path))
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[[self.config.target_column]].values.flatten()
        
        results = {}
        
        for model_name, model in self.candidate_models.items():
            y_pred = model.predict(X_test)
            metrics = self.evaluator.eval_metrics(y_test, y_pred)
            results[model_name] = metrics
            
            # Log to MLflow
            with mlflow.start_run(run_name=model_name):
                mlflow.set_tag('type', 'model_comparison')
                mlflow.set_tag('model_name', model_name)
                for param_name, param_value in self.config.all_params.items():
                    mlflow.log_param(param_name, param_value)
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
        
        return results
    
    def select_best_model(self, results: dict, metric='f1'):
        """Select best model based on metric."""
        best_model = max(results.items(), 
                        key=lambda x: x[1][metric])
        return best_model
    
    def generate_report(self, results: dict, metric='f1'):
        """Generate comparison report."""
        import json
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'metric_used': metric,
            'models': results,
            'best_model': max(results.items(), 
                            key=lambda x: x[1][metric])[0]
        }
        
        return report
```

### Usage
```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle

# Load models
models = {
    'XGBoost': pickle.load(open('models/xgboost.pkl', 'rb')),
    'RandomForest': pickle.load(open('models/rf.pkl', 'rb'))
}

config = ConfigurationManager()
eval_config = config.get_model_evaluation_config()
auto_eval = AutomatedModelEvaluation(config=eval_config, candidate_models=models)

results = auto_eval.evaluate_all()
best_model, best_metrics = auto_eval.select_best_model(results, metric='f1')
report = auto_eval.generate_report(results)

print(f"Best Model: {best_model}")
print(f"Metrics: {best_metrics}")
```

---

## 7. MLflow Advanced Features

### Custom Tags and Metadata

```python
import mlflow

class AdvancedMLflowLogging:
    def __init__(self, config):
        self.config = config
        self.evaluator = ModelEvaluation(config=config)
    
    def log_with_metadata(self):
        """Log with advanced MLflow features."""
        test_data = self.evaluator.load_data(str(self.config.test_data_path))
        model = self.evaluator.load_model(str(self.config.model_path))
        
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[[self.config.target_column]].values.flatten()
        
        y_pred = model.predict(X_test)
        metrics = self.evaluator.eval_metrics(y_test, y_pred)
        
        with mlflow.start_run():
            # Log tags
            mlflow.set_tags({
                'environment': 'production',
                'framework': 'xgboost',
                'task': 'classification',
                'data_version': 'v1.0'
            })
            
            # Log parameters
            mlflow.log_params(self.config.all_params)
            
            # Log metrics with timestamps
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log datasets info
            mlflow.log_dict({
                'test_size': len(y_test),
                'features': len(X_test.columns),
                'class_distribution': pd.Series(y_test).value_counts().to_dict()
            }, 'dataset_info.json')
            
            # Log model with custom signature
            from mlflow.models.signature import ModelSignature
            from mlflow.types.schema import Schema, ColSpec
            
            input_schema = Schema([
                ColSpec("double", col) for col in X_test.columns
            ])
            output_schema = Schema([ColSpec("long")])
            signature = ModelSignature(input_schema, output_schema)
            
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="ProductionModel",
                signature=signature
            )
```

---

## 8. Performance Monitoring

### Track Performance Over Time

```python
class PerformanceMonitor:
    def __init__(self, config):
        self.config = config
        self.evaluator = ModelEvaluation(config=config)
    
    def get_historical_performance(self, experiment_id=None):
        """Retrieve historical performance data."""
        client = mlflow.tracking.MlflowClient()
        
        if experiment_id is None:
            experiment = mlflow.get_experiment_by_name("Default")
            experiment_id = experiment.experiment_id
        
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        
        history = []
        for idx, run in runs.iterrows():
            run_data = {
                'run_id': run['run_id'],
                'timestamp': pd.to_datetime(run['start_time']),
                'metrics': {col: run[col] for col in runs.columns if col.startswith('metrics.')},
                'status': run['status']
            }
            history.append(run_data)
        
        return pd.DataFrame(history)
    
    def detect_performance_drift(self, metric='f1', threshold=0.05):
        """Detect significant performance drops."""
        history = self.get_historical_performance()
        
        if len(history) < 2:
            return None
        
        metric_col = f'metrics.{metric}'
        current = history[metric_col].iloc[-1]
        previous = history[metric_col].iloc[-2]
        
        drop = previous - current
        
        if drop > threshold:
            return {
                'detected': True,
                'metric': metric,
                'previous': previous,
                'current': current,
                'drop': drop,
                'alert': f"Performance drop detected: {metric} dropped by {drop:.4f}"
            }
        
        return {'detected': False}
```

---

## 9. Integration with Feature Store (Future)

```python
class FeatureStoreIntegration:
    """Ready for integration with Tecton, Feast, or similar."""
    
    def __init__(self, config):
        self.config = config
    
    def log_features_used(self, feature_metadata):
        """Log which features were used in evaluation."""
        mlflow.log_dict(feature_metadata, 'features_used.json')
    
    def log_feature_importance(self, model, feature_names):
        """Log feature importance from model."""
        importance = model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        mlflow.log_dict(feature_importance, 'feature_importance.json')
```

---

## Summary of Advanced Patterns

| Pattern | Use Case |
|---------|----------|
| Custom Metrics | Specialized evaluation metrics |
| Model Comparison | A/B testing multiple models |
| Threshold Optimization | Tuning decision threshold |
| Imbalance Analysis | Handling skewed datasets |
| Cross-Validation | Robust evaluation |
| Auto Comparison | Pipeline automation |
| Advanced MLflow | Metadata and tracking |
| Performance Monitoring | Drift detection |
| Feature Store | Feature metadata tracking |

All of these patterns integrate seamlessly with the base `ModelEvaluation` component!
