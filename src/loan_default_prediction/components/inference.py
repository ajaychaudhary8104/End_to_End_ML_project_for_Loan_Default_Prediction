from pathlib import Path
from typing import Any, Dict, List
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler
from loan_default_prediction import logger
from src.loan_default_prediction.entity.config_entity import ModelInferenceConfig
from src.loan_default_prediction.utils.common import create_directories, load_bin


class ModelInference:
    def __init__(self, config: ModelInferenceConfig):
        self.config = config
        create_directories([self.config.root_dir])
        self.model = self.load_model()
        self.scaler = self.load_scaler()

    def load_model(self):
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from: {model_path}")
        return load_bin(path=model_path)

    def load_scaler(self) -> StandardScaler:
        scaler_path = Path(self.config.scaler_path)
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        logger.info(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)

    def drop_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.config.drop_columns:
            drop_list = [col for col in self.config.drop_columns if col in data.columns]
            if drop_list:
                logger.info(f"Dropping columns: {drop_list}")
                data = data.drop(columns=drop_list)
        return data

    def create_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating derived features...")

        if 'LoanAmount' in data.columns and 'Income' in data.columns:
            data['LoanToIncome'] = data['LoanAmount'] / (data['Income'] + 1e-8)
            logger.info("Created feature: LoanToIncome")

        if 'Age' in data.columns:
            data['AgeGroup'] = pd.cut(
                data['Age'],
                bins=[0, 25, 35, 45, 55, 100],
                labels=['<25', '25-35', '35-45', '45-55', '55+'],
                include_lowest=True
            )
            logger.info("Created feature: AgeGroup")

        if 'EmploymentType' in data.columns:
            data['IsEmployed'] = (data['EmploymentType'] != 'Unemployed').astype(int)
            logger.info("Created feature: IsEmployed")

        if 'DTIRatio' in data.columns:
            data['HighDTI'] = (data['DTIRatio'] > 0.5).astype(int)
            logger.info("Created feature: HighDTI")

        if 'CreditScore' in data.columns:
            data['LowCreditScore'] = (data['CreditScore'] < 600).astype(int)
            logger.info("Created feature: LowCreditScore")

        if 'InterestRate' in data.columns:
            data['HighInterestRate'] = (data['InterestRate'] > 15).astype(int)
            logger.info("Created feature: HighInterestRate")

        if all(col in data.columns for col in ['HighDTI', 'LowCreditScore', 'HighInterestRate']):
            data['RiskScore'] = (
                data['HighDTI'] + data['LowCreditScore'] + data['HighInterestRate']
            )
            logger.info("Created feature: RiskScore")

        if 'Income' in data.columns and 'Age' in data.columns:
            data['IncomePerAge'] = data['Income'] / (data['Age'] + 1e-8)
            logger.info("Created feature: IncomePerAge")

        if 'MonthsEmployed' in data.columns:
            data['EmploymentDuration'] = pd.cut(
                data['MonthsEmployed'],
                bins=[-1, 12, 36, 60, 120, 600],
                labels=['0-1yr', '1-3yr', '3-5yr', '5-10yr', '10+yr'],
                include_lowest=True
            )
            logger.info("Created feature: EmploymentDuration")

        logger.info(f"Derived features created. New shape: {data.shape}")
        return data

    def impute_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Imputing missing values...")

        for col in self.config.numeric_columns:
            if col in data.columns:
                if data[col].isnull().any():
                    median_value = data[col].median()
                    data[col] = data[col].fillna(median_value)
                    logger.info(f"Filled missing numeric values in {col} with median: {median_value}")

        for col in self.config.categorical_columns:
            if col in data.columns:
                if data[col].isnull().any():
                    mode_value = data[col].mode()
                    fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
                    data[col] = data[col].fillna(fill_value)
                    logger.info(f"Filled missing categorical values in {col} with: {fill_value}")

        return data

    def encode_categorical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        categorical_cols = [col for col in self.config.categorical_columns if col in data.columns]
        dynamic_cats = ['AgeGroup', 'EmploymentDuration']
        categorical_cols += [col for col in dynamic_cats if col in data.columns and col not in categorical_cols]

        if categorical_cols:
            logger.info(f"Encoding categorical columns: {categorical_cols}")
            data = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=int)
            logger.info(f"Encoded categorical columns. New shape: {data.shape}")
        return data

    def scale_numeric_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        # Get features that the scaler was fitted on
        scaler_features = self.scaler.get_feature_names_out() if hasattr(self.scaler, 'get_feature_names_out') else getattr(self.scaler, 'feature_names_in_', [])
        
        # Only scale columns that exist in data and were in the scaler's fit set
        numeric_cols = [col for col in self.config.numeric_columns if col in data.columns and col in scaler_features]
        
        derived_numeric = [
            'LoanToIncome', 'IncomePerAge', 'HighDTI', 'LowCreditScore',
            'HighInterestRate', 'RiskScore', 'IsEmployed'
        ]
        numeric_cols += [col for col in derived_numeric if col in data.columns and col not in numeric_cols and col in scaler_features]

        if numeric_cols:
            logger.info(f"Scaling numeric columns: {numeric_cols}")
            data[numeric_cols] = self.scaler.transform(data[numeric_cols])
            logger.info("Numeric features scaled using loaded StandardScaler")

        return data

    def align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        # Get expected features
        if hasattr(self.model, 'feature_names_in_'):
            expected_features = list(self.model.feature_names_in_)
        else:
            expected_features = self.model.get_booster().feature_names or []

        # Drop unexpected
        unexpected_features = [col for col in features.columns if col not in expected_features]
        if unexpected_features:
            logger.warning(f"Dropping unexpected features: {unexpected_features}")
            features = features.drop(columns=unexpected_features)

        # Add missing safely
        missing_features = [col for col in expected_features if col not in features.columns]
        if missing_features:
            logger.warning(f"Missing features detected: {missing_features}")
            for col in missing_features:
                features[col] = 0  # fallback (see improvement below)

        # Ensure correct order
        features = features[expected_features]

        # 🔥 Enforce numeric conversion where possible
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='ignore')

        return features

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting data preprocessing for inference...")
        data = self.drop_columns(data)
        data = self.create_derived_features(data)
        data = self.impute_missing_values(data)
        data = self.encode_categorical_columns(data)
        # Get expected features from model before scaling
        if hasattr(self.model, 'feature_names_in_'):
            expected_features = list(self.model.feature_names_in_)
            # Drop any features not in the model's expected features
            unexpected = [col for col in data.columns if col not in expected_features]
            if unexpected:
                logger.info(f"Dropping features not expected by model: {unexpected}")
                data = data.drop(columns=unexpected)
        data = self.scale_numeric_columns(data)
        logger.info("Data preprocessing completed for inference")
        return data

    def load_input_data(self) -> pd.DataFrame:
        input_path = Path(self.config.input_data_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input data file not found: {input_path}")

        logger.info(f"Loading inference input data from: {input_path}")
        return pd.read_csv(input_path)

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.config.target_column in data.columns:
            logger.info(f"Dropping target column '{self.config.target_column}' from inference features")
            data = data.drop(columns=[self.config.target_column])
        return data

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        if input_df.empty:
            raise ValueError("Inference input data is empty")

        features = self.prepare_features(input_df.copy())
        features = self.align_features(features)
        logger.info(f"Running inference on {len(features)} records")

        predictions = self.model.predict(features)
        results = pd.DataFrame({"prediction": predictions})

        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(features)[:, 1]
                results["prediction_probability"] = proba
            except Exception as exc:
                logger.warning(f"Model does not support probability estimates: {exc}")

        return results

    def save_predictions(self, input_df: pd.DataFrame, results: pd.DataFrame) -> Path:
        output_path = Path(self.config.prediction_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        merged = input_df.reset_index(drop=True).copy()
        merged = pd.concat([merged, results.reset_index(drop=True)], axis=1)
        merged.to_csv(output_path, index=False)

        logger.info(f"Saved inference predictions to: {output_path}")
        return output_path

    def run_batch_inference(self) -> Path:
        input_df = self.load_input_data()
        results = self.predict(input_df)
        return self.save_predictions(input_df, results)

    def predict_records(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        if not records:
            raise ValueError("No records were provided for prediction")

        input_df = pd.DataFrame(records)
        input_df = self.preprocess_data(input_df)
        return self.predict(input_df)
