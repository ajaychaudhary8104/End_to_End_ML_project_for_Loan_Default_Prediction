import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from loan_default_prediction import logger
from loan_default_prediction.entity.config_entity import DataPreprocessingConfig


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.scaler = StandardScaler()

    def get_raw_file_path(self) -> str:
        return os.path.join(self.config.raw_data_dir, self.config.input_file_name)

    def load_data(self) -> pd.DataFrame:
        raw_file_path = self.get_raw_file_path()
        if not os.path.exists(raw_file_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_file_path}")

        logger.info(f"Loading raw data from: {raw_file_path}")
        data = pd.read_csv(raw_file_path)
        logger.info(f"Loaded raw data shape: {data.shape}")
        return data

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
        numeric_cols = [col for col in self.config.numeric_columns if col in data.columns]
        derived_numeric = [
            'LoanToIncome', 'IncomePerAge', 'HighDTI', 'LowCreditScore',
            'HighInterestRate', 'RiskScore', 'EmploymentYears'
        ]
        numeric_cols += [col for col in derived_numeric if col in data.columns and col not in numeric_cols]

        if numeric_cols:
            logger.info(f"Scaling numeric columns: {numeric_cols}")
            self.scaler.fit(data[numeric_cols])
            data[numeric_cols] = self.scaler.transform(data[numeric_cols])
            logger.info("Numeric features scaled using StandardScaler")
            self._save_scaler()

        return data

    def _save_scaler(self) -> None:
        scaler_path = os.path.join(self.config.root_dir, 'scaler.pkl')
        os.makedirs(self.config.root_dir, exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to: {scaler_path}")

    def save_processed_data(self, data: pd.DataFrame) -> None:
        os.makedirs(self.config.root_dir, exist_ok=True)
        logger.info(f"Saving processed data to: {self.config.processed_data_file}")
        data.to_csv(self.config.processed_data_file, index=False)
        logger.info(f"Processed data saved. Final shape: {data.shape}")

    def initiate_data_preprocessing(self) -> bool:
        try:
            logger.info("Starting data preprocessing...")
            data = self.load_data()
            data = self.drop_columns(data)
            data = self.create_derived_features(data)
            data = self.impute_missing_values(data)
            data = self.encode_categorical_columns(data)
            data = self.scale_numeric_columns(data)
            self.save_processed_data(data)
            logger.info("Data preprocessing completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during data preprocessing: {str(e)}")
            raise e
