import os
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from loan_default_prediction import logger
from loan_default_prediction.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def load_processed_data(self) -> pd.DataFrame:
        if not os.path.exists(self.config.input_file_path):
            raise FileNotFoundError(f"Processed data not found: {self.config.input_file_path}")

        logger.info(f"Loading processed data from: {self.config.input_file_path}")
        data = pd.read_csv(self.config.input_file_path)
        logger.info(f"Processed data shape: {data.shape}")
        return data

    def split_data(self, data: pd.DataFrame):
        target_col = self.config.target_column
        if target_col not in data.columns:
            raise KeyError(f"Target column '{target_col}' not found in data")

        logger.info("Creating stratified train/validation/test split...")
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # First split out the test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if len(y.unique()) > 1 else None
        )

        # Then split the remaining data into training and validation sets
        relative_val_size = self.config.validation_size / (1.0 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=relative_val_size,
            random_state=self.config.random_state,
            stratify=y_train_val if len(y_train_val.unique()) > 1 else None
        )

        train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        val_df = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
        test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

        logger.info(f"Train shape: {train_df.shape}, Validation shape: {val_df.shape}, Test shape: {test_df.shape}")
        return train_df, val_df, test_df

    def apply_smote(self, train_df: pd.DataFrame):
        target_col = self.config.target_column
        if target_col not in train_df.columns:
            raise KeyError(f"Target column '{target_col}' not found in training data")

        logger.info("Applying SMOTE to the training set...")
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        smote = SMOTE(random_state=self.config.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        train_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X_train.columns),
                                     pd.Series(y_resampled, name=target_col)], axis=1)

        logger.info(f"SMOTE applied. Training shape after resampling: {train_resampled.shape}")
        return train_resampled

    def save_split_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        os.makedirs(self.config.split_artifacts_dir, exist_ok=True)

        train_path = self.config.train_file_path
        val_path = self.config.validation_file_path
        test_path = self.config.test_file_path

        logger.info(f"Saving train data to: {train_path}")
        train_df.to_csv(train_path, index=False)

        logger.info(f"Saving validation data to: {val_path}")
        val_df.to_csv(val_path, index=False)

        logger.info(f"Saving test data to: {test_path}")
        test_df.to_csv(test_path, index=False)

        logger.info(f"All split datasets saved to: {self.config.split_artifacts_dir}")

    def initiate_data_transformation(self) -> bool:
        try:
            data = self.load_processed_data()
            train_df, val_df, test_df = self.split_data(data)
            train_df = self.apply_smote(train_df)
            self.save_split_data(train_df, val_df, test_df)
            logger.info("Data transformation completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during data transformation: {str(e)}")
            raise e
