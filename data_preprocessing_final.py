import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
import logging
import json
import os
import io

# Configure logging
logging.basicConfig(
    filename='auto_forecast_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Global Variables for Data Preprocessing
month_col = "month"
year_col = "year"
date_col = "mon_year"
dependent_var = "Invoices (K Euro)"
columns_to_exclude = [
    "Total Service Gap (K Euro)",
    "month",
    "year",
    "Sourcing Location"
]
start_year = 2019
end_year = 2024
key_variable = "Sourcing Location"
missing_values_treatment_stage = "skip"  # Options: 'do', 'skip'
numeric_fill_method = "mean"  # Options: 'mean', 'median', 'ffill', 'bfill'
categorical_fill_method = "mode"  # Options: 'mode', 'ffill', 'bfill'

class DataPreprocessor:
    def __init__(self, uploadedDataST=None, month_col=None, year_col=None, date_col=None, dependent_var=None,
                 columns_to_exclude=None, start_year=None, end_year=None,
                 key_variable=None, missing_values_treatment_stage=None,
                 numeric_fill_method=None, categorical_fill_method=None):
        """
        Initialize the data preprocessor. Uses configuration variables.
        """
        self.uploadedDataST = uploadedDataST
        self.month_col = month_col
        self.year_col = year_col
        self.date_col = date_col
        self.dependent_var = dependent_var
        self.columns_to_exclude = columns_to_exclude
        self.start_year = start_year
        self.end_year = end_year
        self.key_variable = key_variable
        self.missing_values_treatment_stage = missing_values_treatment_stage
        self.numeric_fill_method = numeric_fill_method
        self.categorical_fill_method = categorical_fill_method

    @staticmethod
    def clean_column_names(df):
        """
        Standardize column names by removing extra spaces and ensuring consistency.
        """
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
        df.columns = df.columns.str.replace(' +', ' ', regex=True)  # Replace multiple spaces with single space
        logging.info("Column names cleaned.")
        return df

            
    def preprocess_data(self, data):
        """
        Preprocess the data using configuration variables.
        Returns:
            Tuple (processed_data, dep_var, independent_vars)
        """
        # Clean column names
        data = self.clean_column_names(data)

        # Filter data by years
        filtered_data = data.loc[(data[self.year_col] >= self.start_year) & (data[self.year_col] <= self.end_year)].copy()
        logging.info(f"Data filtered between years {self.start_year} and {self.end_year}.")

        # Drop and create 'mon_year' column
        if 'mon_year' in filtered_data.columns:
            filtered_data.drop(columns=['mon_year'], inplace=True)
            logging.info("'mon_year' column found and dropped to prevent duplication.")

        # Create 'mon_year' column
        if self.month_col != "NA" and self.year_col != "NA":
            try:
                # Concatenate Month and Year as strings
                filtered_data['mon_year'] = filtered_data[self.month_col].astype(str) + filtered_data[self.year_col].astype(str)
                # Convert 'mon_year' to datetime
                filtered_data['mon_year'] = pd.to_datetime(filtered_data['mon_year'], format='%b%Y')
                logging.info("'mon_year' column created successfully in 'mmmyyyy' format.")
            except ValueError as e:
                try:
                    # Attempt with full month name if abbreviated fails
                    filtered_data['mon_year'] = pd.to_datetime(filtered_data['mon_year'], format='%B%Y')
                    logging.info("'mon_year' column created successfully using full month names.")
                except ValueError as e:
                    raise ValueError(f"Error parsing 'mon_year': {e}")
        elif self.date_col != "NA":
            try:
                filtered_data['mon_year'] = pd.to_datetime(filtered_data[self.date_col])
                logging.info("'mon_year' column created successfully from 'Date' column.")
            except ValueError as e:
                raise ValueError(f"Error parsing 'mon_year' from 'Date' column: {e}")
        else:
            raise ValueError("Date or Month/Year columns missing in the configuration.")

        # Extract 'month' and 'year'
        filtered_data['month'] = filtered_data['mon_year'].dt.month
        filtered_data['year'] = filtered_data['mon_year'].dt.year
        logging.info("'month' and 'year' columns extracted from 'mon_year'.")

        # Define independent variables
        independent_vars = [
            col for col in filtered_data.columns
            if col not in ['Date', self.dependent_var, self.key_variable, self.month_col, self.year_col] + self.columns_to_exclude
        ]
        logging.info("Independent variables defined.")
        dep_var = self.dependent_var

        return filtered_data, dep_var, independent_vars

    def handle_missing_values(self, data):
        """
        Handle missing values based on configuration.
        """
        # Separate numeric and categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns

        # Handle numeric columns
        if self.numeric_fill_method:
            if self.numeric_fill_method == 'mean':
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
                logging.info("Numeric columns imputed with mean.")
            elif self.numeric_fill_method == 'median':
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
                logging.info("Numeric columns imputed with median.")
            elif self.numeric_fill_method == 'ffill':
                data[numeric_cols] = data[numeric_cols].fillna(method='ffill')
                logging.info("Numeric columns imputed with forward fill.")
            elif self.numeric_fill_method == 'bfill':
                data[numeric_cols] = data[numeric_cols].fillna(method='bfill')
                logging.info("Numeric columns imputed with backward fill.")

        # Handle categorical columns
        if self.categorical_fill_method:
            if self.categorical_fill_method == 'mode':
                for col in categorical_cols:
                    mode_value = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                    data[col] = data[col].fillna(mode_value)
                logging.info("Categorical columns imputed with mode.")
            elif self.categorical_fill_method == 'ffill':
                data[categorical_cols] = data[categorical_cols].fillna(method='ffill')
                logging.info("Categorical columns imputed with forward fill.")
            elif self.categorical_fill_method == 'bfill':
                data[categorical_cols] = data[categorical_cols].fillna(method='bfill')
                logging.info("Categorical columns imputed with backward fill.")

        return data

    def run_preprocessing(self):
        """
        Run the complete preprocessing pipeline.
        Returns:
            pd.DataFrame: Preprocessed data.
        """
        data = self.uploadedDataST
        data, dep_var, independent_vars = self.preprocess_data(data)
        if self.missing_values_treatment_stage == "do":
            data = self.handle_missing_values(data)
            logging.info("Missing value treatment applied")
        elif self.missing_values_treatment_stage == "skip":
            logging.info("Missing value treatment skipped")
        else:
            logging.info("Provide defined Value for missing_values_treatment_stage Variable")

        logging.info("Data preprocessing completed.")
        return data, dep_var, independent_vars

# If this script is run directly, execute the following code
if __name__ == "__main__":
    # Example usage
    # Provide your file path here
    # data_file_path = "Service Forecasting_original.xlsx"  

    preprocessor = DataPreprocessor(
        uploadedDataST=uploadedDataST,
        month_col=month_col,
        year_col=year_col,
        date_col=date_col,
        dependent_var=dependent_var,
        columns_to_exclude=columns_to_exclude,
        start_year=start_year,
        end_year=end_year,
        key_variable=key_variable,
        missing_values_treatment_stage=missing_values_treatment_stage,
        numeric_fill_method=numeric_fill_method,
        categorical_fill_method=categorical_fill_method
    )

    pre_processed_data, dep_var, independent_vars = preprocessor.run_preprocessing()

    print("Dependent variable:", dep_var)
    print("Independent variables:", independent_vars)
    print("Pre-Processed data sample:")
    print(pre_processed_data.head())