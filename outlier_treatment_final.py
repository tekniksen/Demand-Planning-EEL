# outlier_treatment_final.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
import logging
import json
import os
from data_preprocessing_final import DataPreprocessor  # Adjust the import as per your module name

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
dependent_var = "Total Service Gap (K Euro)"
columns_to_exclude = [
    "Total Service Gap (K Euro)",
    "month",
    "mon_year",
    "Sourcing Location"
]
start_year = 2019
end_year = 2024
key_variable = "Sourcing Location"
missing_values_treatment_stage = "skip"  # Options: 'do', 'skip'
numeric_fill_method = "ffill"  # Options: 'mean', 'median', 'ffill', 'bfill'
categorical_fill_method = "mode"  # Options: 'mode', 'ffill', 'bfill'

# Global Variables for Outlier Detection
outlier_detection_method = "Percentile"
outlier_detection_columns = [
    "Service",
    "Orders (K Euro)",
    "Invoices (K Euro)",
    "incl. Forecast Gap",
    "incl. Operations Gap",
    "incl. Capacity Gap",
    "incl Materials Gap",
    "incl. Logistics Gap",
    "incl. Others Gap"
]
outlier_detection_timeframe = {
    "start_mon_year": "Jan2019",
    "end_mon_year": "Dec2024"
}
outlier_detection_params = {
    "IsolationForest": {
        "rolling_window": 12,
        "num_lags": 3,
        "contamination": 0.05,
        "n_estimators": 100,
        "random_state": 42,
        "fill_method": "median"
    },
    "LOF": {
        "rolling_window": 3,
        "num_lags": 3,
        "n_neighbors": 20,
        "contamination": 0.05,
        "fill_method": "median"
    },
    "CustomSeasonal": {
        "n": 3,
        "upper_bound_threshold": 2.3,
        "lower_bound_threshold": 0.2,
        "upper_correction_method": "mean",
        "lower_correction_multiplier": 0.2,
        "lower_correction_method": "mean"
    },
    "STL_IQR": {
        "iqr_multiplier": 1.5,
        "seasonal": 13,
        "freq": "MS",
        "fill_method": "mean"
    },
    "STL_Variance": {
        "z_threshold": 2.0,
        "seasonal": 13,
        "freq": "MS"
    },
    "Percentile": {
        "lower_percentile_threshold": 0.02,
        "upper_percentile_threshold": 0.98,
        "upper_correction_method": "mean",
        "lower_correction_method": "mean",
        "lower_correction_multiplier": 0.4
    }
}

class OutlierDetector:
    """
    Class for detecting and correcting anomalies in time series data using various methods.
    """

    def __init__(self, file_path=None, preprocessed_data=None, outlier_detection_method=None,
                 outlier_detection_columns=None, outlier_detection_timeframe=None,
                 outlier_detection_params=None, key_variable=None):
        """
        Initialize the OutlierDetector class.

        Parameters:
        - file_path (str): Path to the raw data file that needs to be preprocessed before outlier detection.
        - preprocessed_data (pd.DataFrame): Preprocessed data that can be used directly for outlier detection.
        - outlier_detection_method (str): The method to be used for outlier detection.
        - outlier_detection_columns (list): List of columns to perform outlier detection on.
        - outlier_detection_timeframe (dict): Dictionary with 'start_mon_year' and 'end_mon_year'.
        - outlier_detection_params (dict): Dictionary containing parameters for outlier detection methods.
        - key_variable (str): The key variable for grouping data.

        If both file_path and preprocessed_data are provided, preprocessed_data will be used.
        """
        self.file_path = file_path
        self.preprocessed_data = preprocessed_data
        self.outlier_detection_method = outlier_detection_method
        self.outlier_detection_columns = outlier_detection_columns
        self.outlier_detection_timeframe = outlier_detection_timeframe
        self.outlier_detection_params = outlier_detection_params
        self.key_variable = key_variable

        if self.preprocessed_data is not None:
            # Use provided preprocessed data
            self.data = self.preprocessed_data.copy()
            logging.info("Using provided preprocessed data.")
        elif self.file_path is not None:
            # Load data from file and preprocess it
            self.preprocessor = DataPreprocessor(
                file_path=self.file_path,
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
            self.preprocessed_data, self.dep_var, self.independent_vars = self.preprocessor.run_preprocessing()
            self.data = self.preprocessed_data.copy()
            logging.info("Data has been preprocessed from file.")
        else:
            raise ValueError("Either file_path or preprocessed_data must be provided.")

    def filter_timeframe(self):
        """Filter the data based on the specified timeframe for outlier detection."""
        timeframe = self.outlier_detection_timeframe
        start_mon_year = timeframe.get('start_mon_year')
        end_mon_year = timeframe.get('end_mon_year')

        try:
            start_date = pd.to_datetime(start_mon_year, format='%b%Y')
        except:
            try:
                start_date = pd.to_datetime(start_mon_year)
                logging.warning(f"Start date '{start_mon_year}' parsed with default format.")
            except Exception as e:
                logging.error(f"Error parsing start_mon_year '{start_mon_year}': {e}")
                raise

        try:
            end_date = pd.to_datetime(end_mon_year, format='%b%Y')
        except:
            try:
                end_date = pd.to_datetime(end_mon_year)
                logging.warning(f"End date '{end_mon_year}' parsed with default format.")
            except Exception as e:
                logging.error(f"Error parsing end_mon_year '{end_mon_year}': {e}")
                raise

        # Filter data to the specified timeframe
        self.data['mon_year'] = pd.to_datetime(self.data['mon_year'])
        data_filtered = self.data.loc[(self.data['mon_year'] >= start_date) & (self.data['mon_year'] <= end_date)].copy()
        logging.info(f"Outlier detection: Data filtered between {start_date} and {end_date}.")

        return data_filtered

    # Isolation Forest Outlier Detection
    def detect_and_correct_anomalies_isolation_forest(self):
        """
        Detect and correct anomalies using Isolation Forest.
        Returns:
        - main_df: DataFrame with corrected values (including only specified columns)
        - helper_columns_df: DataFrame containing all variable values, their _Anomaly and _Corrected columns
        """
        # Extract features
        params = self.outlier_detection_params["IsolationForest"]
        outlier_columns = self.outlier_detection_columns
        self.data = self.data.copy()
        for col in outlier_columns:
            if col not in self.data.columns:
                logging.warning(f"Column '{col}' specified for Isolation Forest method not found in data.")
                continue

            # Ensure the column is numeric
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

            # Initialize helper columns
            self.data[f'{col}_Anomaly'] = 'Normal'
            self.data[f'{col}_Corrected'] = self.data[col].astype(float)

            # Prepare data for Isolation Forest
            # Use data within the timeframe for model fitting
            df_timeframe = self.filter_timeframe()
            feature_df = df_timeframe[[col]].copy()
            feature_df[col].fillna(feature_df[col].mean(), inplace=True)

            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_df)

            # Apply Isolation Forest
            iso_forest = IsolationForest(contamination=params['contamination'], n_estimators=params['n_estimators'], random_state=params['random_state'])
            labels = iso_forest.fit_predict(scaled_features)
            anomalies = labels == -1

            # Map anomalies back to the main dataframe
            anomaly_indices = feature_df.index[anomalies]
            self.data.loc[anomaly_indices, f'{col}_Anomaly'] = 'Anomaly'

            # Correct anomalies
            if params['fill_method'] == 'mean':
                fill_value = self.data.loc[self.data[f'{col}_Anomaly'] == 'Normal', col].mean()
            else:
                fill_value = self.data.loc[self.data[f'{col}_Anomaly'] == 'Normal', col].median()

            self.data.loc[anomaly_indices, f'{col}_Corrected'] = fill_value
            self.data.loc[anomaly_indices, col] = fill_value

            logging.info(f"Isolation Forest method: Anomalies detected and corrected for '{col}'.")

        # Prepare helper_columns_df
        helper_columns = [self.key_variable, 'mon_year'] + outlier_columns
        for col in outlier_columns:
            helper_columns.extend([f'{col}_Anomaly', f'{col}_Corrected'])
        helper_columns_df = self.data[helper_columns].copy()

        # Prepare main_df with only the required columns
        required_columns = ['month', 'year', 'mon_year', self.key_variable]
        for col in outlier_columns:
            required_columns.extend([col, f'{col}_Anomaly', f'{col}_Corrected'])
        main_df = self.data[required_columns].copy()

        return main_df, helper_columns_df

    # Local Outlier Factor (LOF) Outlier Detection
    def detect_and_correct_anomalies_lof(self):
        """
        Detect and correct anomalies using Local Outlier Factor (LOF).
        Returns:
        - main_df: DataFrame with corrected values (including only specified columns)
        - helper_columns_df: DataFrame containing all variable values, their _Anomaly and _Corrected columns
        """
        params = self.outlier_detection_params["LOF"]
        outlier_columns = self.outlier_detection_columns
        self.data = self.data.copy()
        for col in outlier_columns:
            if col not in self.data.columns:
                logging.warning(f"Column '{col}' specified for LOF method not found in data.")
                continue

            # Ensure the column is numeric
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

            # Initialize helper columns
            self.data[f'{col}_Anomaly'] = 'Normal'
            self.data[f'{col}_Corrected'] = self.data[col].astype(float)

            # Prepare data for LOF
            df_timeframe = self.filter_timeframe()
            feature_df = df_timeframe[[col]].copy()
            feature_df[col].fillna(feature_df[col].mean(), inplace=True)

            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_df)

            # Apply LOF
            lof = LocalOutlierFactor(n_neighbors=params['n_neighbors'], contamination=params['contamination'])
            labels = lof.fit_predict(scaled_features)
            anomalies = labels == -1

            # Map anomalies back to the main dataframe
            anomaly_indices = feature_df.index[anomalies]
            self.data.loc[anomaly_indices, f'{col}_Anomaly'] = 'Anomaly'

            # Correct anomalies
            if params['fill_method'] == 'mean':
                fill_value = self.data.loc[self.data[f'{col}_Anomaly'] == 'Normal', col].mean()
            else:
                fill_value = self.data.loc[self.data[f'{col}_Anomaly'] == 'Normal', col].median()

            self.data.loc[anomaly_indices, f'{col}_Corrected'] = fill_value
            self.data.loc[anomaly_indices, col] = fill_value

            logging.info(f"LOF method: Anomalies detected and corrected for '{col}'.")

        # Prepare helper_columns_df
        helper_columns = [self.key_variable, 'mon_year'] + outlier_columns
        for col in outlier_columns:
            helper_columns.extend([f'{col}_Anomaly', f'{col}_Corrected'])
        helper_columns_df = self.data[helper_columns].copy()

        # Prepare main_df with only the required columns
        required_columns = ['month', 'year', 'mon_year', self.key_variable]
        for col in outlier_columns:
            required_columns.extend([col, f'{col}_Anomaly', f'{col}_Corrected'])
        main_df = self.data[required_columns].copy()

        return main_df, helper_columns_df

    # STL IQR Anomaly Detection
    def detect_and_correct_anomalies_stl_iqr(self):
        """
        Detect and correct anomalies using STL decomposition and IQR method.
        Returns:
        - main_df: DataFrame with corrected values (including only specified columns)
        - helper_columns_df: DataFrame containing all variable values, their _Anomaly and _Corrected columns
        """
        params = self.outlier_detection_params["STL_IQR"]
        outlier_columns = self.outlier_detection_columns

        self.data = self.data.copy()
        for col in outlier_columns:
            if col not in self.data.columns:
                logging.warning(f"Column '{col}' specified for STL_IQR method not found in data.")
                continue

            # Ensure the column is numeric
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

            # Initialize helper columns
            self.data[f'{col}_Anomaly'] = 'Normal'
            self.data[f'{col}_Corrected'] = self.data[col].astype(float)

            # Prepare data for STL
            df_timeframe = self.filter_timeframe()
            df_col = df_timeframe.set_index('mon_year')[[col]].copy()
            df_col[col].fillna(df_col[col].mean(), inplace=True)

            # Apply STL decomposition
            stl = STL(df_col[col], period=params['seasonal'], robust=True)
            res = stl.fit()
            resid = res.resid

            # Calculate IQR
            Q1 = resid.quantile(0.25)
            Q3 = resid.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - params['iqr_multiplier'] * IQR
            upper_bound = Q3 + params['iqr_multiplier'] * IQR

            # Identify anomalies
            anomalies = (resid < lower_bound) | (resid > upper_bound)
            anomaly_indices = resid[anomalies].index

            # Map anomalies back to the main dataframe
            self.data.loc[anomaly_indices, f'{col}_Anomaly'] = 'Anomaly'

            # Correct anomalies
            if params.get('fill_method', 'mean') == 'mean':
                fill_value = self.data.loc[self.data[f'{col}_Anomaly'] == 'Normal', col].mean()
            else:
                fill_value = self.data.loc[self.data[f'{col}_Anomaly'] == 'Normal', col].median()

            self.data.loc[anomaly_indices, f'{col}_Corrected'] = fill_value
            self.data.loc[anomaly_indices, col] = fill_value

            logging.info(f"STL_IQR method: Anomalies detected and corrected for '{col}'.")

        # Prepare helper_columns_df
        helper_columns = [self.key_variable, 'mon_year'] + outlier_columns
        for col in outlier_columns:
            helper_columns.extend([f'{col}_Anomaly', f'{col}_Corrected'])
        helper_columns_df = self.data[helper_columns].copy()

        # Prepare main_df with only the required columns
        required_columns = ['month', 'year', 'mon_year', self.key_variable]
        for col in outlier_columns:
            required_columns.extend([col, f'{col}_Anomaly', f'{col}_Corrected'])
        main_df = self.data[required_columns].copy()

        return main_df, helper_columns_df

    # Percentile Based Outlier Detection
    def detect_and_correct_anomalies_percentile(self):
        """
        Detect and correct anomalies using the Percentile Rank method.
        Returns:
        - main_df: DataFrame with corrected values (including only specified columns)
        - helper_columns_df: DataFrame containing all variable values, their _Anomaly and _Corrected columns, and other helper columns
        """
        params = self.outlier_detection_params['Percentile']
        outlier_columns = self.outlier_detection_columns

        # Initialize a DataFrame to collect helper columns
        helper_columns_list = []

        for col in outlier_columns:
            if col not in self.data.columns:
                logging.warning(f"Column '{col}' specified for Percentile method not found in data.")
                continue

            # Ensure the column is numeric
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            self.data[col] = self.data[col].astype(float)  # Ensure it's float to handle decimal corrections

            # Initialize helper columns
            self.data[f'{col}_Anomaly'] = 'Normal'
            self.data[f'{col}_Corrected'] = self.data[col].astype(float)
            self.data[f'{col}_Percentile'] = np.nan

            # Group by key_variable
            df_timeframe = self.filter_timeframe()
            grouped = df_timeframe.groupby(self.key_variable)

            for key, group in grouped:
                group = group.dropna(subset=[col]).copy()
                if group.empty:
                    continue

                # Calculate percentile ranks
                group['percentile'] = group[col].rank(pct=True)

                # Identify anomalies
                lower_outliers = group['percentile'] <= params['lower_percentile_threshold']
                upper_outliers = group['percentile'] >= params['upper_percentile_threshold']
                anomalies = lower_outliers | upper_outliers

                # Update anomalies and percentiles
                self.data.loc[group.index, f'{col}_Percentile'] = group['percentile']
                self.data.loc[group.index[anomalies], f'{col}_Anomaly'] = 'Anomaly'

                # Correct anomalies
                for idx in group.index[upper_outliers]:
                    month = group.loc[idx, 'month']
                    same_period_data = group[(group['month'] == month) & (~anomalies)]

                    if not same_period_data.empty:
                        if params['upper_correction_method'] == 'mean':
                            correction_value = same_period_data[col].mean()
                        else:
                            correction_value = same_period_data[col].median()
                    else:
                        if params['upper_correction_method'] == 'mean':
                            correction_value = group.loc[~anomalies, col].mean()
                        else:
                            correction_value = group.loc[~anomalies, col].median()

                    # Assign corrected value
                    correction_value = float(correction_value)
                    self.data.loc[idx, f'{col}_Corrected'] = correction_value
                    self.data.loc[idx, col] = correction_value

                for idx in group.index[lower_outliers]:
                    month = group.loc[idx, 'month']
                    same_period_data = group[(group['month'] == month) & (~anomalies)]

                    if not same_period_data.empty:
                        if params['lower_correction_method'] == 'mean':
                            correction_value = params['lower_correction_multiplier'] * same_period_data[col].mean()
                        else:
                            correction_value = params['lower_correction_multiplier'] * same_period_data[col].median()
                    else:
                        if params['lower_correction_method'] == 'mean':
                            correction_value = params['lower_correction_multiplier'] * group.loc[~anomalies, col].mean()
                        else:
                            correction_value = params['lower_correction_multiplier'] * group.loc[~anomalies, col].median()

                    # Assign corrected value
                    correction_value = float(correction_value)
                    self.data.loc[idx, f'{col}_Corrected'] = correction_value
                    self.data.loc[idx, col] = correction_value

            logging.info(f"Percentile Rank method: Anomalies detected and corrected for '{col}'.")

        # Prepare helper_columns_df
        helper_columns = [self.key_variable, 'mon_year'] + outlier_columns
        for col in outlier_columns:
            helper_columns.extend([f'{col}_Anomaly', f'{col}_Corrected', f'{col}_Percentile'])
        helper_columns_df = self.data[helper_columns].copy()

        # Prepare main_df with only the required columns
        required_columns = ['month', 'year', 'mon_year', self.key_variable]
        for col in outlier_columns:
            required_columns.extend([col, f'{col}_Anomaly', f'{col}_Corrected'])
        main_df = self.data[required_columns].copy()

        return main_df, helper_columns_df

    def run_method(self):
        """ Run the selected outlier detection methods.
        Returns:
        - main_df: DataFrame with corrected values (including only specified columns)
        - helper_columns_df: DataFrame containing all variable values, their _Anomaly and _Corrected columns, and other helper columns
        """

        method = self.outlier_detection_method
        # Apply the selected outlier detection method
        if method == 'Percentile':
            main_df, helper_columns_df = self.detect_and_correct_anomalies_percentile()
        elif method == 'IsolationForest':
            main_df, helper_columns_df = self.detect_and_correct_anomalies_isolation_forest()
        elif method == 'LOF':
            main_df, helper_columns_df = self.detect_and_correct_anomalies_lof()
        elif method == 'STL_IQR':
            main_df, helper_columns_df = self.detect_and_correct_anomalies_stl_iqr()
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")

        logging.info(f"Outlier detection and correction using {self.outlier_detection_method} completed.")

        return main_df, helper_columns_df

# Example usage
if __name__ == "__main__":
    # Option 1: Run with a raw file path for preprocessing
    data_file_path = "path_to_your_data_file.xlsx"  # Replace with your actual file path

    outlier_detector = OutlierDetector(
        file_path=data_file_path,
        outlier_detection_method=outlier_detection_method,
        outlier_detection_columns=outlier_detection_columns,
        outlier_detection_timeframe=outlier_detection_timeframe,
        outlier_detection_params=outlier_detection_params,
        key_variable=key_variable
    )

    main_df, helper_columns_df = outlier_detector.run_method()
    print("\nPreprocessed Data:")
    print(outlier_detector.preprocessed_data)
    print("\nFinal Data After Outlier Detection and Corrections:")
    print(outlier_detector.data)

