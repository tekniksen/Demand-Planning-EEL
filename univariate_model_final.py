# univariate_model_final.py

import pandas as pd
import numpy as np
import logging
import warnings
import streamlit as st

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Nixtla's libraries
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA, AutoETS, AutoTheta, AutoCES, ARIMA, HoltWinters,
    OptimizedTheta, DynamicOptimizedTheta, DynamicTheta, Holt,
    AutoMFLES, SeasonalExponentialSmoothing, SeasonalExponentialSmoothingOptimized,
    SimpleExponentialSmoothing, SimpleExponentialSmoothingOptimized, Theta,
    ARCH, GARCH,
    ADIDA, CrostonClassic, CrostonOptimized, CrostonSBA, IMAPA,
)

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Hyperparameter tuning
import optuna
from scipy import stats  # For Box-Cox transformation

# Import TimeSeriesCharacteristics class
from ts_characteristics import TimeSeriesCharacteristics

# Import SeasonalityDetector class
from seasonality_acf_test import SeasonalityDetector

# Configure logging
logging.basicConfig(
    filename='auto_forecast_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Global Parameters

# Time Series Characteristics parameters
series_identifier_cols = ['Sourcing Location']  # Use the id_col as series identifier
date_col = 'mon_year'   # Date column
target_col = 'Invoices (K Euro)'  # Target variable
dep_var = target_col  # Dependent variable
frequency = 'MS'        # Monthly start frequency
ts_freq = frequency  # Time series frequency

# Default season length based on frequency
frequency_to_season_length = {
    'D': 7,    # Daily data, weekly seasonality
    'W': 52,   # Weekly data, yearly seasonality
    'M': 12,   # Monthly data, yearly seasonality
    'MS': 12,  # Monthly start data, yearly seasonality
    'Q': 4,    # Quarterly data, yearly seasonality
    'QS': 4,   # Quarterly start data, yearly seasonality
    'Y': 1,    # Yearly data, no seasonality
    'YS': 1    # Yearly start data, no seasonality
}
sp = frequency_to_season_length.get(frequency, 1)  # Seasonal period

delimiter = '_'
npi_cutoff = 12  # New product intro timeline cutoff
eol_cutoff = 12  # End of life timeline cutoff
segment_time = 12  # Segmentation time period in weeks or months depending on time series frequency
abc_cutoff = [0.3, 0.8]  # Cutoffs for volume segments
cov_cutoff = [0.5, 1]  # Cutoffs for variability/CoV segments
adi_cutoff = 1.32  # Average demand interval cutoff for demand classes
nz_cov_cutoff = 0.49  # Non-zero CoV cutoff for demand classes
imt_cutoff = 0.5  # Intermittency cutoff (ratio of zeros to total observations)
train_end_date = None  # Training end date, will be set after loading data

# Other Global Parameters
train_size_ratio = 0.8  # Ratio for train-test split
cv_folds = 3            # Number of cross-validation folds
future_periods = 12     # Number of future periods to forecast

# Evaluation metric weights
mse_weight = 0.7
bias_magnitude_weight = 0.2
bias_direction_weight = 0.1

# Transformation flags
use_transformation = True  # Set to True to use data transformation
use_log1p = True           # Set to True to use log1p transformation
use_boxcox = False         # Set to True to use Box-Cox transformation

# Model inclusion flags
use_ARIMA = True
use_HoltWinters = False
use_ARCH = True
use_DOT = False
use_DSTM = False
use_GARCH = False
use_Holt = False
use_MFLES = False
use_OptimizedTheta = False
use_SeasonalES = False
use_SeasonalESOptimized = False
use_SESOptimized = False
use_SES = False
use_Theta = False

# Intermittent models flag (user can overwrite this)
use_intermittent_models = True  # Set to True to include intermittent models

# Auto models inclusion flag
use_auto_models = True  # Set to False to exclude auto models like AutoARIMA and AutoETS

# Seasonality detection parameters
get_seasonality = False  # Set to True to detect seasonality
seasonality_lags = 12  # Number of lags to consider in ACF
skip_lags = 11
lower_ci_threshold = -0.10
upper_ci_threshold = 0.90

# ts_characteristics flag
ts_characteristics_flag = True  # Set to True to compute time series characteristics

# detect_intermittency flag
detect_intermittency = True  # Set to True to detect intermittency

class ModelBuilder:
    def __init__(self, data_path=None, data=None):
        """
        Initialize the ModelBuilder with the path to the data or a DataFrame.
        Parameters:
        - data_path (str): Path to the data file.
        - data (pd.DataFrame): DataFrame containing the data.
        """
        self.data_path = data_path
        self.data = data
        self.future_forecasts = None  # DataFrame to store future forecasts for all series
        self.models_results = {}       # Dictionary to store results for all models and series
        self.ts_char_df = None         # DataFrame to store time series characteristics
        self.train_end_date = None     # Training end date, to be set after loading data
        self.fitted_lambdas = {}       # To store fitted lambdas for Box-Cox
        self.shifts = {}               # To store shifts applied for Box-Cox

    def load_data(self):
        """
        Load the data from the provided DataFrame or file path.
        """
        if self.data is not None:
            logging.info("Using provided DataFrame as data.")
            # Ensure date column is datetime
            self.data[date_col] = pd.to_datetime(self.data[date_col], errors='coerce')
        elif self.data_path is not None:
            if self.data_path.endswith('.csv'):
                self.data = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.xls') or self.data_path.endswith('.xlsx'):
                self.data = pd.read_excel(self.data_path, engine='openpyxl')
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or XLS/XLSX file.")
            logging.info(f"Data loaded successfully from {self.data_path}.")
            # Ensure date column is datetime
            self.data[date_col] = pd.to_datetime(self.data[date_col], errors='coerce')
        else:
            raise ValueError("No data provided. Please provide a data_path or a DataFrame.")

        if self.data[date_col].isnull().any():
            raise ValueError(f"Date column '{date_col}' contains invalid datetime entries.")

        # Set train_end_date to the maximum date in the data
        self.train_end_date = self.data[date_col].max()

    def compute_ts_characteristics(self):
        """
        Compute time series characteristics if the flag is set.
        """
        if ts_characteristics_flag:
            logging.info("Computing time series characteristics...")
            ts_char_obj = TimeSeriesCharacteristics(
                data=self.data,
                series_identifier_cols=series_identifier_cols,
                date_col=date_col,
                dep_var=dep_var,
                ts_freq=ts_freq,
                sp=sp,
                delimiter=delimiter,
                npi_cutoff=npi_cutoff,
                eol_cutoff=eol_cutoff,
                segment_time=segment_time,
                abc_cutoff=abc_cutoff,
                cov_cutoff=cov_cutoff,
                adi_cutoff=adi_cutoff,
                nz_cov_cutoff=nz_cov_cutoff,
                imt_cutoff=imt_cutoff,
                train_end_date=self.train_end_date  # Pass as datetime object
            )
            self.ts_char_df = ts_char_obj.ts_characteristics()
            self.ts_char_df.to_csv('ts_characteristics.csv', index=False)
            logging.info("Time series characteristics saved to 'ts_characteristics.csv'.")
        else:
            logging.info("Time series characteristics computation skipped.")

    def evaluate_model(self, y_true, y_pred, train_data):
        """
        Evaluate the model using the combined metric with IQR normalization.
        """
        if np.isnan(y_pred).any() or np.isnan(y_true).any():
            logging.error("NaN values encountered in y_true or y_pred.")
            return {
                'RMSE': np.nan,
                'MAE': np.nan,
                'Bias': np.nan,
                'CombinedMetric': np.nan
            }

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        bias_magnitude = np.abs(bias)
        bias_direction = np.sign(bias)

        # Use IQR for normalization
        Q1 = train_data['y'].quantile(0.25)
        Q3 = train_data['y'].quantile(0.75)
        iqr = Q3 - Q1
        nrmse = rmse / iqr if iqr != 0 else rmse
        normalized_bias_magnitude = bias_magnitude / iqr if iqr != 0 else bias_magnitude

        # Combine metrics using specified weights
        combined_metric = (
            mse_weight * nrmse +
            bias_magnitude_weight * normalized_bias_magnitude +
            bias_direction_weight * bias_direction
        )

        return {
            'RMSE': rmse,
            'MAE': mae,
            'Bias': bias,
            'CombinedMetric': combined_metric
        }

    def inverse_transform(self, y_transformed, series_id):
        """
        Apply the inverse transformation to the forecasted values.
        """
        if use_transformation:
            if use_log1p:
                y_inversed = np.expm1(y_transformed)
            elif use_boxcox:
                fitted_lambda = self.fitted_lambdas.get(series_id)
                shift = self.shifts.get(series_id, 0)
                if fitted_lambda is not None:
                    y_inversed = stats.inv_boxcox(y_transformed, fitted_lambda) - shift
                else:
                    logging.error(f"Fitted lambda not found for series {series_id}. Cannot inverse transform.")
                    y_inversed = y_transformed  # Return without inverse transformation
            else:
                y_inversed = y_transformed  # No transformation
        else:
            y_inversed = y_transformed  # No transformation
        return y_inversed

    def get_models(self, season_length, is_intermittent):
        """
        Generate the list of models based on the inclusion flags and detected seasonality.

        Parameters:
        - season_length (int): Detected seasonality length for the series.
        - is_intermittent (bool): Whether the series is intermittent.

        Returns:
        - models (list): List of auto models.
        - manual_models (list): List of manual models with default parameters.
        """
        models = []
        manual_models = []

        # Manual models with hyperparameter tuning
        if use_ARIMA:
            manual_models.append(('ARIMA', ARIMA(order=(1, 1, 1), alias='ARIMA')))

        if use_HoltWinters:
            manual_models.append(('HoltWinters', HoltWinters(season_length=season_length, alias='HoltWinters')))

        if use_ARCH:
            manual_models.append(('ARCH', ARCH(p=2, alias='ARCH')))

        if use_GARCH:
            manual_models.append(('GARCH', GARCH(p=1, q=1, alias='GARCH')))

        if use_Holt:
            manual_models.append(('Holt', Holt(season_length=season_length, alias='Holt')))

        if use_OptimizedTheta:
            manual_models.append(('OptimizedTheta', OptimizedTheta(season_length=season_length, alias='OptimizedTheta')))

        if use_SeasonalES:
            manual_models.append(('SeasonalES', SeasonalExponentialSmoothing(alpha=0.1, season_length=season_length, alias='SeasonalES')))

        if use_SeasonalESOptimized:
            manual_models.append(('SeasonalESOptimized', SeasonalExponentialSmoothingOptimized(season_length=season_length, alias='SeasonalESOptimized')))

        if use_SESOptimized:
            manual_models.append(('SESOptimized', SimpleExponentialSmoothingOptimized(alias='SESOptimized')))

        if use_SES:
            manual_models.extend([
                ('SES_alpha', SimpleExponentialSmoothing(alpha=0.1, alias='SES_alpha')),
            ])

        if use_Theta:
            manual_models.append(('Theta', Theta(season_length=season_length, alias='Theta')))

        if use_DOT:
            manual_models.append(('DynamicOptimizedTheta', DynamicOptimizedTheta(season_length=season_length, alias='DynamicOptimizedTheta')))

        if use_DSTM:
            manual_models.append(('DynamicTheta', DynamicTheta(season_length=season_length, alias='DynamicTheta')))

        # Include intermittent models only if use_intermittent_models is True and the series is intermittent
        if is_intermittent:
            if use_intermittent_models:
                manual_models.extend([
                    ('ADIDA', ADIDA(alias='ADIDA')),
                    ('CrostonClassic', CrostonClassic(alias='CrostonClassic')),
                    ('CrostonOptimized', CrostonOptimized(alias='CrostonOptimized')),
                    ('CrostonSBA', CrostonSBA(alias='CrostonSBA')),
                    ('IMAPA', IMAPA(alias='IMAPA'))
                ])

        # Auto models
        if use_auto_models:
            auto_models = [
                AutoETS(season_length=season_length, alias='AutoETS'),
                AutoTheta(season_length=season_length, decomposition_type="additive", model="STM", alias='AutoTheta'),
                AutoARIMA(season_length=season_length, alias='AutoARIMA'),
                AutoCES(season_length=season_length, alias='AutoCES')
            ]
            models.extend(auto_models)

        return models, manual_models

    def hyperparameter_tuning_manual_models(self, series_data, train_data, model_name, season_length):
        """
        Perform hyperparameter tuning for a single manual model using Optuna hyperparameter tuning.
        Returns the best hyperparameters for the model.
        """
        logging.info(f"Hyperparameter tuning for model: {model_name} on series.")

        def objective(trial):
            # Define hyperparameter space based on model
            if model_name == 'ARIMA':
                p = trial.suggest_int('p', 0, 5)
                d = trial.suggest_int('d', 0, 2)
                q = trial.suggest_int('q', 0, 5)
                order = (p, d, q)
                current_model = ARIMA(order=order, alias=model_name)
            elif model_name == 'HoltWinters':
                error_type = trial.suggest_categorical('error_type', ['A', 'M'])
                current_model = HoltWinters(season_length=season_length, error_type=error_type, alias=model_name)
            elif model_name == 'OptimizedTheta':
                decomposition_type = trial.suggest_categorical('decomposition_type', ['additive', 'multiplicative'])
                current_model = OptimizedTheta(season_length=season_length, decomposition_type=decomposition_type, alias=model_name)
            elif model_name == 'ARCH':
                p = trial.suggest_int('p', 1, 5)
                current_model = ARCH(p=p, alias=model_name)
            elif model_name == 'GARCH':
                p = trial.suggest_int('p', 1, 5)
                q = trial.suggest_int('q', 1, 5)
                current_model = GARCH(p=p, q=q, alias=model_name)
            elif model_name == 'Holt':
                error_type = trial.suggest_categorical('error_type', ['A', 'M'])
                current_model = Holt(season_length=season_length, error_type=error_type, alias=model_name)
            elif model_name == 'SeasonalES':
                alpha = trial.suggest_float('alpha', 0.1, 1.0)
                current_model = SeasonalExponentialSmoothing(alpha=alpha, season_length=season_length, alias=model_name)
            elif model_name == 'SeasonalESOptimized':
                current_model = SeasonalExponentialSmoothingOptimized(season_length=season_length, alias=model_name)
            elif model_name == 'SESOptimized':
                current_model = SimpleExponentialSmoothingOptimized(alias=model_name)
            elif model_name == 'SES_alpha':
                alpha = trial.suggest_float('alpha', 0.1, 1.0)
                current_model = SimpleExponentialSmoothing(alpha=alpha, alias=model_name)
            elif model_name == 'Theta':
                decomposition_type = trial.suggest_categorical('decomposition_type', ['additive', 'multiplicative'])
                current_model = Theta(season_length=season_length, decomposition_type=decomposition_type, alias=model_name)
            elif model_name == 'DynamicOptimizedTheta':
                decomposition_type = trial.suggest_categorical('decomposition_type', ['additive', 'multiplicative'])
                current_model = DynamicOptimizedTheta(season_length=season_length, decomposition_type=decomposition_type, alias=model_name)
            elif model_name == 'DynamicTheta':
                decomposition_type = trial.suggest_categorical('decomposition_type', ['additive', 'multiplicative'])
                current_model = DynamicTheta(season_length=season_length, decomposition_type=decomposition_type, alias=model_name)
            elif model_name in ['ADIDA', 'CrostonClassic', 'CrostonOptimized', 'CrostonSBA', 'IMAPA']:
                # Intermittent models
                alpha = trial.suggest_float('alpha', 0.1, 1.0)
                if model_name == 'ADIDA':
                    current_model = ADIDA(alias=model_name)
                elif model_name == 'CrostonClassic':
                    current_model = CrostonClassic(alpha=alpha, alias=model_name)
                elif model_name == 'CrostonOptimized':
                    current_model = CrostonOptimized(alias=model_name)
                elif model_name == 'CrostonSBA':
                    current_model = CrostonSBA(alpha=alpha, alias=model_name)
                elif model_name == 'IMAPA':
                    current_model = IMAPA(alias=model_name)
            else:
                return float('inf')  # Skip unknown models

            # Initialize StatsForecast with the current model
            sf_cv = StatsForecast(
                models=[current_model],
                freq=frequency,
                n_jobs=1
            )

            # Perform cross-validation
            try:
                cv_results = sf_cv.cross_validation(
                    df=train_data[['unique_id', 'ds', 'y']],
                    h=future_periods,
                    step_size=season_length,
                    n_windows=cv_folds
                )
            except Exception as e:
                logging.error(f"Error during cross-validation in hyperparameter tuning for model {model_name}: {e}")
                return float('inf')

            # Check if model_name is in cv_results columns
            if model_name not in cv_results.columns:
                logging.error(f"Model {model_name} did not produce predictions during cross-validation.")
                return float('inf')

            # Compute combined metric over folds
            combined_metrics = []
            for fold in cv_results['cutoff'].unique():
                fold_df = cv_results[cv_results['cutoff'] == fold]
                y_true = fold_df['y'].values
                y_pred = fold_df[model_name].values

                if np.isnan(y_pred).any() or np.isnan(y_true).any():
                    logging.error(f"NaN values encountered in predictions or true values for model {model_name} during hyperparameter tuning.")
                    return float('inf')

                try:
                    metrics = self.evaluate_model(y_true, y_pred, train_data)
                    combined_metrics.append(metrics['CombinedMetric'])
                except Exception as e:
                    logging.error(f"Error during metric computation in hyperparameter tuning for model {model_name}: {e}")
                    return float('inf')

            avg_combined_metric = np.mean(combined_metrics)

            return avg_combined_metric

        # Create Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)  # Adjust n_trials as needed

        best_params = study.best_params
        logging.info(f"Best params for model {model_name}: {best_params}")

        return best_params

    def build_univariate_models(self):
        """
        Build and evaluate univariate models for each time series using cross-validation and hyperparameter tuning.
        """
        logging.info("Building univariate models with cross-validation and hyperparameter tuning...")

        # Prepare data
        sf_data = self.data[[series_identifier_cols[0], date_col, dep_var]].copy()
        sf_data = sf_data.rename(columns={series_identifier_cols[0]: 'unique_id', date_col: 'ds', dep_var: 'y'})
        sf_data = sf_data.sort_values(by=['unique_id', 'ds'])
        sf_data.reset_index(drop=True, inplace=True)

        # Resample data to ensure consistent frequency
        def resample_group(group):
            group = group.set_index('ds').asfreq(frequency)
            group['unique_id'] = group['unique_id'].iloc[0]
            return group.reset_index()

        sf_data = sf_data.groupby('unique_id').apply(resample_group).reset_index(drop=True)

        # Fill missing values if any
        sf_data['y'] = sf_data['y'].fillna(0)

        # Apply data transformations if specified
        self.fitted_lambdas = {}  # To store fitted lambdas for Box-Cox
        self.shifts = {}          # To store shifts applied for Box-Cox

        if use_transformation:
            for series_id in sf_data['unique_id'].unique():
                series_mask = sf_data['unique_id'] == series_id
                y_series = sf_data.loc[series_mask, 'y']

                if use_log1p:
                    # Apply log1p transformation
                    sf_data.loc[series_mask, 'y'] = np.log1p(y_series)
                elif use_boxcox:
                    # Shift data to be positive
                    min_value = y_series.min()
                    shift = abs(min_value) + 1 if min_value <= 0 else 0
                    y_shifted = y_series + shift

                    # Apply Box-Cox transformation
                    try:
                        y_transformed, fitted_lambda = stats.boxcox(y_shifted)
                        sf_data.loc[series_mask, 'y'] = y_transformed

                        # Store fitted lambda and shift for inverse transformation
                        self.fitted_lambdas[series_id] = fitted_lambda
                        self.shifts[series_id] = shift
                    except Exception as e:
                        logging.error(f"Box-Cox transformation failed for series {series_id}: {e}")
                        sf_data.loc[series_mask, 'y'] = y_series  # Revert to original
                else:
                    logging.warning("No transformation method selected despite 'use_transformation' being True.")
        else:
            logging.info("No data transformation applied.")

        # List of unique series identifiers
        unique_series = sf_data['unique_id'].unique()

        # Initialize dictionaries to store results
        self.models_results = {}
        future_forecasts_list = []

        # Process each time series individually
        for series_id in unique_series:
            logging.info(f"Processing series: {series_id}")
            series_data = sf_data[sf_data['unique_id'] == series_id].copy()

            # Remove any rows with missing target values
            series_data = series_data.dropna(subset=['y'])

            # Split data into training and testing sets
            split_index = int(len(series_data) * train_size_ratio)
            train_data = series_data.iloc[:split_index].copy()
            test_data = series_data.iloc[split_index:].copy()

            if len(train_data) < 24:
                logging.warning(f"Not enough data to train for series {series_id}. Skipping.")
                continue

            # Initialize a dictionary to store the results for the current series
            self.models_results[series_id] = {}

            # Initialize test_metrics
            test_metrics = {}

            # Detect seasonality for the series if get_seasonality is True
            if get_seasonality:
                seasonality_detector = SeasonalityDetector(
                    data=series_data,
                    date_col='ds',
                    target_col='y',
                    lags=seasonality_lags,
                    skip_lags=skip_lags,
                    lower_ci_threshold=lower_ci_threshold,
                    upper_ci_threshold=upper_ci_threshold
                )
                seasonality_detected, seasonality_detected_flag = seasonality_detector.detect_seasonality()

                if seasonality_detected_flag and seasonality_detected:
                    season_length_list = seasonality_detected  # It's already a list of integers
                    season_length = season_length_list[0]  # Use the most significant seasonality
                    logging.info(f"Seasonality detected for series {series_id}: {season_length}")
                else:
                    # Use default season length based on frequency
                    season_length = sp
                    logging.info(f"No significant seasonality detected for series {series_id}. Using default season_length={season_length}.")
            else:
                # Use default season length based on frequency
                season_length = sp
                logging.info(f"Seasonality detection skipped. Using default season_length={season_length} for series {series_id}.")

            # Determine if the series is intermittent using ts_characteristics
            if ts_characteristics_flag and self.ts_char_df is not None:
                ts_char_series = self.ts_char_df[self.ts_char_df[series_identifier_cols[0]] == series_id]
                if not ts_char_series.empty:
                    intermittency = ts_char_series['Intermittency'].values[0]
                    demand_class = ts_char_series['Demand Class'].values[0]
                    is_stationary = ts_char_series['Stationary'].values[0]
                    trend_category = ts_char_series['trend_category'].values[0]
                    trend_strength = ts_char_series['trend_strength'].values[0]
                else:
                    intermittency = "Non-Intermittent"
                    demand_class = "Unknown"
                    is_stationary = "Unknown"
                    trend_category = "Unknown"
                    trend_strength = np.nan
            else:
                intermittency = "Non-Intermittent"
                demand_class = "Unknown"
                is_stationary = "Unknown"
                trend_category = "Unknown"
                trend_strength = np.nan

            # Decide whether to use intermittent models
            is_intermittent_series = False
            if detect_intermittency and use_intermittent_models and intermittency == "Intermittent":
                is_intermittent_series = True

            # Get models with the detected or default season_length
            models, manual_models = self.get_models(season_length, is_intermittent=is_intermittent_series)

            # Perform hyperparameter tuning for manual models
            best_manual_params = {}
            tuned_manual_models = []
            for model_name, model in manual_models:
                best_params = self.hyperparameter_tuning_manual_models(series_data, train_data, model_name, season_length)
                best_manual_params[model_name] = best_params
                # Update manual models with best parameters
                if model_name == 'ARIMA':
                    p = best_params.get('p', 1)
                    d = best_params.get('d', 1)
                    q = best_params.get('q', 1)
                    tuned_model = ARIMA(order=(p, d, q), alias=model_name)
                elif model_name == 'HoltWinters':
                    error_type = best_params.get('error_type', "A")
                    tuned_model = HoltWinters(season_length=season_length, error_type=error_type, alias=model_name)
                elif model_name == 'OptimizedTheta':
                    decomposition_type = best_params.get('decomposition_type', "additive")
                    tuned_model = OptimizedTheta(season_length=season_length, decomposition_type=decomposition_type, alias=model_name)
                elif model_name == 'ARCH':
                    p = best_params.get('p', 2)
                    tuned_model = ARCH(p=p, alias=model_name)
                elif model_name == 'GARCH':
                    p = best_params.get('p', 2)
                    q = best_params.get('q', 2)
                    tuned_model = GARCH(p=p, q=q, alias=model_name)
                elif model_name == 'Holt':
                    error_type = best_params.get('error_type', 'A')
                    tuned_model = Holt(season_length=season_length, error_type=error_type, alias=model_name)
                elif model_name == 'SeasonalES':
                    alpha = best_params.get('alpha', 0.1)
                    tuned_model = SeasonalExponentialSmoothing(alpha=alpha, season_length=season_length, alias=model_name)
                elif model_name == 'SeasonalESOptimized':
                    tuned_model = SeasonalExponentialSmoothingOptimized(season_length=season_length, alias=model_name)
                elif model_name == 'SESOptimized':
                    tuned_model = SimpleExponentialSmoothingOptimized(alias=model_name)
                elif model_name == 'SES_alpha':
                    alpha = best_params.get('alpha', 0.1)
                    tuned_model = SimpleExponentialSmoothing(alpha=alpha, alias=model_name)
                elif model_name == 'Theta':
                    decomposition_type = best_params.get('decomposition_type', "additive")
                    tuned_model = Theta(season_length=season_length, decomposition_type=decomposition_type, alias=model_name)
                elif model_name == 'DynamicOptimizedTheta':
                    decomposition_type = best_params.get('decomposition_type', "additive")
                    tuned_model = DynamicOptimizedTheta(season_length=season_length, decomposition_type=decomposition_type, alias=model_name)
                elif model_name == 'DynamicTheta':
                    decomposition_type = best_params.get('decomposition_type', "additive")
                    tuned_model = DynamicTheta(season_length=season_length, decomposition_type=decomposition_type, alias=model_name)
                elif model_name in ['ADIDA', 'CrostonClassic', 'CrostonOptimized', 'CrostonSBA', 'IMAPA']:
                    alpha = best_params.get('alpha', 0.1)
                    if model_name == 'ADIDA':
                        tuned_model = ADIDA(alias=model_name)
                    elif model_name == 'CrostonClassic':
                        tuned_model = CrostonClassic(alpha=alpha, alias=model_name)
                    elif model_name == 'CrostonOptimized':
                        tuned_model = CrostonOptimized(alias=model_name)
                    elif model_name == 'CrostonSBA':
                        tuned_model = CrostonSBA(alpha=alpha, alias=model_name)
                    elif model_name == 'IMAPA':
                        tuned_model = IMAPA(alias=model_name)
                    else:
                        tuned_model = model  # Default
                else:
                    tuned_model = model  # Default

                tuned_manual_models.append(tuned_model)

            # Combine all models with tuned manual models
            final_models = models + tuned_manual_models

            # Perform cross-validation on train data for all models
            cv_metrics = {}
            if final_models:
                for model_instance in final_models:
                    model_name = model_instance.alias
                    logging.info(f"Performing cross-validation for model: {model_name} on series {series_id}.")

                    sf_cv = StatsForecast(
                        models=[model_instance],
                        freq=frequency,
                        n_jobs=1
                    )

                    try:
                        cv_results = sf_cv.cross_validation(
                            df=train_data[['unique_id', 'ds', 'y']],
                            h=future_periods,
                            step_size=season_length,
                            n_windows=cv_folds
                        )

                        # Check if model_name is in cv_results columns
                        if model_name not in cv_results.columns:
                            logging.error(f"Model {model_name} did not produce predictions during cross-validation.")
                            cv_metrics[model_name] = float('inf')
                            continue  # Skip to next model

                        # Compute combined metric over folds
                        combined_metrics = []
                        for fold in cv_results['cutoff'].unique():
                            fold_df = cv_results[cv_results['cutoff'] == fold]
                            y_true = fold_df['y'].values
                            y_pred = fold_df[model_name].values

                            if np.isnan(y_pred).any() or np.isnan(y_true).any():
                                logging.error(f"NaN values encountered in predictions or true values for model {model_name} during cross-validation.")
                                cv_metrics[model_name] = float('inf')
                                break  # Exit the loop since we cannot compute metrics
                            try:
                                metrics = self.evaluate_model(y_true, y_pred, train_data)
                                combined_metrics.append(metrics['CombinedMetric'])
                            except Exception as e:
                                logging.error(f"Error during metric computation in cross-validation for model {model_name}: {e}")
                                cv_metrics[model_name] = float('inf')
                                break  # Exit the loop since we cannot compute metrics

                        else:
                            avg_combined_metric = np.mean(combined_metrics)
                            cv_metrics[model_name] = avg_combined_metric
                            logging.info(f"Average Combined Metric for model {model_name}: {avg_combined_metric}")
                    except Exception as e:
                        logging.error(f"Error during cross-validation for model {model_name}: {e}")
                        cv_metrics[model_name] = float('inf')
            else:
                logging.warning(f"No models to evaluate during cross-validation for series {series_id}.")

            # Evaluate each model on test data
            if final_models:
                # Forecast on test data
                h_test = len(test_data)
                sf_all = StatsForecast(
                    models=final_models,
                    freq=frequency,
                    n_jobs=-1
                )
                sf_all.fit(df=train_data[['unique_id', 'ds', 'y']])

                forecast_test = sf_all.predict(h=h_test)
                forecast_test = forecast_test.set_index('ds')

                # Evaluate each model on test data
                for model_instance in final_models:
                    model_name = model_instance.alias
                    if model_name in forecast_test.columns:
                        y_pred_test = forecast_test[model_name].values
                    else:
                        logging.error(f"Forecasts for model '{model_name}' not found on test data for series '{series_id}'.")
                        y_pred_test = np.full(len(test_data), np.nan)

                    y_true_test = test_data['y'].values

                    if np.isnan(y_pred_test).any() or np.isnan(y_true_test).any():
                        logging.error(f"NaN values encountered in predictions or true values for model {model_name} on test data.")
                        metrics = {
                            'RMSE': np.nan,
                            'MAE': np.nan,
                            'Bias': np.nan,
                            'CombinedMetric': np.nan
                        }
                    else:
                        # Evaluate the model
                        metrics = self.evaluate_model(y_true_test, y_pred_test, train_data)
                    test_metrics[model_name] = metrics
            else:
                logging.warning(f"No models to evaluate on test data for series {series_id}.")

            # Handle AutoMFLES separately
            mfles_ran_successfully = False  # Initialize flag for MFLES
            if use_MFLES:
                logging.info(f"Processing AutoMFLES for series {series_id}.")

                # Define configuration for AutoMFLES
                season_length_list = [season_length]
                config = {
                    'seasonality_weights': [True, False],
                    'smoother': [False],
                    'ma': [season_length, season_length // 2, None],
                    'seasonal_period': [None, season_length],
                }

                # Prepare the y series for AutoMFLES
                y_series = train_data['y'].values

                try:
                    mfles_model = AutoMFLES(
                        season_length=season_length_list,
                        test_size=len(test_data),
                        n_windows=cv_folds,
                        metric='smape',
                        config=config
                    )
                    # Fit the model on training data
                    mfles_model.fit(y=y_series)
                    # Predict on test data
                    predicted = mfles_model.predict(len(test_data))['mean']
                    # Evaluate the model
                    y_true_test = test_data['y'].values

                    if np.isnan(predicted).any() or np.isnan(y_true_test).any():
                        logging.error(f"NaN values encountered in AutoMFLES predictions or true values on test data.")
                        metrics = {
                            'RMSE': np.nan,
                            'MAE': np.nan,
                            'Bias': np.nan,
                            'CombinedMetric': np.nan
                        }
                    else:
                        metrics = self.evaluate_model(y_true_test, predicted, train_data)
                        test_metrics['AutoMFLES'] = metrics
                        mfles_ran_successfully = True
                        logging.info(f"AutoMFLES Test Combined Metric: {metrics['CombinedMetric']}")

                    # Store the MFLES model and config
                    best_mfles_model = mfles_model
                    best_mfles_config = config

                except Exception as e:
                    logging.error(f"Error during AutoMFLES fitting: {e}")
                    logging.warning(f"AutoMFLES did not produce a valid model for series {series_id}.")

            # Select the best model based on test CombinedMetric
            if not test_metrics:
                logging.warning(f"No test metrics available for series {series_id}. Skipping.")
                continue

            valid_test_metrics = {k: v for k, v in test_metrics.items() if not np.isnan(v['CombinedMetric'])}

            if not valid_test_metrics:
                logging.warning(f"No valid test metrics for series {series_id}. Skipping.")
                continue

            best_model_name = min(valid_test_metrics.items(), key=lambda x: x[1]['CombinedMetric'])[0]
            best_model_metrics = valid_test_metrics[best_model_name]
            logging.info(f"Best model for series {series_id} based on test data: {best_model_name}")

            # Retrain the best model on the full data (train + test)
            if best_model_name == 'AutoMFLES':
                # Use the best AutoMFLES model with the same config
                y_full_series = series_data['y'].values
                mfles_model_full = AutoMFLES(
                    season_length=season_length_list,
                    test_size=future_periods,
                    n_windows=2,
                    metric='smape',
                    config=best_mfles_config
                )
                mfles_model_full.fit(y=y_full_series)
                # Forecast future periods
                y_pred_future = mfles_model_full.predict(future_periods)['mean']
            else:
                best_model_instance = next((model for model in final_models if model.alias == best_model_name), None)
                if best_model_instance is None:
                    logging.error(f"Best model instance not found for model {best_model_name}")
                    continue

                sf_best_full = StatsForecast(
                    models=[best_model_instance],
                    freq=frequency,
                    n_jobs=1
                )
                full_data = series_data[['unique_id', 'ds', 'y']].copy()
                sf_best_full.fit(df=full_data)

                # Forecast future periods
                forecast_future = sf_best_full.predict(h=future_periods)
                forecast_future = forecast_future.set_index('ds')

                if best_model_name in forecast_future.columns:
                    y_pred_future = forecast_future[best_model_name].values
                else:
                    logging.error(f"Forecasts for model '{best_model_name}' not found for future periods for series '{series_id}'.")
                    y_pred_future = np.full(future_periods, np.nan)

            # Get future dates
            last_date = series_data['ds'].max()
            future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=future_periods, freq=frequency)

            # Inverse transform forecasts if necessary
            y_pred_future_inversed = self.inverse_transform(y_pred_future, series_id)

            # Ensure non-negative forecasts
            y_pred_future_inversed = np.maximum(y_pred_future_inversed, 0)

            # Create a DataFrame with future forecasts
            future_forecast_df = pd.DataFrame({
                'unique_id': series_id,
                'ds': future_dates,
                'forecast': y_pred_future_inversed
            })
            future_forecasts_list.append(future_forecast_df)

            # Store results for the current series
            self.models_results[series_id] = {
                'best_model_name': best_model_name,
                'test_metrics': best_model_metrics,
                'best_model_params': best_manual_params.get(best_model_name, {}) if best_model_name != 'AutoMFLES' else best_mfles_config,
                'season_length': season_length,  # Store the detected season length
                'mfles_ran_successfully': mfles_ran_successfully,
                'Intermittency': intermittency,
                'Demand_Class': demand_class,
                'Stationary': is_stationary,
                'Trend_Category': trend_category,
                'Trend_Strength': trend_strength
            }

        # Combine all future forecasts
        if future_forecasts_list:
            self.future_forecasts = pd.concat(future_forecasts_list, ignore_index=True)
        else:
            self.future_forecasts = None

    def run(self):
        """
        Run the entire model building and forecasting pipeline.
        """
        logging.info("Starting the model building and forecasting pipeline.")
        self.load_data()
        self.compute_ts_characteristics()
        self.build_univariate_models()

        # Save model parameters and metrics to CSV
        st.title('Model Results for Each Series:')
        results_list = []
        for series_id, results in self.models_results.items():
            best_model_name = results.get('best_model_name', '')
            test_metrics = results.get('test_metrics', {})
            # best_model_params = results.get('best_model_params', {})
            season_length = results.get('season_length', '')
            mfles_ran_successfully = results.get('mfles_ran_successfully', False)

            # Additional characteristics
            intermittency = results.get('Intermittency', 'Unknown')
            demand_class = results.get('Demand_Class', 'Unknown')
            is_stationary = results.get('Stationary', 'Unknown')
            trend_category = results.get('Trend_Category', 'Unknown')
            # trend_strength = results.get('Trend_Strength', np.nan)

            results_list.append({
                'Series_ID': series_id,
                'Best_Model': best_model_name,
                # 'Best_Model_Params': best_model_params,
                'Season_Length': season_length,
                'Test_RMSE': test_metrics.get('RMSE', 'N/A'),
                'Test_MAE': test_metrics.get('MAE', 'N/A'),
                'Test_Bias': test_metrics.get('Bias', 'N/A'),
                'Test_CombinedMetric': test_metrics.get('CombinedMetric', 'N/A'),
                'MFLES_Ran_Successfully': mfles_ran_successfully,
                'Intermittency': intermittency,
                'Demand_Class': demand_class,
                'Stationary': is_stationary,
                'Trend_Category': trend_category,
                # 'Trend_Strength': trend_strength
            })

        results_df = pd.DataFrame(results_list)
        # results_df.to_csv('model_results.csv', index=False)
        logging.info("Model results saved to 'model_results.csv'.")

        # Save future forecasts
        if self.future_forecasts is not None:
            self.future_forecasts.to_csv('future_forecasts.csv', index=False)
            logging.info("Future forecasts saved to 'future_forecasts.csv'.")
        else:
            logging.warning("No future forecasts to save.")

# Usage example
if __name__ == "__main__":
    # Option 1: Provide data as a DataFrame
    # data = pd.read_csv('your_data.csv')  # Replace with your actual data loading method
    # model_builder = ModelBuilder(data=data)

    # Option 2: Provide data via file path
    # data_path = "Service Forecasting_original.xlsx"  # Replace with your actual file path
    model_builder = ModelBuilder(data_path=data_path)

    # Run the model builder
    model_builder.run()

    # Print results
    print("\nModel Results for Each Series:")
    for series_id, results in model_builder.models_results.items():
        print(f"Series ID: {series_id}")
        # print(f"Best Model: {results.get('best_model_name', '')}")
        # print(f"Best Model Params: {results.get('best_model_params', {})}")
        print(f"Season Length: {results.get('season_length', '')}")
        print(f"Test RMSE: {results.get('test_metrics', {}).get('RMSE', 'N/A')}")
        print(f"Test MAE: {results.get('test_metrics', {}).get('MAE', 'N/A')}")
        print(f"Test Bias: {results.get('test_metrics', {}).get('Bias', 'N/A')}")
        # print(f"Test Combined Metric: {results.get('test_metrics', {}).get('CombinedMetric', 'N/A')}")
        # print(f"MFLES Ran Successfully: {results.get('mfles_ran_successfully', False)}")
        print(f"Intermittency: {results.get('Intermittency', 'Unknown')}")
        print(f"Demand Class: {results.get('Demand_Class', 'Unknown')}")
        print(f"Stationary: {results.get('Stationary', 'Unknown')}")
        print(f"Trend Category: {results.get('Trend_Category', 'Unknown')}")
        # print(f"Trend Strength: {results.get('Trend_Strength', np.nan)}")
        print("-" * 40)
    
    # Print future forecasts
    if model_builder.future_forecasts is not None:
        print("\nFuture Periods Forecast:")
        print(model_builder.future_forecasts)
    else:
        print("\nNo future forecasts available.")
