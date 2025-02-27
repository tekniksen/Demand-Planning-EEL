# main_pipeline.py

import pandas as pd
import numpy as np
import logging
import io
import streamlit as st

def run_forecast_pipeline(df_input, dependentVariable, developer_mode=False):
    # Create a StringIO object to capture the printed output
    log_output = io.StringIO()

    # Import modules and classes
    import data_preprocessing_final
    import outlier_treatment_final
    import univariate_model_final
    from data_preprocessing_final import DataPreprocessor
    from outlier_treatment_final import OutlierDetector
    from univariate_model_final import ModelBuilder


    # Configure logging
    logging.basicConfig(
        filename='auto_forecast_pipeline.log',
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

    if developer_mode:
        st.text(" Using User-provived Model Parameters...")
        st.sidebar.subheader("🔧 Developer Settings")

        # Global Parameters for Data Preprocessing
        month_col = "month"
        year_col = "year"
        date_col = "mon_year"
        dependent_var = dependentVariable
        # dependent_var = st.sidebar.text_input("Dependent Variable", "dependentVariable")
        columns_to_exclude = st.sidebar.multiselect("Columns to Exclude", 
            ["Total Service Gap (K Euro)", "month", "year", "Sourcing Location"], 
            default=["Total Service Gap (K Euro)", "month", "year", "Sourcing Location"])
        start_year = st.sidebar.slider("Start Year", 2000, 2030, 2019)
        end_year = st.sidebar.slider("End Year", 2000, 2030, 2024)
        key_variable = st.sidebar.text_input("Key Variable", "Sourcing Location")

        # Missing Values Treatment
        missing_values_treatment_stage = st.sidebar.selectbox("Missing Values Treatment Stage", ["do", "skip"], index=1)
        numeric_fill_method = st.sidebar.selectbox("Numeric Fill Method", ["mean", "median", "ffill", "bfill"], index=0)
        categorical_fill_method = st.sidebar.selectbox("Categorical Fill Method", ["mode", "ffill", "bfill"], index=0)

        # Train-Test Split and Forecasting
        train_size_ratio = st.sidebar.slider("Train Size Ratio", 0.5, 1.0, 0.8, step=0.05)
        cv_folds = st.sidebar.slider("Cross-Validation Folds", 1, 10, 3)
        future_periods = st.sidebar.slider("Future Periods to Forecast", 1, 24, 12)

        # Evaluation Metric Weights
        mse_weight = st.sidebar.slider("MSE Weight", 0.0, 1.0, 0.7, step=0.1)
        bias_magnitude_weight = st.sidebar.slider("Bias Magnitude Weight", 0.0, 1.0, 0.2, step=0.1)
        bias_direction_weight = st.sidebar.slider("Bias Direction Weight", 0.0, 1.0, 0.1, step=0.1)

        # Transformation Flags
        use_transformation = st.sidebar.checkbox("Use Transformation", False)
        use_log1p = st.sidebar.checkbox("Use log1p", True)
        use_boxcox = st.sidebar.checkbox("Use Box-Cox", False)

        # Model Inclusion Flags
        st.sidebar.subheader("📈 Model Selection")
        use_ARIMA = st.sidebar.checkbox("Use ARIMA", True)
        use_HoltWinters = st.sidebar.checkbox("Use Holt-Winters", False)
        use_ARCH = st.sidebar.checkbox("Use ARCH", False)
        use_DOT = st.sidebar.checkbox("Use DOT", False)
        use_DSTM = st.sidebar.checkbox("Use DSTM", False)
        use_GARCH = st.sidebar.checkbox("Use GARCH", False)
        use_Holt = st.sidebar.checkbox("Use Holt", False)
        use_MFLES = st.sidebar.checkbox("Use MFLES", False)
        use_OptimizedTheta = st.sidebar.checkbox("Use Optimized Theta", False)
        use_SeasonalES = st.sidebar.checkbox("Use Seasonal ES", False)
        use_SeasonalESOptimized = st.sidebar.checkbox("Use Seasonal ES Optimized", False)
        use_SESOptimized = st.sidebar.checkbox("Use SES Optimized", True)
        use_SES = st.sidebar.checkbox("Use SES", True)
        use_Theta = st.sidebar.checkbox("Use Theta", True)

        # Additional Flags
        use_intermittent_models = st.sidebar.checkbox("Use Intermittent Models", True)
        use_auto_models = st.sidebar.checkbox("Use Auto Models", True)

        # Seasonality Detection Parameters
        st.sidebar.subheader("🕰️ Seasonality Detection")
        get_seasonality = st.sidebar.checkbox("Detect Seasonality", False)
        seasonality_lags = st.sidebar.slider("Seasonality Lags", 1, 24, 12)
        skip_lags = st.sidebar.slider("Skip Lags", 1, 24, 11)
        lower_ci_threshold = st.sidebar.slider("Lower CI Threshold", -1.0, 0.0, -0.10, step=0.01)
        upper_ci_threshold = st.sidebar.slider("Upper CI Threshold", 0.0, 1.0, 0.90, step=0.01)

        # Time Series Characteristics & Intermittency
        ts_characteristics_flag = st.sidebar.checkbox("Compute Time Series Characteristics", True)
        detect_intermittency = st.sidebar.checkbox("Detect Intermittency", True)

        st.sidebar.success("Developer Mode Activated 🚀")


    else:
        st.text("Using Default Model Parameters...")
        # Global Parameters for Data Preprocessing
        month_col = "month"
        year_col = "year"
        date_col = "mon_year"
        dependent_var = dependentVariable
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


        # Other Global Parameters
        train_size_ratio = 0.8  # Ratio for train-test split
        cv_folds = 3            # Number of cross-validation folds
        future_periods = 12     # Number of future periods to forecast

        # Evaluation metric weights
        mse_weight = 0.7
        bias_magnitude_weight = 0.2
        bias_direction_weight = 0.1

        # Transformation flags
        use_transformation = False  # Set to True to use data transformation
        use_log1p = True           # Set to True to use log1p transformation
        use_boxcox = False         # Set to True to use Box-Cox transformation

        # Model inclusion flags
        use_ARIMA = True
        use_HoltWinters = False
        use_ARCH = False
        use_DOT = False
        use_DSTM = False
        use_GARCH = False
        use_Holt = False
        use_MFLES = False
        use_OptimizedTheta = False
        use_SeasonalES = False
        use_SeasonalESOptimized = False
        use_SESOptimized = True
        use_SES = True
        use_Theta = True

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




    # Global Variables for Outlier Detection
    outlier_detection_method = "Percentile"
    outlier_detection_columns = [
        "Service",
        # "Orders (K Euro)",
        "Invoices (K Euro)",
        # "incl. Forecast Gap",
        # "incl. Operations Gap",
        # "incl. Capacity Gap",
        # "incl Materials Gap",
        # "incl. Logistics Gap",
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

    # Global Parameters for Univariate Modeling
    # Time Series Characteristics parameters
    frequency = 'MS'  # Monthly start frequency
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
    segment_time = 12  # Segmentation time period
    abc_cutoff = [0.3, 0.8]  # Cutoffs for volume segments
    cov_cutoff = [0.5, 1]  # Cutoffs for variability/CoV segments
    adi_cutoff = 1.32  # Average demand interval cutoff for demand classes
    nz_cov_cutoff = 0.49  # Non-zero CoV cutoff for demand classes
    imt_cutoff = 0.5  # Intermittency cutoff (ratio of zeros to total observations)
    train_end_date = None  # Training end date, will be set after loading data



    # Path to the data file
    # data_file_path = "Service Forecasting_original.xlsx"  # Replace with your actual file path

    # Parameter to choose whether to use the original target variable or the corrected one
    use_corrected_target = True  # Set to True to use the corrected target variable

    # Set the global variables in data_preprocessing_final module
    data_preprocessing_final.uploadedDataST = df_input
    data_preprocessing_final.month_col = month_col
    data_preprocessing_final.year_col = year_col
    data_preprocessing_final.date_col = date_col
    data_preprocessing_final.dependent_var = dependent_var
    data_preprocessing_final.columns_to_exclude = columns_to_exclude
    data_preprocessing_final.start_year = start_year
    data_preprocessing_final.end_year = end_year
    data_preprocessing_final.key_variable = key_variable
    data_preprocessing_final.missing_values_treatment_stage = missing_values_treatment_stage
    data_preprocessing_final.numeric_fill_method = numeric_fill_method
    data_preprocessing_final.categorical_fill_method = categorical_fill_method
    
    # Initialize DataPreprocessor without passing parameters
    preprocessor = DataPreprocessor(uploadedDataST = df_input,
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
        categorical_fill_method=categorical_fill_method)

    # Run preprocessing
    pre_processed_data, dep_var, independent_vars = preprocessor.run_preprocessing()
    logging.info("Data preprocessing completed.")
    st.text("Data preprocesssing completed...")

    # Save preprocessed data if needed
    # pre_processed_data.to_csv('pre_processed_data.csv', index=False)

    # Set the global variables in outlier_treatment_final module
    outlier_treatment_final.outlier_detection_method = outlier_detection_method
    outlier_treatment_final.outlier_detection_columns = outlier_detection_columns
    outlier_treatment_final.outlier_detection_timeframe = outlier_detection_timeframe
    outlier_treatment_final.outlier_detection_params = outlier_detection_params
    outlier_treatment_final.key_variable = key_variable
    outlier_treatment_final.date_col = date_col
    outlier_treatment_final.dependent_var = dependent_var

    # Initialize OutlierDetector without passing parameters (uses module globals)
    outlier_detector = OutlierDetector(
        preprocessed_data=pre_processed_data,
        outlier_detection_method=outlier_detection_method,
        outlier_detection_columns=outlier_detection_columns,
        outlier_detection_timeframe=outlier_detection_timeframe,
        outlier_detection_params=outlier_detection_params,
        key_variable=key_variable
    )

    # Run outlier detection
    main_df, helper_columns_df = outlier_detector.run_method()
    logging.info("Outlier Treatment completed...")
    st.text("Outlier Treatment completed...")

    # Save the main DataFrame and helper columns DataFrame
    # main_df.to_csv('outlier_processed_data.csv', index=False)
    # helper_columns_df.to_csv('helper_columns.csv', index=False)
    # logging.info("Outlier processed data saved.")

    # Step 3: Univariate Modeling
    # Decide which target variable to use
    if use_corrected_target:
        # Use the corrected target variable
        corrected_target_col = dependent_var + '_Corrected'
        if corrected_target_col in main_df.columns:
            modeling_data = main_df.copy()
            modeling_data[dependent_var] = modeling_data[corrected_target_col]
        else:
            logging.error(f"Corrected target column '{corrected_target_col}' not found. Using original target column '{dependent_var}'.")
            modeling_data = main_df.copy()
    else:
        # Use the original target variable
        modeling_data = main_df.copy()

    # Ensure that the date column is in datetime format
    modeling_data[date_col] = pd.to_datetime(modeling_data[date_col], errors='coerce')

    # Set the global variables in the univariate_model_final module
    univariate_model_final.date_col = date_col
    univariate_model_final.series_identifier_cols = [key_variable]
    univariate_model_final.dep_var = dependent_var
    univariate_model_final.frequency = frequency
    univariate_model_final.ts_freq = frequency
    univariate_model_final.sp = sp
    univariate_model_final.delimiter = delimiter
    univariate_model_final.npi_cutoff = npi_cutoff
    univariate_model_final.eol_cutoff = eol_cutoff
    univariate_model_final.segment_time = segment_time
    univariate_model_final.abc_cutoff = abc_cutoff
    univariate_model_final.cov_cutoff = cov_cutoff
    univariate_model_final.adi_cutoff = adi_cutoff
    univariate_model_final.nz_cov_cutoff = nz_cov_cutoff
    univariate_model_final.imt_cutoff = imt_cutoff
    univariate_model_final.train_end_date = None  # Will be set after loading data

    # Other global parameters
    univariate_model_final.train_size_ratio = train_size_ratio
    univariate_model_final.cv_folds = cv_folds
    univariate_model_final.future_periods = future_periods

    univariate_model_final.mse_weight = mse_weight
    univariate_model_final.bias_magnitude_weight = bias_magnitude_weight
    univariate_model_final.bias_direction_weight = bias_direction_weight

    univariate_model_final.use_transformation = use_transformation
    univariate_model_final.use_log1p = use_log1p
    univariate_model_final.use_boxcox = use_boxcox

    univariate_model_final.use_ARIMA = use_ARIMA
    univariate_model_final.use_HoltWinters = use_HoltWinters
    univariate_model_final.use_ARCH = use_ARCH
    univariate_model_final.use_DOT = use_DOT
    univariate_model_final.use_DSTM = use_DSTM
    univariate_model_final.use_GARCH = use_GARCH
    univariate_model_final.use_Holt = use_Holt
    univariate_model_final.use_MFLES = use_MFLES
    univariate_model_final.use_OptimizedTheta = use_OptimizedTheta
    univariate_model_final.use_SeasonalES = use_SeasonalES
    univariate_model_final.use_SeasonalESOptimized = use_SeasonalESOptimized
    univariate_model_final.use_SESOptimized = use_SESOptimized
    univariate_model_final.use_SES = use_SES
    univariate_model_final.use_Theta = use_Theta

    univariate_model_final.use_intermittent_models = use_intermittent_models
    univariate_model_final.use_auto_models = use_auto_models

    univariate_model_final.get_seasonality = get_seasonality
    univariate_model_final.seasonality_lags = seasonality_lags
    univariate_model_final.skip_lags = skip_lags
    univariate_model_final.lower_ci_threshold = lower_ci_threshold
    univariate_model_final.upper_ci_threshold = upper_ci_threshold

    univariate_model_final.ts_characteristics_flag = ts_characteristics_flag
    univariate_model_final.detect_intermittency = detect_intermittency

    st.text("Building Model...")
    # Initialize and run the model builder
    model_builder = ModelBuilder(
        data=modeling_data  # Pass the modeling data
    )
    model_builder.run()
    logging.info("Univariate modeling completed.")
    st.text("Building Models completed...")

    # Print results
    print("\nModel Results for Each Series:")

    for series_id, results in model_builder.models_results.items():
        st.text(f"Series ID: {series_id} | Best Model: {results.get('best_model_name', '')} | Test RMSE: {results.get('test_metrics', {}).get('RMSE', 'N/A')} | Test MAE: {results.get('test_metrics', {}).get('MAE', 'N/A')} | Intermittency: {results.get('Intermittency', 'Unknown')} | Demand Class: {results.get('Demand_Class', 'Unknown')} | Stationary: {results.get('Stationary', 'Unknown')} | Trend Category: {results.get('Trend_Category', 'Unknown')} ")

    # Print future forecasts
    if model_builder.future_forecasts is not None:
        print("\nFuture Periods Forecast:")
        print(model_builder.future_forecasts)
        prediction_df = model_builder.future_forecasts
        # st.dataframe(prediction_df)
        # model_builder.future_forecasts.to_csv('future_forecasts.csv', index=False)
    else:
        prediction_df = None
        print("\nNo future forecasts available.")
    
    # Get the log output as a string
    log_output_str = log_output.getvalue()
    log_output.close()

    return prediction_df, log_output_str


def main():
    future_forecasts = run_forecast_pipeline()
    # You can handle the returned DataFrame here if needed
    print(future_forecasts)

if __name__ == "__main__":
    main()