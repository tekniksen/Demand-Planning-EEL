import streamlit as st
import pandas as pd
import io
import logging
from contextlib import redirect_stdout
import plotly.express as px
import json
from transformers import pipeline

from AutoForecastPipeline_ST import run_forecast_pipeline  # Forecasting Engine
from rag_generatorAnswer import AnswerGenerator
from rag_doc2vectorDb import RAG_store_and_retrieve
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Forecast Pipeline with LLM Insights",
    page_icon="📊",
    layout="wide"
)
st.title("Demand Planning Experience Enrichment using LLMs (DPEEL)")
# Upload additional documents to the pdf folder
st.subheader("Upload Docs For Intel-Augmented Generation")
uploaded_docs = st.file_uploader("", accept_multiple_files=True)
if uploaded_docs:
    for doc in uploaded_docs:
        with open(os.path.join("/Users/anupshanker/Documents/Rest/BITS Masters/Sem 4/Dissertation_Code_and_Work/DPEEL_ForecastPipeline/SupportingDocsOrg", doc.name), "wb") as f:
            f.write(doc.getbuffer())
    st.success("✅ Additional documents uploaded successfully!")

# Button to refresh RAG documents
if st.button("Refresh RAG Documents"):
    rag = RAG_store_and_retrieve(pdf_folder="./SupportingDocsOrg", forecast_csv="future_forecasts.csv")
    rag.store_pdf_in_chromadb()
    rag.store_forecasts_in_chromadb()
    st.success("✅ RAG documents refreshed successfully!")

# Sidebar Branding
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.title("📊 Forecasting Dashboard")

# Page Title
st.title("Run a Forecast")
st.subheader("Upload your Input File")

# dependent_var = st.sidebar.text_input("Dependent Variable", "dependent_var")

################
#####################
# Global Parameters for Data Preprocessing
month_col = "month"
year_col = "year"
date_col = "mon_year"
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

# Developer Mode Toggle
developer_mode = st.sidebar.checkbox("Enable Developer Mode", False)
if developer_mode:
    st.text(" Using User-provived Model Parameters...")
    st.sidebar.subheader("🔧 Developer Settings")

    # Global Parameters for Data Preprocessing
    month_col = "month"
    year_col = "year"
    date_col = "mon_year"
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


############
###################################

params = {
    'month_col' :  month_col,
    'year_col' :  year_col,
    'date_col' :  date_col,
    'columns_to_exclude' :  columns_to_exclude,
    'start_year' :  start_year,
    'end_year' :  end_year,
    'key_variable' :  key_variable,
    'missing_values_treatment_stage' :  missing_values_treatment_stage,
    'numeric_fill_method' :  numeric_fill_method,
    'categorical_fill_method' :  categorical_fill_method,
    'train_size_ratio' :  train_size_ratio,
    'cv_folds' :  cv_folds,
    'future_periods' :  future_periods,
    'mse_weight' :  mse_weight,
    'bias_magnitude_weight' :  bias_magnitude_weight,
    'bias_direction_weight' :  bias_direction_weight,
    'use_transformation' :  use_transformation,
    'use_log1p' :  use_log1p,
    'use_boxcox' :  use_boxcox,
    'use_ARIMA' :  use_ARIMA,
    'use_HoltWinters' :  use_HoltWinters,
    'use_ARCH' :  use_ARCH,
    'use_DOT' :  use_DOT,
    'use_DSTM' :  use_DSTM,
    'use_GARCH' :  use_GARCH,
    'use_Holt' :  use_Holt,
    'use_MFLES' :  use_MFLES,
    'use_OptimizedTheta' :  use_OptimizedTheta,
    'use_SeasonalES' :  use_SeasonalES,
    'use_SeasonalESOptimized' :  use_SeasonalESOptimized,
    'use_SESOptimized' :  use_SESOptimized,
    'use_SES' :  use_SES,
    'use_Theta' :  use_Theta,
    'use_intermittent_models' :  use_intermittent_models,
    'use_auto_models' :  use_auto_models,
    'get_seasonality' :  get_seasonality,
    'seasonality_lags' :  seasonality_lags,
    'skip_lags' :  skip_lags,
    'lower_ci_threshold' :  lower_ci_threshold,
    'upper_ci_threshold' :  upper_ci_threshold,
    'ts_characteristics_flag' :  ts_characteristics_flag,
    'detect_intermittency' :  detect_intermittency
}

#####################
################

# File Upload Section
uploaded_file = st.file_uploader("Upload a file", type=["txt", "csv", "xlsx"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    file_ext = uploaded_file.name.split(".")[-1]

    if file_ext == "csv":
        df_input = pd.read_csv(uploaded_file)
        st.info("📂 CSV file detected and loaded!")
    elif file_ext == "xlsx":
        df_input = pd.read_excel(uploaded_file)
        st.info("📂 XLSX file detected and loaded!")
    else:
        st.error("❌ Invalid file format. Please upload a CSV or XLSX file.")
        st.stop()

    # Show dataset preview
    st.subheader("📋 Preview of Uploaded Data")
    st.dataframe(df_input.head())

    # Tabbed Interface: Forecasting, Logs, About, AI Insights
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecasting", "📜 Logs", "ℹ️ About", "🤖 AI Insights"])

    with tab1:
        dependent_variable = st.selectbox(
            "🎯 Select Dependent Variable", [""] + list(df_input.columns),  # Empty string as the first option
            index=0)

        # Capture logs
        log_stream = io.StringIO()

        if dependent_variable:
            with redirect_stdout(log_stream):
                try:
                    # Run the forecast pipeline
                    if st.button("Run Forecast Pipeline"):
                        st.info("⏳ Running forecast pipeline...")
                        predictions_df, log_output = run_forecast_pipeline(df_input, dependentVariable=dependent_variable, params=params)
                        log_stream.write(log_output)

                        if predictions_df is not None:
                            st.subheader("📊 Forecast Results")
                            x_col, y_col = 'ds', 'forecast'
                            
                            # Plotly interactive chart
                            fig = px.line(predictions_df, x=x_col, y=y_col, title=f"📈 {y_col} vs {x_col}")
                            st.plotly_chart(fig)

                            # Provide a download link for the predictions
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                predictions_df.to_excel(writer, index=False, sheet_name='Predictions')
                            output.seek(0)

                            st.download_button(
                                label="📥 Download Predictions as XLSX",
                                data=output,
                                file_name='predictions.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            )
                    else:
                        st.warning("⚠ No predictions available.")

                except Exception as e:
                    logger.error(f"❌ Error running forecast pipeline: {e}")
                    st.error(f"An error occurred: {e}")
    if dependent_variable:
        with tab2:
            st.subheader("📝 Logs & Execution Details")
            st.text_area("Logs", log_stream.getvalue(), height=300)

        with tab3:
            st.subheader("ℹ️ About This App")
            st.markdown("""
            **Forecast Pipeline** is a data-driven forecasting tool designed to help businesses and analysts predict future trends using **machine learning**. 
            """)

        # 🔹 AI Insights Tab (SLM Integration)
        with tab4:
            st.subheader("🤖 AI Insights from Predictions")
            st.markdown("Ask an AI model to analyze and summarize the predictions!")

            user_query = st.chat_input("Ask the AI about predictions...")
                
            if user_query:
                with st.spinner("🤖 Thinking..."):
                    # Call OpenAI API or local LLM
                    response = AnswerGenerator()
                    response = response.answer_query(user_query)
                    # ai_response = response['choices'][0]['message']['content']
                    st.markdown(f"**🤖 AI Response:** {response}")

else:
    st.info("📂 Please upload an XLSX or CSV file to proceed.")