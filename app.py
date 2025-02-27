import streamlit as st
import pandas as pd
import io
import logging
from contextlib import redirect_stdout
import plotly.express as px
# import openai  # OpenAI API for SLM-based insights
import json
from transformers import pipeline
from AutoForecastPipeline_ST import run_forecast_pipeline  # Forecasting Engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Forecast Pipeline with LLM Insights",
    page_icon="📊",
    layout="wide"
)

# API Key for OpenAI (or local model)
OPENAI_API_KEY = "your-openai-api-key"  # Replace with your API Key
# openai.api_key = OPENAI_API_KEY

# Sidebar Branding
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.title("📊 Forecasting Dashboard")

# Page Title
st.title("🔮 Forecast Pipeline")
st.subheader("Upload your Input File")

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
                    st.info("⏳ Running forecast pipeline...")

                    # Run the forecast pipeline
                    predictions_df, log_output = run_forecast_pipeline(df_input,dependentVariable=dependent_variable)

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

            # Convert Predictions DataFrame to JSON (for LLM processing)
            if predictions_df is not None:
                predictions_json = predictions_df.to_json()

                # Chat Interface
                user_query = st.chat_input("Ask the AI about predictions...")
                
                if user_query:
                    with st.spinner("🤖 Thinking..."):
                        # Call OpenAI API or local LLM
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",  # Use small LLM or local model
                            messages=[
                                {"role": "system", "content": "You are an AI analyst helping with forecasting insights."},
                                {"role": "user", "content": f"Here are the prediction results: {predictions_json}.\nQuestion: {user_query}"}
                            ],
                            temperature=0.7
                        )

                        ai_response = response['choices'][0]['message']['content']
                        st.markdown(f"**🤖 AI Response:** {ai_response}")

else:
    st.info("📂 Please upload an XLSX or CSV file to proceed.")