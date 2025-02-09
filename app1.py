import streamlit as st
import pandas as pd
import io
import logging
from contextlib import redirect_stdout
import plotly.express as px
import matplotlib.pyplot as plt
from AutoForecastPipeline_ST import run_forecast_pipeline  # Import the function from your module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Forecast Pipeline",
    page_icon="📊",  # Emoji icon
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
        /* Main background color */
        .main {
            background-color: #F0F2F6;
        }
        
        /* Center title */
        h1 {
            text-align: center;
            color: #333333;
        }

        /* Custom button styling */
        .stButton>button {
            color: white !important;
            background-color: #007BFF !important;
            border-radius: 8px;
            font-size: 16px;
            padding: 8px 20px;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #E3E3E3 !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            font-size: 18px;
            font-weight: bold;
            padding: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# Load Logo (Ensure you have 'logo.png' in the same directory or provide a URL)
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.title("📊 Forecasting Dashboard")

# File Upload Section
st.title("🔮 Forecast Pipeline")
st.subheader("Upload your Input File")

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

    # Tabbed Interface
    tab1, tab2, tab3 = st.tabs(["📈 Forecasting", "📜 Logs", "ℹ️ About"])

    with tab1:
        dependent_variable = st.selectbox("🎯 Select Dependent Variable", df_input.columns)

        # Capture logs
        log_stream = io.StringIO()
        with redirect_stdout(log_stream):
            try:
                st.info("⏳ Running forecast pipeline...")

                # Run the forecast pipeline
                predictions_df, log_output = run_forecast_pipeline(df_input)

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

    with tab2:
        st.subheader("📝 Logs & Execution Details")
        st.text_area("Logs", log_stream.getvalue(), height=300)

    with tab3:
        st.subheader("ℹ️  About the forecasting engine- DPEEL")
        st.markdown("""
        ### 📂 Excel Export  
                The predictions DataFrame (`predictions_df`) is written to an Excel file using `pandas` and `xlsxwriter`.

                 💾 Memory Buffer  
                The Excel file is saved to an in-memory buffer (`io.BytesIO()`) for efficient handling.

                 📥 Download Button  
                A download button is provided in the Streamlit app, allowing users to download predictions as an XLSX file.

                 ❌ Error Handling  
                Any exceptions during the forecasting process are caught, logged, and displayed as error messages in the app.

                 📝 Logs Display  
                Execution logs are shown in a text area within a separate tab labeled "Logs & Execution Details".


                 ℹ️ About Section  
                An additional tab provides information about the forecasting engine, named DPEEL.

                 📂 File Upload Prompt  
                If no file is uploaded, the user is prompted to upload an XLSX or CSV file to proceed.

                 ⚠️ Conditional Display  
                Warnings are displayed if no predictions are available to ensure clear communication.

                 🗂️ Streamlit Tabs  
                The app uses Streamlit tabs to organize the display of predictions, logs, and information about the forecasting engine.

                 🎨 User Interface  
                The app provides a user-friendly interface for:  
                - Uploading files 📂  
                - Viewing predictions 📊  
                - Downloading results 📥  
                - Accessing logs 📝  
                - Exploring forecasting engine details ℹ️  
        """)

else:
    st.info("📂 Please upload an XLSX or CSV file to proceed.")