import streamlit as st
import pandas as pd
import io
import logging
from contextlib import redirect_stdout
import plotly.express as px

from AutoForecastPipeline_ST import run_forecast_pipeline  # Import the function from your module
import matplotlib.pyplot as plt


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit UI
st.set_page_config(page_title="Forecast Pipeline", layout="wide")

st.title('Forecast Pipeline')
st.markdown("""
    <style>
        .main {
            background-color: blue;
        }
        .stButton>button {
            color: white;
            background-color: #4CAF50;
        }
    </style>
    """, unsafe_allow_html=True)

st.header("Upload your input XLSX file")
uploaded_file = st.file_uploader("Upload a file", type=["txt", "csv","xlsx"])



if uploaded_file is not None:
    st.text("File uploaded successfully.")
    file_ext = uploaded_file.name.split(".")[-1]
    if file_ext == "csv":
        st.text("CSV File uploaded")
        file_contents = uploaded_file.read().decode("utf-8")
        df_input = pd.read_csv(io.StringIO(file_contents))
        # Use df as input to another code
    elif file_ext == "xlsx":
        st.text("XLSX File uploaded")
        df_input = pd.read_excel(uploaded_file)
        st.dataframe(df_input)
        # Use df as input to another code
    else:
        st.text("Couldn't read the file")
        st.error("Invalid file format. Please upload a CSV or XLSX file.")
    
    dependent_variable = st.selectbox("Select Dependent Variable", df_input.columns)

    # Capture logs
    log_stream = io.StringIO()
    with redirect_stdout(log_stream):
        try:
            st.text("Running forecast pipeline...")
            # Run the forecast pipeline
            predictions_df, log_output = run_forecast_pipeline(df_input)
            
            # Display the predictions
            if predictions_df is not None:
                st.subheader("Predictions")
                x_col = 'ds'
                y_col = 'forecast'
                fig = px.line(predictions_df, x=x_col, y=y_col, title=f"Line Chart of {y_col} vs {x_col}")
                st.plotly_chart(fig)
              
                # Provide a download link for the predictions
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    predictions_df.to_excel(writer, index=False, sheet_name='Predictions')
                output.seek(0)
                
                st.download_button(
                    label="Download Predictions as XLSX",
                    data=output,
                    file_name='predictions.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )
            else:
                st.warning("No predictions available.")
        except Exception as e:
            logger.error(f"Error running forecast pipeline: {e}")
# Draw line chart of predictions_df
else:
    st.info("Please upload an XLSX file to proceed.")