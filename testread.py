import streamlit as st
import streamlit as st
import pandas as pd
def main():
    st.title("File Upload and Display")

    # File upload
    uploaded_file = st.file_uploader("Upload XLSX or CSV file", type=["xlsx", "csv"])

    if uploaded_file is not None:
        # Read the file
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)

        # Display the contents
        st.dataframe(df)

if __name__ == "__main__":
    main()