# seasonality_acf_test.py

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

class SeasonalityDetector:
    def __init__(
        self,
        data=None,
        data_path=None,
        date_col=None,
        target_col=None,
        lags=12,
        alpha=0.05,
        lower_ci_threshold=-0.10,
        upper_ci_threshold=0.90,
        skip_lags=11
    ):
        """
        Initialize the SeasonalityDetector.

        Parameters:
        - data: pd.DataFrame containing the time series data. (Optional)
        - data_path: str, path to the data file (CSV or Excel). (Optional)
        - date_col: str, name of the date column in data.
        - target_col: str, name of the target column in data.
        - lags: int, number of lags to consider in ACF.
        - alpha: float, significance level for confidence intervals.
        - lower_ci_threshold: float, lower threshold for confidence interval.
        - upper_ci_threshold: float, upper threshold for confidence interval.
        - skip_lags: int, number of lags to skip when identifying seasonality.
        """
        self.date_col = date_col
        self.target_col = target_col
        self.lags = lags
        self.alpha = alpha
        self.lower_ci_threshold = lower_ci_threshold
        self.upper_ci_threshold = upper_ci_threshold
        self.skip_lags = skip_lags

        # Load data from file path or use provided DataFrame
        if data is not None:
            self.data = data.copy()
        elif data_path is not None:
            self.data = self.load_data(data_path)
        else:
            raise ValueError("Either 'data' or 'data_path' must be provided.")

        # Will be set after detection
        self.seasonality = None
        self.seasonality_detected = False

    def load_data(self, data_path):
        """
        Load data from the specified file path.

        Parameters:
        - data_path: str, path to the data file (CSV or Excel).

        Returns:
        - data: pd.DataFrame containing the loaded data.
        """
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.xls') or data_path.endswith('.xlsx'):
            data = pd.read_excel(data_path, engine='openpyxl')
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or XLS/XLSX file.")

        # Ensure date column is datetime
        data[self.date_col] = pd.to_datetime(data[self.date_col], errors='coerce')

        if data[self.date_col].isnull().any():
            raise ValueError(f"Date column '{self.date_col}' contains invalid datetime entries.")

        # Sort data by date
        data = data.sort_values(by=self.date_col)

        # Check if target column exists
        if self.target_col not in data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in the data.")

        return data

    def get_seasonality_length(self, d):
        out = []

        if len(d) > 1:
            if all(np.diff(d) == np.diff(d)[0]):
                out.append(max(d))
                return out
        while d:
            k = d.pop(0)
            d = [i for i in d if i % k != 0]
            out.append(k)

        # Reducing the options to avoid consecutive (up to 2) lags
        out.sort(reverse=True)

        cleaned_out = []
        for val in out:
            if len(cleaned_out) < 1:
                cleaned_out.append(val)
            else:
                if cleaned_out[-1] - val <= self.skip_lags:
                    pass
                else:
                    cleaned_out.append(val)
        cleaned_out.sort(reverse=True)
        cleaned_out = cleaned_out[:3]  # Top 3 periods only

        return cleaned_out

    def detect_seasonality(self):
        # Ensure data is sorted by date
        self.data = self.data.sort_values(by=self.date_col)
        ts_diff = self.data[self.target_col].values

        # Compute ACF
        ac, confint, qstat, qval = acf(
            ts_diff, nlags=self.lags, qstat=True, alpha=self.alpha
        )

        raw_seasonality = []
        for i, _int in enumerate(confint):
            # Skip lag 0
            if i == 0:
                continue
            lower_ci, upper_ci = _int
            if (
                (lower_ci >= self.lower_ci_threshold) or (upper_ci >= self.upper_ci_threshold)
            ):
                raw_seasonality.append(i)

        seasonality_lengths = self.get_seasonality_length(raw_seasonality.copy())
        self.seasonality_detected = True if len(seasonality_lengths) >= 1 else False
        self.seasonality = seasonality_lengths if self.seasonality_detected else None

        return self.seasonality, self.seasonality_detected

# Example usage (uncomment to test)
if __name__ == "__main__":
    # Using data path
    data_path = 'Service Forecasting_original.xlsx'  # or 'path_to_your_data.xlsx'
    detector = SeasonalityDetector(
        data_path=data_path,
        date_col='mon_year',
        target_col='Invoices (K Euro)',
        lags=12
    )
    seasonality, detected = detector.detect_seasonality()
    print(f"Seasonality Detected: {detected}")
    print(f"Seasonality Lengths: {seasonality}")

#     # Or using an existing DataFrame
#     # data = pd.read_csv('path_to_your_data.csv')
#     # detector = SeasonalityDetector(
#     #     data=data,
#     #     date_col='Date',
#     #     target_col='Actual',
#     #     lags=12
#     # )
#     # seasonality, detected = detector.detect_seasonality()
#     # print(f"Seasonality Detected: {detected}")
#     # print(f"Seasonality Lengths: {seasonality}")
