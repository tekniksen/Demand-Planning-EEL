# ts_characteristics.py

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from statsmodels.tsa.stattools import adfuller, acf
from sklearn.linear_model import LinearRegression

import tsfeatures as tf
import pymannkendall as mk
import logging
from scipy.stats import kurtosis, skew
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

class TimeSeriesCharacteristics:
    def __init__(self, data, series_identifier_cols, date_col, dep_var, ts_freq, sp, delimiter,
                 npi_cutoff, eol_cutoff, segment_time, abc_cutoff, cov_cutoff, adi_cutoff,
                 nz_cov_cutoff, imt_cutoff, train_end_date=None):
        """
        Initialize the TimeSeriesCharacteristics class with required parameters.
        """
        self.data = data.copy()
        self.series_identifier_cols = series_identifier_cols
        self.date_col = date_col
        self.dep_var = dep_var
        self.ts_freq = ts_freq
        self.sp = sp
        self.npi_cutoff = npi_cutoff
        self.eol_cutoff = eol_cutoff
        self.segment_time = segment_time
        self.abc_cutoff = abc_cutoff
        self.cov_cutoff = cov_cutoff
        self.adi_cutoff = adi_cutoff
        self.nz_cov_cutoff = nz_cov_cutoff
        self.imt_cutoff = imt_cutoff
        self.delimiter = delimiter

        # Handle train_end_date
        if train_end_date is None:
            self.train_end_date = self.data[self.date_col].max()
        else:
            # Parse date in MM/DD/YYYY format
            self.train_end_date = pd.to_datetime(train_end_date, format='%m/%d/%Y')

    def imputer(self, base_data, resampled):
        """
        Impute missing values for identifier columns.
        """
        for the_col in self.series_identifier_cols:
            resampled[the_col] = base_data[the_col].iloc[0]
        return resampled

    def fill_missing_date_wrapper(self, group):
        """
        Fill missing dates in the time series data.
        """
        base_data = group.copy()
        date_range = pd.date_range(
            start=base_data[self.date_col].min(),
            end=self.train_end_date,
            freq=self.ts_freq,
        )
        resampled = (
            base_data.set_index(self.date_col)
            .reindex(date_range)
            .rename_axis(self.date_col)
            .reset_index()
        )
        resampled[[self.dep_var]] = resampled[[self.dep_var]].fillna(0)
        resampled = self.imputer(base_data, resampled)
        return resampled

    def get_plc_segments(self, df):
        """
        Determine the Product Life Cycle (PLC) segments.
        """
        np_cutoff_date = self.train_end_date + relativedelta(months=-self.npi_cutoff)
        dicon_cutoff_date = self.train_end_date + relativedelta(months=-self.eol_cutoff)
        df = df[df[self.dep_var] > 0]
        plc_data = (
            df.assign(
                start_date=df[self.date_col],
                end_date=df[self.date_col],
            )
            .groupby(self.series_identifier_cols)
            .agg(dict(start_date='min', end_date='max'))
            .reset_index()
        )
        plc_data["PLC"] = "Mature"
        plc_data.loc[plc_data["start_date"] > np_cutoff_date, "PLC"] = "NPI"
        plc_data.loc[plc_data["end_date"] < dicon_cutoff_date, "PLC"] = "EOL"
        plc_data.drop(columns=["start_date", "end_date"], inplace=True)
        return plc_data

    def get_vol_segments(self, df):
        """
        Determine the volume segments based on sales volume.
        """
        segment_data = df[
            df[self.date_col] >= (df[self.date_col].max() + relativedelta(months=-self.segment_time))
        ]
        sales = (
            segment_data.groupby(self.series_identifier_cols)[self.dep_var]
            .agg(["sum"])
            .reset_index()
        )
        sales.rename(columns={"sum": self.dep_var}, inplace=True)
        sales.sort_values([self.dep_var], ascending=False, inplace=True)
        sales["vol_perc"] = sales[self.dep_var] / sales[self.dep_var].sum()
        sales["cum_vol_perc"] = sales.vol_perc.cumsum()
        sales["Volume_Segment"] = "A"
        sales.loc[sales.cum_vol_perc > self.abc_cutoff[0], "Volume_Segment"] = "B"
        sales.loc[sales.cum_vol_perc > self.abc_cutoff[1], "Volume_Segment"] = "C"
        if len(self.abc_cutoff) > 2:
            sales.loc[sales.cum_vol_perc > self.abc_cutoff[2], "Volume_Segment"] = "D"
        sales["Vol_12M"] = sales[self.dep_var]
        sales.drop(columns=["vol_perc", "cum_vol_perc", self.dep_var], inplace=True)
        return sales

    def calc_cov(self, group):
        """
        Calculate the Coefficient of Variation (CoV) for variability segmentation.
        """
        data = group.copy().reset_index()
        x = data[self.dep_var]
        if (len(x) > 2) and (x.sum() > 0):
            cov = np.std(x) / np.mean(x)
        else:
            cov = np.nan
        cov_df = pd.DataFrame()
        for i in self.series_identifier_cols:
            cov_df[i] = data[i]
        cov_df = cov_df.head(1)
        cov_df['CoV'] = cov
        return cov_df

    def calc_adi(self, x):
        """
        Calculate the Average Demand Interval (ADI).
        """
        df = pd.DataFrame({"X": x}).reset_index(drop=True).reset_index()
        df = df[df.X > 0]
        df["index_shift"] = df["index"].shift(-1)
        df["interval"] = df["index_shift"] - df["index"]
        df = df.dropna(subset=["interval"])
        return np.mean(df["interval"])

    def calc_cov2(self, x):
        """
        Calculate the Coefficient of Variation (CoV) for non-zero demand periods.
        """
        if (len(x) > 2) and (x.sum() > 0):
            cov = np.std(x) / np.mean(x)
        else:
            cov = np.nan
        return cov

    def get_demand_class(self, group):
        """
        Determine the demand class based on ADI and CoV squared.
        """
        sku_data = group.copy().reset_index()
        adi = self.calc_adi(sku_data[self.dep_var])
        non_zero_cov = self.calc_cov2(sku_data[self.dep_var][sku_data[self.dep_var] > 0])
        cov_sqr = non_zero_cov ** 2
        if adi >= self.adi_cutoff:
            if cov_sqr < self.nz_cov_cutoff:
                demand_class = "Intermittent"
            else:
                demand_class = "Lumpy"
        else:
            if cov_sqr > self.nz_cov_cutoff:
                demand_class = "Erratic"
            else:
                demand_class = "Smooth"
        demand_class_df = pd.DataFrame()
        for i in self.series_identifier_cols:
            demand_class_df[i] = sku_data[i]
        demand_class_df = demand_class_df.head(1)
        demand_class_df['Demand Class'] = demand_class
        return demand_class_df

    def calculate_trend(self, group):
        """
        Calculate the trend of the time series data.
        """
        data = group.copy().reset_index()
        values = data[self.dep_var].to_numpy()
        trend_value = 0
        try:
            x = np.array([x for x in range(1, len(values) + 1)]).reshape(-1, 1)
            linear_model = LinearRegression().fit(x, values)
            trend_value = linear_model.coef_[0]
            trend_threshold = 0.01
            conditions = [
                trend_value > trend_threshold,
                (trend_value <= trend_threshold) & (trend_value >= -trend_threshold),
                trend_value < -trend_threshold,
            ]
            choices = ["UPWARD", "NO TREND", "DOWNWARD"]
            trend_category = np.select(conditions, choices, default=None)
            # Corrected call to stl_features
            features = tf.stl_features(data[self.dep_var].values, self.sp)
            trend_strength = np.nan
            if trend_category in ["UPWARD", "DOWNWARD"]:
                trend_strength = features.get("trend", np.nan)
            trend_df = pd.DataFrame()
            for i in self.series_identifier_cols:
                trend_df[i] = data[i]
            trend_df = trend_df.head(1)
            trend_df['trend_value'] = trend_value
            trend_df['trend_category'] = trend_category
            trend_df['trend_strength'] = trend_strength
        except Exception as e:
            logging.exception(e)
            trend_df = pd.DataFrame()
        return trend_df

    def tsf_features(self, group):
        """
        Extract time series features using tsfeatures package.
        """
        data = group.copy().reset_index()
        entropy = tf.entropy(data[self.dep_var].values, self.sp)["entropy"]
        acf_features = tf.acf_features(data[self.dep_var].values)
        x_acf1 = acf_features["x_acf1"]
        x_acf10 = acf_features["x_acf10"]
        diff1_acf1 = acf_features["diff1_acf1"]
        diff1_acf10 = acf_features["diff1_acf10"]
        diff2_acf1 = acf_features["diff2_acf1"]
        diff2_acf10 = acf_features["diff2_acf10"]
        pacf_features = tf.pacf_features(data[self.dep_var].values)
        x_pacf5 = pacf_features["x_pacf5"]
        diff1_pacf5 = pacf_features["diff1x_pacf5"]
        diff2_pacf5 = pacf_features["diff2x_pacf5"]
        stl_features = tf.stl_features(data[self.dep_var].values, self.sp)
        linearity = stl_features["linearity"]
        curvature = stl_features["curvature"]
        heterogeneity = tf.heterogeneity(data[self.dep_var].values)
        arch_acf = heterogeneity["arch_acf"]
        garch_acf = heterogeneity["garch_acf"]
        arch_r2 = heterogeneity["arch_r2"]
        garch_r2 = heterogeneity["garch_r2"]
        hurst = tf.hurst(data[self.dep_var].values)["hurst"]
        hp = tf.holt_parameters(data[self.dep_var].values, self.sp)
        hp_alpha = hp['alpha']
        hp_beta = hp['beta']
        hw = tf.hw_parameters(data[self.dep_var].values, self.sp)
        hw_alpha = hw['hw_alpha']
        hw_beta = hw['hw_beta']
        hw_gamma = hw['hw_gamma']
        kurtosis1 = data[self.dep_var].kurt()
        kurtosis2 = kurtosis(data[self.dep_var], axis=0, bias=True)
        skewed = skew(data[self.dep_var], axis=0, bias=True)
        tsf_df = pd.DataFrame()
        for i in self.series_identifier_cols:
            tsf_df[i] = data[i]
        tsf_df = tsf_df.head(1)
        tsf_df['Entropy'] = entropy
        tsf_df["x_acf1"] = x_acf1
        tsf_df["x_acf10"] = x_acf10
        tsf_df["diff1_acf1"] = diff1_acf1
        tsf_df["diff1_acf10"] = diff1_acf10
        tsf_df["diff2_acf1"] = diff2_acf1
        tsf_df["diff2_acf10"] = diff2_acf10
        tsf_df["x_pacf5"] = x_pacf5
        tsf_df["diff1x_pacf5"] = diff1_pacf5
        tsf_df["diff2x_pacf5"] = diff2_pacf5
        tsf_df["linearity"] = linearity
        tsf_df["curvature"] = curvature
        tsf_df["arch_acf"] = arch_acf
        tsf_df["garch_acf"] = garch_acf
        tsf_df["arch_r2"] = arch_r2
        tsf_df["garch_r2"] = garch_r2
        tsf_df["hurst"] = hurst
        tsf_df['hp_alpha'] = hp_alpha
        tsf_df['hp_beta'] = hp_beta
        tsf_df['hw_alpha'] = hw_alpha
        tsf_df['hw_beta'] = hw_beta
        tsf_df['hw_gamma'] = hw_gamma
        tsf_df['kurtosis1'] = kurtosis1
        tsf_df["kurtosis2"] = kurtosis2
        tsf_df["skewed"] = skewed
        return tsf_df

    def is_stationary(self, group):
        """
        Perform stationarity test on the time series data.
        """
        data = group.copy().reset_index()
        x = data[self.dep_var]
        pval = np.nan
        if len(x) > 10:
            try:
                result = adfuller(x, maxlag=2)
                pval = result[1]
                if result[1] < 0.05:
                    stat = "True"
                else:
                    stat = "False"
            except:
                stat = "Constant data error"
        else:
            stat = "Insufficient data"
        stat_data = pd.DataFrame()
        for i in self.series_identifier_cols:
            stat_data[i] = data[i]
        stat_data = stat_data.head(1)
        stat_data['Stationary'] = stat
        stat_data['Stationary_pvalue'] = pval
        return stat_data

    def get_intermittency(self, group):
        """
        Determine the intermittency of the time series.
        """
        data = group.copy().reset_index()
        x = data[self.dep_var]
        zero_count = x.isin([0]).sum()
        tot_count = len(x)
        perc_zero = (zero_count / tot_count)
        density = 1 - perc_zero
        if perc_zero > self.imt_cutoff:
            intermit = "Intermittent"
        else:
            intermit = "Non-Intermittent"
        intermit_data = pd.DataFrame()
        for i in self.series_identifier_cols:
            intermit_data[i] = data[i]
        intermit_data = intermit_data.head(1)
        intermit_data['Intermittency'] = intermit
        intermit_data["Demand_Density"] = density
        return intermit_data

    def ACFDetector(
            self,
            group,
            lags,
            skip_lags,
            diff,
            alpha,
            lower_ci_threshold,
            upper_ci_threshold
    ):
        """
        Detect seasonality using Auto-Correlation Function (ACF).
        """
        df_sub = group.copy().reset_index()
        if len(df_sub) >= self.sp:
            ts_diff = df_sub[self.dep_var].values
            for _ in range(diff):
                ts_diff = np.diff(ts_diff)
            ac_values, confint, qstat, qval = acf(
                ts_diff, nlags=lags, qstat=True, alpha=alpha
            )
            raw_seasonality = []
            for i, _int in enumerate(confint):
                if (
                        (
                                (_int[0] >= lower_ci_threshold)
                                or (_int[1] >= upper_ci_threshold)
                        )
                        and (i > skip_lags)
                ):
                    raw_seasonality.append(i)
            seasonality_detected = True if len(raw_seasonality) >= 1 else False
            seasonality_col = (
                "Exists" if seasonality_detected else "Does not Exist"
            )
            # Corrected call to stl_features
            features = tf.stl_features(df_sub[self.dep_var].values, self.sp)
            seasonality_strength = np.nan
            if seasonality_col == "Exists":
                seasonality_strength = features.get("seasonal_strength", np.nan)
            seasonality_df = pd.DataFrame()
            for i in self.series_identifier_cols:
                seasonality_df[i] = df_sub[i]
            seasonality_df = seasonality_df.head(1)
            seasonality_df["seasonality"] = seasonality_col
            seasonality_df["seasonal_strength"] = seasonality_strength
            seasonality_df['seasonal_periods'] = ",".join(
                map(str, raw_seasonality)) if seasonality_detected else np.nan
        else:
            seasonality_df = pd.DataFrame()
            for i in self.series_identifier_cols:
                seasonality_df[i] = df_sub[i]
            seasonality_df = seasonality_df.head(1)
            seasonality_df["seasonality"] = "insufficient data"
            seasonality_df["seasonal_strength"] = np.nan
            seasonality_df['seasonal_periods'] = np.nan
        return seasonality_df

    def ts_characteristics(self):
        """
        Compute time series characteristics for all series in the dataset.
        """
        essential_columns = self.series_identifier_cols + [self.date_col, self.dep_var]
        data = self.data[essential_columns]
        data[self.date_col] = pd.to_datetime(data[self.date_col])
        dates_filled = (
            data.groupby(self.series_identifier_cols, as_index=False)
            .apply(lambda x: self.fill_missing_date_wrapper(group=x))
            .reset_index(drop=True)
        )
        # PLC Segments
        plc_data = self.get_plc_segments(df=dates_filled)
        # Volume Segments
        volume_segments = self.get_vol_segments(df=dates_filled)
        ts_char = pd.merge(volume_segments, plc_data, how='left', on=self.series_identifier_cols)
        # Variability Segments
        var_df = (
            dates_filled.groupby(self.series_identifier_cols, as_index=False)
            .apply(lambda x: self.calc_cov(group=x))
            .reset_index(drop=True)
        )
        var_df['Variability_segment'] = "Low"
        var_df.loc[var_df['CoV'] > self.cov_cutoff[0], 'Variability_segment'] = "Medium"
        if len(self.cov_cutoff) > 1:
            var_df.loc[var_df['CoV'] > self.cov_cutoff[1], 'Variability_segment'] = "High"
        if len(self.cov_cutoff) > 2:
            var_df.loc[var_df['CoV'] > self.cov_cutoff[2], 'Variability_segment'] = "Very High"
        ts_char = pd.merge(ts_char, var_df, how='left', on=self.series_identifier_cols)
        # Demand Class
        dem_cls_df = (
            dates_filled.groupby(self.series_identifier_cols, as_index=False)
            .apply(lambda x: self.get_demand_class(group=x))
            .reset_index(drop=True)
        )
        ts_char = pd.merge(ts_char, dem_cls_df, how='left', on=self.series_identifier_cols)
        # Trend
        trend_df = (
            dates_filled.groupby(self.series_identifier_cols, as_index=False)
            .apply(lambda x: self.calculate_trend(group=x))
            .reset_index(drop=True)
        )
        ts_char = pd.merge(ts_char, trend_df, how='left', on=self.series_identifier_cols)
        # Stationarity
        stat_df = (
            dates_filled.groupby(self.series_identifier_cols, as_index=False)
            .apply(lambda x: self.is_stationary(group=x))
            .reset_index(drop=True)
        )
        ts_char = pd.merge(ts_char, stat_df, how='left', on=self.series_identifier_cols)
        # Intermittency
        intermit_df = (
            dates_filled.groupby(self.series_identifier_cols, as_index=False)
            .apply(lambda x: self.get_intermittency(group=x))
            .reset_index(drop=True)
        )
        ts_char = pd.merge(ts_char, intermit_df, how='left', on=self.series_identifier_cols)
        # Seasonality
        seasonality_acf = (
            dates_filled.groupby(self.series_identifier_cols, as_index=False)
            .apply(
                lambda x: self.ACFDetector(
                    group=x,
                    lags=self.sp,
                    skip_lags=2,
                    diff=1,
                    lower_ci_threshold=-0.10,
                    upper_ci_threshold=0.90,
                    alpha=0.05,
                )
            )
            .reset_index(drop=True)
        )
        ts_char = pd.merge(ts_char, seasonality_acf, how='left', on=self.series_identifier_cols)
        # Time Series Features
        tsf_df = (
            dates_filled.groupby(self.series_identifier_cols, as_index=False)
            .apply(lambda x: self.tsf_features(group=x))
            .reset_index(drop=True)
        )
        ts_char = pd.merge(ts_char, tsf_df, how='left', on=self.series_identifier_cols)
        return ts_char

# Usage example:
if __name__ == "__main__":
    # Import logging module
    import logging
    logging.basicConfig(level=logging.INFO)

    # Global Parameters for Time Series Characteristics
    series_identifier_cols = ["Sourcing Location"]  # Series identifier columns
    date_col = "mon_year"  # Time column
    dep_var = "Invoices (K Euro)"  # Target/dependent variable
    ts_freq = "MS"  # Time series frequency
    delimiter = "_"

    # For monthly frequency, give number of months
    npi_cutoff = 12  # New product intro timeline cutoff
    eol_cutoff = 12  # End of life timeline cutoff
    segment_time = 12  # Segmentation time period in months
    abc_cutoff = [0.3, 0.8]  # Cutoffs for volume segments
    cov_cutoff = [0.5, 1]  # Cutoffs for variability/CoV segments
    adi_cutoff = 1.32  # Average demand interval cutoff for demand classes
    nz_cov_cutoff = 0.49  # Non-zero CoV cutoff for demand classes
    imt_cutoff = 0.5  # Intermittency cutoff (ratio of zeros to total observations)
    train_end_date = None  # Training end date in 'MM/DD/YYYY' format or None
    sp = 12  # Seasonal period (e.g., 12 for monthly data)

    # Path to the data file
    # data_path = "Service Forecasting_original.xlsx"  # Replace with your actual file path

    # Read the data
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.xls') or data_path.endswith('.xlsx'):
        data = pd.read_excel(data_path, engine='openpyxl')
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or XLS/XLSX file.")

    # Ensure date column is datetime
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    if data[date_col].isnull().any():
        raise ValueError(f"Date column '{date_col}' contains invalid datetime entries.")

    # Initialize the class with data and parameters
    ts_char_obj = TimeSeriesCharacteristics(
        data=data,
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
        train_end_date=train_end_date
    )

    # Get the time series characteristics
    final_df = ts_char_obj.ts_characteristics()
