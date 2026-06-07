"""Shared paths and data-quality helpers for the analysis project."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "sales.csv"
RAW_EXCEL_PATH = PROJECT_ROOT / "data" / "raw" / "Du_Bao_Nhu_Cau_Xe_Tai_Mua_Vu_5Nam.xlsx"
PROCESSED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "sales_cleaned.csv"
PROCESSED_GZIP_PATH = PROJECT_ROOT / "data" / "processed" / "sales_cleaned.csv.gz"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

KEY_COLUMNS = ["Year", "Week", "Route", "Fleet_Type"]
METRIC_COLUMNS = ["Actual_Trips", "Total_Volume_Tons"]


def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize strings and use compact, analysis-friendly dtypes."""
    result = df.copy()
    for column in ["Year_Week", "Route", "Fleet_Type"]:
        result[column] = result[column].astype("string").str.strip()

    result["Route"] = result["Route"].astype("category")
    result["Fleet_Type"] = result["Fleet_Type"].astype("category")
    result["Year"] = pd.to_numeric(result["Year"], downcast="integer")
    result["Week"] = pd.to_numeric(result["Week"], downcast="integer")
    result["Actual_Trips"] = pd.to_numeric(result["Actual_Trips"], downcast="integer")
    result["Total_Volume_Tons"] = pd.to_numeric(
        result["Total_Volume_Tons"], downcast="float"
    )
    result["Is_Peak_Event"] = pd.to_numeric(
        result["Is_Peak_Event"], downcast="integer"
    )
    return result


def quality_report(df: pd.DataFrame) -> pd.Series:
    """Return counts of violations for the project's cleaning rules."""
    expected_year_week = (
        df["Year"].astype(str) + "-W" + df["Week"].astype(str).str.zfill(2)
    )
    invalid_iso_week = 0
    for year, week in df[["Year", "Week"]].drop_duplicates().itertuples(index=False):
        try:
            date.fromisocalendar(int(year), int(week), 4)
        except ValueError:
            invalid_iso_week += int(
                (df["Year"].eq(year) & df["Week"].eq(week)).sum()
            )

    checks = {
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_keys": int(df.duplicated(KEY_COLUMNS, keep=False).sum()),
        "actual_trips_non_positive": int(df["Actual_Trips"].le(0).sum()),
        "volume_non_positive": int(df["Total_Volume_Tons"].le(0).sum()),
        "invalid_week": int((~df["Week"].between(1, 53)).sum()),
        "invalid_iso_week": invalid_iso_week,
        "invalid_peak_event": int((~df["Is_Peak_Event"].isin([0, 1])).sum()),
        "inconsistent_year_week": int(df["Year_Week"].ne(expected_year_week).sum()),
    }
    return pd.Series(checks, name="violation_count")


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ISO-week dates using Thursday as the representative day."""
    result = df.copy()
    result["Date"] = pd.to_datetime(
        result["Year"].astype(str)
        + "-W"
        + result["Week"].astype(str).str.zfill(2)
        + "-4",
        format="%G-W%V-%u",
    )
    result["YearMonth"] = result["Date"].dt.to_period("M").astype("string")
    result["Month"] = result["Date"].dt.month.astype("int8")
    result["Time_Index"] = (
        (result["Year"] - result["Year"].min()) * 53 + result["Week"]
    ).astype("int16")
    result["Is_Peak_Week"] = (
        result.groupby(["Year", "Week"], observed=True)["Is_Peak_Event"]
        .transform("max")
        .astype("int8")
    )
    return result


def add_residual_outlier_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag non-peak residual anomalies after removing trend and seasonality."""
    result = df.copy()
    flag_columns: list[str] = []

    for column in METRIC_COLUMNS:
        trend_column = f"{column}_Trend"
        detrended_column = f"{column}_Detrended"
        seasonal_column = f"{column}_Seasonal_Baseline"
        residual_column = f"{column}_Residual"
        flag_column = f"{column}_Outlier_Candidate"

        def estimate_linear_trend(series: pd.Series) -> pd.Series:
            time_index = result.loc[series.index, "Time_Index"].to_numpy()
            normal_mask = result.loc[series.index, "Is_Peak_Event"].eq(0).to_numpy()
            slope, intercept = np.polyfit(
                time_index[normal_mask], series.to_numpy()[normal_mask], 1
            )
            return pd.Series(slope * time_index + intercept, index=series.index)

        result[trend_column] = (
            result.groupby(["Route", "Fleet_Type"], observed=True)[column]
            .transform(estimate_linear_trend)
        )
        result[detrended_column] = result[column] - result[trend_column]
        result[seasonal_column] = (
            result.groupby(["Route", "Fleet_Type", "Week"], observed=True)[
                detrended_column
            ].transform(
                lambda series: series[
                    result.loc[series.index, "Is_Peak_Event"].eq(0)
                ].median()
            )
        )
        fallback_seasonal = (
            result.groupby(["Route", "Fleet_Type"], observed=True)[detrended_column]
            .transform(
                lambda series: series[
                    result.loc[series.index, "Is_Peak_Event"].eq(0)
                ].median()
            )
        )
        result[seasonal_column] = result[seasonal_column].fillna(fallback_seasonal)
        result[residual_column] = result[detrended_column] - result[seasonal_column]

        def flag_iqr(series: pd.Series) -> pd.Series:
            normal_series = series[result.loc[series.index, "Is_Peak_Event"].eq(0)]
            q1, q3 = normal_series.quantile([0.25, 0.75])
            iqr = q3 - q1
            return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)

        result[flag_column] = (
            result.groupby(["Route", "Fleet_Type"], observed=True)[residual_column]
            .transform(flag_iqr)
            .astype(bool)
            & result["Is_Peak_Event"].eq(0)
        )
        flag_columns.append(flag_column)

    result["Is_Outlier_Candidate"] = result[flag_columns].any(axis=1)
    return result


def build_weekly_monthly_tables(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build weekly and monthly tables without conflating peak rows with weeks."""
    weekly = (
        df.groupby(
            ["Year", "Week", "Year_Week", "YearMonth", "Date"], observed=True
        )
        .agg(
            Total_Trips=("Actual_Trips", "sum"),
            Total_Volume=("Total_Volume_Tons", "sum"),
            Is_Peak_Week=("Is_Peak_Week", "max"),
        )
        .reset_index()
    )
    monthly = (
        weekly.groupby("YearMonth", observed=True)
        .agg(
            Total_Trips=("Total_Trips", "sum"),
            Total_Volume=("Total_Volume", "sum"),
            Avg_Weekly_Trips=("Total_Trips", "mean"),
            Avg_Weekly_Volume=("Total_Volume", "mean"),
            Weeks_In_Month=("Year_Week", "nunique"),
            Peak_Weeks=("Is_Peak_Week", "sum"),
        )
        .reset_index()
    )
    monthly["Date"] = pd.to_datetime(monthly["YearMonth"] + "-01")
    return weekly, monthly
