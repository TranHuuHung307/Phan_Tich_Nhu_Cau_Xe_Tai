"""Load raw sales data, validate it, clean it, and export the processed dataset."""

import pandas as pd

try:
    from .utils import (
        METRIC_COLUMNS,
        PROCESSED_CSV_PATH,
        PROCESSED_GZIP_PATH,
        RAW_CSV_PATH,
        add_residual_outlier_flags,
        add_time_columns,
        normalize_dtypes,
        quality_report,
    )
except ImportError:
    from utils import (
        METRIC_COLUMNS,
        PROCESSED_CSV_PATH,
        PROCESSED_GZIP_PATH,
        RAW_CSV_PATH,
        add_residual_outlier_flags,
        add_time_columns,
        normalize_dtypes,
        quality_report,
    )


def load_raw_data(path=RAW_CSV_PATH) -> pd.DataFrame:
    """Load raw CSV without modifying it."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found: {path}. Restore the immutable source file."
        )
    return pd.read_csv(path)


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the reproducible cleaning pipeline and retain anomaly candidates."""
    cleaned = normalize_dtypes(df)
    initial_report = quality_report(cleaned)

    blocking_checks = initial_report.drop(labels=["missing_values"])
    non_metric_missing = int(cleaned.drop(columns=METRIC_COLUMNS).isna().sum().sum())
    if non_metric_missing:
        raise ValueError(
            f"Raw data contains {non_metric_missing} missing non-metric values."
        )
    if blocking_checks.any():
        raise ValueError(
            "Raw data violates blocking quality checks:\n"
            + blocking_checks[blocking_checks.gt(0)].to_string()
        )

    if cleaned[METRIC_COLUMNS].isna().any().any():
        cleaned = cleaned.sort_values(
            ["Route", "Fleet_Type", "Year", "Week"]
        ).reset_index(drop=True)
        cleaned[METRIC_COLUMNS] = (
            cleaned.groupby(["Route", "Fleet_Type"], observed=True)[METRIC_COLUMNS]
            .transform(lambda series: series.ffill().bfill())
        )

    cleaned = add_time_columns(cleaned)
    cleaned = add_residual_outlier_flags(cleaned)
    if cleaned.isna().any().any():
        raise ValueError("Cleaning pipeline produced unresolved missing values.")
    return cleaned.sort_values(["Year", "Week", "Route", "Fleet_Type"]).reset_index(
        drop=True
    )


def export_cleaned_data(
    df: pd.DataFrame,
    output_path=PROCESSED_CSV_PATH,
    compressed_path=PROCESSED_GZIP_PATH,
):
    """Export required CSV and a gzip-compressed copy without changing raw data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    df.to_csv(compressed_path, index=False, compression="gzip")
    return output_path


def main() -> None:
    raw = load_raw_data()
    cleaned = clean_sales_data(raw)
    output_path = export_cleaned_data(cleaned)
    print(f"Raw rows: {len(raw):,}")
    print(f"Cleaned rows: {len(cleaned):,}")
    print(f"Outlier candidates retained: {int(cleaned['Is_Outlier_Candidate'].sum()):,}")
    print(f"Exported: {output_path}")
    csv_size = output_path.stat().st_size
    gzip_size = PROCESSED_GZIP_PATH.stat().st_size
    compression_percent = (1 - gzip_size / csv_size) * 100
    print(f"Compressed: {PROCESSED_GZIP_PATH}")
    print(f"Storage reduction: {compression_percent:.1f}%")


if __name__ == "__main__":
    main()
