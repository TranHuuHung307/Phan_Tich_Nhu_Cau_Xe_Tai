"""Reusable chart functions for the cleaned freight-demand dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

try:
    from .utils import FIGURES_DIR, PROCESSED_CSV_PATH, build_weekly_monthly_tables
except ImportError:
    from utils import FIGURES_DIR, PROCESSED_CSV_PATH, build_weekly_monthly_tables


def save_outlier_boxplots(
    df: pd.DataFrame, output_dir: Path = FIGURES_DIR
) -> Path:
    """Save grouped boxplots that preserve route and fleet context."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "01_outlier_boxplot.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x="Fleet_Type", y="Actual_Trips", hue="Route", ax=axes[0])
    sns.boxplot(
        data=df, x="Fleet_Type", y="Total_Volume_Tons", hue="Route", ax=axes[1]
    )
    axes[0].set_title("Actual trips by route and fleet")
    axes[1].set_title("Volume by route and fleet")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def save_monthly_trend(
    df: pd.DataFrame, output_dir: Path = FIGURES_DIR
) -> Path:
    """Save monthly average-weekly demand to avoid 4/5-week month bias."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "02_monthly_trend.png"
    _, monthly = build_weekly_monthly_tables(df)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(monthly["Date"], monthly["Avg_Weekly_Volume"])
    axes[0].set_ylabel("Average weekly volume")
    axes[1].plot(monthly["Date"], monthly["Avg_Weekly_Trips"])
    axes[1].set_ylabel("Average weekly trips")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def save_bar_chart(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Save route totals and average weekly trips by calendar month."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "03_bar_chart.png"
    weekly, _ = build_weekly_monthly_tables(df)
    route_totals = (
        df.groupby("Route", observed=True)["Actual_Trips"].sum().reset_index()
    )
    weekly["Month"] = weekly["Date"].dt.month
    month_average = (
        weekly.groupby("Month", observed=True)["Total_Trips"].mean().reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(route_totals["Route"], route_totals["Actual_Trips"])
    axes[0].set_title("Total trips by route")
    axes[0].set_ylabel("Trips")
    axes[1].bar(month_average["Month"], month_average["Total_Trips"])
    axes[1].set_title("Average weekly trips by month")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Average weekly trips")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def save_pie_chart(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Save freight-volume share by fleet type."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "04_pie_chart.png"
    fleet_volume = (
        df.groupby("Fleet_Type", observed=True)["Total_Volume_Tons"].sum()
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.pie(
        fleet_volume.values,
        labels=fleet_volume.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Freight volume share by fleet type")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def save_scatter_plot(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Save the relationship between trips and freight volume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "05_scatter_plot.png"
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="Actual_Trips",
        y="Total_Volume_Tons",
        hue="Fleet_Type",
        style="Route",
        ax=ax,
    )
    ax.set_title("Actual trips versus freight volume")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def save_decomposition(
    df: pd.DataFrame, output_dir: Path = FIGURES_DIR
) -> Path:
    """Decompose average weekly monthly demand into trend/seasonality/residual."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "06_decomposition.png"
    _, monthly = build_weekly_monthly_tables(df)
    series = monthly.set_index("Date")["Avg_Weekly_Volume"].asfreq("MS")
    decomposition = seasonal_decompose(
        series, model="additive", period=12, extrapolate_trend="freq"
    )
    figure = decomposition.plot()
    figure.set_size_inches(14, 10)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(figure)
    return output_path


def main() -> None:
    """Generate the reusable chart set from the processed dataset."""
    df = pd.read_csv(PROCESSED_CSV_PATH, parse_dates=["Date"])
    outputs = [
        save_outlier_boxplots(df),
        save_monthly_trend(df),
        save_bar_chart(df),
        save_pie_chart(df),
        save_scatter_plot(df),
        save_decomposition(df),
    ]
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
