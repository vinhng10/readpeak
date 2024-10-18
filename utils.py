import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.axes import Axes
from matplotlib.figure import Figure

warnings.filterwarnings("ignore")


def normalize_os(item: str | None) -> str | None:
    if item == None:
        return None
    elif "android" in item:
        return "android"
    elif "ios" in item:
        return "ios"
    elif "mac" in item:
        return "mac"
    elif "windows" in item:
        return "windows"
    elif "chrome" in item:
        return "chrome"
    elif "linux" in item:
        return "linux"
    elif "x11" in item:
        return "x11"
    elif "ipados" in item:
        return "ios"
    else:
        return item


def normalize_make(item: str | None) -> str | None:
    if item == None:
        return None

    item = item.replace("tablet", "")
    item = item.split()[0]
    if "apple" in item:
        return "apple"
    elif "iphone" in item:
        return "apple"
    elif "mac" in item:
        return "apple"
    elif "ipad" in item:
        return "apple"
    elif "nokia" in item:
        return "nokia"
    elif "mi" in item and "microsoft" not in item:
        return "xiaomi"
    else:
        return item


def normalize_lang(item: str | None) -> str | None:
    return item.split("-")[0] if item else None


def plot_category_vs_ctr(df: pd.DataFrame, column: str, n: int = 100) -> Figure:
    # Compute CTR per category of the column:
    df = (
        df.groupby(column)
        .agg(impressions=("label", "size"), clicks=("label", "sum"))
        .reset_index()
    )
    df["CTR"] = (100 * df["clicks"] / df["impressions"]).round(2)
    if column in ["hour", "weekday"]:
        df = df.sort_values(column, ascending=True)
    else:
        df = df.sort_values("impressions", ascending=False)
    df = df.iloc[:n, :]

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(15, 3))

    # Position for bars on x-axis
    bar_width = 0.35  # Width of the bars
    index = np.arange(len(df))  # X locations for the groups

    # Bar plot for 'impressions' on the left y-axis
    ax1.bar(
        index - bar_width / 2,
        df["impressions"],
        bar_width,
        label="Impressions",
    )

    # Set up the first y-axis for impressions
    ax1.set_ylabel("Total Impressions", fontsize=12)
    ax1.set_xlabel(column.capitalize(), fontsize=12)

    # Create the second y-axis for CTR
    ax2 = ax1.twinx()

    # Force the y-axis limit for CTR to be 0-30
    ax2.set_ylim(0, 30)

    # Bar plot for 'CTR' on the right y-axis, scaled as a percentage
    ax2.bar(
        index + bar_width / 2, df["CTR"], bar_width, label="CTR (%)", color="tomato"
    )

    # Set up the second y-axis for CTR
    ax2.set_ylabel("CTR (%)", fontsize=12)

    # Adjust x-axis labels and ticks
    ax1.set_xticks(index)
    ax1.set_xticklabels(df[column], rotation=90)

    # Get handles and labels from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine both legends
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    # Display plot
    ax1.set_title(f"{column.capitalize()} Impressions vs CTR", fontsize=14)

    plt.close(fig)

    return fig


def barplot(
    series: pd.Series,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[int] = (10, 6),
    ax: Axes | None = None,
) -> None:
    if ax == None:
        _, ax = plt.subplots(figsize=figsize)
    series = series.sort_values(ascending=False)
    sns.barplot(series, ax=ax)
    ax.set_title(title)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
