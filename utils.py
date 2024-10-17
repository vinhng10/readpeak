import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.axes import Axes

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
