import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
        return "unknown"


def normalize_make(item: str | None) -> str | None:
    if item == None:
        return None

    item = item.replace("tablet", "")
    if "samsung" in item:
        return "samsung"
    elif "apple" in item:
        return "apple"
    elif "iphone" in item:
        return "apple"
    elif "mac" in item:
        return "apple"
    elif "ipad" in item:
        return "apple"
    elif "huawei" in item:
        return "huawei"
    elif "oneplus" in item:
        return "oneplus"
    elif "nokia" in item:
        return "nokia"
    elif "mi" in item and "microsoft" not in item:
        return "xiaomi"
    elif "lenovo" in item:
        return "lenovo"
    elif "motorola" in item:
        return "motorola"
    elif "sony" in item:
        return "sony"
    elif "unknown" in item:
        return "unknown"
    else:
        return "others"


def barplot(
    series: pd.Series,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[int] = (10, 6),
) -> None:
    series = series.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(series, ax=ax)
    ax.set_title(title)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()
