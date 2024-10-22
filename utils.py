import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.utils.random import check_random_state
from category_encoders import LeaveOneOutEncoder
from matplotlib.figure import Figure
from ipywidgets import widgets
from functools import reduce


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


def posterior_ctr(
    clicks: int, impressions: int, alpha: float = 4, beta: float = 996, rounding=True
) -> float:
    ctr = 100 * (alpha + clicks) / (alpha + beta + impressions)
    if rounding:
        return round(ctr, 2)
    else:
        return ctr


def select_for_plot(data: pd.DataFrame, column: str, N: int = 15) -> list[any]:
    CTR = (
        data.groupby(column)
        .agg(impressions=("label", "size"), clicks=("label", "sum"))
        .reset_index()
    )
    CTR["CTR"] = posterior_ctr(CTR["clicks"], CTR["impressions"])
    selected = list(CTR.sort_values("CTR", ascending=False)[column])[: N // 2]
    # print(data[column].value_counts(ascending=False))
    counts = list(data[column].value_counts(ascending=False).index)
    for count in counts:
        if len(selected) >= N:
            break
        if count not in selected:
            selected.append(count)
        else:
            continue
    return selected


def group_df(data: pd.DataFrame, filters: dict[str, str], view_choice: str):
    # Filter the data:
    filtered_data = filter_df(data, filters)

    # Group by 'os' and 'lt' (using pd.Grouper to resample by hour)
    hourly_impressions = (
        filtered_data.groupby(["os", pd.Grouper(key="lt", freq="H")])
        .agg(
            impressions=("label", "size"),  # Count the number of impressions
            clicks=("label", "sum"),  # Sum the 'label' column to count clicks
        )
        .reset_index()
    )
    total_impressions = (
        hourly_impressions.groupby("lt")["impressions"].sum().reset_index()
    )

    # Calculate CTR for both hourly and cumulative cases
    if view_choice == "hourly":
        total_ctr = (
            hourly_impressions.groupby("lt")
            .apply(lambda x: posterior_ctr(x["clicks"].sum(), x["impressions"].sum()))
            .reset_index(name="CTR")
        )
        total_cumulative_impressions = None
    elif view_choice == "cumulative":
        # Calculate cumulative impressions and clicks
        hourly_impressions["cumulative_impressions"] = hourly_impressions.groupby("os")[
            "impressions"
        ].cumsum()
        hourly_impressions["cumulative_clicks"] = hourly_impressions.groupby("os")[
            "clicks"
        ].cumsum()

        total_cumulative_impressions = (
            hourly_impressions.groupby("lt")["impressions"]
            .sum()
            .cumsum()
            .reset_index(name="cumulative_impressions")
        )
        total_cumulative_clicks = (
            hourly_impressions.groupby("lt")["clicks"]
            .sum()
            .cumsum()
            .reset_index(name="cumulative_clicks")
        )

        # Calculate cumulative CTR
        total_ctr = pd.merge(
            total_cumulative_impressions, total_cumulative_clicks, on="lt"
        )
        total_ctr["CTR"] = total_ctr.apply(
            lambda row: posterior_ctr(
                row["cumulative_clicks"], row["cumulative_impressions"]
            ),
            axis=1,
        )

    return (
        hourly_impressions,
        total_impressions,
        total_cumulative_impressions,
        total_ctr,
    )


def update_plot(
    data: pd.DataFrame,
    art_id: str,
    view_choice: str,
    os_choice: str,
    output: widgets.Output,
):
    # Clear the output widget before rendering the new plot
    output.clear_output()
    (
        hourly_impressions,
        total_impressions,
        total_cumulative_impressions,
        total_ctr,
    ) = group_df(data, {"art": art_id}, view_choice)

    with output:
        # Create a figure and axes
        fig, ax1 = plt.subplots(figsize=(13, 3))

        if view_choice == "hourly":
            # Plot the hourly impressions over time
            if os_choice == "individual":
                sns.lineplot(
                    data=hourly_impressions, x="lt", y="impressions", hue="os", ax=ax1
                )
            elif os_choice == "all":
                sns.lineplot(
                    data=total_impressions,
                    x="lt",
                    y="impressions",
                    color="red",
                    label="Total",
                    ax=ax1,
                )
        elif view_choice == "cumulative":
            # Plot the cumulative impressions over time
            if os_choice == "individual":
                sns.lineplot(
                    data=hourly_impressions,
                    x="lt",
                    y="cumulative_impressions",
                    hue="os",
                    ax=ax1,
                )
            elif os_choice == "all":
                sns.lineplot(
                    data=total_cumulative_impressions,
                    x="lt",
                    y="cumulative_impressions",
                    color="red",
                    label="Total",
                    ax=ax1,
                )

        # Create a secondary y-axis for CTR
        ax2 = ax1.twinx()
        ax2.set_ylabel("CTR (%)")
        ax2.set_ylim(-0.5, max(total_ctr["CTR"].max() + 5, 10))

        # Plot the CTR as a bar plot
        bars = ax2.bar(
            total_ctr["lt"],
            total_ctr["CTR"],
            alpha=0.2,
            color="black",
            width=0.03,
            label="CTR",
        )

        ax1.set_title(f"Ad Performance")
        ax1.set_ylabel("Impressions")
        ax1.set_xlabel("")
        ax1.set_xlim([data["lt"].min(), data["lt"].max()])
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + [bars], labels1 + labels2, loc="upper right")

        # Show the plot
        plt.show()


def update_detailed_plot(
    data: pd.DataFrame,
    art_id: str,
    tag_id: str | None,
    view_choice: str,
    os_choice: str,
    output: widgets.Output,
):
    # Clear the output widget before rendering the new plot
    output.clear_output()

    (
        art_hourly_impressions,
        art_total_impressions,
        art_total_cumulative_impressions,
        art_total_ctr,
    ) = group_df(data, {"art": art_id, "tag": tag_id}, view_choice)

    (
        tag_hourly_impressions,
        tag_total_impressions,
        tag_total_cumulative_impressions,
        tag_total_ctr,
    ) = group_df(data, {"tag": tag_id}, view_choice)

    hourly_impressions = [
        art_hourly_impressions,
        tag_hourly_impressions,
    ]
    total_impressions = [
        art_total_impressions,
        tag_total_impressions,
    ]
    total_cumulative_impressions = [
        art_total_cumulative_impressions,
        tag_total_cumulative_impressions,
    ]
    total_ctr = [art_total_ctr, tag_total_ctr]

    titles = [
        "Ad Performance Per Tag",
        "Tag Performance Across All Ads",
    ]

    # Determine the global min and max datetime (x-axis limits)
    min_datetime = data["lt"].min()
    max_datetime = data["lt"].max()

    with output:
        # Create a figure and axes
        fig, ax = plt.subplots(len(titles), 1, figsize=(13, 7))
        plt.subplots_adjust(hspace=1)
        for i in range(len(titles)):
            if view_choice == "hourly":
                # Plot the hourly impressions over time
                if os_choice == "individual":
                    sns.lineplot(
                        data=hourly_impressions[i],
                        x="lt",
                        y="impressions",
                        hue="os",
                        ax=ax[i],
                    )
                elif os_choice == "all":
                    sns.lineplot(
                        data=total_impressions[i],
                        x="lt",
                        y="impressions",
                        color="red",
                        label="Total",
                        ax=ax[i],
                    )
                ax[i].set_ylim(
                    0, max(total_impressions[i]["impressions"].max() + 100, 500)
                )
            elif view_choice == "cumulative":
                # Plot the cumulative impressions over time
                if os_choice == "individual":
                    sns.lineplot(
                        data=hourly_impressions[i],
                        x="lt",
                        y="cumulative_impressions",
                        hue="os",
                        ax=ax[i],
                    )
                elif os_choice == "all":
                    sns.lineplot(
                        data=total_cumulative_impressions[i],
                        x="lt",
                        y="cumulative_impressions",
                        color="red",
                        label="Total",
                        ax=ax[i],
                    )
                ax[i].set_ylim(
                    0,
                    max(
                        total_cumulative_impressions[i]["cumulative_impressions"].max()
                        + 5000,
                        500,
                    ),
                )

            # Create a secondary y-axis for CTR
            ax2 = ax[i].twinx()
            ax2.set_ylabel("CTR (%)")
            ax2.set_ylim(0, max(total_ctr[i]["CTR"].max() + 5, 10))

            # Plot the CTR as a bar plot
            bars = ax2.bar(
                total_ctr[i]["lt"],
                total_ctr[i]["CTR"],
                alpha=0.2,
                color="black",
                width=0.03,
                label="CTR",
            )

            # Set the same x-axis limits for all subplots
            ax[i].set_title(titles[i])
            ax[i].set_ylabel("Impressions")
            ax[i].set_xlabel("")
            ax[i].set_xlim([min_datetime, max_datetime])
            ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45)

            # Combine legends
            lines1, labels1 = ax[i].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax[i].legend(lines1 + [bars], labels1 + labels2, loc="upper right")

        # Show the plot
        plt.show()


def filter_df(data: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    return data[reduce(lambda x, y: x & y, [data[k] == v for k, v in filters.items()])]


def create_content(
    data: pd.DataFrame,
    art_id: str,
    tag_id: str | None = None,
) -> widgets.Box:
    output = widgets.Output()

    # Create radio buttons for "hourly" and "cumulative"
    view_radio = widgets.RadioButtons(
        options=["hourly", "cumulative"],
        description="View:",
        disabled=False,
    )
    os_radio = widgets.RadioButtons(
        options=["all", "individual"],
        description="OS:",
        disabled=False,
    )
    radio_buttons = widgets.HBox([view_radio, os_radio])

    # Display the initial plot (hourly by default)
    if tag_id:
        update_detailed_plot(data, art_id, tag_id, "hourly", "all", output)
    else:
        update_plot(data, art_id, "hourly", "all", output)

    # Add a callback to the radio buttons to update the plot when the option changes
    view_radio.observe(
        lambda change, output=output: (
            update_detailed_plot(
                data,
                art_id,
                tag_id,
                change["new"],
                os_radio.value,
                output,
            )
            if tag_id
            else update_plot(
                data,
                art_id,
                change["new"],
                os_radio.value,
                output,
            )
        ),
        names="value",
    )
    os_radio.observe(
        lambda change, output=output: (
            update_detailed_plot(
                data,
                art_id,
                tag_id,
                view_radio.value,
                change["new"],
                output,
            )
            if tag_id
            else update_plot(
                data,
                art_id,
                view_radio.value,
                change["new"],
                output,
            )
        ),
        names="value",
    )

    # Organize the layout with radio buttons and plot
    content = widgets.HBox([radio_buttons, output])

    return content


def plot_category_vs_ctr(data: pd.DataFrame, column: str, n: int = 100) -> Figure:
    # Compute CTR per category of the column:
    data = (
        data.groupby(column)
        .agg(impressions=("label", "size"), clicks=("label", "sum"))
        .reset_index()
    )
    data["CTR"] = posterior_ctr(data["clicks"], data["impressions"])
    data = data.sort_values("impressions", ascending=False).iloc[:n, :]

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(15, 3))

    # Position for bars on x-axis
    bar_width = 0.35  # Width of the bars
    index = np.arange(len(data))  # X locations for the groups

    # Bar plot for 'impressions' on the left y-axis
    ax1.bar(
        index - bar_width / 2,
        data["impressions"],
        bar_width,
        label="Impressions",
    )

    # Set up the first y-axis for impressions
    ax1.set_ylabel("Total Impressions", fontsize=12)

    # Create the second y-axis for CTR
    ax2 = ax1.twinx()

    # Force the y-axis limit for CTR to be 0-30
    ax2.set_ylim(0, 25)

    # Bar plot for 'CTR' on the right y-axis, scaled as a percentage
    ax2.bar(
        index + bar_width / 2, data["CTR"], bar_width, label="CTR (%)", color="tomato"
    )

    # Set up the second y-axis for CTR
    ax2.set_ylabel("CTR (%)", fontsize=12)

    # Adjust x-axis labels and ticks
    ax1.set_xticks(index)
    ax1.set_xticklabels(data[column], rotation=90)

    # Get handles and labels from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine both legends
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    # Display plot
    ax1.set_title(f"{column.capitalize()}", fontsize=16)

    plt.close(fig)

    return fig


class CTRLeaveOneOutEncoder(LeaveOneOutEncoder):

    def fit_leave_one_out(self, X_in, y, cols=None):
        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values

        self._mean = posterior_ctr(y.sum(), y.count(), rounding=False)

        return {col: self.fit_column_map(X[col], y) for col in cols}

    def transform_leave_one_out(self, X, y, mapping=None):
        """
        Leave one out encoding uses a single column of floats to represent the means of the target variables.
        """

        random_state_ = check_random_state(self.random_state)

        for col, colmap in mapping.items():
            level_notunique = colmap["count"] > 1

            unique_train = colmap.index
            unseen_values = pd.Series(
                [x for x in X[col].unique() if x not in unique_train],
                dtype=unique_train.dtype,
            )

            is_nan = X[col].isnull()
            is_unknown_value = X[col].isin(unseen_values.dropna().astype(object))

            if (
                X[col].dtype.name == "category"
            ):  # Pandas 0.24 tries hard to preserve categorical data type
                index_dtype = X[col].dtype.categories.dtype
                X[col] = X[col].astype(index_dtype)

            if self.handle_unknown == "error" and is_unknown_value.any():
                raise ValueError("Columns to be encoded can not contain new values")

            if (
                y is None
            ):  # Replace level with its mean target; if level occurs only once, use global mean
                level_means = posterior_ctr(
                    colmap["sum"], colmap["count"], rounding=False
                ).where(level_notunique, self._mean)
                X[col] = X[col].map(level_means)
            else:  # Replace level with its mean target, calculated excluding this row's target
                # The y (target) mean for this level is normally just the sum/count;
                # excluding this row's y, it's (sum - y) / (count - 1)
                level_means = posterior_ctr(
                    X[col].map(colmap["sum"]) - y,
                    X[col].map(colmap["count"]) - 1,
                    rounding=False,
                )
                # The 'where' fills in singleton levels (count = 1 -> div by 0) with the global mean
                X[col] = level_means.where(
                    X[col].map(colmap["count"][level_notunique]).notnull(), self._mean
                )

            if self.handle_unknown == "value":
                X.loc[is_unknown_value, col] = self._mean
            elif self.handle_unknown == "return_nan":
                X.loc[is_unknown_value, col] = np.nan

            if self.handle_missing == "value":
                X.loc[is_nan & unseen_values.isnull().any(), col] = self._mean
            elif self.handle_missing == "return_nan":
                X.loc[is_nan, col] = np.nan

            if self.sigma is not None and y is not None:
                X[col] = X[col] * random_state_.normal(1.0, self.sigma, X[col].shape[0])

        return X
