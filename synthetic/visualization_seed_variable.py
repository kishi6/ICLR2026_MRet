import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
import conf
from typing import List, Optional
from utils import relative_by_policy
from pathlib import Path

plt.style.use("ggplot")
plt.rcParams["font.size"] = 16
COLOR_DICT = conf.color_dict

def _calculate_statistics(df: pd.DataFrame,
                          value_col: str,
                          group_cols: list[str],
                          ci: float = 0.95) -> pd.DataFrame:
    grouped = df.groupby(group_cols)[value_col]
    stats = grouped.agg(mean="mean", sem=sem).reset_index()
    z = 1.96 if np.isclose(ci, 0.95) else 1.0
    stats["ci_low"] = stats["mean"] - z * stats["sem"]
    stats["ci_high"] = stats["mean"] + z * stats["sem"]
    return stats

def plot_metric(df: pd.DataFrame,
                *,
                x_col: str,
                y_col: str,
                xlabel: str,
                ylabel: str,
                method_order: Optional[List[str]] = None,
                ci: float = 0.95,
                color_dict: dict[str, str] = COLOR_DICT,
                figsize: tuple[int, int] = (10, 6),
                ax: Optional[plt.Axes] = None,
                x_log_scale: bool = False) -> None:
    df = df.copy()
    if method_order is not None:
        df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    stats = _calculate_statistics(df, y_col, [x_col, "method"], ci)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    for method in stats["method"].unique():
        sub = stats[stats["method"] == method]
        color = color_dict.get(method, "black")
        ax.plot(sub[x_col], sub["mean"], label=method, linewidth=3, color=color, marker="o", markersize=8)
        ax.fill_between(sub[x_col], sub["ci_low"], sub["ci_high"], alpha=0.2, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x_log_scale:
        ax.set_xscale("log")
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style="plain", axis="x")
        ax.set_xticks(sorted(df[x_col].unique()))

def run_visual(
    all_data: pd.DataFrame,
    variable: str,
    n_x: int,
    n_y: int,
    T: int,
    x_log_scale: bool = False
):
    all_data_T = relative_by_policy(all_data[all_data["t"] == T - 1], variable, conf.relative)
    if variable == "n_train":
        x_label = "training data size"
    elif variable == "kappa":
        x_label = "level of popularity"
    elif variable == "ranking_metric":
        x_label = "examination function"
    elif variable == "alpha_param":
        x_label = "MRet (best) matches for satisfaction"
    elif variable == "K":
        x_label = "length of ranking"
    elif variable == "n_xy":
        x_label = "number of users"
    elif variable == "proportion":
        x_label = "group size ratio"
    elif variable == "lambda_":
        x_label = "parameter of FairCo"
    else:
        x_label = variable
    show_method_list = conf.show_method_list
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    plot_metric(
        all_data_T,
        x_col=variable,
        y_col="match_x",
        xlabel=x_label,
        ylabel="cumulative number of matches",
        method_order=show_method_list,
        ax=axs[0],
        x_log_scale=x_log_scale,
    )
    tmp = all_data_T.copy()
    tmp["total_active_users"] = (tmp["active_users_x"] * n_x + tmp["active_users_y"] * n_y) / (n_x + n_y)
    plot_metric(
        tmp,
        x_col=variable,
        y_col="total_active_users",
        xlabel=x_label,
        ylabel="user retention rate",
        method_order=show_method_list,
        ax=axs[1],
        x_log_scale=x_log_scale,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    tmp = all_data_T.copy()
    tmp["total_retain"] = (tmp["true_user_retain_x"] * n_x + tmp["true_user_retain_y"] * n_y) / (n_x + n_y)
    plot_metric(
        tmp,
        x_col=variable,
        y_col="total_retain",
        xlabel=x_label,
        ylabel="User Retention",
        method_order=show_method_list,
        x_log_scale=x_log_scale,
    )

def plot_match_and_user_retain_variable(
    all_data: pd.DataFrame,
    variable: str,
    n_x: int,
    n_y: int,
    T: int,
    x_log_scale: bool = False,
    figsize: tuple[int, int] = (14, 6),
    save_path: Optional[Path] = None,
) -> None:
    all_data_T = relative_by_policy(all_data[all_data["t"] == T - 1], variable, conf.relative)
    if variable == "n_train":
        x_label = "training data size"
    elif variable == "kappa":
        x_label = "level of popularity"
    elif variable == "ranking_metric":
        x_label = "examination function"
    elif variable == "alpha_param":
        x_label = "MRet (best) matches for satisfaction"
    elif variable == "K":
        x_label = "length of ranking"
    elif variable == "n_xy":
        x_label = "number of users"
    elif variable == "proportion":
        x_label = "group size ratio"
    elif variable == "lambda_":
        x_label = "parameter of FairCo"
    elif variable == "rel_noise":
        x_label = "noise level"
    else:
        x_label = variable
    show_method_list = conf.show_method_list
    if variable == "n_xy":
        x_label = "number of users"
        show_method_list = ["MRet (best)", "Uniform", "MaxMatch", "FairCo (lam=100)", "MRet", "FairCo (equal exposure) (lam=100)"]
    if variable == "lambda_":
        show_method_list = ["FairCo", "FairCo (equal exposure)"]
    plt.style.use("ggplot")
    plt.rcParams["font.size"] = 20
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True)
    plot_metric(
        all_data_T,
        x_col=variable,
        y_col="match_x",
        xlabel=x_label,
        ylabel="number of matches",
        method_order=show_method_list,
        ax=axs[0],
        x_log_scale=x_log_scale,
    )
    tmp = all_data_T.copy()
    tmp["total_active_users"] = (tmp["active_users_x"] * n_x + tmp["active_users_y"] * n_y) / (n_x + n_y)
    plot_metric(
        tmp,
        x_col=variable,
        y_col="total_active_users",
        xlabel=x_label,
        ylabel="user retention rate",
        method_order=show_method_list,
        ax=axs[1],
        x_log_scale=x_log_scale,
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


# def output_table(
#     all_data: pd.DataFrame,
#     variable: str,
#     n_x: int,
#     n_y: int,
#     T: int,
# ) -> None:
#     all_data_T = relative_by_policy(all_data[all_data["t"] == T - 1], variable, conf.relative)
#     if variable == "n_train":
#         x_label = "training data size"
#     elif variable == "kappa":
#         x_label = "level of popularity"
#     elif variable == "ranking_metric":
#         x_label = "examination function"
#     elif variable == "alpha_param":
#         x_label = "optimal matches for satisfaction"
#     elif variable == "K":
#         x_label = "length of ranking"
#     elif variable == "n_xy":
#         x_label = "number of users"
#     elif variable == "proportion":
#         x_label = "group size ratio"
#     elif variable == "lambda_":
#         x_label = "parameter of FairCo"
#     else:
#         x_label = variable

#     show_method_list = conf.show_method_list
#     if variable == "lambda_":
#         show_method_list = ["optimal", "uniform", "naive", "fairco", "proposed"]

#     tmp = all_data_T.copy()
#     tmp["total_active_users"] = (tmp["active_users_x"] * n_x + tmp["active_users_y"] * n_y) / (n_x + n_y)
#     tmp["total_retain"] = (tmp["true_user_retain_x"] * n_x + tmp["true_user_retain_y"] * n_y) / (n_x + n_y)

#     # Calculate mean and 95% CI (same as run_visual)
#     def _stats(value_col: str, prefix: str) -> pd.DataFrame:
#         s = _calculate_statistics(tmp, value_col, [variable, "method"], ci=0.95)
#         s = s.drop(columns=["sem"]).rename(
#             columns={
#                 "mean": f"{prefix}_mean",
#                 "ci_low": f"{prefix}_ci_low",
#                 "ci_high": f"{prefix}_ci_high",
#             }
#         )
#         return s

#     stats = (
#         _stats("match_x", "match_x")
#         .merge(_stats("total_active_users", "total_active_users"), on=[variable, "method"])
#         .merge(_stats("total_retain", "total_retain"), on=[variable, "method"])
#     )

#     # Filter only target methods and fix order
#     stats = stats[stats["method"].isin(show_method_list)].copy()
#     stats["method"] = pd.Categorical(stats["method"], categories=show_method_list, ordered=True)
#     stats = stats.sort_values(by=["method", variable])

#     # Output
#     cols = [
#         "method",
#         variable,
#         # "match_x_mean", "match_x_ci_low", "match_x_ci_high",
#         "total_active_users_mean", 
#         # "total_active_users_ci_low", "total_active_users_ci_high",
#         # "total_retain_mean", "total_retain_ci_low", "total_retain_ci_high",
#     ]
#     print(stats[cols].to_string(index=False))


def output_table(
    all_data: pd.DataFrame,
    variable: str,
    n_x: int,
    n_y: int,
    T: int,
) -> None:
    all_data_T = relative_by_policy(all_data[all_data["t"] == T - 1], variable, conf.relative)
    if variable == "n_train":
        x_label = "training data size"
    elif variable == "kappa":
        x_label = "level of popularity"
    elif variable == "ranking_metric":
        x_label = "examination function"
    elif variable == "alpha_param":
        x_label = "MRet (best) matches for satisfaction"
    elif variable == "K":
        x_label = "length of ranking"
    elif variable == "n_xy":
        x_label = "number of users"
    elif variable == "proportion":
        x_label = "group size ratio"
    elif variable == "lambda_":
        x_label = "parameter of FairCo"
    else:
        x_label = variable

    show_method_list = conf.show_method_list
    if variable == "lambda_":
        show_method_list = ["FairCo", "FairCo (equal exposure)"]

    tmp = all_data_T.copy()
    tmp["total_active_users"] = (tmp["active_users_x"] * n_x + tmp["active_users_y"] * n_y) / (n_x + n_y)
    tmp["total_retain"] = (tmp["true_user_retain_x"] * n_x + tmp["true_user_retain_y"] * n_y) / (n_x + n_y)

    # Calculate statistics
    def _stats(value_col: str, prefix: str) -> pd.DataFrame:
        s = _calculate_statistics(tmp, value_col, [variable, "method"], ci=0.95)
        s = s.drop(columns=["sem"]).rename(
            columns={
                "mean": f"{prefix}_mean",
                "ci_low": f"{prefix}_ci_low",
                "ci_high": f"{prefix}_ci_high",
            }
        )
        return s

    # Add all statistics to be merged
    stats = (
        _stats("match_x", "match_x")
        .merge(_stats("total_active_users", "total_active_users"), on=[variable, "method"])
        .merge(_stats("total_retain", "total_retain"), on=[variable, "method"])
        .merge(_stats("active_users_x", "active_users_x"), on=[variable, "method"])
        .merge(_stats("active_users_y", "active_users_y"), on=[variable, "method"])
    )

    # Filter by target methods
    stats = stats[stats["method"].isin(show_method_list)].copy()
    stats["method"] = pd.Categorical(stats["method"], categories=show_method_list, ordered=True)
    stats = stats.sort_values(by=["method", variable])

    # Add columns to output
    cols = [
        "method",
        variable,
        # "active_users_x_mean",
        # "active_users_y_mean",
        "total_active_users_mean",
        # "match_x_mean", "match_x_ci_low", "match_x_ci_high",
        # "total_active_users_ci_low", "total_active_users_ci_high",
        # "total_retain_mean", "total_retain_ci_low", "total_retain_ci_high",
    ]

    print(stats[cols].to_string(index=False))
