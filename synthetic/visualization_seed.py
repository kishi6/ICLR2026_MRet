import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from typing import Optional
import conf
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from matplotlib.axes import Axes

color_dict = conf.color_dict


def calculate_statistics(all_data, value_col, group_cols):
    grouped = all_data.groupby(group_cols)[value_col]
    stats = grouped.agg(mean="mean", sem=sem).reset_index()
    stats["ci95_low"] = stats["mean"] - 1.96 * stats["sem"]
    stats["ci95_high"] = stats["mean"] + 1.96 * stats["sem"]
    return stats


def _plot_line_with_ci(
    stats: pd.DataFrame,
    *,
    ylabel: str,
    figsize: tuple[int, int] = (10, 6),
    legend_ncol: int = 8,
    ax: Optional[Axes] = None,
    y_sig_digits: Optional[int] = None,
) -> None:
    plt.style.use("ggplot")
    plt.rcParams["font.size"] = 18
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    for method in stats["method"].unique():
        m = stats[stats["method"] == method]
        color = color_dict.get(method, "black")
        ax.plot(m["t"], m["mean"], label=method, linewidth=2.5, color=color)
        ax.fill_between(m["t"], m["ci95_low"], m["ci95_high"], alpha=.2, color=color)
    ax.set_xlabel("timestep")
    ax.set_ylabel(ylabel)
    if y_sig_digits:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, pos: "0" if v == 0 else f"{v:.{y_sig_digits}g}")
        )


def _extract_metric(
    df: pd.DataFrame,
    base: str,
    *,
    side: str = "both",
    n_x: Optional[int] = None,
    n_y: Optional[int] = None,
) -> pd.Series:
    if side == "x":
        return df[f"{base}_x"]
    if side == "y":
        return df[f"{base}_y"]
    if n_x is None or n_y is None:
        raise ValueError("Please specify n_x, n_y when side='both'")
    return (df[f"{base}_x"] * n_x + df[f"{base}_y"] * n_y) / (n_x + n_y)


_DEFAULT_METHOD_ORDER = [
    "MRet (best)",
    "MRet",
    "optimal_ranking",
    "Uniform",
    "MaxMatch",
    "FairCo (lam=0.01)",
    "FairCo (lam=0.1)",
    "FairCo (lam=100)",
    "FairCo (equal exposure) (lam=0.1)",
    "FairCo (equal exposure) (lam=1)",
    "FairCo (equal exposure) (lam=10)",
    "FairCo (equal exposure) (lam=100)",
]


def _prepare_stats(
    all_data: pd.DataFrame,
    metric_base: str,
    *,
    side: str = "both",
    n_x: Optional[int] = None,
    n_y: Optional[int] = None,
    extra_transform: Optional[callable] = None,
    method_order=_DEFAULT_METHOD_ORDER,
) -> pd.DataFrame:
    tmp = all_data.copy()
    tmp["metric"] = _extract_metric(tmp, metric_base, side=side, n_x=n_x, n_y=n_y)
    if extra_transform:
        tmp["metric"] = extra_transform(tmp)
    tmp["method"] = pd.Categorical(tmp["method"], categories=method_order, ordered=True)
    return calculate_statistics(tmp, "metric", ["t", "method"])


def plot_match_per(all_data, *, side="both", n_x=None, n_y=None, ax=None, y_sig_digits=None, **kwargs):
    stats = _prepare_stats(all_data, "match", side=side, n_x=n_x, n_y=n_y)
    ylabel = f"number of matches ({side})" if side != "both" else "number of matches"
    _plot_line_with_ci(stats, ylabel=ylabel, ax=ax, y_sig_digits=y_sig_digits, **kwargs)


def plot_number_user_retain(all_data, *, side="both", n_x=None, n_y=None, ax=None, y_sig_digits=None, **kwargs):
    stats = _prepare_stats(all_data, "active_users", side=side, n_x=n_x, n_y=n_y)
    ylabel = "user retention rate"
    if side != "both":
        ylabel += f" ({side})"
    _plot_line_with_ci(stats, ylabel=ylabel, ax=ax, y_sig_digits=y_sig_digits, **kwargs)


def plot_user_retain(all_data, *, side="both", n_x=None, n_y=None, ax=None, **kwargs):
    stats = _prepare_stats(all_data, "true_user_retain", side=side, n_x=n_x, n_y=n_y)
    ylabel = "user retain"
    if side != "both":
        ylabel += f" ({side})"
    _plot_line_with_ci(stats, ylabel=ylabel, ax=ax, **kwargs)


def plot_histogram(
    results: dict,
    method_list: list,
    metric: str,
    T: Optional[int] = None,
    bins: int = 75,
    xlabel: Optional[str] = None,
    legend: str = "upper left",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    last_index = -1 if T is None else T
    for m in method_list:
        data = results[m][metric][last_index]
        ax.hist(data, bins=bins, color=color_dict[m], alpha=0.5, label=m)
    ax.set_xlabel(xlabel or metric)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


def plot_match_and_user_retain(
    all_data: pd.DataFrame,
    *,
    side: str = "both",
    n_x: Optional[int] = None,
    n_y: Optional[int] = None,
    figsize: tuple[int, int] = (14, 6),
    x_log_scale: bool = False,
    y_sig_digits: Optional[int] = None,
    legend_ncol: int = 8,
    save_path: Optional[Path] = None,
) -> None:
    plt.style.use("ggplot")
    plt.rcParams["font.size"] = 20
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True)
    plot_match_per(
        all_data,
        side=side,
        n_x=n_x,
        n_y=n_y,
        ax=axs[0],
        y_sig_digits=y_sig_digits,
        legend_ncol=legend_ncol,
    )
    plot_number_user_retain(
        all_data,
        side=side,
        n_x=n_x,
        n_y=n_y,
        ax=axs[1],
        y_sig_digits=y_sig_digits,
        legend_ncol=legend_ncol,
    )
    if x_log_scale:
        for ax in axs:
            ax.set_xscale("log")
            from matplotlib.ticker import ScalarFormatter
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(style="plain", axis="x")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()
