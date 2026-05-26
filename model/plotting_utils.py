"""Petits utilitaires graphiques utilisés par les notebooks MODAL_ORDER_BOOK.

This project is conducted by Lizhan Hong and Tom Zhang under the supervision of Professor Charles-Albert Lehalle.

Ces utilitaires restent volontairement proches de matplotlib/pandas afin que les notebooks
restent faciles à auditer. Ils ne modifient pas les résultats de simulation ; ils permettent
seulement de rendre explicites les diagnostics dégénérés et de maintenir une mise en forme cohérente des figures.
"""

from __future__ import annotations

from typing import Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_barplot_metric(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    group: Optional[str] = None,
    zero_policy: str = "annotate",
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Trace un diagramme en barres métrique, en annotant les valeurs dégénérées toutes nulles.

    Paramètres
    ----------
    df:
        Tableau source.
    x, y:
        Noms des colonnes pour les catégories et les valeurs métriques.
    group:
        Colonne de groupement optionnelle. Si elle est fournie, les étiquettes
        de catégorie deviennent ``x`` suivi de ``group`` sur une nouvelle ligne.
    zero_policy:
        ``"annotate"`` remplace les barres entièrement nulles par un message explicatif.
        Toute autre valeur trace les barres nulles normalement.
    """
    if x not in df or y not in df:
        raise KeyError(f"safe_barplot_metric requires columns {x!r} and {y!r}")
    data = df.copy()
    if data[y].isna().any():
        warnings.warn(f"Missing values in {y}; dropping NaN rows.", RuntimeWarning)
        data = data.dropna(subset=[y])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
    else:
        fig = ax.figure

    if data.empty:
        ax.text(0.5, 0.5, f"No finite values for {y}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax

    values = data[y].to_numpy(dtype=float)
    if zero_policy == "annotate" and np.allclose(values, 0.0):
        message = (
            "All displayed values are zero.\n"
            "For probability CI widths this often means the Bernoulli estimator is degenerate,\n"
            "not that the estimator has no uncertainty."
        )
        print(
            "Degenerate Bernoulli estimator: all estimated probabilities are equal to one. "
            "Probability CI width is not informative."
        )
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title or y)
        ax.set_axis_off()
        return fig, ax

    labels = data[x].astype(str)
    if group is not None:
        if group not in data:
            raise KeyError(f"group column {group!r} is not present")
        labels = labels + "\n" + data[group].astype(str)

    bars = ax.bar(np.arange(len(data)), values)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(labels)
    if labels.map(len).max() > 14 or len(labels) > 5:
        ax.tick_params(axis="x", rotation=20)
    ax.set_title(title or y)
    ax.set_ylabel(ylabel or y)

    ymin, ymax = ax.get_ylim()
    offset = 0.02 * (ymax - ymin if ymax > ymin else 1.0)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    return fig, ax


def plot_hist_with_ci(
    values: np.ndarray,
    mean: float,
    ci_low: float,
    ci_high: float,
    title: str,
    xlabel: str,
    ax: Optional[plt.Axes] = None,
    bins: int = 35,
) -> tuple[plt.Figure, plt.Axes]:
    """Histogram with a mean line and bootstrap-confidence band."""
    arr = np.asarray(values, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    ax.hist(arr, bins=bins, alpha=0.65, color="#4C78A8", edgecolor="white")
    ax.axvline(mean, color="#B22222", linewidth=2, label=f"mean = {mean:.3g}")
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        ax.axvspan(ci_low, ci_high, color="#B22222", alpha=0.15, label="bootstrap CI")
    ax.set_title(f"{title} (n={len(arr)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend()
    return fig, ax


def plot_delta_with_ci(
    df: pd.DataFrame,
    x: str = "a_cross",
    y: str = "delta_mean",
    low: str = "ci_low",
    high: str = "ci_high",
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot an effect-size curve with confidence bands."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    ordered = df.sort_values(x)
    xx = ordered[x].to_numpy(dtype=float)
    yy = ordered[y].to_numpy(dtype=float)
    lo = ordered[low].to_numpy(dtype=float)
    hi = ordered[high].to_numpy(dtype=float)
    ax.plot(xx, yy, marker="o", linewidth=2)
    ax.fill_between(xx, lo, hi, alpha=0.2)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Effect size with bootstrap confidence band")
    return fig, ax


def plot_joint_scatter_or_hexbin(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    sample_size_threshold: int = 2000,
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Use scatter for small samples and hexbin for large samples."""
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    corr = np.corrcoef(xx, yy)[0, 1] if len(xx) > 1 and np.std(xx) > 0 and np.std(yy) > 0 else np.nan
    if len(xx) >= sample_size_threshold:
        hb = ax.hexbin(xx, yy, gridsize=35, mincnt=1, cmap="viridis")
        fig.colorbar(hb, ax=ax, label="count")
    else:
        ax.scatter(xx, yy, alpha=0.45, s=18)
    suffix = f"n={len(xx)}, corr={corr:.3f}" if np.isfinite(corr) else f"n={len(xx)}"
    ax.set_title(f"{title} ({suffix})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def format_metric_table(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Return a notebook-friendly copy with rounded numeric columns."""
    out = df.copy()
    numeric = out.select_dtypes(include=[np.number]).columns
    out[numeric] = out[numeric].round(decimals)
    return out
