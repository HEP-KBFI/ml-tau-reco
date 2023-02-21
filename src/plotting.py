import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import confusion_matrix

hep.style.use(hep.styles.CMS)


def plot_roc_curve(
    efficiencies: list, fakerates: list, marker: str = "^", label: str = "ROC", output_path: str = "roc.png"
) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(fakerates, efficiencies, marker=marker, label=label)
    plt.xlabel("Fake rate", fontdict={"size": 20})
    plt.ylabel("Efficiency", fontdict={"size": 20})
    plt.xscale("log")
    plt.grid(True, which="both")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close("all")


def plot_regression_confusion_matrix(
    y_true: np.array,
    y_pred: np.array,
    output_path: str,
    left_bin_edge: float = 0.0,
    right_bin_edge: float = 1.0,
    n_bins: int = 24,
    figsize: tuple = (12, 12),
    cmap: str = "Greys",
    y_label: str = "Predicted",
    x_label: str = "Truth",
    title: str = "Confusion matrix",
) -> None:
    """Plots the confusion matrix for the regression task. Although confusion
    matrix is in principle meant for classification task, the problem can be
    solved by binning the predictions and truth values.

    Args:
        y_true : np.array
            The array containing the truth values with shape (n,)
        y_pred : np.array
            The array containing the predicted values with shape (n,)
        output_path : str
            The path where output plot will be saved
        left_bin_edge : float
            [default: 0.0] The smallest value
        right_bin_edge : float
            [default: 1.0] The largest value
        n_bins : int
            [default: 24] The number of bins the values will be divided into
            linearly. The number of bin edges will be n_bin_edges = n_bins + 1
        figsize : tuple
            [default: (12, 12)] The size of the figure that will be created
        cmap : str
            [default: "Greys"] Name of the colormap to be used for the
            confusion matrix
        y_label : str
            [default: "Predicted"] The label for the y-axis
        x_label : str
            [default: "Truth"] The label for the x-axis
        title : str
            [default: "Confusion matrix"] The title for the plot

    Returns:
        None
    """
    bin_edges = np.linspace(left_bin_edge, right_bin_edge, num=n_bins + 1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.label_outer()
    bin_counts = np.histogram2d(y_true, y_pred, bins=[bin_edges, bin_edges])[0]
    im = ax.pcolor(bin_edges, bin_edges, bin_counts.transpose(), cmap=cmap, norm=colors.LogNorm())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_aspect("equal")
    ax.set_ylabel(f"{y_label}")
    ax.set_xlabel(f"{x_label}")
    plt.title(title, fontsize=18, loc="center", fontweight="bold", style="italic", family="monospace")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close("all")


def plot_classification_confusion_matrix(
    true_cats: np.array,
    pred_cats: np.array,
    categories: list,
    output_path: str,
    cmap: str = "gray",
    bin_text_color: str = "r",
    y_label: str = "Prediction",
    x_label: str = "Truth",
    title: str = "",
    figsize: tuple = (12, 12),
) -> None:
    """Plots the confusion matrix for the classification task. Confusion
    matrix functions has the categories in the other way in order to have the
    truth on the x axis.
    Args:
        true_cats : np.array,
            The true categories
        pred_cats : np.array
            The predicted categories
        categories : list
            Category labels in the correct order
        output_path : str
            The path where the plot will be outputted
        cmap : str
            [default: "gray"] The colormap to be used
        bin_text_color : str
            [default: "r"] The color of the text on bins
        y_label : str
            [default: "Predicted"] The label for the y-axis
        x_label : str
            [default: "Truth"] The label for the x-axis
        title : str
            [default: "Confusion matrix"] The title for the plot
        figsize : tuple
            The size of the figure drawn
    Returns:
        None
    """
    histogram = confusion_matrix(true_cats, pred_cats, normalize="true")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    hep.style.use(hep.style.ROOT)
    xbins = ybins = np.arange(len(categories) + 1)
    tick_values = np.arange(len(categories)) + 0.5
    hep.hist2dplot(histogram, xbins, ybins, cmap=cmap, cbar=True)
    plt.xticks(tick_values, categories, fontsize=14, rotation=0)
    plt.yticks(tick_values + 0.2, categories, fontsize=14, rotation=90)
    plt.xlabel(f"{x_label}", fontdict={"size": 14})
    plt.ylabel(f"{y_label}", fontdict={"size": 14})
    ax.tick_params(axis="both", which="both", length=0)
    for i in range(len(ybins) - 1):
        for j in range(len(xbins) - 1):
            bin_value = histogram.T[i, j]
            ax.text(
                xbins[j] + 0.5,
                ybins[i] + 0.5,
                f"{bin_value:.2f}",
                color=bin_text_color,
                ha="center",
                va="center",
                fontweight="bold",
            )
    plt.savefig(output_path, bbox_inches="tight")
    plt.close("all")


def plot_histogram(
    entries: np.array,
    output_path: str,
    left_bin_edge: float = 0.0,
    right_bin_edge: float = 1.0,
    n_bins: int = 24,
    figsize: tuple = (12, 12),
    y_label: str = "",
    x_label: str = "",
    title: str = "",
    integer_bins: bool = False,
) -> None:
    """Plots the confusion matrix for the regression task. Although confusion
    matrix is in principle meant for classification task, the problem can be
    solved by binning the predictions and truth values.

    Args:
        entries : np.array
            The array containing the truth values with shape (n,)
        output_path : str
            The path where output plot will be saved
        left_bin_edge : float
            [default: 0.0] The smallest value
        right_bin_edge : float
            [default: 1.0] The largest value
        n_bins : int
            [default: 24] The number of bins the values will be divided into
            linearly. The number of bin edges will be n_bin_edges = n_bins + 1
        figsize : tuple
            [default: (12, 12)] The size of the figure that will be created
        y_label : str
            [default: "Predicted"] The label for the y-axis
        x_label : str
            [default: "Truth"] The label for the x-axis
        title : str
            [default: "Confusion matrix"] The title for the plot

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize)
    if integer_bins:
        bin_diff = np.min(np.diff(np.unique(entries)))
        left_of_first_bin = np.min(entries) - float(bin_diff) / 2
        right_of_last_bin = np.max(entries) + float(bin_diff) / 2
        hist, bin_edges = np.histogram(
            entries,
            bins=np.arange(left_of_first_bin, right_of_last_bin + bin_diff, bin_diff)
        )
    else:
        hist, bin_edges = np.histogram(
            entries,
            bins=np.linspace(left_bin_edge, right_bin_edge, num=n_bins + 1)
        )
    hep.histplot(hist, bin_edges, yerr=True, label=title)
    plt.xlabel(x_label, fontdict={"size": 20})
    plt.ylabel(y_label, fontdict={"size": 20})
    # plt.grid(True, which="both")
    plt.yscale("log")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close("all")
