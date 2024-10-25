import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# This file contains functions to visualise the dataset


def kde_plots(
    data: pd.DataFrame,
    x_axis: str,
    label_letter: str,
    x_axis_title: str,
    output_file: str,
    figsize: tuple[int] = (8, 5),
    dpi: int = 1200,
    font_size: int = 15,
) -> None:
    """
    Create a KDE plot for each author.

    Parameters:
    data (pd.DataFrame): The dataset to plot.
    x_axis (str): The aggregated column to prepare the distribution for.
    label_letter (str): The letter to use for the quantity.
    x_axis_title (str): The title of the x-axis.
    output_file (str): The file to save the plot to.
    figsize (tuple[int]): The size of the figure.
    dpi (int): The resolution of the figure.
    font_size (int): The font size of the plot.

    Returns:
    None
    """
    # Set the font size
    plt.rcParams.update({"font.size": font_size})
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # Create a color palette
    palette = sns.diverging_palette(145, 300, n=2)

    # Calculate the means
    human_mean = np.mean(data[data["author"] == "human"][x_axis])
    gpt_mean = np.mean(data[data["author"] == "chatgpt"][x_axis])

    human_count = data[data["author"] == "human"]
    gpt_count = data[data["author"] == "chatgpt"]

    # sns.reset_orig()
    sns.kdeplot(
        human_count[x_axis],
        fill=True,
        color=palette[0],
        ax=ax,
        label=f"Human:\n${label_letter}_{{avg}}$ = {human_mean:.1f}",
    )
    sns.kdeplot(
        gpt_count[x_axis],
        fill=True,
        color=palette[1],
        ax=ax,
        label=f"ChatGPT:\n${label_letter}_{{avg}}$ = {gpt_mean:.1f}",
    )

    human_count = human_count[x_axis]
    gpt_count = gpt_count[x_axis]

    # Calculate the KDE to get the maximum density values
    kde1 = gaussian_kde(human_count)
    kde2 = gaussian_kde(gpt_count)

    # Calculate the KDE values at the means
    kde_value_at_human_mean = kde1(human_mean)[0]
    kde_value_at_gpt_mean = kde2(gpt_mean)[0]

    # Add vertical lines at the means, stopping at the max density

    plt.plot(
        [human_mean, human_mean],
        [0, kde_value_at_human_mean],
        color=palette[0],
        linestyle="--",
    )
    plt.plot(
        [gpt_mean, gpt_mean],
        [0, kde_value_at_gpt_mean],
        color=palette[1],
        linestyle="--",
    )
    ax.set_xlabel(x_axis_title)

    plt.legend()
    plt.yticks([])
    ax.set_ylabel("")
    # export the plot
    plt.savefig(output_file, bbox_inches="tight", dpi=dpi)

    return None
