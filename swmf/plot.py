import matplotlib.pyplot as plt
import numpy as np


def plot_PR(P, R, labels, title="", color_P='b', color_R='r', x_axis="x", y_axis="y", f_decimals=2):
    """
    Plot a grouped bar chart for two arrays P and R, grouped by index.

    Args:
        P (list or array): First array of values.
        R (list or array): Second array of values, same length as P.
        labels (list or array): Labels for the x-axis, same length as P and R.
        color_P (str): Color for bars representing P.
        color_R (str): Color for bars representing R.
    """
    if len(P) != len(R) or len(P) != len(labels):
        raise ValueError("P, R, and labels must have the same length.")

    N = len(P)
    i = np.arange(N)
    w = 0.35  # Width of each bar

    fig, ax = plt.subplots()

    # Plot bars for P and R
    ax.bar(i - w/2, P, w, color=color_P, label='Precision')
    ax.bar(i + w/2, R, w, color=color_R, label='Recall')

    # Format x_labels to show only `decimals` decimal places
    formatted_labels = [f"{label:.{f_decimals}f}" for label in labels]

    # Add labels and title
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(title)
    ax.set_xticks(i)
    ax.set_xticklabels(formatted_labels)
    ax.set_ylim(0, 1)  # Set y-axis range to (0, 1)
    ax.legend()

    plt.tight_layout()
    plt.show()


