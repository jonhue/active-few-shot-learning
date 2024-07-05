import numpy as np
import matplotlib.pyplot as plt


from utils import *


#
#   Plot the results
#


def plot_condition_number():
    plt.figure(figsize=(18, 6))

    #   ITL-noiseless with fine-tuned jitter

    func = "ITL-noiseless"

    df = group_data("CIFAR-general_performance")
    func_group = df.loc[df["alg"] == func]

    n = np.arange(func_group["round"].shape[0])
    # Plot variance
    plt.plot(
        n,
        func_group["accuracy_mean"],
        label=ALG_TITLE[func] + " fine tuned jitter",
        color=FUNC_COLORS[func],
        linestyle="solid",
    )

    # Plot standard error
    plt.fill_between(
        n,
        func_group["accuracy_mean"] - func_group["accuracy_stderr"],
        func_group["accuracy_mean"] + func_group["accuracy_stderr"],
        color=FUNC_COLORS[func],
        alpha=ALPHA,
    )

    #   ITL-noiseless with condition number based jitter

    func = "ITL-noiseless"

    df = group_data("CIFAR-general_performance")
    func_group = df.loc[df["alg"] == func]

    n = np.arange(func_group["round"].shape[0])
    # Plot variance
    plt.plot(
        n,
        func_group["accuracy_mean"],
        label=ALG_TITLE[func] + " condition number jitter",
        color="red",
        linestyle="dashed",
    )

    # Plot standard error
    plt.fill_between(
        n,
        func_group["accuracy_mean"] - func_group["accuracy_stderr"],
        func_group["accuracy_mean"] + func_group["accuracy_stderr"],
        color="red",
        alpha=ALPHA,
    )

    # Set labels
    plt.xlabel("Round", fontsize=FONTSIZE)
    plt.ylabel("Round Accuracy", fontsize=FONTSIZE)

    plt.title("Fine tuned vs condition number jitter", fontsize=FONTSIZE)

    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=FONTSIZE
    )

    plt.savefig(IMG_PATH + "conditionNumberCIFAR.pdf", bbox_inches="tight", dpi=600)
