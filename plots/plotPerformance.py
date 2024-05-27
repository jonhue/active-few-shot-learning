import pandas as pd
import numpy as np
import wandb as wb
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm



from utils import *



#
#   Plot the results
#

def plotPerformance(fig, df):
    for func in FUNCTIONS:
        func_group = df.loc[df['alg'] == func]

        n = np.arange(func_group["round"].shape[0])
        # Plot variance
        fig.plot(
            n, 
            func_group["accuracy_mean"], 
            label=ALG_TITLE[func], 
            color=FUNC_COLORS[func]
        )
        
        # Plot standard error
        fig.fill_between(
            n, 
            func_group["accuracy_mean"] - func_group["accuracy_stderr"], 
            func_group["accuracy_mean"] + func_group["accuracy_stderr"], 
            color=FUNC_COLORS[func], 
            alpha=ALPHA
        )

    # Set labels
    fig.set_xlabel('Round', fontsize=FONTSIZE)
    fig.set_ylabel("Round Accuracy", fontsize=FONTSIZE)

def plotGeneralPerformance():    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    df = group_data("CIFAR-general_performance")
    plotPerformance(axes[0], df)
    axes[0].set_title("CIFAR", fontsize=FONTSIZE)

    df = group_data("MNIST-general_performance")
    plotPerformance(axes[1], df)
    axes[1].set_title("MNIST", fontsize=FONTSIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0.03), 
        ncol=4,
        fontsize=FONTSIZE
    )

    plt.savefig(IMG_PATH + "generalPerformanceMNISTandCIFAR.pdf", bbox_inches="tight", dpi=600)
