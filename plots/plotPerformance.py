import pandas as pd
import numpy as np
import wandb as wb
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm



IMG_PATH = "images/"
DATA_PATH = "data/"

#
#   Get the data
#

def wandb_get_data(dataset: str, tag: str):
    print(f"Trying to load data with tag \"{tag}\"")
    api = wb.Api()
    runs = api.runs(
        path=f"bongni/Fine-tuning {dataset}", 
        filters={"tags": {"$in": [tag]}}
    )

    # Iterate over runs and fetch data
    data = []
    print("Getting runs")
    for run in tqdm(runs):
        df_history = run.history(keys=["round_accuracy", "round", "_step"])
        if df_history.shape[0] == 0:
            continue

        for row in df_history.iterrows():
            data.append({
                "run_id": run.id,
                "step": row[1]['_step'],
                "alg": run.config['alg'],
                "round_accuracy": row[1]['round_accuracy'],
                "round": row[1]['round']
            })

    df = pd.DataFrame(data)
    df.to_csv(f"./data/{dataset}-{tag}.csv", index=False)



def group_data(file_name):
    df = pd.read_csv(f'./data/{file_name}.csv')
    grouped_df = df.groupby(["alg", "round"], as_index=False)

    accuracy_mean = grouped_df["round_accuracy"].mean().rename(columns={"round_accuracy": "accuracy_mean"})
    accuracy_stderr = grouped_df["round_accuracy"].sem().rename(columns={"round_accuracy": "accuracy_stderr"})

    merged = pd.merge(accuracy_mean, accuracy_stderr, on=['alg', 'round'])

    return merged

#
#   Plot the results
#

FONTSIZE = 16

FUNCTIONS = ['ITL-noiseless', 'ITL', 'UncertaintySampling', 'Random']

ALG_TITLE = {
    "ITL-noiseless":            "ITL noiseless",
    "ITL":                      "ITL",
    "UncertaintySampling":      "Uncertainty sampling",
    "Random":                   "Uniform sampling"
}

FUNC_COLORS = {
    'UncertaintySampling' : 'gray', 
    'Random' : 'red', 
    'ITL-noiseless' : 'green', 
    'ITL' : 'black'
}

ALPHA = 0.15

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
