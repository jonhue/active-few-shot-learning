import pandas as pd
import wandb as wb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


IMG_PATH = "images/"
DATA_PATH = "data/"


FONTSIZE = 16

FUNCTIONS = ["ITL-noiseless", "ITL", "UncertaintySampling", "Random"]

ALG_TITLE = {
    "ITL-noiseless": "ITL noiseless",
    "ITL": "ITL",
    "UncertaintySampling": "Uncertainty sampling",
    "Random": "Uniform sampling",
}

FUNC_COLORS = {
    "UncertaintySampling": "gray",
    "Random": "red",
    "ITL-noiseless": "green",
    "ITL": "black",
}

ALPHA = 0.15

#
#   Get the data
#


def wandb_get_data(dataset: str, tag: str):
    print(f'Trying to load data with tag "{tag}"')
    api = wb.Api()
    runs = api.runs(
        path=f"bongni/Fine-tuning {dataset}", filters={"tags": {"$in": [tag]}}
    )

    # Iterate over runs and fetch data
    data = []
    print("Getting runs")
    for run in tqdm(runs):
        df_history = run.history(keys=["round_accuracy", "round", "_step"])
        if df_history.shape[0] == 0:
            continue

        for row in df_history.iterrows():
            data.append(
                {
                    "run_id": run.id,
                    "step": row[1]["_step"],
                    "alg": run.config["alg"],
                    "round_accuracy": row[1]["round_accuracy"],
                    "round": row[1]["round"],
                }
            )

    df = pd.DataFrame(data)
    df.to_csv(f"./data/{dataset}-{tag}.csv", index=False)


def group_data(file_name):
    df = pd.read_csv(f"./data/{file_name}.csv")
    grouped_df = df.groupby(["alg", "round"], as_index=False)

    accuracy_mean = grouped_df["round_accuracy"].mean().rename(columns={"round_accuracy": "accuracy_mean"})  # type: ignore
    accuracy_stderr = grouped_df["round_accuracy"].sem().rename(columns={"round_accuracy": "accuracy_stderr"})  # type: ignore

    merged = pd.merge(accuracy_mean, accuracy_stderr, on=["alg", "round"])

    return merged
