import argparse
from examples.launch_utils import generate_base_command, generate_run_commands
import examples.fine_tuning.llama.experiment as experiment

DEBUG = False

applicable_configs = {
    "seed": [42], #[i for i in range(10)],
    "noise-std": [1],
    "n-init": [100],  # [4, 10, 20, 50, 100, 200, 500]
    "query-batch-size": [2],
    "subsampled-target-frac": [0.1],  # [0.05, 0.1, 0.2, 0.5, 1.0]
    "max-target-size": ["None"],
    "subsample-acquisition": [1],
    "update-target": [0],
    "algs": [
        # "OracleRandom",
        #"Random",
        "ITL",
        #"ITL-noisy",
        #"ITL-noiseless",
        #"ITL-noiseless-old",
        # "ITL-nonsequential",
        # "VTL",
        # "CTL",
        # "CosineSimilarity",
        # "InformationDensity",
        # "UndirectedITL",
        # "UndirectedVTL",
        # "UncertaintySampling",
        # "MinMargin",
        # "MaxEntropy",
        # "LeastConfidence",
        # "MaxDist",
        # "KMeansPP",
    ],
}

def main(args):
    command_list = []
    for seed in applicable_configs["seed"]:
        for noise_std in applicable_configs["noise-std"]:
            for n_init in applicable_configs["n-init"]:
                for query_batch_size in applicable_configs["query-batch-size"]:
                    for subsampled_target_frac in applicable_configs[
                        "subsampled-target-frac"
                    ]:
                        for max_target_size in applicable_configs["max-target-size"]:
                            for subsample_acquisition in applicable_configs[
                                "subsample-acquisition"
                            ]:
                                for update_target in applicable_configs[
                                    "update-target"
                                ]:
                                    for alg in applicable_configs["algs"]:
                                        flags = {
                                            "seed": seed,
                                            "noise-std": noise_std,
                                            "n-init": n_init,
                                            "query-batch-size": query_batch_size,
                                            "subsampled-target-frac": subsampled_target_frac,
                                            "max-target-size": max_target_size,
                                            "subsample-acquisition": subsample_acquisition,
                                            "update-target": update_target,
                                            "alg": alg,
                                            "debug": DEBUG,
                                        }
                                        cmd = generate_base_command(
                                            experiment, flags=flags
                                        )
                                        command_list.append(cmd)

    generate_run_commands(
        command_list,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        mode="euler",
        num_hours=args.num_hours,
        promt=True,
        mem=args.mem,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpus", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-hours", type=int, default=8)
    parser.add_argument("--mem", type=int, default=16000)
    parser.add_argument("--gpumem", type=int, default=24)
    args = parser.parse_args()
    main(args)
