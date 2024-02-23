import argparse
from examples.launch_utils import generate_base_command, generate_run_commands
import full_training_experiment

applicable_configs = {
    "seed": [i for i in range(10)],
    "noise-std": [1],
    "n-init": [100],
    "query-batch-size": [10],
    "subsampled-target-frac": [0.5],
    "max-target-size": [None],
    "algs": [
        # "Oracle Random",
        # "Random",
        "ITL",
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
                            for alg in applicable_configs["algs"]:
                                flags = {
                                    "seed": seed,
                                    "noise-std": noise_std,
                                    "n-init": n_init,
                                    "query-batch-size": query_batch_size,
                                    "subsampled-target-frac": subsampled_target_frac,
                                    "max-target-size": max_target_size,
                                    "alg": alg,
                                }
                                cmd = generate_base_command(
                                    full_training_experiment, flags=flags
                                )
                                command_list.append(cmd)

    generate_run_commands(
        command_list,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        mode="euler",
        num_hours=args.num_hours,
        promt=True,
        mem=16000,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--num-hours", type=int, default=24)
    parser.add_argument("--mem", type=int, default=6000)
    parser.add_argument("--gpumem", type=int, default=0)
    args = parser.parse_args()
    main(args)
