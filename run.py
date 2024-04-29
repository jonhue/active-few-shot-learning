import os
import sys
import argparse

import examples.llama as llama

def main(args):
    base_cmd = sys.executable + " -u " + os.path.abspath(llama.__file__)

    sbatch_cmd = (
        "sbatch -A ls_krausea "
        + f"--time={args.num_hours}:59:00 "
        + f"--mem-per-cpu={args.mem}G "
        + f"-n {args.num_cpus} "
        + f"--gpus={args.num_gpus} "
        + f"--gres=gpumem:{args.gpumem}g "
        + f'--wrap="{base_cmd}"'
    )
    
    os.system(sbatch_cmd)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpus", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-hours", type=int, default=8)
    parser.add_argument("--mem", type=int, default=16)
    parser.add_argument("--gpumem", type=int, default=24)
    args = parser.parse_args()
    main(args)