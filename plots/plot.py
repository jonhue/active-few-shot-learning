import sys
import argparse



from plotPerformance import *



if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("\nUsage: Need to pass at least one argument\n")
        print("--plot \t generalPerformance \n\t ... \n\t ...\n")
        print("--load \t tag of wandb project\n")
        exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, default="")
    parser.add_argument("--MNIST", action='store_true', default=False)
    parser.add_argument("--CIFAR", action='store_true', default=False)
    parser.add_argument("--load", type=str, default="")
    args = parser.parse_args()

    if args.load != "":
        if args.MNIST:
            wandb_get_data("MNIST", args.load)
        elif args.CIFAR:
            wandb_get_data("CIFAR", args.load)
        else:
            print("Invalid argument missing --CIFAR or --MNIST")
            exit(-1)
    elif args.plot != "":
        if args.plot.strip() == "generalPerformance":
            plotGeneralPerformance()
        else:
            print(f"Invalid argument for --plot got \"{args.plot}\"")
            exit(-1)
    
    print("Success!")