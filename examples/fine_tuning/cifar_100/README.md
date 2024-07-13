Building the indexes:

```
sbatch --wrap="python examples/fine_tuning/cifar_100/build_index.py" --gpus=1 --gres=gpumem:20g --mem-per-cpu=10G
```
