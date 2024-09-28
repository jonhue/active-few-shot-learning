r"""
*Active Fine-Tuning* (`activeft`) is a Python package for informative data selection.

## Why Active Data Selection?

As opposed to random data selection, active data selection chooses data adaptively utilizing the current model.
In other words, <p style="text-align: center;">active data selection pays *attention* to the most useful data</p> which allows for faster learning and adaptation.
There are mainly two reasons for why some data may be particularly useful:

1. **Informativeness**: The data contains information that the model had previously been uncertain about.
2. **Relevance**: The data is closely related to a particular task, such as answering a specific prompt.

This is related to memory recall, where the brain recalls informative and relevant memories (think "data") to make sense of the current sensory input.
Focusing recall on useful data enables efficient few-shot learning.

`activeft` provides a simple interface for active data selection, which can be used as a drop-in replacement for random data selection.

## Getting Started

You can install `activeft` from [PyPI](https://pypi.org/project/activeft/) via pip:

```bash
pip install activeft
```

We briefly discuss how to use `activeft` for [fine-tuning](#example-fine-tuning) and [in-context learning / retrieval-augmented generation](#example-in-context-learning).

### Example: Fine-tuning

Given a [PyTorch](https://pytorch.org) model which may (but does not have to be!) pre-trained, we can use `activeft` to efficiently fine-tune the model.
This model may be generative (e.g., a language model) or discriminative (e.g., a classifier), and can use any architecture.

We only need the following things:
- A dataset of inputs `dataset` (such that `dataset[i]` returns a vector of length $d$) from which we want to select batches for fine-tuning. If one has a supervised dataset returning input-label pairs, then `activeft.data.InputDataset(dataset)` can be used to obtain a dataset over the input space.
- A tensor of prediction targets `target` ($m \times d$) which specifies the task we want to fine-tune the model for.
Here, $m$ can be quite small, e.g., equal to the number of classes in a classification task.
If there is no *specific* task for training, then active data selection can still be useful as we will see [later](#undirected-data-selection).
- The `model` can be any PyTorch `nn.Module` with an `embed(x)` method that computes (latent) embeddings for the given inputs `x`, e.g., the representation of `x` from the penultimate layer.
See `activeft.model.ModelWithEmbedding` for more details. Alternatively, the model can have a `kernel(x1,x2)` method that computes a kernel for given inputs `x1` and `x2` (see `activeft.model.ModelWithKernel`).

.. note::

   For active data selection to be effective, it is important that the model's embeddings are somewhat representative of the data.
   In particular, embeddings should capture the relationship between the data and the task.

With this in place, we can initialize the "active" data loader

```python
from activeft import ActiveDataLoader

data_loader = ActiveDataLoader.initialize(dataset, target, batch_size=64)
```

To obtain the next batch from `data`, we can then simply call

```python
batch = data[data_loader.next(model)]
```

Note that the active data selection of the next batch is utilizing the current `model` to select the most relevant data with respect to the given `target`.

Combining the data selection with a model update step, we can implement a simple training loop as follows:

```python
while not converged:
    batch = dataset[data_loader.next(model)]
    model.step(batch)
```

Notice the feedback loop(!): the batch selection improves as the model learns and the model learns faster as the batch selection improves.

This is it!
Training with active data selection is as simple as that.

#### "Undirected" Data Selection

If there is no specific task for training then all data is equally relevant, yet, we can still use active data selection to select the most informative data.
To do this, simply initialize

```python
data_loader = ActiveDataLoader.initialize(dataset, target=None, batch_size=64)
```

### Example: In-context Learning

We can also use the intelligent retrieval of informative and relevant data outside a training loop — for example, for in-context learning and retrieval-augmented generation.

The setup is analogous to the previous section: we have a pre-trained `model`, a dataset `data` to query from, and `target`s (e.g., a prompt) for which we want to retrieve relevant data.
We can use `activeft` to query the most useful data and then add it to the model's context:

```python
from activeft import ActiveDataLoader

data_loader = ActiveDataLoader.initialize(dataset, target, batch_size=5)
context = dataset[data_loader.next(model)]
model.add_to_context(context)
```

Again: very simple!

## Citation

If you use the code in a publication, please cite our papers:

```bibtex
# Active fine-tuning:
@inproceedings{huebotter2024active,
    title={Active Few-Show Fine-Tuning},
    author={Jonas Hübotter and Bhavya Sukhija and Lenart Treven and Yarden As and Andreas Krause},
    booktitle={ICLR Workshop on Bridging the Gap Between Practice and Theory in Deep Learning},
    year={2024},
    pdf={https://arxiv.org/pdf/2402.15898.pdf},
    url={https://github.com/jonhue/activeft}
}

# Theoretical analysis of "directed" active learning:
@inproceedings{huebotter2024information,
    title={Information-based Transductive Active Learning},
    author={Jonas Hübotter and Bhavya Sukhija and Lenart Treven and Yarden As and Andreas Krause},
    booktitle={ICML},
    year={2024},
    pdf={https://arxiv.org/pdf/2402.15441.pdf},
    url={https://github.com/jonhue/activeft}
}
```

---
"""

from activeft.active_data_loader import ActiveDataLoader
from activeft import acquisition_functions, data, embeddings, model, sift

__all__ = [
    "ActiveDataLoader",
    "acquisition_functions",
    "data",
    "embeddings",
    "model",
    "sift",
]
__version__ = "0.1.0"
__author__ = "Jonas Hübotter"
__credits__ = "ETH Zurich, Switzerland"
