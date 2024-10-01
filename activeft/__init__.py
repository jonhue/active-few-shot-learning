r"""
*Active Fine-Tuning* (`activeft`) is a Python package for intelligent active data selection.

## Why Active Data Selection?

As opposed to random data selection, active data selection chooses data adaptively utilizing the current model.
In other words, <p style="text-align: center;">active data selection pays *attention* to the most useful data</p> which allows for faster learning and adaptation.
There are mainly two reasons for why some data may be particularly useful:

1. **Relevance**: The data is closely related to a particular task, such as answering a specific prompt.
2. **Diversity**: The data contains non-redundant information that is not yet captured by the model.

A dataset that is both relevant and diverse is *informative* for the model.
This is related to memory recall, where the brain recalls informative and relevant memories (think "data") to make sense of the current sensory input.
Focusing recall on useful data enables efficient learning from few examples.

`activeft` provides a simple interface for active data selection, which can be used as a drop-in replacement for random data selection or nearest neighbor retrieval.

## Getting Started

You can install `activeft` from [PyPI](https://pypi.org/project/activeft/) via pip:

```bash
pip install activeft
```

We briefly discuss how to use `activeft` for standard [fine-tuning](#example-fine-tuning) and [test-time fine-tuning](#example-test-time-fine-tuning).

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

### Example: Test-Time Fine-Tuning

The above example described active data selection in the context of training a model with multiple batches. This usually happens at "train-time" or during "post-training".

The following example demonstrates how to use `activeft` at "test-time" to obtain a model that is as good as possible on a specific test instance.
For example, with a language model, this would fine-tune the model for a few gradient steps on data selected specifically for a given prompt.
We refer to the following paper for more details: [Efficiently Learning at Test-Time: Active Fine-Tuning of LLMs](TODO).

We can also use the intelligent retrieval of informative and relevant data outside a training loop — for example, for in-context learning and retrieval-augmented generation.

The setup is analogous to the previous section: we have a pre-trained `model`, a dataset `data` to query from, and `target`s (e.g., a prompt) for which we want to retrieve relevant data.
We can use `activeft` to query the most useful data and then add it to the model's context:

```python
from activeft import ActiveDataLoader

data_loader = ActiveDataLoader.initialize(dataset, target, batch_size=10)
data = dataset[data_loader.next(model)]
model.step(data)
```

Again: very simple!

### Scaling to Large Datasets

By default `activeft` maintains a matrix of size of the dataset in memory. This is not feasible for very large datasets.
Some acquisition functions (such as `activeft.acquisition_functions.LazyVTL`) allow for efficient computation of the acquisition function without storing the entire dataset in memory.
An alternative approach is to pre-select a subset of the data using nearest neighbor retrieval (using [Faiss](https://github.com/facebookresearch/faiss)), before initializing the `ActiveDataLoader`.
The following is an example of this approach in the context of [test-time fine-tuning](#example-test-time-fine-tuning):

```python
import torch
import faiss
from activeft.sift import Retriever

# Before Test-Time
embeddings = torch.randn(1000, 768)
index = faiss.IndexFlatIP(embeddings.size(1))
index.add(embeddings)
retriever = Retriever(index)

# At Test-Time, given query
query_embeddings = torch.randn(1, 768)
indices = retriever.search(query_embeddings, N=10, K=1_000)
data = embeddings[indices]
model.step(data)  # Use data to fine-tune base model, then forward pass query
```

`activeft.sift.Retriever` first pre-selects `K` nearest neighbors and then uses `activeft` to select the `N` most informative data for the given query from this subset.

## Citation

If you use the code in a publication, please cite our papers:

```bibtex
# Large-Scale Learning at Test-Time with SIFT
@article{hubotter2024efficiently,
	title        = {Efficiently Learning at Test-Time: Active Fine-Tuning of LLMs},
	author       = {H{\"u}botter, Jonas and Bongni, Sascha and Hakimi, Ido and Krause, Andreas},
	year         = 2024,
	journal      = {arXiv Preprint}
}

# Theory and Fundamental Algorithms for Transductive Active Learning
@inproceedings{hubotter2024transductive,
	title        = {Transductive Active Learning: Theory and Applications},
	author       = {H{\"u}botter, Jonas and Sukhija, Bhavya and Treven, Lenart and As, Yarden and Krause, Andreas},
	year         = 2024,
	booktitle    = {Advances in Neural Information Processing Systems}
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
