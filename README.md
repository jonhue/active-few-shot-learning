# Active Fine-Tuning

A library for automatic data selection in active fine-tuning of large neural networks.

**[Website](https://jonhue.github.io/activeft)** | **[Documentation](https://jonhue.github.io/activeft/docs)**

Please cite our work if you use this library in your research ([bibtex below](#citation)):

- [Efficiently Learning at Test-Time: Active Fine-Tuning of LLMs]()
- [Transductive Active Learning: Theory and Applications](https://arxiv.org/abs/2402.15898) (Section 4)

## Installation

```
pip install activeft
```

## Usage Example

```python
from activeft.sift import Retriever

# Load embeddings
embeddings = np.random.rand(1000, 512)
query_embeddings = np.random.rand(1, 512)

index = faiss.IndexFlatIP(d)
index.add(embeddings)
retriever = Retriever(index)
indices = retriever.search(query_embeddings, N=10)
```

## Development

### CI checks

* The code is auto-formatted using `black .`.
* Static type checks can be run using `pyright`.
* Tests can be run using `pytest test`.

### Documentation

To start a local server hosting the documentation run ```pdoc ./activeft --math```.

### Publishing

1. update version number in `pyproject.toml` and `activeft/__init__.py`
2. build: `poetry build`
3. publish: `poetry publish`
4. push version update to GitHub
5. create new release on GitHub

## Citation

```bibtex
@article{hubotter2024efficiently,
	title        = {Efficiently Learning at Test-Time: Active Fine-Tuning of LLMs},
	author       = {H{\"u}botter, Jonas and Bongni, Sascha and Hakimi, Ido and Krause, Andreas},
	year         = 2024,
	journal      = {arXiv Preprint}
}

@inproceedings{hubotter2024transductive,
	title        = {Transductive Active Learning: Theory and Applications},
	author       = {H{\"u}botter, Jonas and Sukhija, Bhavya and Treven, Lenart and As, Yarden and Krause, Andreas},
	year         = 2024,
	booktitle    = {Advances in Neural Information Processing Systems}
}
```
