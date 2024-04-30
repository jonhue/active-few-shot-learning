# Active Few-Shot Learning

## Setup

1. Create venv `python -m venv --system-site-packages llama`
2. Activate venv `source llama/bin/activate`
3. Install afsl
    1. Navigate to the root folder of the project
    2. Run `pip install -e .`
4. Install dependencies for LM-Fine-Tuning
    1. `pip install trl`
    2. `pip install peft`
    3. `pip install -U datasets`
    4. `pip install -U bitsandbytes`
## Maintenance

### CI checks

* The code is auto-formatted using `black .`.
* Static type checks can be run using `pyright`.
* Tests can be run using `pytest test`.

### Documentation

To start a local server hosting the documentation run ```pdoc ./afsl --math```.

### Publishing

1. update version number in `pyproject.toml` and `afsl/__init__.py`
2. build: `poetry build`
3. publish: `poetry publish`
4. push version update to GitHub
5. create new release on GitHub
