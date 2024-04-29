# Active Few-Shot Learning

## Setup

1. Create venv `python -m venv --system-site-packages llama`
2. Activate venv `source myenv/bin/activate`
3. Install afsl
    1. Navigate to the root folder of the project
    2. Run `pip install -e .`

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
