# Active Few-Shot Learning

## Maintenance

### CI checks

* The code is auto-formatted using `black .`.
* Static type checks can be run using `pyright`.
* Tests can be run using `pytest tests`.

### Documentation

To start a local server hosting the documentation run ```pdoc ./afsl --math```.

### Publishing

1. update version number in `setup.py` and `afsl/__init__.py`
2. test package metadata: `python setup.py check`
3. generate distribution archives: `python setup.py sdist`
4. *(optional)* upload to test PyPI: `twine upload --repository-url https://test.pypi.org/legacy/ dist/active-few-shot-learning-VERSION.tar.gz`
5. *(optional)* test installation from test PyPI: `pip install --index-url https://test.pypi.org/simple/ active-few-shot-learning --user`
6. upload to PyPI: `twine upload dist/active-few-shot-learning-VERSION.tar.gz`
