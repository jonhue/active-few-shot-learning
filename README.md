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

### Use Flan v2 dataset
1. Go to root directory of project
2. Run `git clone git@github.com:google-research/FLAN.git`
3. Install dependencies
    1. `pip install tfds_nightly`
    2. `pip install frozendict`
    3. `pip install seqio`
    4. `pip install t5`
4. Add FLAN to PYTHONPATH by adding these lines to `~/llama/bin/activate`
    ```bash
    deactivate () {
        ...

        PYTHONPATH="$_OLD_PYTHONPATH"
        export PYTHONPATH

        ...
    }

    ...

    _OLD_PYTHONPATH="$PYTHONPATH"
    PYTHONPATH="/cluster/home/sbongni/afsl/FLAN:$PYTHONPATH"
    export PYTHONPATH

    ...
    ```
5. Run in FLAN directory via bash
    ```bash
    unzip flan/v2/niv2_few_shot_data/niv2_exemplars.jsonl-00000-of-00010.zip -d flan/v2/niv2_few_shot_data/
    unzip flan/v2/niv2_few_shot_data/niv2_exemplars.jsonl-00001-of-00010.zip -d flan/v2/niv2_few_shot_data/
    unzip flan/v2/niv2_few_shot_data/niv2_exemplars.jsonl-00002-of-00010.zip -d flan/v2/niv2_few_shot_data/
    unzip flan/v2/niv2_few_shot_data/niv2_exemplars.jsonl-00003-of-00010.zip -d flan/v2/niv2_few_shot_data/
    unzip flan/v2/niv2_few_shot_data/niv2_exemplars.jsonl-00004-of-00010.zip -d flan/v2/niv2_few_shot_data/
    unzip flan/v2/niv2_few_shot_data/niv2_exemplars.jsonl-00005-of-00010.zip -d flan/v2/niv2_few_shot_data/
    unzip flan/v2/niv2_few_shot_data/niv2_exemplars.jsonl-00006-of-00010.zip -d flan/v2/niv2_few_shot_data/
    unzip flan/v2/niv2_few_shot_data/niv2_exemplars.jsonl-00007-of-00010.zip -d flan/v2/niv2_few_shot_data/
    unzip flan/v2/niv2_few_shot_data/niv2_exemplars.jsonl-00008-of-00010.zip -d flan/v2/niv2_few_shot_data/
    unzip flan/v2/niv2_few_shot_data/niv2_exemplars.jsonl-00009-of-00010.zip -d flan/v2/niv2_few_shot_data/
    ```

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
