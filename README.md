This python package forecasts the earnings of a wind operator selling the energy produced in the spot market. It is written in Python 3.10.11.

## Installation

This package uses a `poetry` environment encoded in a `pyproject.toml`- to install it, make sure to have Python 3.10.11 activated (we recommend using `pyenv` for this) and then type:
```bash
poetry install
```

Alternatively, you can use the built in virtual environments module `venv`. This depends on the `requirements.txt` which can be found in the root directory. To install it, type:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the package
First activate the environment with `source .venv/bin/activate` (assuming that your poetry environment is installed in the same project, otherwise you must point at the adequate folder).

To run the helper, type:
```bash
python -m forecaster -h
```

To run an example:
```bash
python -m forecaster -i instances/example.toml -o results_example -tts price
```
This example forecasts the prices using one year of data as validation. To check the validation of this prediction, you can go to `notebooks/validation.ipynb`.

Another example can be run with:
```bash
python -m forecaster -i instances/example.toml -o results_example -tts energy --roll-predict
```
This example runs the training with the full dataset and then predicts hour by hour for one extra year.

### Notes
* Only `price` or `energy` can be used as `-tts` (refer to the helper for more info)
* Currently the behaviour of both `--roll-predict` (with or without) is quite strict (not tons of flexibility there at this stage)

## Notebooks
Four notebooks are included with the package:
* `clean_input_dataset.ipynb`: this simply removes NaNs from the original dataset
* `explore_data.ipynb`: basic exploration of the input data
* `validation.ipynb`: validates the predictions on unseen data
* `projected_earnings.ipynb`: this is the most important one - it runs the python module and computes the projected earnings (p10, p50, p90)
