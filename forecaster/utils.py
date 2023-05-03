import toml

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Any


def read_input_file(inputs_file_path: str) -> dict[str, Any]:
    """Reads TOML file with inputs
    Parameters
    ----------
    inputs_file_path : str
        Path to the inputs file in TOML format
    Returns
    -------
    data : dict[str, Any]
        dictionary with the input data
    """
    def key_change_recursive(data: dict):
        for key, vals in data.items():
            if isinstance(vals, dict):
                key_change_recursive(vals)
            else:
                if not vals:
                    vals = None
                else:
                    ...

    with open(inputs_file_path) as inputs_file:
        data = toml.load(inputs_file)

    key_change_recursive(data)

    return data


def infer_scaler(scaler: str) -> StandardScaler | MinMaxScaler:
    """Maps the scaler sting given to the appropriate scaler
    Parameters
    ----------
    scaler : str
        scaler to use
    Returns
    -------
    Scaler object
    """
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler()
    }
    return scalers[scaler]
