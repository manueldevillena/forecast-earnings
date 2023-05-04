import argparse
import os

import pandas as pd

from numpy.typing import NDArray
from pathlib import Path

from .utils import read_input_file
from .create_features import CreateFeatures
from .quantile_forest import QuantileForest


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parsing the inputs to run the module.")
    parser.add_argument("-i", "--inputs", dest="inputs", help="TOML file with input data.")
    parser.add_argument("-o", "--outputs", dest="outputs", help="Output folder.")
    parser.add_argument(
        "-tts", "--target_timeseries", dest="target_timeseries", help="Timeseries to forecast."
    )
    parser.add_argument("--roll-predict", dest="roll_predict", action="store_true",
                        help="flag to activate the rolling prediction for the future.")

    return parser.parse_args()


def read_data(data_path: Path) -> dict[str, pd.DataFrame]:
    data = {
        dataset: {
            param: pd.read_csv(data_path / dataset / f"{param}.csv")
            for param in ["price", "energy"]
        }
        for dataset in ["full", "train_test", "validate"]
    }

    return data


def create_features(
        inputs: dict[str, str | int],
        data: dict[str, pd.DataFrame],
        target: str,
        roll_redict: bool
) -> dict[str, dict[str, NDArray] | None]:
    """Creates the feature objects for train, test, and validation

    Parameters
    ----------
    inputs : dict
        hyper-parameters
    data : dict
        input data
    target : str
        target to forecast

    Returns
    -------
    features : dict
        features, targets, and scalers
    """
    if roll_redict:
        data_train = data["full"][target]
    else:
        data_train = data["train_test"][target]
    X_y, scaler = CreateFeatures(
        shift=inputs["shift"],
        which_scalers=inputs["which_scalers"],
        original_data=data_train,
        column_target=target
    )()
    if not roll_redict:
        X_y_validate, scaler = CreateFeatures(
            shift=inputs["shift"],
            which_scalers=inputs["which_scalers"],
            original_data=data["validate"][target],
            column_target=target,
            scalers=scaler,
            validation=True
        )()
    else:
        X_y_validate = None

    return X_y, X_y_validate, scaler


def estimator(X_y: NDArray, X_y_validate: NDArray, qf_params: dict[str, dict[str, int]]):
    estimator_ = QuantileForest(
        X_y=X_y,
        X_y_validate=X_y_validate,
        kf_params=qf_params["kf_params"],
        rfqr_params=qf_params["rfqr_params"]
    )
    estimator_.fit()

    return estimator_


def main() -> None:
    # Parse inputs
    args = parse_arguments()

    # Read inputs
    inputs = read_input_file(args.inputs)
    target = args.target_timeseries

    # Create output folder
    os.makedirs(args.outputs, exist_ok=True)

    # Read data
    data = read_data(Path(inputs["path_to_dataset"]))

    # Create features
    X_y, X_y_validate, scaler = create_features(inputs, data, target, args.roll_predict)
    estimator_ = estimator(X_y, X_y_validate, inputs["qf_params"])
    if not args.roll_predict:
        predictions = estimator_.predict(X_y_validate, scaler, args.outputs, args.target_timeseries)
    else:
        estimator_.roll_predict(
            data["full"][args.target_timeseries],
            inputs["shift"],
            args.outputs,
            args.target_timeseries,
            scaler,
            pd.date_range(start="2023-04-01 00:00:00", end="2024-03-31 23:00:00", freq="H")
        )


if __name__ == "__main__":
    main()
