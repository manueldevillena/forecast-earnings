import argparse
import pandas as pd

from numpy.typing import NDArray
from pathlib import Path

from .utils import read_input_file
from .create_features import CreateFeatures
from .quantile_forest import QuantileForest


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parsing the inputs to run the module.")
    parser.add_argument("-i", "--inputs", dest="inputs", help="TOML file with input data.")
    parser.add_argument(
        "its" "--inputs_timeseries", dest="inputs_timeseries", help="Input features."
    )
    parser.add_argument(
        "tts" "--target_timeseries", dest="target_timeseries", help="Timeseries to forecast."
    )

    return parser.parse_args()


def read_data(data_path: Path) -> dict[str, pd.DataFrame]:
    data = {
        dataset: {
            param: pd.read_csv(data_path / dataset / f"{param}.csv", index_col=0)
            for param in ["spot_price", "energy"]
        }
        for dataset in ["train_test", "validate"]
    }

    return data


def create_features(
        inputs: dict[str, str | int],
        data: dict[str, pd.DataFrame]
) -> dict[str, dict[str, NDArray] | None]:
    X_y, scaler_train = CreateFeatures(
        shift=inputs["shift"],
        which_scalers=inputs["which_scalers"],
        original_data=data["train_test"]
    )()
    X_y_validate, scaler_validate = CreateFeatures(
        shift=inputs["shift"],
        which_scalers=inputs["which_scalers"],
        original_data=data["validate"],
        scalers=scaler_train,
        validation=True
    )()

    return X_y, X_y_validate, scaler_train


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

    # Read data
    data = read_data(Path(inputs["path_to_dataset"]))

    # Create features
    X_y, X_y_validate, scaler_validate = create_features(inputs, data)
    estimator_ = estimator(X_y, X_y_validate, inputs["qf_params"])
    predictions = estimator_.predict(X_y_validate, scaler_validate)
    estimator_.plot_results(predictions)


if __name__ == "__main__":
    main()
