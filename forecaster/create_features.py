import copy
import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from numpy.typing import NDArray

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .utils import infer_scaler


@dataclass
class CreateFeatures:
    """Creates features
    """
    shift: int
    which_scalers: dict[str, str]
    original_data: pd.DataFrame
    column_target: str
    features: dict[str, NDArray] = field(default_factory=dict)
    targets: dict[str, NDArray] = field(default_factory=dict)
    scalers: dict[str, MinMaxScaler | StandardScaler] = field(default_factory=dict)
    split_train_test: bool = False
    validation: bool = False

    def __post_init__(self) -> None:
        if not self.validation:
            for key in self.which_scalers.keys():
                self.scalers[key] = infer_scaler(self.which_scalers[key])

    def __call__(
            self
    ) -> tuple[dict[str, dict[str, NDArray]], dict[str, MinMaxScaler | StandardScaler]]:

        scalers = self.autocorrelation_features()
        self.time_features()

        features = self._merge_features(self.features)
        targets = self.targets["autocorrelation"]

        if self.split_train_test:
            X_y = self._split_train_test(features, targets)
        else:
            if not self.validation:
                X_y = {
                    "train": {
                        "X": features,
                        "y": targets
                    },
                    "test": None
                }
            else:
                X_y = {
                    "validate": {
                        "X": features,
                        "y": targets
                    }
                }

        return X_y, scalers

    def autocorrelation_features(self):
        """Creates the autocorrelation features
        """
        X_raw = self.original_data[self.column_target]
        y_raw = self.original_data[self.column_target]
        X_cols = pd.DataFrame()
        for t in range(self.shift):
            X_cols[t] = X_raw.shift(periods=-t)
        y_cols = y_raw.shift(periods=-t-1)
        X = X_cols.values[:-self.shift]
        y = y_cols.values[:-self.shift].reshape(-1, 1)

        scaled, scalers = self._scale_features(X, y, self.scalers)

        self.features["autocorrelation"] = scaled["features"]
        self.targets["autocorrelation"] = scaled["targets"]

        return scalers

    def time_features(self) -> None:
        """Creates the time dependent features
        """
        Xy_raw = self.original_data.shift(periods=-self.shift)[:-self.shift]
        X_raw = pd.DataFrame(pd.to_datetime(Xy_raw["datetime"]))
        y_raw = Xy_raw.values

        X_raw["day"] = X_raw["datetime"].dt.day
        X_raw["weekend"] = (
                X_raw["datetime"].dt.weekday > 4
        ).replace(to_replace=False, value=0).replace(to_replace=True, value=1)
        X_raw["week"] = X_raw["datetime"].dt.isocalendar().week

        X = X_raw[["day", "weekend", "week"]].values
        y = y_raw.reshape(-1, 1)

        self.features["time"] = X
        self.targets["time"] = y

    @staticmethod
    def _merge_features(features: dict[str, NDArray]) -> NDArray:
        """Merges the different types of features
        Parameters
        ----------
        features : dict[str, NDArray]
            dictionary containing all the different types of features
        Returns
        -------
        """
        all_features = []
        for key, vals in features.items():
            all_features.append(vals)

        return np.concatenate(all_features, axis=1)

    def _scale_features(
            self,
            features: NDArray,
            targets: NDArray,
            scalers: dict[str, StandardScaler | MinMaxScaler]
    ) -> dict[str, NDArray]:
        """Scales the features (e.g., minmax)

        Parameters
        ----------
        features : NDArray
            Array with features to be scaled
        targets : NDArray
            Array with targets to be scaled
        scalers : dict[str, StandardScaler | MinMaxScaler]
            Scaler objects to use with the features and the targets

        Returns
        -------
        scaled features and targets
        """
        if not self.validation:
            X_scaled = scalers["features"].fit_transform(features)
            y_scaled = scalers["targets"].fit_transform(targets)
        else:
            X_scaled = scalers["features"].transform(features)
            y_scaled = targets

        return {
            "features": X_scaled,
            "targets": y_scaled
        }, scalers

    @staticmethod
    def _split_train_test(X: NDArray, y: NDArray, test_size: float = 0.33
                          ) -> dict[str, dict[str, NDArray] | None]:
        """Splits the dataset into train and test

        Parameters
        ----------
        X : NDArray
            feature set
        y : NDArray
            targets set
        test_size : float
            proportion of test set after splitting

        Returns
        -------
        train_test : dict[str, dict[str, NDArray]]
            dictionary with train and test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        return {
            "train": {
                "X": X_train,
                "y": y_train
            },
            "test": {
                "X": X_test,
                "y": y_test
            }
        }
