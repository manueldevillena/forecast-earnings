import numpy as np
import pandas as pd

from numpy.typing import NDArray

from sklearn.model_selection import KFold
from quantile_forest import RandomForestQuantileRegressor


class QuantileForest:
    """Estimator based on Random Forest Quantile Regressor
    Attributes
    ----------
    X_y : dict[str, dict[str, NDArray] | None]
    """
    def __init__(
            self,
            X_y: dict[str, dict[str, NDArray] | None],
            X_y_validate: dict[str, NDArray],
            kf_params: dict[str, int],
            rfqr_params: dict[str, int]
    ) -> None:
        """Constructor
        """
        self.X_y = X_y
        self.X_y_validate = X_y_validate
        if not kf_params["random_state"]:
            kf_params["random_state"] = None
        if not rfqr_params["random_state"]:
            rfqr_params["random_state"] = None

        self.kf = KFold(
            n_splits=kf_params["n_splits"],
            random_state=kf_params["random_state"]
        )
        self.rfqr = RandomForestQuantileRegressor(
            random_state=rfqr_params["random_state"],
            min_samples_split=rfqr_params["min_samples_split"],
            n_estimators=rfqr_params["n_estimators"],
            n_jobs=rfqr_params["n_jobs"]
        )
        self.predictions: list = []

    def fit(self) -> None:
        """Fit model
        """
        X = self.X_y["train"]["X"]
        y = self.X_y["train"]["y"]

        for train_index, test_index in self.kf.split(X):
            X_train, X_test, y_train, y_test = (
                X[train_index], X[test_index], y[train_index], y[test_index])

            self.rfqr.set_params(max_features=X_train.shape[1] // 3)
            self.rfqr.fit(X_train, y_train)

    def predict(
            self,
            X_y_predict: NDArray,
            scaler,
            output_dir: str,
            target_predictions: str,
            quantiles: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ) -> dict[str, NDArray]:
        X = X_y_predict["validate"]["X"]
        y = X_y_predict["validate"]["y"]

        predictions_scaled = self.rfqr.predict(X, quantiles=quantiles)

        predictions = scaler["targets"].inverse_transform(predictions_scaled)
        predictions = pd.DataFrame(predictions, columns=[f"p{p*100:.0f}" for p in quantiles])
        predictions["target"] = y
        predictions.to_csv(f"{output_dir}/{target_predictions}_predictions.csv")

        return {
            "y_pred": predictions,
            "y": y
        }

    def roll_predict(
            self,
            X_raw: pd.DataFrame,
            shift: int,
            output_dir: str,
            target: str,
            scaler,
            date_range,
            quantiles: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ) -> None:
        """
        Predicts in a rolling window there the last prediction becomes part of the feature space
        for the subsequent predictions
        """
        X_raw = X_raw[-shift:]
        X = X_raw[target].values
        X_scaled = scaler["features"].transform(X.reshape(-1, len(X)))

        t = pd.to_datetime(X_raw["datetime"][-1:].values[0]) + pd.Timedelta(hours=1)
        time_features = self.__get_time_features(t)

        X_scaled = np.concatenate((X_scaled.ravel(), time_features))
        X_scaled = X_scaled.reshape(1, len(X_scaled))

        prediction_scaled = self.rfqr.predict(X_scaled, quantiles=quantiles)
        predictions = scaler["targets"].inverse_transform(prediction_scaled)
        predictions = pd.DataFrame(
            predictions, columns=[f"p{p*100:.0f}" for p in quantiles],
            index=[t]
        )

        quantiles = predictions.columns
        X_quantiles = {q: X for q in quantiles}
        for t in date_range:
            X_quantiles = {
                q: scaler["features"].transform(
                    np.concatenate(
                        (X_quantiles[q].ravel()[1:], predictions[q].values[-1].ravel())
                    ).reshape(1, len(X_quantiles[q].ravel()))
                )
                for q in quantiles
            }
            time_features = self.__get_time_features(t)
            predictions_aux = pd.Series(index=quantiles, dtype=np.float64)
            for key, vals in X_quantiles.items():
                vals = np.concatenate((vals.ravel(), time_features))
                vals = vals.reshape(1, len(vals))
                predictions_aux[key] = scaler["targets"].inverse_transform(
                    self.rfqr.predict(
                        vals, quantiles=[(int(key[1:]) / 100)]
                    ).reshape(-1, 1)).ravel()

            predictions.loc[t] = predictions_aux

        predictions.to_csv(f"{output_dir}/{target}_predictions.csv")

    @staticmethod
    def __get_time_features(t) -> np.ndarray:
        day = t.day
        weekend = 1 if t.day_of_week > 4 else 0
        week = t.week

        return np.array([day, weekend, week])

