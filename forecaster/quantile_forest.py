import matplotlib.pyplot as plt

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
            X_y_validate: NDArray,
            scaler,
            quantiles: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ) -> dict[str, NDArray]:
        X = X_y_validate["validate"]["X"]
        y = X_y_validate["validate"]["y"]

        predictions_scaled = self.rfqr.predict(X, quantiles=quantiles)

        predictions = scaler["targets"].inverse_transform(predictions_scaled)

        return {
            "y_pred": predictions,
            "y": y
        }

    @staticmethod
    def plot_results(results: dict[str, NDArray]) -> None:
        """Plots results
        """
        y_pred = results["y_pred"]
        y_true = results["y"]

        plt.plot(y_true, "ro")
        for i in range(y_pred.shape[1]):
            plt.plot(y_pred[:, i], "b", alpha=0.3)
        # plt.fill_between(
            # np.arange(len(y_pred)), lower, upper, alpha=0.2, color="r",
            # label="Pred. interval")
        plt.xlabel("Ordered samples.")
        plt.ylabel("Values and prediction intervals.")
        plt.xlim([0, 500])
        plt.show()

    def save_predictions(self) -> None:
        pass
