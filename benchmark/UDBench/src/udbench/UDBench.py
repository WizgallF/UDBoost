from typing import List, Tuple, Callable, Any, Dict
import numpy as np

# --- Within Lib --- #
from utils.my_logging import Logging
from utils.Types import UncertaintyDict
from DatasetLoader import TabularDataLoader

class UDBench:
    """Unified Benchmarking framework for uncertainty quantification models.

    This class provides a standardized interface to evaluate predictive models
    and their uncertainty estimates. It supports both deterministic and
    probabilistic predictions, enabling consistent comparison across different
    model types.
    """

    def __init__(
        self,
        y_test_pred: np.array = None,
        y_test_predictive_uncertainty: np.array = None,
        y_test_aleatoric_uncertainty: np.array = None,
        y_test_epistemic_uncertainty: np.array = None,
        dataset_name: str = None,
    ):
        """Initialize the UDBench object.

        Args:
            y_test_pred (np.array): Predictions for the test set.
            y_test_predictive_uncertainty (np.array, optional): Predictive uncertainty estimates for the test set.
            y_test_aleatoric_uncertainty (np.array, optional): Aleatoric uncertainty estimates for the test set.
            y_test_epistemic_uncertainty (np.array, optional): Epistemic uncertainty estimates for the test set.
            dataset_name (str): Name of the dataset being evaluated.
        """
        # --- PREDICTIONS --- #
        self.y_test_pred = y_test_pred

        # --- UNCERTAINTY ESTIMATES --- #
        self.y_test_predictive_uncertainty = y_test_predictive_uncertainty
        self.y_test_aleatoric_uncertainty = y_test_aleatoric_uncertainty
        self.y_test_epistemic_uncertainty = y_test_epistemic_uncertainty

        # --- DATASET INFO --- #
        self.dataset_name = dataset_name
      
        # --- LOGGING --- #
        self.logger = Logging('log').get_logger()


    def predictive_performance(self, plot: bool = False, verbose: int = 1) -> dict:
        """Evaluate the predictive performance of the model.

        This method calculates standard performance metrics such as accuracy,
        mean squared error, or negative log-likelihood, depending on the type of
        model and predictions provided. Optionally, results can be plotted.

        Args:
            plot: If ``True``, generates plots to visualize performance results.
                Defaults to ``False``.
            verbose: Verbosity level. ``0`` = silent, ``1`` = basic logs,
                ``2`` = detailed logs. Defaults to ``1``.

        Returns:
            dict: A dictionary containing performance metrics. Keys vary depending
            on the model and evaluation strategy. Example:

            ```python
            {
                "mse": 0.87,
                "r2": 0.45,
                "scatterplot": <plot object>
            }
            ```

        Raises:
            ValueError: If the predictions or uncertainty estimates are invalid or
                incompatible.
        """
        # --- Load True Labels --- #
        y_test_true = TabularDataLoader.load_labels(self.dataset_name)


        # --- Compute Metrics --- #
        mse = ((self.y_test_pred - y_test_true) ** 2).mean()
        r2 = 1 - (mse / ((y_test_true - y_test_true.mean()) ** 2).mean())

        # --- Report Results --- #
        results = {"mse": mse, "r2": r2}
        self.logger.performance_report(mse, r2)

        # --- Generate Plots if Required --- #
        if plot:
            # TODO: Implement plotting
            pass
        else:
            pass

        return results

    def calibration(self, plot: bool = False, verbose: int = 1) -> dict:
        """Assess the calibration of the model's uncertainty estimates.

        # --- Generate Plots if Required --- #
        if plot:
            # TODO: Implement plotting
            pass
        else:
            pass

    def calibration(self, plot: bool = False, verbose: int = 1) -> dict:
        """Assess the calibration of the model's uncertainty estimates.

        This method evaluates how well the predicted uncertainties fit empiric uncertainty distributions.
        It computes calibration metrics such as reliability diagrams, expected calibration error (ECE), maximum
        calibration error (MCE), continuous ranked probability score (CRPS), and negative log-likelihood (NLL). 
        Optionally, results can be visualized through plots.

        Args:
            plot: If ``True``, generates plots to visualize calibration results.
                Defaults to ``False``.
            verbose: Verbosity level. ``0`` = silent, ``1`` = basic logs,
                ``2`` = detailed logs. Defaults to ``1``.

        Returns:
            dict: A dictionary containing calibration metrics. Keys may include:

            ```python
            {
                "ece": 0.05,
                "mce": 0.10,
                "CRPS": 0.15,
                "NLL": 0.20,
                "reliability_diagram": <plot object>
            }
            ```

        Raises:
            ValueError: If the predictions or uncertainty estimates are invalid or
                incompatible.
        """
        pass

    def aleatoric_uncertainty(self, plot: bool = False, verbose: int = 1) -> dict:
        """Evaluate the aleatoric uncertainty of the model's predictions.

        This method assesses the model's ability to quantify uncertainty due to
        inherent noise in the data. It computes metrics related to aleatoric
        uncertainty, such as predictive entropy, mutual information, and
        uncertainty intervals. Optionally, results can be visualized through plots.

        Args:
            plot: If ``True``, generates plots to visualize uncertainty results.
                Defaults to ``False``.
            verbose: Verbosity level. ``0`` = silent, ``1`` = basic logs,
                ``2`` = detailed logs. Defaults to ``1``.

        Returns:
            dict: A dictionary containing aleatoric uncertainty metrics. Keys may include:

            ```python
            {
                "MSE": 0.05,
                "R2": 0.10,
                "uncertainty_plot": <plot object>
            }
            ```

        Raises:
            ValueError: If the predictions or uncertainty estimates are invalid or
                incompatible.
        """
        pass

    def featurewise_aleatoric_uncertainty(self, plot: bool = False, verbose: int = 1) -> dict:
        """Evaluate the feature-wise aleatoric uncertainty of the model's predictions.

        This method assesses the model's ability to quantify uncertainty due to
        inherent noise in the data at the feature level. It computes metrics related
        to feature-wise aleatoric uncertainty, such as feature importance scores
        and uncertainty intervals for each feature. Optionally, results can be
        visualized through plots.

        Args:
            plot: If ``True``, generates plots to visualize uncertainty results.
                Defaults to ``False``.
            verbose: Verbosity level. ``0`` = silent, ``1`` = basic logs,
                ``2`` = detailed logs. Defaults to ``1``.

        Returns:
            dict: A dictionary containing feature-wise aleatoric uncertainty metrics.
            Keys may include:

            ```python
            {
                "feature_1": {
                    "MSE": 0.05,
                    "R2": 0.10,
                    "uncertainty_plot": <plot object>
                },
                "feature_2": {
                    "MSE": 0.07,
                    "R2": 0.12,
                    "uncertainty_plot": <plot object>
                }
            }
            ```

        Raises:
            ValueError: If the predictions or uncertainty estimates are invalid or
                incompatible.
        """
        pass

    def epistemic_uncertainty(self, plot: bool = False, verbose: int = 1) -> dict:
        """Evaluate the epistemic uncertainty of the model's predictions.

        This method assesses the model's ability to quantify uncertainty due to
        lack of knowledge or data. It computes metrics related to epistemic
        uncertainty, such as model variance, ensemble uncertainty, and
        uncertainty intervals. Optionally, results can be visualized through plots.

        Args:
            plot: If ``True``, generates plots to visualize uncertainty results.
                Defaults to ``False``.
            verbose: Verbosity level. ``0`` = silent, ``1`` = basic logs,
                ``2`` = detailed logs. Defaults to ``1``.

        Returns:
            dict: A dictionary containing epistemic uncertainty metrics. Keys may include:

            ```python
            {
                "F1": 0.05,
                "AUC": 0.10,
                "AUC_plot": <plot object>
            }
            ```

        Raises:
            ValueError: If the predictions or uncertainty estimates are invalid or
                incompatible.
        """
        pass

    def featurewise_epistemic_uncertainty(self, plot: bool = False, verbose: int = 1) -> dict:
        """Evaluate the feature-wise epistemic uncertainty of the model's predictions.

        This method assesses the model's ability to quantify uncertainty due to
        lack of knowledge or data at the feature level. It computes metrics related
        to feature-wise epistemic uncertainty, such as feature importance scores
        and uncertainty intervals for each feature. Optionally, results can be
        visualized through plots.

        Args:
            plot: If ``True``, generates plots to visualize uncertainty results.
                Defaults to ``False``.
            verbose: Verbosity level. ``0`` = silent, ``1`` = basic logs,
                ``2`` = detailed logs. Defaults to ``1``.

        Returns:
            dict: A dictionary containing feature-wise epistemic uncertainty metrics.
            Keys may include:

            ```python
            {
                "feature_1": {
                    "F1": 0.05,
                    "AUC": 0.10,
                    "AUC_plot": <plot object>
                },
                "feature_2": {
                    "F1": 0.07,
                    "AUC": 0.12,
                    "AUC_plot": <plot object>
                }
            }
            ```

        Raises:
            ValueError: If the predictions or uncertainty estimates are invalid or
                incompatible.
        """
        pass

    def epistmic_data_density():
        pass

    def epistemic_capacity():
        pass
    
    def epistemic_aleatoric_correlation():
        pass