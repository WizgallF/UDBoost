from typing import List, Tuple, Callable, Any, Dict
import numpy as np

# --- Within Lib --- #
from utils.my_logging import Logging
from utils.Types import UncertaintyDict

class UDBench:
    """Unified Benchmarking framework for uncertainty quantification models.

    This class provides a standardized interface to evaluate predictive models
    and their uncertainty estimates. It supports both deterministic and
    probabilistic predictions, enabling consistent comparison across different
    model types.

    Attributes:
        predict (callable): A function or method that generates predictions
            given input data.
        predict_uncertainty (callable): A function or method that generates
            uncertainty estimates for the given input data.
    """

    def __init__(
        self,
        predict: Callable[[np.array], np.array],
        predict_uncertainty: Callable[[np.array], UncertaintyDict]
    ):
        """Initialize the UDBench object.

        Args:
            predict: Callable function that generates predictions given input data.
                Must accept input data (e.g., numpy arrays, pandas DataFrames) and
                return predictions in a compatible format.
            predict_uncertainty: Callable function that generates uncertainty
                estimates for input data. Must return uncertainties aligned with the
                predictions from ``predict``.
        """
        # --- Functions --- #
        self.predict = predict
        self.predict_uncertainty = predict_uncertainty

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
                "accuracy": 0.87,
                "nll": 0.45,
                "brier_score": 0.12,
                "barplot": <plot object>
            }
            ```

        Raises:
            ValueError: If the predictions or uncertainty estimates are invalid or
                incompatible.
        """
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