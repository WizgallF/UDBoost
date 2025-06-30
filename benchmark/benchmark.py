import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BenchmarkUncertainty:
    """
    A class to benchmark uncertainty quantification methods.

    Inputs:
        X: Input features
        y: Target variable
        true_total_uncertainty: True total uncertainty in the data
        true_aleatoric_uncertainty: True aleatoric uncertainty in the data
        true_epistemic_uncertainty: True epistemic uncertainty in the data
        Pro Modell:
            - predictions
            - epistemic_uncertainty
            - aleatoric_uncertainty
            - total_uncertainty

    Methods:
        performance: MSE, R2 Daten und Plots
        predictive_uncertainty: One plot per model showing the predictive uncertainty on top of the predictions.
    """

    def benchmark_uncertainty(self, dataset, uncertainties):
        """
        Generates a plot with four panels:
        - The first panel shows the true function and noisy observations and the data density
        - The second panels shows the estimations of total uncertainty
        - the third panel shows the estimations of aleatoric uncertainty
        - the fourth panel shows the estimations of epistemic uncertainty
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Benchmarking Uncertainty Quantification Methods', fontsize=16)

        # --- First panel: True function and noisy observations --- #
        dataset.plot(ax=axs[0, 0], show=False)

        # --- Second panel: Total uncertainty with predictive variance --- #
        axs[0, 1].scatter(
            dataset.dataspace,
            uncertainties['mean'],
            label='Predictive Mean',
            color='#264653',
            s=10
        )

        # Shaded region: predictive uncertainty
        predictive_std = uncertainties['predictive']  # assumed to be standard deviation (or sqrt of variance)
        axs[0, 1].fill_between(
            dataset.dataspace,
            uncertainties['mean'] - predictive_std,
            uncertainties['mean'] + predictive_std,
            color='#264653',
            alpha=0.3,
            label='Predictive Uncertainty'
        )

        axs[0, 1].set_title('Estimated Total Uncertainty')
        axs[0, 1].set_xlabel('Input Feature')
        axs[0, 1].set_ylabel('Output')
        axs[0, 1].legend()
        axs[0, 1].grid()

        # --- Third panel: Aleatoric uncertainty --- #
        axs[1, 0].scatter(
        dataset.dataspace,
        uncertainties['mean'],
        label='Predictive Mean',
        color='#540b0e',
        s=10
        )

        # Shaded region: aleatoric uncertainty
        aleatoric_std = uncertainties['aleatoric']  # assumed to be std
        axs[1, 0].fill_between(
            dataset.dataspace,
            uncertainties['mean'] - aleatoric_std,
            uncertainties['mean'] + aleatoric_std,
            color='#540b0e',
            alpha=0.3,
            label='Aleatoric Uncertainty'
        )

        axs[1, 0].set_title('Estimated Aleatoric Uncertainty')
        axs[1, 0].set_xlabel('Input Feature')
        axs[1, 0].set_ylabel('Output')
        axs[1, 0].legend()
        axs[1, 0].grid()

        # --- Fourth panel: Epistemic uncertainty --- #
        axs[1, 1].scatter(
            dataset.dataspace,
            uncertainties['mean'],
            label='Predictive Mean',
            color='#344e41',
            s=10
        )

        # Shaded region: epistemic uncertainty
        epistemic_std = uncertainties['epistemic']  # assumed to be std
        axs[1, 1].fill_between(
            dataset.dataspace,
            uncertainties['mean'] - epistemic_std,
            uncertainties['mean'] + epistemic_std,
            color='#344e41',
            alpha=0.3,
            label='Epistemic Uncertainty'
        )
        axs[1, 1].set_title('Estimated Epistemic Uncertainty')
        axs[1, 1].set_xlabel('Input Feature')
        axs[1, 1].set_ylabel('Output')
        axs[1, 1].legend()
        axs[1, 1].grid()

        plt.tight_layout()
        plt.show()