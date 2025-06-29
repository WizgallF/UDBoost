import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BenchmarkUncertainty:
    """
    A class to benchmark uncertainty quantification methods.
    """

    def within_sample_aleatoric_uncertainty(self, X: np.ndarray, y: np.ndarray, y_pred: np.array, true_aleatoric_uncertainty: np.array, aleatoric_uncertainty: np.array):
        """
        Plots the aleatoric uncertainty of a model within the sample space.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Subplot 1: True aleatoric uncertainty
        sc1 = axes[0].scatter(X, y_pred, c=aleatoric_uncertainty, cmap='viridis', s=10)
        plt.colorbar(sc1, ax=axes[0], label='Aleatoric Uncertainty')
        axes[0].set_title('Aleatoric Uncertainty of Model Predictions')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('y')
        axes[0].grid(True)

        # Subplot 2: True Noise level in the Data
        sc2 = axes[1].scatter(X, y, c=true_aleatoric_uncertainty, cmap='viridis', s=10)
        plt.colorbar(sc2, ax=axes[1], label='Noise Level')
        axes[1].set_title('True Noise Level in the Data')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('y')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
        
    def within_sample_epistemic_uncertainty(self, X: np.ndarray, y: np.ndarray, y_pred: np.array, true_epistemic_uncertainty: np.array, epistemic_uncertainty: np.array):
        """        Plots the epistemic uncertainty of a model within the sample space.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Subplot 1: True aleatoric uncertainty
        sc1 = axes[0].scatter(X, y_pred, c=epistemic_uncertainty, cmap='viridis', s=10)
        plt.colorbar(sc1, ax=axes[0], label='Aleatoric Uncertainty')
        axes[0].set_title('Epistemic Uncertainty of Model Predictions')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('y')
        axes[0].grid(True)

        # Subplot 2: True Noise level in the Data
        sc2 = axes[1].scatter(X, y, c=true_epistemic_uncertainty, cmap='viridis', s=10)
        plt.colorbar(sc2, ax=axes[1], label='Noise Level')
        axes[1].set_title('True amount of evidence in the Data')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('y')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
        