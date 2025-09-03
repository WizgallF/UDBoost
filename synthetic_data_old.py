import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Optional


class SyntheticDataGenerator:
    '''
    A class to generate synthetic data for regression tasks with uncertainty disentanglement.
    '''
    def gen_1d_synthetic_benchmark(
        self,
        n_samples: int = 1000,
        noise_levels: list = [0.1, 0.1],
        data_densities: list = [1.0, 1.0],
        random_seed: int = 42,
        func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalized_y: bool = False
    ) -> 'DataSet':
        """
        Generate synthetic regression data with configurable aleatoric (noise) and epistemic (density) uncertainty.

        The ground-truth noise and density values are returned over the entire dataspace.

        Parameters:
            n_samples (int): Total number of points in the [0, 1] domain.
            noise_levels (list): Standard deviation of noise per region.
            data_densities (list): Fraction of data kept per region (0 to 1).
            random_seed (int): Seed for reproducibility.
            func (Callable): True underlying function. Defaults to sin(2πx).
            normalized_y (bool): Whether to z-normalize outputs.

        Returns:
            DataSet: Contains dataspace, selected samples X, noisy and clean outputs,
                    and full-length ground-truth noise/density profiles.
        """
        np.random.seed(random_seed)
        func = func or (lambda x: np.sin(2 * np.pi * x))

        # Full 1D input space and clean function values
        dataspace = np.linspace(0, 1, n_samples)
        y_clean = func(dataspace)

        # Initialize full-length noise and density arrays
        true_noise_std = np.zeros(n_samples)
        true_data_density = np.zeros(n_samples)

        # Define regions
        num_regions = len(noise_levels)
        region_edges = np.linspace(0, 1, num_regions + 1)
        keep_indices = []

        y_noisy = np.zeros(n_samples)  # Initialize noisy output

        for i in range(num_regions):
            # Region indices
            mask = (dataspace >= region_edges[i]) & (dataspace < region_edges[i + 1])
            indices = np.where(mask)[0]

            # Assign region-wide noise and density
            true_noise_std[indices] = noise_levels[i]
            true_data_density[indices] = data_densities[i]

            # Add noise to the clean function values in this region
            noise = np.random.normal(0, noise_levels[i], size=len(indices))
            y_noisy[indices] = y_clean[indices] + noise

            # Subsample according to density (epistemic uncertainty)
            n_keep = int(len(indices) * data_densities[i])
            selected = np.random.choice(indices, size=n_keep, replace=False)
            keep_indices.extend(selected)

        # Only keep noisy outputs for selected indices
        y_noisy = y_noisy[keep_indices]

        # Selected input samples
        X = dataspace[keep_indices]

        # Normalize if requested
        if normalized_y:
            mean = np.mean(y_clean)
            std = np.std(y_clean)
            y_noisy = (y_noisy - mean) / std
            y_clean = (y_clean - mean) / std

        return DataSet(
            dataspace=dataspace,
            X=X,
            y=y_noisy,
            y_clean=y_clean,
            true_function=func,
            true_noise_std=true_noise_std,
            true_data_density=true_data_density
        )



class DataSet:
    """
    A class to represent a dataset for regression tasks with uncertainty quantification.
    
    Attributes:
        dataspace: np.ndarray The space of the input features
        X: np.ndarray The input features
        y: np.ndarray The target variable
        y_clean: np.ndarray The clean target variable without noise
        true_function: Callable The true function used to generate the target variable
        true_noise_std: np.ndarray The true noise standard deviation over the dataspace
        true_data_density: np.ndarray The true data density over the dataspace
    """
    
    def __init__(self, dataspace: np.ndarray, X: np.ndarray, y: np.ndarray, y_clean: np.array, true_function: Callable, true_noise_std: np.ndarray, true_data_density: np.ndarray):
        self.dataspace = dataspace
        self.X = X
        self.y = y
        self.y_clean = y_clean
        self.true_function = true_function
        self.true_noise_std = true_noise_std
        self.true_data_density = true_data_density

    def plot_1d_syn_benchmark(self, ax: Optional[plt.Axes] = None, show: bool = False):
        """
        Plots the dataset with the true function, noisy observations, and true noise std as a shaded area.
        """
        # Create a new axes if none provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot true function
        ax.plot(self.dataspace, self.y_clean, label='True Function', color='black')

        # Plot aleatoric noise as shaded region
        ax.fill_between(
            self.dataspace,
            self.y_clean - self.true_noise_std,
            self.y_clean + self.true_noise_std,
            color='orange',
            alpha=0.3,
            label='True Noise Std'
        )

        # Background shading for epistemic uncertainty (data density)
        norm_density = (self.true_data_density - np.min(self.true_data_density)) / (
            np.ptp(self.true_data_density) + 1e-8
        )
        for i in range(len(self.dataspace) - 1):
            ax.axvspan(
                self.dataspace[i],
                self.dataspace[i + 1],
                color=plt.cm.Blues(norm_density[i]),
                alpha=0.2,
                linewidth=0
            )

        # Plot noisy observations
        ax.scatter(self.X, self.y, label='Noisy Observations', color='blue', s=10)

        ax.set_title('Synthetic Dataset')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid()
        if show:
            plt.show()
