import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SyntheticDataGenerator:
    '''
    A class to generate synthetic data for regression tasks with uncertainty disentanglement.
    '''
    def gen_aleatoric_benchmark(self, n_samples: int = 1000, noise_levels: list = [0, 1], noise_scale: int = 1, random_seed: int = 42, plot: bool = False, func=None) -> pd.DataFrame:
        """
        Generates synthetic data with aleatoric uncertainty by varying noise levels in the target variable.
        
        Parameters:
            n_samples (int): Number of samples to generate.
            noise_levels (list): List of noise levels to apply at different sample spaces.
            noise_scale (int): Scale factor for the noise.
            random_seed (int): Random seed for reproducibility.
            func (callable, optional): Function to generate the clean signal. Should accept a numpy array and return a numpy array.
        
        Returns:
            pd.DataFrame: DataFrame containing the generated data with columns 'x', 'y_clean', 'y_noisy', and 'true_noise_std'.
        """
        np.random.seed(random_seed)
        X = np.linspace(0, n_samples, n_samples)
        # Use provided function or default to identity
        if func is not None:
            y_clean = func(X)
        else:
            y_clean = X

        noise_regions = np.array_split(np.arange(n_samples), len(noise_levels))

        # Generate y with region-specific aleatoric noise
        y_noisy = np.copy(y_clean)
        noise_std_per_point = np.zeros_like(y_clean)

        for region, noise_level in zip(noise_regions, noise_levels):
            noise = np.random.normal(0, noise_level, size=len(region))
            y_noisy[region] += noise * noise_scale
            noise_std_per_point[region] = noise_level * noise_scale

        # Create dataframe
        df = pd.DataFrame({
            "x": X,
            "y_clean": y_clean,
            "y_noisy": y_noisy,
            "true_noise_std": noise_std_per_point
        })

        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(X, y_clean, label="Clean Signal", color="black", linestyle="--")
            plt.scatter(X, y_noisy, c=noise_std_per_point, cmap="viridis", s=10, label="Noisy Observations")
            plt.colorbar(label="True Noise Std Dev")
            plt.title("Synthetic Dataset with Region-Specific Aleatoric Noise")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return df
    
    def gen_epistemic_benchmark(
        self,
        n_samples: int = 1000,
        noise_std: float = 0.1,
        data_densities: list = [0, 1],
        random_seed: int = 42,
        plot: bool = False,
        func=None
    ) -> pd.DataFrame:
        '''
        Generates synthetic data with epistemic uncertainty by varying data density in the feature space.

        Parameters:
            n_samples (int): Number of samples to generate.
            noise_std (float): Standard deviation of the Gaussian noise added to the target variable.
            data_densities (list): List of data densities to apply to the feature space.
            random_seed (int): Random seed for reproducibility.
        '''
        np.random.seed(random_seed)

        # Step 1: Generate full input space and compute noisy targets
        X_full = np.linspace(0, 1, n_samples)
        func = func or (lambda x: np.sin(2 * np.pi * x))
        y_full = func(X_full) + np.random.normal(0, noise_std, size=n_samples)

        # Step 2: Define density regions
        num_regions = len(data_densities)
        region_edges = np.linspace(0, 1, num_regions + 1)

        # Step 3: Subsample points based on regional densities
        keep_indices = []
        density_lookup = []

        for i in range(num_regions):
            # Region bounds
            start, end = region_edges[i], region_edges[i + 1]
            region_mask = (X_full >= start) & (X_full < end)
            region_idx = np.where(region_mask)[0]

            # Subsample points based on density
            n_keep = int(len(region_idx) * data_densities[i])
            if n_keep > 0:
                selected = np.random.choice(region_idx, size=n_keep, replace=False)
                keep_indices.extend(selected)
                density_lookup.extend([data_densities[i]] * n_keep)

        # Step 4: Construct final dataset
        keep_indices = np.array(keep_indices)
        X = X_full[keep_indices]
        y = y_full[keep_indices]
        densities = np.array(density_lookup)

        df = pd.DataFrame({
            'x': X,
            'y_noisy': y,
            'true_noise_std': densities,
        })

        # Optional plotting
        if plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            X_plot = np.linspace(0, 1, 500)
            y_plot = func(X_plot)
            plt.plot(X_plot, y_plot, label="Clean Signal", color="black", linestyle="--")

            for i in range(num_regions):
                x_start = region_edges[i]
                x_end = region_edges[i + 1]
                density = data_densities[i]
                alpha = 0.1 + 0.4 * density
                plt.axvspan(x_start, x_end, color='purple', alpha=alpha, label=f'Density {density}' if i == 0 else None)

            plt.scatter(X, y, color="blue", s=10, label="Sampled Points")
            plt.title("Synthetic Dataset with Region-Specific Epistemic Uncertainty")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return df



