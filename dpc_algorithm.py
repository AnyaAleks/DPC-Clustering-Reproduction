import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


class DensityPeaksClustering:
    """
    Implementation of "Clustering by fast search and find of density peaks"
    Rodriguez A., Laio A. Science 344, 1492 (2014)
    """

    def __init__(self, dc=None, kernel='gaussian', method='standard'):
        """
        :param dc: cutoff distance
        :param kernel: 'gaussian' or 'cutoff'
        :param method: 'standard' or 'percentile'
        """
        self.dc = dc
        self.kernel = kernel
        self.method = method
        self.distances = None
        self.density = None
        self.delta = None
        self.nearest_higher_density = None
        self.clusters = None
        self.halo = None
        self.centers = None

    def fit(self, X):
        """
        Main clustering function
        """
        n = X.shape[0]

        # 1. Compute distance matrix
        print("Computing distance matrix...")
        self.distances = squareform(pdist(X))

        # 2. Determine dc if not provided
        if self.dc is None:
            self.dc = self._estimate_dc(self.distances)
            print(f"Automatically selected dc: {self.dc:.4f}")

        # 3. Compute density
        print("Computing density...")
        self.density = self._compute_density(self.distances)

        # 4. Compute delta and nearest higher density
        print("Computing delta...")
        self.delta, self.nearest_higher_density = self._compute_delta(self.distances, self.density)

        return self

    def _estimate_dc(self, distances, percentile=2.0):
        """
        Estimate dc as percentile of all distances
        """
        triu_indices = np.triu_indices_from(distances, k=1)
        all_distances = distances[triu_indices]
        dc = np.percentile(all_distances, percentile)
        return dc

    def _compute_density(self, distances):
        """
        Compute density for each point
        """
        n = distances.shape[0]
        density = np.zeros(n)

        if self.kernel == 'cutoff':
            # Original formula from the paper
            for i in tqdm(range(n), desc="Density (cutoff)"):
                density[i] = np.sum(distances[i, :] < self.dc) - 1  # exclude self
        elif self.kernel == 'gaussian':
            # Gaussian kernel (better for small datasets)
            for i in tqdm(range(n), desc="Density (gaussian)"):
                density[i] = np.sum(np.exp(-(distances[i, :] / self.dc) ** 2)) - 1

        return density

    def _compute_delta(self, distances, density):
        """
        Compute minimum distance to point with higher density
        """
        n = len(density)
        delta = np.zeros(n)
        nearest_higher_density = np.zeros(n, dtype=int)

        # Sort indices by density in descending order
        sorted_indices = np.argsort(density)[::-1]

        # For point with highest density
        delta[sorted_indices[0]] = np.max(distances[sorted_indices[0]])
        nearest_higher_density[sorted_indices[0]] = -1

        # For other points
        for i in tqdm(range(1, n), desc="Computing delta"):
            idx = sorted_indices[i]
            # Find points with higher density
            higher_density_indices = sorted_indices[:i]
            # Minimum distance to them
            min_dist = np.min(distances[idx, higher_density_indices])
            delta[idx] = min_dist
            # Index of nearest point with higher density
            nearest_idx = higher_density_indices[np.argmin(distances[idx, higher_density_indices])]
            nearest_higher_density[idx] = nearest_idx

        return delta, nearest_higher_density

    def predict(self, centers):
        """
        Perform clustering after selecting centers
        :param centers: list of indices of cluster centers
        """
        n = len(self.density)
        self.centers = centers
        self.clusters = -np.ones(n, dtype=int)  # -1 means "not assigned"

        # Assign centers
        for cluster_id, center_idx in enumerate(centers):
            self.clusters[center_idx] = cluster_id

        # Sort points by density in descending order
        sorted_indices = np.argsort(self.density)[::-1]

        # Assign remaining points
        for idx in sorted_indices:
            if self.clusters[idx] == -1:  # not yet assigned
                nearest_higher = self.nearest_higher_density[idx]
                self.clusters[idx] = self.clusters[nearest_higher]

        # Compute halo (boundary points)
        self._compute_halo()

        return self.clusters

    def _compute_halo(self):
        """
        Determine halo points (boundary/noise)
        """
        n = len(self.clusters)
        self.halo = np.zeros(n, dtype=bool)  # True = halo point

        # Compute border density for each cluster
        border_density = {}

        for i in range(n):
            for j in range(i + 1, n):
                if self.clusters[i] != self.clusters[j] and self.distances[i, j] < self.dc:
                    # Points from different clusters but close
                    cluster_i = self.clusters[i]
                    cluster_j = self.clusters[j]

                    # Update maximum density on border
                    density_max = max(self.density[i], self.density[j])

                    if cluster_i not in border_density:
                        border_density[cluster_i] = density_max
                    else:
                        border_density[cluster_i] = max(border_density[cluster_i], density_max)

                    if cluster_j not in border_density:
                        border_density[cluster_j] = density_max
                    else:
                        border_density[cluster_j] = max(border_density[cluster_j], density_max)

        # Identify halo points
        for i in range(n):
            cluster_id = self.clusters[i]
            if cluster_id in border_density and self.density[i] < border_density[cluster_id]:
                self.halo[i] = True

    def get_decision_graph(self):
        """
        Return data for decision graph (rho, delta)
        """
        return self.density, self.delta

    def get_gamma(self):
        """
        gamma = rho * delta (for sorting)
        """
        return self.density * self.delta