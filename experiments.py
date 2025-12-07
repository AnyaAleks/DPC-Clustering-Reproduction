import numpy as np
import matplotlib.pyplot as plt
from dpc_algorithm import DensityPeaksClustering
from generate_data import *
import warnings

warnings.filterwarnings('ignore')


def plot_decision_graph(rho, delta, centers=None, title="Decision Graph"):
    """
    Visualize decision graph as in the paper
    """
    plt.figure(figsize=(8, 6))

    if centers is not None:
        plt.scatter(rho, delta, s=20, alpha=0.6, c='blue', label='Data points')
        plt.scatter(rho[centers], delta[centers], s=100, c='red',
                    marker='o', edgecolors='black', linewidth=2, label='Centers')
    else:
        plt.scatter(rho, delta, s=20, alpha=0.6)

    plt.title(title)
    plt.xlabel(r"$\rho$ (density)")
    plt.ylabel(r"$\delta$ (distance)")
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_gamma_sorted(gamma, title="Gamma = ρ × δ (sorted)"):
    """
    Plot gamma values sorted in decreasing order
    """
    sorted_gamma = np.sort(gamma)[::-1]
    ranks = np.arange(1, len(sorted_gamma) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(ranks, sorted_gamma, 'b-', linewidth=2)
    plt.scatter(ranks, sorted_gamma, s=30, alpha=0.6)

    plt.title(title)
    plt.xlabel("Rank")
    plt.ylabel(r"$\gamma = \rho \times \delta$")
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_clusters(X, clusters, halo=None, centers=None, title="Clustering Results"):
    """
    Visualize clustering results
    """
    plt.figure(figsize=(10, 8))

    # Unique clusters (excluding -1 if present)
    unique_clusters = np.unique(clusters)
    unique_clusters = unique_clusters[unique_clusters >= 0]
    n_clusters = len(unique_clusters)

    # Colors for clusters
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

    # Plot halo points (if any)
    if halo is not None and np.any(halo):
        halo_points = X[halo]
        plt.scatter(halo_points[:, 0], halo_points[:, 1],
                    s=20, alpha=0.5, c='black', label='Halo/Noise')

    # Plot clusters
    for idx, cluster_id in enumerate(unique_clusters):
        mask = (clusters == cluster_id)
        if halo is not None:
            mask = mask & (~halo)  # exclude halo from core
        cluster_points = X[mask]

        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        s=30, alpha=0.7, c=colors[idx % len(colors)],
                        label=f'Cluster {cluster_id}')

    # Plot centers
    if centers is not None:
        center_points = X[centers]
        plt.scatter(center_points[:, 0], center_points[:, 1],
                    s=200, c='yellow', marker='*',
                    edgecolors='black', linewidth=2, label='Centers')

    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    if len(unique_clusters) < 20:  # Avoid too many legend entries
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def experiment_figure_2():
    """
    Experiment as in Figure 2 of the paper
    """
    print("=" * 60)
    print("EXPERIMENT: FIGURE 2 (Synthetic Data with 5 Peaks)")
    print("=" * 60)

    # Generate data
    X, y_true = generate_figure_2a(size=4000)

    # Create and fit model
    dpc = DensityPeaksClustering(dc=0.1, kernel='cutoff')
    dpc.fit(X)

    # Get decision graph
    rho, delta = dpc.get_decision_graph()

    # Visualize decision graph
    plot_decision_graph(rho, delta, title="Decision Graph for Figure 2 Data")

    # Select centers manually (based on graph)
    # In practice, you'd choose points with large delta and rho
    gamma = rho * delta
    sorted_indices = np.argsort(gamma)[::-1]

    # Choose 5 points with highest gamma as centers
    centers = sorted_indices[:5]
    print(f"Selected centers (indices): {centers}")
    print(f"Gamma values: {gamma[centers]}")

    # Clustering
    clusters = dpc.predict(centers)

    # Visualization
    plot_clusters(X, clusters, dpc.halo, centers,
                  title="Clustering Results (Figure 2)")

    # Evaluate quality (only core points)
    core_mask = ~dpc.halo
    y_pred_core = clusters[core_mask]
    y_true_core = y_true[core_mask]

    # Accuracy (simple evaluation)
    from sklearn.metrics import adjusted_rand_score
    if len(np.unique(y_true_core)) > 1:
        ari = adjusted_rand_score(y_true_core[y_true_core >= 0],
                                  y_pred_core[y_true_core >= 0])
        print(f"Adjusted Rand Index (core points): {ari:.4f}")

    # Plot gamma sorted
    plot_gamma_sorted(gamma, title="Gamma Values for Figure 2 Data")

    return dpc, X, clusters


def experiment_figure_3(dataset='a'):
    """
    Experiments with data from Figure 3
    """
    print("=" * 60)
    print(f"EXPERIMENT: FIGURE 3{dataset.upper()}")
    print("=" * 60)

    # Generate data
    if dataset == 'a':
        X, y_true = generate_figure_3a()
        expected_clusters = 2
    elif dataset == 'b':
        X, y_true = generate_figure_3b()
        expected_clusters = 15
    elif dataset == 'c':
        X, y_true = generate_figure_3c()
        expected_clusters = 3
    elif dataset == 'd':
        X, y_true = generate_figure_3d()
        expected_clusters = 3

    # Use Gaussian kernel for better density estimation
    dpc = DensityPeaksClustering(dc=None, kernel='gaussian')
    dpc.fit(X)

    # Decision graph
    rho, delta = dpc.get_decision_graph()
    plot_decision_graph(rho, delta, title=f"Decision Graph (Figure 3{dataset.upper()})")

    # Automatic center selection based on gamma
    gamma = dpc.get_gamma()
    sorted_indices = np.argsort(gamma)[::-1]

    # Take points with anomalously large gamma
    gamma_diff = np.diff(gamma[sorted_indices])
    # Find large jump
    if len(gamma_diff) > expected_clusters:
        # Simple heuristic
        centers = sorted_indices[:expected_clusters]
    else:
        centers = sorted_indices[:expected_clusters]

    print(f"Selected {len(centers)} centers")

    # Clustering
    clusters = dpc.predict(centers)

    # Visualization
    plot_clusters(X, clusters, dpc.halo, centers,
                  title=f"Clustering Results (Figure 3{dataset.upper()})")

    # Plot gamma sorted
    plot_gamma_sorted(gamma, title=f"Gamma Values (Figure 3{dataset.upper()})")

    return dpc, X, clusters


def compare_with_kmeans(X, n_clusters):
    """
    Compare with K-means (as in the paper)
    """
    from sklearn.cluster import KMeans

    print("=" * 60)
    print("COMPARISON WITH K-MEANS")
    print("=" * 60)

    # Run K-means 10 times and take the best
    best_kmeans = None
    best_inertia = float('inf')

    for i in range(10):
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=i)
        kmeans.fit(X)
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_kmeans = kmeans

    kmeans_labels = best_kmeans.labels_

    # Visualization
    plt.figure(figsize=(8, 6))

    for cluster_id in range(n_clusters):
        mask = (kmeans_labels == cluster_id)
        cluster_points = X[mask]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    s=30, alpha=0.7, label=f'Cluster {cluster_id}')

    plt.scatter(best_kmeans.cluster_centers_[:, 0], best_kmeans.cluster_centers_[:, 1],
                s=200, c='red', marker='X', edgecolors='black', linewidth=2, label='K-means Centers')

    plt.title(f"K-means Clustering (K={n_clusters})")
    plt.xlabel("X1")
    plt.ylabel("X2")
    if n_clusters < 20:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return best_kmeans


def experiment_random_uniform():
    """
    Experiment with random uniform distribution
    """
    print("=" * 60)
    print("EXPERIMENT: RANDOM UNIFORM DISTRIBUTION")
    print("=" * 60)

    # Generate random uniform data
    X, y = generate_random_uniform(n_points=1000, dim=2)

    # Fit DPC model
    dpc = DensityPeaksClustering(dc=None, kernel='cutoff')
    dpc.fit(X)

    # Get decision graph
    rho, delta = dpc.get_decision_graph()
    plot_decision_graph(rho, delta, title="Decision Graph (Random Uniform)")

    # Plot gamma for random data
    gamma = dpc.get_gamma()
    plot_gamma_sorted(gamma, title="Gamma Values for Random Uniform Data")

    # Compare with clustered data
    X_clustered, _ = generate_figure_2a(1000)
    dpc_clustered = DensityPeaksClustering(dc=0.1, kernel='cutoff')
    dpc_clustered.fit(X_clustered)
    gamma_clustered = dpc_clustered.get_gamma()

    # Compare gamma distributions
    plt.figure(figsize=(10, 6))

    sorted_gamma_random = np.sort(gamma)[::-1]
    sorted_gamma_clustered = np.sort(gamma_clustered)[::-1]

    ranks_random = np.arange(1, len(sorted_gamma_random) + 1)
    ranks_clustered = np.arange(1, len(sorted_gamma_clustered) + 1)

    plt.loglog(ranks_random, sorted_gamma_random, 'b-', linewidth=2, label='Random Uniform')
    plt.loglog(ranks_clustered, sorted_gamma_clustered, 'r-', linewidth=2, label='Clustered Data')

    plt.title("Comparison of Gamma Values: Random vs Clustered Data")
    plt.xlabel("Rank")
    plt.ylabel(r"$\gamma = \rho \times \delta$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return dpc


def plot_decision_graph(rho, delta, centers=None, title="Decision Graph"):
    """
    Visualize decision graph as in the paper
    """
    plt.figure(figsize=(8, 6))

    if centers is not None:
        plt.scatter(rho, delta, s=20, alpha=0.6, c='blue', label='Data points')
        plt.scatter(rho[centers], delta[centers], s=100, c='red',
                    marker='o', edgecolors='black', linewidth=2, label='Centers')
    else:
        plt.scatter(rho, delta, s=20, alpha=0.6)

    plt.title(title)
    plt.xlabel(r"$\rho$ (density)")
    plt.ylabel(r"$\delta$ (distance)")
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()  # <-- ДОБАВЬТЕ ЭТУ СТРОКУ


def plot_gamma_sorted(gamma, title="Gamma = ρ × δ (sorted)"):
    """
    Plot gamma values sorted in decreasing order
    """
    sorted_gamma = np.sort(gamma)[::-1]
    ranks = np.arange(1, len(sorted_gamma) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(ranks, sorted_gamma, 'b-', linewidth=2)
    plt.scatter(ranks, sorted_gamma, s=30, alpha=0.6)

    plt.title(title)
    plt.xlabel("Rank")
    plt.ylabel(r"$\gamma = \rho \times \delta$")
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()  # <-- ДОБАВЬТЕ ЭТУ СТРОКУ


def plot_clusters(X, clusters, halo=None, centers=None, title="Clustering Results"):
    """
    Visualize clustering results
    """
    plt.figure(figsize=(10, 8))

    # Unique clusters (excluding -1 if present)
    unique_clusters = np.unique(clusters)
    unique_clusters = unique_clusters[unique_clusters >= 0]
    n_clusters = len(unique_clusters)

    # Colors for clusters
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

    # Plot halo points (if any)
    if halo is not None and np.any(halo):
        halo_points = X[halo]
        plt.scatter(halo_points[:, 0], halo_points[:, 1],
                    s=20, alpha=0.5, c='black', label='Halo/Noise')

    # Plot clusters
    for idx, cluster_id in enumerate(unique_clusters):
        mask = (clusters == cluster_id)
        if halo is not None:
            mask = mask & (~halo)  # exclude halo from core
        cluster_points = X[mask]

        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        s=30, alpha=0.7, c=colors[idx % len(colors)],
                        label=f'Cluster {cluster_id}')

    # Plot centers
    if centers is not None:
        center_points = X[centers]
        plt.scatter(center_points[:, 0], center_points[:, 1],
                    s=200, c='yellow', marker='*',
                    edgecolors='black', linewidth=2, label='Centers')

    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    if len(unique_clusters) < 20:  # Avoid too many legend entries
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()  # <-- ДОБАВЬТЕ ЭТУ СТРОКУ


def compare_with_kmeans(X, n_clusters):
    """
    Compare with K-means (as in the paper)
    """
    from sklearn.cluster import KMeans

    print("=" * 60)
    print("COMPARISON WITH K-MEANS")
    print("=" * 60)

    # Run K-means 10 times and take the best
    best_kmeans = None
    best_inertia = float('inf')

    for i in range(10):
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=i)
        kmeans.fit(X)
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_kmeans = kmeans

    kmeans_labels = best_kmeans.labels_

    # Visualization
    plt.figure(figsize=(8, 6))

    for cluster_id in range(n_clusters):
        mask = (kmeans_labels == cluster_id)
        cluster_points = X[mask]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    s=30, alpha=0.7, label=f'Cluster {cluster_id}')

    plt.scatter(best_kmeans.cluster_centers_[:, 0], best_kmeans.cluster_centers_[:, 1],
                s=200, c='red', marker='X', edgecolors='black', linewidth=2, label='K-means Centers')

    plt.title(f"K-means Clustering (K={n_clusters})")
    plt.xlabel("X1")
    plt.ylabel("X2")
    if n_clusters < 20:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()  # <-- ДОБАВЬТЕ ЭТУ СТРОКУ


if __name__ == "__main__":
    # Run experiments

    # 1. Figure 2 experiment
    print("\n" + "=" * 60)
    print("STARTING EXPERIMENTS")
    print("=" * 60 + "\n")

    dpc1, X1, clusters1 = experiment_figure_2()

    # 2. Figure 3 experiments
    for dataset in ['a', 'b', 'c', 'd']:
        print("\n")
        dpc, X, clusters = experiment_figure_3(dataset)

        # Comparison with K-means
        if dataset == 'a':
            compare_with_kmeans(X, n_clusters=2)
        elif dataset == 'b':
            compare_with_kmeans(X, n_clusters=15)
        elif dataset == 'c':
            compare_with_kmeans(X, n_clusters=3)
        elif dataset == 'd':
            compare_with_kmeans(X, n_clusters=3)

    # 3. Random uniform distribution experiment
    print("\n")
    experiment_random_uniform()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 60)