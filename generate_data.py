import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons


def generate_figure_2a(size=4000):
    """
    Generate data like Figure 2A of the paper (distribution with 5 peaks)
    """
    np.random.seed(42)

    # Parameters for 5 Gaussian distributions
    means = [
        [0.2, 0.2],
        [0.8, 0.2],
        [0.5, 0.5],
        [0.2, 0.8],
        [0.8, 0.8]
    ]

    covs = [
        [[0.02, 0], [0, 0.05]],
        [[0.03, 0.01], [0.01, 0.03]],
        [[0.02, 0], [0, 0.02]],
        [[0.05, 0], [0, 0.02]],
        [[0.03, -0.01], [-0.01, 0.03]]
    ]

    # Cluster sizes (proportional to density)
    sizes = [int(size * 0.3), int(size * 0.2), int(size * 0.25), int(size * 0.15), int(size * 0.1)]

    data = []
    labels = []

    for i, (mean, cov, sz) in enumerate(zip(means, covs, sizes)):
        cluster_data = np.random.multivariate_normal(mean, cov, sz)
        data.append(cluster_data)
        labels.extend([i] * sz)

    # Add uniform noise (20% background)
    noise_size = int(size * 0.2)
    noise = np.random.rand(noise_size, 2)
    noise_labels = [-1] * noise_size

    data.append(noise)
    labels.extend(noise_labels)

    X = np.vstack(data)
    y = np.array(labels)

    return X, y


def generate_figure_3a():
    """
    Data from Figure 3A (from Gionis et al. 2007)
    """
    np.random.seed(42)
    n_points = 300

    # Two moons with noise
    X, y = make_moons(n_samples=n_points, noise=0.05, random_state=42)

    # Add additional noise
    noise = np.random.uniform(-1.5, 2.5, (50, 2))
    noise_labels = [-1] * 50

    X = np.vstack([X, noise])
    y = np.hstack([y, noise_labels])

    return X, y


def generate_figure_3b():
    """
    Data from Figure 3B (15 overlapping clusters)
    """
    np.random.seed(42)

    centers = []
    for i in range(5):
        for j in range(3):
            centers.append([i * 0.25 + 0.1, j * 0.3 + 0.1])

    X, y = make_blobs(n_samples=1500, centers=centers, cluster_std=0.03, random_state=42)

    return X, y


def generate_figure_3c():
    """
    Data from Figure 3C (FLAME test)
    """
    np.random.seed(42)

    # Three concentric circles
    X1, y1 = make_circles(n_samples=300, factor=0.3, noise=0.05, random_state=42)
    X2, _ = make_circles(n_samples=500, factor=0.6, noise=0.05, random_state=42)
    X3, _ = make_circles(n_samples=800, factor=0.9, noise=0.05, random_state=42)

    X = np.vstack([X1 * 0.3 + [0.5, 0.5],
                   X2 * 0.5 + [0.5, 0.5],
                   X3 * 0.7 + [0.5, 0.5]])
    y = np.hstack([np.zeros(300), np.ones(500), np.ones(800) * 2])

    return X, y


def generate_figure_3d():
    """
    Data from Figure 3D (path-based spectral clustering test)
    """
    np.random.seed(42)

    # Three curved clusters
    t = np.linspace(0, 2 * np.pi, 400)

    # Cluster 1
    x1 = 0.5 + 0.3 * np.cos(t) + 0.1 * np.random.randn(400)
    y1 = 0.5 + 0.3 * np.sin(t) + 0.1 * np.random.randn(400)

    # Cluster 2
    x2 = 0.3 + 0.15 * np.cos(t) + 0.05 * np.random.randn(400)
    y2 = 0.7 + 0.15 * np.sin(t) + 0.05 * np.random.randn(400)

    # Cluster 3
    x3 = 0.7 + 0.15 * np.cos(t) + 0.05 * np.random.randn(400)
    y3 = 0.3 + 0.15 * np.sin(t) + 0.05 * np.random.randn(400)

    X = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2]),
        np.column_stack([x3, y3])
    ])

    y = np.hstack([np.zeros(400), np.ones(400), np.ones(400) * 2])

    return X, y


def generate_random_uniform(n_points=1000, dim=2):
    """
    Generate random uniform distribution (for comparison)
    """
    np.random.seed(42)
    X = np.random.rand(n_points, dim)
    y = np.ones(n_points) * -1  # all noise
    return X, y


def plot_data(X, y, title="Generated Data"):
    """
    Visualize generated data
    """
    plt.figure(figsize=(8, 6))

    if len(np.unique(y)) > 10:
        plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.6)
    else:
        for label in np.unique(y):
            if label == -1:
                plt.scatter(X[y == label, 0], X[y == label, 1],
                            s=10, alpha=0.3, c='gray', label='Noise')
            else:
                plt.scatter(X[y == label, 0], X[y == label, 1],
                            s=10, alpha=0.6, label=f'Cluster {int(label)}')

    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    if len(np.unique(y)) < 10:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test data generation
    X, y = generate_figure_2a(1000)
    plot_data(X, y, "Figure 2A-like Data")