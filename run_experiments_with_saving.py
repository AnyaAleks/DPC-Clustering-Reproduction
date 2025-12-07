"""
Run experiments and save all plots with proper labels and annotations
"""

import numpy as np
import matplotlib.pyplot as plt
from dpc_algorithm import DensityPeaksClustering
from generate_data import *
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory
output_dir = "experiment_results"
os.makedirs(output_dir, exist_ok=True)


def save_decision_graph(rho, delta, centers=None, dc_value=None,
                       filename="decision_graph.png", title="Decision Graph"):
    """
    Save decision graph with proper annotations
    """
    plt.figure(figsize=(10, 8))

    # Plot all points
    if centers is not None:
        non_centers = np.ones(len(rho), dtype=bool)
        non_centers[centers] = False

        plt.scatter(rho[non_centers], delta[non_centers],
                   s=30, alpha=0.6, c='blue', label='Data points')

        # Plot centers with larger markers
        plt.scatter(rho[centers], delta[centers], s=200, c='red',
                   marker='*', edgecolors='black', linewidth=2,
                   label=f'Cluster centers ({len(centers)} found)')

        # Annotate centers with their indices
        for i, center_idx in enumerate(centers):
            plt.annotate(f'C{i+1}',
                        xy=(rho[center_idx], delta[center_idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor="yellow", alpha=0.7))
    else:
        plt.scatter(rho, delta, s=30, alpha=0.6, c='blue', label='Data points')

    plt.title(f"{title}\n", fontsize=14, fontweight='bold')
    plt.xlabel(r"Local density $\rho_i$", fontsize=12)
    plt.ylabel(r"Minimum distance $\delta_i$", fontsize=12)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)

    # Add information box
    info_text = f"Total points: {len(rho)}"
    if dc_value is not None:
        info_text += f"\nCutoff distance d_c: {dc_value:.3f}"
    if centers is not None:
        info_text += f"\nCluster centers: {len(centers)}"

    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.legend(loc='best')
    plt.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def save_gamma_sorted(gamma, centers_indices=None, filename="gamma_sorted.png",
                     title="Gamma = ρ × δ (sorted)"):
    """
    Save gamma plot with annotations
    """
    sorted_gamma = np.sort(gamma)[::-1]
    sorted_indices = np.argsort(gamma)[::-1]
    ranks = np.arange(1, len(sorted_gamma) + 1)

    plt.figure(figsize=(10, 8))

    # Plot gamma values
    plt.plot(ranks, sorted_gamma, 'b-', linewidth=1.5, alpha=0.7, label='γ values')
    plt.scatter(ranks, sorted_gamma, s=40, alpha=0.6, c='blue')

    # Highlight cluster centers if provided
    if centers_indices is not None:
        center_ranks = []
        center_gammas = []
        for idx in centers_indices:
            # Find rank of this index
            rank_pos = np.where(sorted_indices == idx)[0][0] + 1
            center_ranks.append(rank_pos)
            center_gammas.append(gamma[idx])

        plt.scatter(center_ranks, center_gammas, s=200, c='red',
                   marker='*', edgecolors='black', linewidth=2,
                   label=f'Cluster centers ({len(centers_indices)} found)')

        # Add annotations for centers
        for i, (rank, gamma_val) in enumerate(zip(center_ranks, center_gammas)):
            plt.annotate(f'C{i+1} (rank {rank})',
                        xy=(rank, gamma_val),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        arrowprops=dict(arrowstyle="->", color='red', alpha=0.7),
                        bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor="yellow", alpha=0.7))

    plt.title(f"{title}\n", fontsize=14, fontweight='bold')
    plt.xlabel("Rank (sorted by γ in descending order)", fontsize=12)
    plt.ylabel(r"$\gamma_i = \rho_i \times \delta_i$", fontsize=12)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, alpha=0.3, which='both')

    # Add statistics box
    mean_gamma = np.mean(gamma)
    std_gamma = np.std(gamma)
    max_gamma = np.max(gamma)

    info_text = (f"Statistics:\n"
                 f"Max γ: {max_gamma:.2f}\n"
                 f"Mean γ: {mean_gamma:.2f}\n"
                 f"Std γ: {std_gamma:.2f}\n"
                 f"Total points: {len(gamma)}")

    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.legend(loc='best')
    plt.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def save_clusters(X, clusters, halo=None, centers=None, dc_value=None,
                 filename="clusters.png", title="Clustering Results"):
    """
    Save clustering results with annotations
    """
    plt.figure(figsize=(12, 10))

    # Unique clusters (excluding -1 if present)
    unique_clusters = np.unique(clusters)
    unique_clusters = unique_clusters[unique_clusters >= 0]
    n_clusters = len(unique_clusters)

    # Colors for clusters
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

    # Plot halo points (if any)
    if halo is not None and np.any(halo):
        halo_points = X[halo]
        halo_count = np.sum(halo)
        plt.scatter(halo_points[:, 0], halo_points[:, 1],
                   s=40, alpha=0.5, c='gray',
                   marker='x', label=f'Halo/Noise ({halo_count} points)')

    # Plot clusters
    core_counts = []
    for idx, cluster_id in enumerate(unique_clusters):
        mask = (clusters == cluster_id)
        if halo is not None:
            mask = mask & (~halo)  # exclude halo from core
        cluster_points = X[mask]
        core_counts.append(len(cluster_points))

        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       s=50, alpha=0.7, c=colors[idx % len(colors)],
                       edgecolors='black', linewidth=0.5,
                       label=f'Cluster {cluster_id+1} ({len(cluster_points)} points)')

    # Plot centers with annotations
    if centers is not None:
        center_points = X[centers]
        plt.scatter(center_points[:, 0], center_points[:, 1],
                   s=300, c='yellow', marker='*',
                   edgecolors='black', linewidth=3, label='Cluster centers')

        # Annotate centers
        for i, (x, y) in enumerate(center_points):
            plt.annotate(f'Center {i+1}', xy=(x, y), xytext=(10, 10),
                        textcoords='offset points', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor="yellow", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", color='black', alpha=0.7))

    plt.title(f"{title}\n", fontsize=16, fontweight='bold')
    plt.xlabel("Feature 1 (X)", fontsize=12)
    plt.ylabel("Feature 2 (Y)", fontsize=12)

    # Add statistics box
    total_points = len(X)
    core_points = total_points - (np.sum(halo) if halo is not None else 0)

    info_text = (f"Clustering Statistics:\n"
                 f"Total points: {total_points}\n"
                 f"Core points: {core_points}\n"
                 f"Halo points: {total_points - core_points}\n"
                 f"Number of clusters: {n_clusters}")

    if dc_value is not None:
        info_text += f"\nCutoff distance d_c: {dc_value:.3f}"

    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))

    # Add algorithm info
    plt.text(0.98, 0.02, "Algorithm: DPC\n(Rodriguez & Laio, 2014)",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def experiment_figure_2_save():
    """
    Experiment as in Figure 2 with BETTER center selection
    """
    print("=" * 60)
    print("EXPERIMENT: FIGURE 2 (Synthetic Data with 5 Peaks)")
    print("=" * 60)

    # Generate data
    X, y_true = generate_figure_2a(size=4000)

    # Create and fit model - try different dc
    dc_value = 0.08  # Changed from 0.1 to 0.08 (often works better)
    dpc = DensityPeaksClustering(dc=dc_value, kernel='cutoff')
    dpc.fit(X)

    # Get decision graph
    rho, delta = dpc.get_decision_graph()

    # BETTER center selection - not just top 5 by gamma!
    # Method from paper: centers have anomalously large delta

    # Calculate thresholds
    rho_threshold = np.percentile(rho, 85)  # Top 15% density
    delta_threshold = np.percentile(delta, 90)  # Top 10% distance

    # Find points that are BOTH high density AND high distance
    candidate_mask = (rho > rho_threshold) & (delta > delta_threshold)
    candidates = np.where(candidate_mask)[0]

    print(f"Found {len(candidates)} candidates with high ρ and δ")

    if len(candidates) >= 5:
        # Among candidates, take those with highest gamma
        gamma = rho * delta
        candidate_gamma = gamma[candidates]
        sorted_candidate_indices = np.argsort(candidate_gamma)[::-1]

        # Take top 5 candidates
        centers = candidates[sorted_candidate_indices[:5]]
    else:
        # Fallback: use top 5 by gamma
        gamma = rho * delta
        sorted_indices = np.argsort(gamma)[::-1]
        centers = sorted_indices[:5]
        print("Warning: Using fallback method (top 5 by gamma)")

    print(f"Selected centers (indices): {centers}")
    print(f"Center densities (ρ): {rho[centers]}")
    print(f"Center distances (δ): {delta[centers]}")

    # Save decision graph with annotations
    save_decision_graph(rho, delta, centers, dc_value,
                        filename="fig2_decision_graph.png",
                        title="Decision Graph - Figure 2 Data\n(5 Expected Clusters)")

    # Clustering
    clusters = dpc.predict(centers)

    # Save clustering results
    save_clusters(X, clusters, dpc.halo, centers, dc_value,
                  filename="fig2_clusters.png",
                  title="Clustering Results - Figure 2\n(5 Density Peaks with Halo)")

    # Save gamma plot
    save_gamma_sorted(gamma, centers, filename="fig2_gamma.png",
                      title="Gamma Values - Figure 2\nγ = ρ × δ (Sorted)")

    # Calculate accuracy
    from sklearn.metrics import adjusted_rand_score
    core_mask = ~dpc.halo
    y_pred_core = clusters[core_mask]
    y_true_core = y_true[core_mask]

    if len(np.unique(y_true_core)) > 1:
        ari = adjusted_rand_score(y_true_core[y_true_core >= 0],
                                  y_pred_core[y_true_core >= 0])
        print(f"Adjusted Rand Index (core points): {ari:.4f}")

        if ari < 0.5:
            print("⚠️ Warning: Low ARI. Centers might be incorrect!")
            print("Try adjusting dc_value or center selection criteria.")

    return dpc, X, clusters


def experiment_figure_3_save(dataset='a'):
    """
    Experiments with data from Figure 3 with annotations
    """
    print("=" * 60)
    print(f"EXPERIMENT: FIGURE 3{dataset.upper()}")
    print("=" * 60)

    # Generate data
    if dataset == 'a':
        X, y_true = generate_figure_3a()
        expected_clusters = 2
        description = "Two Crescent Moons"
    elif dataset == 'b':
        X, y_true = generate_figure_3b()
        expected_clusters = 15
        description = "15 Overlapping Clusters"
    elif dataset == 'c':
        X, y_true = generate_figure_3c()
        expected_clusters = 3
        description = "Three Concentric Circles"
    elif dataset == 'd':
        X, y_true = generate_figure_3d()
        expected_clusters = 3
        description = "Three Curved Clusters"

    # Use Gaussian kernel
    dpc = DensityPeaksClustering(dc=None, kernel='gaussian')
    dpc.fit(X)

    # Get dc value used
    dc_value = dpc.dc

    # Decision graph
    rho, delta = dpc.get_decision_graph()

    # Automatic center selection
    gamma = dpc.get_gamma()
    sorted_indices = np.argsort(gamma)[::-1]
    centers = sorted_indices[:expected_clusters]

    print(f"Selected {len(centers)} centers")
    print(f"DC value: {dc_value:.4f}")

    # Save decision graph
    save_decision_graph(rho, delta, centers, dc_value,
                       filename=f"fig3{dataset}_decision_graph.png",
                       title=f"Decision Graph - Figure 3{dataset.upper()}\n({description})")

    # Clustering
    clusters = dpc.predict(centers)

    # Save clustering results
    save_clusters(X, clusters, dpc.halo, centers, dc_value,
                  filename=f"fig3{dataset}_clusters.png",
                  title=f"Clustering Results - Figure 3{dataset.upper()}\n({description})")

    # Save gamma plot
    save_gamma_sorted(gamma, centers, filename=f"fig3{dataset}_gamma.png",
                     title=f"Gamma Values - Figure 3{dataset.upper()}\n{description}")

    return dpc, X, clusters


def create_comparison_plot():
    """
    Create a comparison plot showing all test cases
    """
    datasets = ['a', 'b', 'c', 'd']
    titles = [
        "A: Two Crescent Moons",
        "B: 15 Overlapping Clusters",
        "C: Three Concentric Circles",
        "D: Three Curved Clusters"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (dataset, title) in enumerate(zip(datasets, titles)):
        # Generate data
        if dataset == 'a':
            X, _ = generate_figure_3a()
            expected_clusters = 2
        elif dataset == 'b':
            X, _ = generate_figure_3b()
            expected_clusters = 15
        elif dataset == 'c':
            X, _ = generate_figure_3c()
            expected_clusters = 3
        elif dataset == 'd':
            X, _ = generate_figure_3d()
            expected_clusters = 3

        # Run DPC
        dpc = DensityPeaksClustering(dc=None, kernel='gaussian')
        dpc.fit(X)

        gamma = dpc.get_gamma()
        sorted_indices = np.argsort(gamma)[::-1]
        centers = sorted_indices[:expected_clusters]
        clusters = dpc.predict(centers)

        # Plot
        ax = axes[idx]
        unique_clusters = np.unique(clusters[clusters >= 0])
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

        # Plot halo
        if dpc.halo is not None and np.any(dpc.halo):
            halo_points = X[dpc.halo]
            ax.scatter(halo_points[:, 0], halo_points[:, 1],
                      s=20, alpha=0.3, c='gray', marker='x')

        # Plot clusters
        for cluster_idx, cluster_id in enumerate(unique_clusters):
            mask = (clusters == cluster_id)
            if dpc.halo is not None:
                mask = mask & (~dpc.halo)
            cluster_points = X[mask]

            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      s=30, alpha=0.7, c=colors[cluster_idx],
                      label=f'Cluster {cluster_id+1}')

        # Plot centers
        center_points = X[centers]
        ax.scatter(center_points[:, 0], center_points[:, 1],
                  s=150, c='yellow', marker='*',
                  edgecolors='black', linewidth=2)

        ax.set_title(f"Figure 3{dataset.upper()}: {title}", fontsize=12, fontweight='bold')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)

        # Add info box
        info_text = f"Clusters: {len(unique_clusters)}\nPoints: {len(X)}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.suptitle("DPC Algorithm Performance on Various Test Cases\n",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_dir, "all_test_cases_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {filepath}")


def main():
    """
    Main function to run all experiments and save annotated plots
    """
    print("=" * 70)
    print("RUNNING ANNOTATED EXPERIMENTS")
    print("Output directory:", output_dir)
    print("=" * 70)

    # Run Figure 2 experiment
    print("\n1. Running Figure 2 experiment (5 density peaks)...")
    dpc1, X1, clusters1 = experiment_figure_2_save()

    # Run Figure 3 experiments
    for i, dataset in enumerate(['a', 'b', 'c', 'd'], 2):
        print(f"\n{i}. Running Figure 3{dataset} experiment...")
        dpc, X, clusters = experiment_figure_3_save(dataset)

    # Create comparison plot
    print("\n6. Creating comparison plot of all test cases...")
    create_comparison_plot()

    print("\n" + "=" * 70)
    print("ALL ANNOTATED EXPERIMENTS COMPLETED!")
    print(f"All plots saved in '{output_dir}' directory")
    print("=" * 70)

    # Create summary text file
    summary_file = os.path.join(output_dir, "experiment_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("DPC CLUSTERING EXPERIMENT SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write("Paper: 'Clustering by fast search and find of density peaks'\n")
        f.write("Authors: Alex Rodriguez and Alessandro Laio\n")
        f.write("Journal: Science 344, 1492 (2014)\n\n")
        f.write("Experiments conducted:\n")
        f.write("1. Figure 2: Synthetic data with 5 density peaks\n")
        f.write("2. Figure 3A: Two crescent moons\n")
        f.write("3. Figure 3B: 15 overlapping clusters\n")
        f.write("4. Figure 3C: Three concentric circles\n")
        f.write("5. Figure 3D: Three curved clusters\n\n")
        f.write(f"Total plots generated: {len([name for name in os.listdir(output_dir) if name.endswith('.png')])}\n")

    print(f"Summary file created: {summary_file}")


if __name__ == "__main__":
    main()