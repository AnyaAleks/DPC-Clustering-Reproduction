import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def display_all_plots():
    """Display all saved plots"""
    output_dir = "experiment_results"

    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist!")
        return

    files = sorted(os.listdir(output_dir))

    if not files:
        print("No files found in the directory!")
        return

    print(f"Found {len(files)} plot files:")
    for f in files:
        print(f"  - {f}")

    # Display key plots
    key_files = [
        'fig2_clusters.png',
        'fig2_decision_graph.png',
        'fig3a_clusters.png',
        'fig3c_clusters.png',
        'fig3d_clusters.png'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, filename in enumerate(key_files):
        if i >= len(axes):
            break

        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            img = mpimg.imread(filepath)
            axes[i].imshow(img)
            axes[i].set_title(filename)
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f"Not found:\n{filename}",
                         ha='center', va='center')
            axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(key_files), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def analyze_results():
    """Quick analysis of the clustering results"""
    import numpy as np
    from dpc_algorithm import DensityPeaksClustering
    from generate_data import generate_figure_2a

    print("=" * 60)
    print("ANALYZING CLUSTERING RESULTS")
    print("=" * 60)

    # Test with smaller dataset for quick analysis
    X, y_true = generate_figure_2a(size=1000)

    dpc = DensityPeaksClustering(dc=0.1, kernel='cutoff')
    dpc.fit(X)

    rho, delta = dpc.get_decision_graph()
    gamma = rho * delta

    # Find potential centers (points with high gamma)
    sorted_indices = np.argsort(gamma)[::-1]

    print(f"\nTop 10 points with highest gamma:")
    print("-" * 40)
    for i in range(min(10, len(gamma))):
        idx = sorted_indices[i]
        print(f"Rank {i + 1}: Point {idx} - γ = {gamma[idx]:.2f}, ρ = {rho[idx]:.0f}, δ = {delta[idx]:.3f}")

    # Check if we have clear cluster centers
    gamma_diff = np.diff(gamma[sorted_indices])
    large_gaps = np.where(gamma_diff < -0.1 * gamma[sorted_indices[:-1]])[0]

    if len(large_gaps) > 0:
        suggested_clusters = large_gaps[0] + 1
        print(f"\nSuggested number of clusters: {suggested_clusters}")
        print(f"(Based on gap in gamma values after point {sorted_indices[suggested_clusters - 1]})")
    else:
        print("\nNo clear gap in gamma values found.")

    print(f"\nStatistics:")
    print(f"- Number of points: {len(X)}")
    print(f"- Max density (ρ): {rho.max():.0f}")
    print(f"- Min density (ρ): {rho.min():.0f}")
    print(f"- Max delta (δ): {delta.max():.3f}")
    print(f"- Min delta (δ): {delta.min():.3f}")


if __name__ == "__main__":
    print("Checking experiment results...")
    print("\n1. Listing saved plot files:")
    print("-" * 40)

    output_dir = "experiment_results"
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        for f in sorted(files):
            size = os.path.getsize(os.path.join(output_dir, f))
            print(f"{f} ({size / 1024:.1f} KB)")

    print("\n2. Quick analysis of clustering:")
    print("-" * 40)
    analyze_results()

    print("\n3. Displaying key plots (if running interactively)...")
    try:
        display_all_plots()
    except:
        print("Could not display plots. Open them manually from the experiment_results/ directory.")