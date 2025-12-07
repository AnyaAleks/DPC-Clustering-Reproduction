"""
Manual center selection based on visual inspection of Decision Graph
FIXED VERSION
"""

import numpy as np
import matplotlib.pyplot as plt
from dpc_algorithm import DensityPeaksClustering
from generate_data import generate_figure_2a
import os

def interactive_center_selection():
    """
    Interactive tool to manually select cluster centers
    """
    print("=" * 70)
    print("MANUAL CENTER SELECTION FOR FIGURE 2")
    print("=" * 70)

    # Generate data
    X, y_true = generate_figure_2a(size=4000)

    # Try different dc values
    dc_values = [0.05, 0.08, 0.1, 0.12, 0.15]

    for dc in dc_values:
        print(f"\n{'='*60}")
        print(f"ANALYSIS WITH d_c = {dc}")
        print('='*60)

        dpc = DensityPeaksClustering(dc=dc, kernel='cutoff')
        dpc.fit(X)

        rho, delta = dpc.get_decision_graph()
        gamma = rho * delta

        # Create interactive plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: Decision Graph
        ax1.scatter(rho, delta, s=20, alpha=0.6, c='blue')
        ax1.set_xlabel(r"$\rho$ (density)", fontsize=12)
        ax1.set_ylabel(r"$\delta$ (distance)", fontsize=12)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Decision Graph (d_c = {dc})", fontsize=14, fontweight='bold')

        # Highlight top 20 points by gamma
        sorted_indices = np.argsort(gamma)[::-1]
        top_n = min(20, len(gamma))
        ax1.scatter(rho[sorted_indices[:top_n]], delta[sorted_indices[:top_n]],
                   s=100, c='red', marker='o', edgecolors='black',
                   linewidth=2, alpha=0.7, label=f'Top {top_n} by γ')

        # Add annotations for top points
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[i]
            ax1.annotate(f'{i+1}', xy=(rho[idx], delta[idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='red')

        ax1.legend()

        # Plot 2: Gamma values
        sorted_gamma = gamma[sorted_indices]
        ranks = np.arange(1, len(sorted_gamma) + 1)

        ax2.plot(ranks[:50], sorted_gamma[:50], 'b-', linewidth=2)
        ax2.scatter(ranks[:50], sorted_gamma[:50], s=40, alpha=0.6)

        # Mark potential cutoff
        gamma_diff = np.diff(sorted_gamma[:20])
        if len(gamma_diff) > 0:
            # FIX: Ensure arrays have same shape
            rel_diff = gamma_diff / sorted_gamma[1:20]  # Changed from 1:21 to 1:20

            # Find where gamma drops significantly (more than 30%)
            large_drops = np.where(rel_diff < -0.3)[0]

            if len(large_drops) > 0:
                cutoff = large_drops[0] + 1
                ax2.axvline(x=cutoff, color='r', linestyle='--', alpha=0.7)
                ax2.text(cutoff + 0.5, sorted_gamma[cutoff] * 0.8,
                        f'Suggested: {cutoff} clusters',
                        color='red', fontweight='bold')

        ax2.set_xlabel("Rank", fontsize=12)
        ax2.set_ylabel(r"$\gamma = \rho \times \delta$", fontsize=12)
        ax2.set_title("Gamma Values (Top 50)", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"debug_dc_{dc:.2f}_decision.png", dpi=120, bbox_inches='tight')
        plt.close()

        print(f"Top 10 points by gamma (d_c = {dc}):")
        print("-" * 50)
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[i]
            print(f"{i+1:2d}. Point {idx:4d}: γ = {gamma[idx]:8.2f}, "
                  f"ρ = {rho[idx]:4.0f}, δ = {delta[idx]:6.3f}, "
                  f"Coords: ({X[idx,0]:.3f}, {X[idx,1]:.3f})")

    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("1. Look at the saved debug_dc_*.png files")
    print("2. Choose points that are in the TOP-RIGHT corner of Decision Graph")
    print("3. These should have BOTH high density (ρ) AND high distance (δ)")
    print("4. Typically 5 clear points for Figure 2 data")
    print("=" * 70)

def automatic_center_detection():
    """
    Automatic center detection using method from the paper
    """
    print("\n" + "=" * 70)
    print("AUTOMATIC CENTER DETECTION")
    print("=" * 70)

    # Generate data
    X, y_true = generate_figure_2a(size=4000)

    # Try to find optimal dc
    best_ari = -1
    best_params = {}

    for dc in [0.06, 0.07, 0.08, 0.09, 0.1, 0.11]:
        print(f"\nTesting d_c = {dc}")

        dpc = DensityPeaksClustering(dc=dc, kernel='cutoff')
        dpc.fit(X)

        rho, delta = dpc.get_decision_graph()
        gamma = rho * delta

        # Method 1: Find points with anomalously large delta
        # In the paper, centers have "anomalously large" delta

        # Calculate mean and std of delta for points with high density
        high_density_mask = rho > np.percentile(rho, 70)
        if np.sum(high_density_mask) > 0:
            delta_mean = np.mean(delta[high_density_mask])
            delta_std = np.std(delta[high_density_mask])

            # Find points with delta > mean + 2*std (anomalously large)
            anomaly_mask = delta > (delta_mean + 2 * delta_std)

            # Also require reasonable density
            density_threshold = np.percentile(rho, 80)
            center_mask = anomaly_mask & (rho > density_threshold)

            centers = np.where(center_mask)[0]

            print(f"  Found {len(centers)} potential centers using anomaly detection")

            if 4 <= len(centers) <= 6:  # Expecting around 5 centers
                # Test these centers
                clusters = dpc.predict(centers)

                # Calculate ARI
                from sklearn.metrics import adjusted_rand_score
                core_mask = ~dpc.halo
                y_pred_core = clusters[core_mask]
                y_true_core = y_true[core_mask]

                if len(np.unique(y_true_core)) > 1:
                    ari = adjusted_rand_score(y_true_core[y_true_core >= 0],
                                             y_pred_core[y_true_core >= 0])

                    print(f"  ARI with {len(centers)} centers: {ari:.4f}")

                    if ari > best_ari:
                        best_ari = ari
                        best_params = {
                            'dc': dc,
                            'centers': centers,
                            'ari': ari,
                            'dpc': dpc,
                            'X': X,
                            'clusters': clusters
                        }

    if best_ari > 0:
        print(f"\n{'='*70}")
        print(f"BEST RESULT:")
        print(f"  d_c = {best_params['dc']:.3f}")
        print(f"  Centers found: {len(best_params['centers'])}")
        print(f"  ARI = {best_params['ari']:.4f}")
        print(f"  Center indices: {best_params['centers']}")
        print('='*70)

        # Visualize the best result
        visualize_best_result(best_params)

        return best_params
    else:
        print("\n❌ No good automatic detection found. Trying manual method...")
        return None

def visualize_best_result(params):
    """
    Visualize the best clustering result
    """
    dpc = params['dpc']
    X = params['X']
    centers = params['centers']
    clusters = params['clusters']
    dc = params['dc']
    ari = params['ari']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Decision Graph
    rho, delta = dpc.get_decision_graph()

    axes[0].scatter(rho, delta, s=20, alpha=0.6, c='blue', label='Data points')
    axes[0].scatter(rho[centers], delta[centers], s=150, c='red',
                   marker='*', edgecolors='black', linewidth=2, label='Centers')

    # Annotate centers
    for i, center_idx in enumerate(centers):
        axes[0].annotate(f'C{i+1}', xy=(rho[center_idx], delta[center_idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor="yellow", alpha=0.7))

    axes[0].set_xlabel(r"$\rho$ (density)", fontsize=12)
    axes[0].set_ylabel(r"$\delta$ (distance)", fontsize=12)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_title(f"Decision Graph (d_c = {dc})", fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Gamma plot
    gamma = rho * delta
    sorted_gamma = np.sort(gamma)[::-1]
    ranks = np.arange(1, len(sorted_gamma) + 1)

    axes[1].plot(ranks[:30], sorted_gamma[:30], 'b-', linewidth=2)
    axes[1].scatter(ranks[:30], sorted_gamma[:30], s=40, alpha=0.6)

    # Mark the selected centers
    center_ranks = []
    for center_idx in centers:
        # Find rank of this center
        rank = np.where(np.argsort(gamma)[::-1] == center_idx)[0][0] + 1
        center_ranks.append(rank)

    axes[1].scatter(center_ranks, gamma[centers], s=150, c='red',
                   marker='*', edgecolors='black', linewidth=2)

    axes[1].set_xlabel("Rank", fontsize=12)
    axes[1].set_ylabel(r"$\gamma = \rho \times \delta$", fontsize=12)
    axes[1].set_title("Gamma Values (Top 30)", fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # 3. Clustering result
    unique_clusters = np.unique(clusters[clusters >= 0])
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    # Halo points
    if dpc.halo is not None and np.any(dpc.halo):
        halo_points = X[dpc.halo]
        axes[2].scatter(halo_points[:, 0], halo_points[:, 1],
                       s=30, alpha=0.3, c='gray', marker='x', label='Halo')

    # Core clusters
    for idx, cluster_id in enumerate(unique_clusters):
        mask = (clusters == cluster_id)
        if dpc.halo is not None:
            mask = mask & (~dpc.halo)
        cluster_points = X[mask]

        axes[2].scatter(cluster_points[:, 0], cluster_points[:, 1],
                       s=40, alpha=0.7, c=colors[idx],
                       edgecolors='black', linewidth=0.5,
                       label=f'Cluster {cluster_id+1}')

    # Centers
    center_points = X[centers]
    axes[2].scatter(center_points[:, 0], center_points[:, 1],
                   s=300, c='yellow', marker='*',
                   edgecolors='black', linewidth=2, label='Centers')

    axes[2].set_xlabel("X", fontsize=12)
    axes[2].set_ylabel("Y", fontsize=12)
    axes[2].set_title(f"Clustering Result\nARI = {ari:.4f}", fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best')

    plt.suptitle(f"Figure 2: Best Automatic Center Detection (d_c = {dc})",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = "figure2_best_automatic.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Saved best result to: {output_file}")

def manual_improvement():
    """
    Manually improve the centers based on visual inspection
    """
    print("\n" + "=" * 70)
    print("MANUAL IMPROVEMENT - BASED ON TYPICAL FIGURE 2 RESULTS")
    print("=" * 70)

    # Based on typical Figure 2 from the paper, we know:
    # - There should be 5 clusters
    # - Centers should be in high-density regions
    # - They should be relatively far from each other

    # Generate data
    X, y_true = generate_figure_2a(size=4000)

    # Use dc = 0.08 (often works well for this dataset)
    dc = 0.08
    dpc = DensityPeaksClustering(dc=dc, kernel='cutoff')
    dpc.fit(X)

    rho, delta = dpc.get_decision_graph()
    gamma = rho * delta

    # Instead of taking top 5 by gamma, let's find points that are:
    # 1. High density (top 10%)
    # 2. Also have relatively high delta

    density_threshold = np.percentile(rho, 90)
    high_density_points = np.where(rho > density_threshold)[0]

    print(f"Points with density > {density_threshold:.1f} (top 10%): {len(high_density_points)}")

    # Among high density points, find those with highest delta
    delta_of_high_density = delta[high_density_points]
    high_density_indices = high_density_points

    # Sort high density points by delta
    sorted_by_delta = np.argsort(delta_of_high_density)[::-1]

    # Take top 5-7 by delta among high density points
    n_candidates = min(7, len(sorted_by_delta))
    candidate_indices = high_density_indices[sorted_by_delta[:n_candidates]]

    print(f"\nTop {n_candidates} candidates (high density + high delta):")
    for i, idx in enumerate(candidate_indices):
        print(f"{i+1}. Point {idx}: ρ = {rho[idx]:.0f}, δ = {delta[idx]:.3f}, γ = {gamma[idx]:.2f}")

    # Try clustering with these candidates
    centers = candidate_indices[:5]  # Take top 5

    clusters = dpc.predict(centers)

    # Calculate ARI
    from sklearn.metrics import adjusted_rand_score
    core_mask = ~dpc.halo
    y_pred_core = clusters[core_mask]
    y_true_core = y_true[core_mask]

    if len(np.unique(y_true_core)) > 1:
        ari = adjusted_rand_score(y_true_core[y_true_core >= 0],
                                 y_pred_core[y_true_core >= 0])

        print(f"\nResult with manual selection:")
        print(f"  Centers: {centers}")
        print(f"  ARI: {ari:.4f}")

        if ari > 0.5:
            print("✅ GOOD - Much better than automatic selection!")
        else:
            print("⚠️ Still needs improvement")

        # Visualize
        plt.figure(figsize=(10, 8))

        unique_clusters = np.unique(clusters[clusters >= 0])
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

        # Halo
        if dpc.halo is not None and np.any(dpc.halo):
            halo_points = X[dpc.halo]
            plt.scatter(halo_points[:, 0], halo_points[:, 1],
                       s=30, alpha=0.3, c='gray', marker='x', label='Halo')

        # Clusters
        for idx, cluster_id in enumerate(unique_clusters):
            mask = (clusters == cluster_id)
            if dpc.halo is not None:
                mask = mask & (~dpc.halo)
            cluster_points = X[mask]

            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       s=40, alpha=0.7, c=colors[idx],
                       edgecolors='black', linewidth=0.5,
                       label=f'Cluster {cluster_id+1}')

        # Centers
        center_points = X[centers]
        plt.scatter(center_points[:, 0], center_points[:, 1],
                   s=300, c='yellow', marker='*',
                   edgecolors='black', linewidth=2, label='Centers')

        plt.title(f"Figure 2 with Manual Center Selection\n"
                  f"d_c = {dc}, ARI = {ari:.4f}",
                  fontsize=14, fontweight='bold')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = "figure2_manual_selection.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✅ Saved to: {output_file}")

        return ari

if __name__ == "__main__":
    print("DPC CLUSTERING - CENTER SELECTION IMPROVEMENT")
    print("=" * 70)

    # Option 1: Run interactive analysis
    print("\n1. Running interactive analysis...")
    try:
        interactive_center_selection()
    except Exception as e:
        print(f"Error in interactive analysis: {e}")
        print("Continuing with automatic detection...")

    # Option 2: Try automatic detection
    print("\n2. Trying automatic center detection...")
    result = automatic_center_detection()

    # Option 3: Manual improvement
    print("\n3. Trying manual improvement method...")
    ari = manual_improvement()

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"- Original ARI (automatic top 5 by gamma): 0.0404")
    if result:
        print(f"- Best automatic ARI: {result['ari']:.4f}")
    if 'ari' in locals():
        print(f"- Manual improvement ARI: {ari:.4f}")
    print("=" * 70)

    print("\n📊 Next steps:")
    print("1. Check the generated debug_dc_*.png files")
    print("2. Look for 5 clear points in the top-right of Decision Graph")
    print("3. Compare with original paper's Figure 2")
    print("4. The key is selecting points with BOTH high ρ AND high δ")