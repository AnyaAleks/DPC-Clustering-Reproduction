"""
Main file for running all experiments from the paper
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments import *

if __name__ == "__main__":
    print("=" * 70)
    print("REPRODUCING EXPERIMENTS FROM:")
    print("'Clustering by fast search and find of density peaks'")
    print("Rodriguez A., Laio A. Science 344, 1492 (2014)")
    print("=" * 70)

    # Run Figure 2 experiment
    print("\n1. Running Figure 2 experiment...")
    dpc_fig2, X_fig2, clusters_fig2 = experiment_figure_2()

    # Run Figure 3 experiments
    for i, dataset in enumerate(['a', 'b', 'c', 'd'], 2):
        print(f"\n{i}. Running Figure 3{dataset} experiment...")
        dpc, X, clusters = experiment_figure_3(dataset)

    # Run random uniform experiment
    print("\n6. Running random uniform distribution experiment...")
    experiment_random_uniform()

    print("\n" + "=" * 70)
    print("EXPERIMENT REPRODUCTION COMPLETE")
    print("=" * 70)

    # Keep plots open
    plt.show(block=True)  # This will keep all plots open