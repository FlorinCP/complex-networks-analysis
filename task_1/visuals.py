import os

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

from task_1.constants import RESULTS_DIR
from task_1.task_1_utils import communities_to_node2comm


def visualize_network(G, communities, filename, node_positions=None):
    plt.figure(figsize=(10, 8))

    node2comm = communities_to_node2comm(communities)
    node_colors = [node2comm.get(node, -1) for node in G.nodes()]

    if node_positions is None:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = node_positions

    nx.draw_networkx(
        G,
        pos=pos,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        node_size=50,
        with_labels=False,
        edge_color='lightgray',
        alpha=0.8
    )

    plt.title(f"Communities: {len(communities)}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close()


def plot_results(results_df):
    if results_df.empty:
        print("No results to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = ['num_communities', 'modularity', 'nmi', 'jaccard']
    titles = ['Number of Communities', 'Modularity', 'Normalized Mutual Information', 'Jaccard Index']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]

        for algo in ['Infomap', 'Louvain', 'Leiden']:
            algo_data = results_df[results_df['algorithm'] == algo]
            if not algo_data.empty:
                ax.plot(algo_data['prr'], algo_data[metric], 'o-', label=algo)

        ax.set_xlabel('prr (intra-community connection probability)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if not results_df['prr'].empty and len(results_df['prr'].unique()) > 1:
            if all(x > 0 for x in results_df['prr'].unique()):
                ax.set_xscale('log')

        prr_values_present = sorted(results_df['prr'].unique())
        if len(prr_values_present) > 0:
            ax.set_xticks(prr_values_present)
            ax.set_xticklabels([str(x) for x in prr_values_present], rotation=45)

        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "task1_metrics_evolution.png"), dpi=300)
    plt.close()


def compare_algorithms(results_df):
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from task_1.constants import RESULTS_DIR

    prr_groups = results_df.groupby('prr')

    comparison_table = []

    for prr, group in prr_groups:
        row = {'prr': prr}

        for algo in ['Infomap', 'Louvain', 'Leiden']:
            algo_data = group[group['algorithm'] == algo]
            if not algo_data.empty:
                row[f'{algo}_communities'] = algo_data['num_communities'].values[0]
                row[f'{algo}_modularity'] = algo_data['modularity'].values[0]
                row[f'{algo}_jaccard'] = algo_data['jaccard'].values[0]
                row[f'{algo}_nmi'] = algo_data['nmi'].values[0]

        comparison_table.append(row)

    # Convert to DataFrame and print
    comparison_df = pd.DataFrame(comparison_table)
    print("\nAlgorithm Comparison Table:")
    print(comparison_df.to_string())

    print("\nAverage metrics per algorithm:")
    algorithm_averages = results_df.groupby('algorithm').mean()
    print(algorithm_averages[['num_communities', 'modularity', 'jaccard', 'nmi']])

    print("\nCases with largest differences in number of communities:")
    for prr, group in prr_groups:
        if len(group) >= 3:  # Ensure we have all three algorithms
            max_comm = group['num_communities'].max()
            min_comm = group['num_communities'].min()
            if max_comm - min_comm > 0:
                print(f"prr={prr}: Range of communities: {min_comm} to {max_comm}")

    # VISUALIZATION 1: Bar chart comparing number of communities for each prr value
    plt.figure(figsize=(12, 6))

    # Get unique prr values and algorithms
    prr_values = sorted(results_df['prr'].unique())
    algorithms = ['Infomap', 'Louvain', 'Leiden']

    # Set the positions for grouped bars
    bar_width = 0.25
    r1 = np.arange(len(prr_values))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Create bars for each algorithm
    for i, algo in enumerate(['Infomap', 'Louvain', 'Leiden']):
        algo_data = results_df[results_df['algorithm'] == algo]

        # Create a position-to-value mapping
        value_map = {}
        for _, row in algo_data.iterrows():
            value_map[row['prr']] = row['num_communities']

        # Get values in the same order as prr_values
        values = [value_map.get(prr, 0) for prr in prr_values]

        # Plot at the correct position
        r = [r1, r2, r3][i]
        plt.bar(r, values, width=bar_width, label=algo)

    # Add labels and legend
    plt.xlabel('prr value')
    plt.ylabel('Number of Communities')
    plt.title('Number of Communities Detected by Each Algorithm')
    plt.xticks([r + bar_width for r in range(len(prr_values))], [str(prr) for prr in prr_values])
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "algorithm_community_comparison.png"), dpi=300)
    plt.close()

    # VISUALIZATION 2: Line chart comparing modularity values
    plt.figure(figsize=(10, 6))

    for algo in algorithms:
        algo_data = results_df[results_df['algorithm'] == algo]

        # Sort by prr for line plotting
        sorted_data = algo_data.sort_values('prr')

        plt.plot(sorted_data['prr'], sorted_data['modularity'], 'o-', label=algo)

    plt.xlabel('prr value')
    plt.ylabel('Modularity')
    plt.title('Modularity Values by Algorithm')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Use logarithmic x-axis if values are suitable
    if all(x > 0 for x in prr_values):
        plt.xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "algorithm_modularity_comparison.png"), dpi=300)
    plt.close()