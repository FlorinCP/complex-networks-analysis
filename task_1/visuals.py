import os

import networkx as nx
from matplotlib import pyplot as plt

from task_1.constants import RESULTS_DIR
from task_1.task_1_utils import communities_to_node2comm


def visualize_network(G, communities, filename, node_positions=None):
    plt.figure(figsize=(10, 8))

    # Create color mapping for nodes based on communities
    node2comm = communities_to_node2comm(communities)
    node_colors = [node2comm.get(node, -1) for node in G.nodes()]

    # Use provided node positions or calculate new ones
    if node_positions is None:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = node_positions

    # Draw the network
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

    metrics = ['num_communities', 'modularity', 'nmi', 'ami']  # Changed jaccard to ami
    titles = ['Number of Communities', 'Modularity', 'Normalized Mutual Information', 'Adjusted Mutual Information']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]

        # Plot each algorithm
        for algo in ['Infomap', 'Louvain', 'Leiden']:
            algo_data = results_df[results_df['algorithm'] == algo]
            if not algo_data.empty:
                ax.plot(algo_data['prr'], algo_data[metric], 'o-', label=algo)

        ax.set_xlabel('prr (intra-community connection probability)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Only use logarithmic x-axis if we have positive values and multiple values
        if not results_df['prr'].empty and len(results_df['prr'].unique()) > 1:
            # Check if all values are positive before using log scale
            if all(x > 0 for x in results_df['prr'].unique()):
                ax.set_xscale('log')

        # Set x-ticks for the prr values we have
        prr_values_present = sorted(results_df['prr'].unique())
        if len(prr_values_present) > 0:
            ax.set_xticks(prr_values_present)
            ax.set_xticklabels([str(x) for x in prr_values_present], rotation=45)

        # Add legend
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "task1_metrics_evolution.png"), dpi=300)
    plt.close()
