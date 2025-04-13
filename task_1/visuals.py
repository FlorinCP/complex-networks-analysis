import os

import networkx as nx
from matplotlib import pyplot as plt

from task_1.constants import RESULTS_DIR


def visualize_network(G, communities, filename, node_positions=None):
    plt.figure(figsize=(10, 8))

    node2comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node2comm[node] = i

    custom_colors = [
        '#377eb8',  # Blue
        '#ff7f00',  # Orange
        '#4daf4a',  # Green
        '#f781bf',  # Pink
        '#a65628',  # Brown
        '#984ea3',  # Purple
        '#e41a1c',  # Red
        '#dede00'  # Yellow
    ]

    while len(custom_colors) < len(communities):
        custom_colors.extend(custom_colors)

    node_colors = [custom_colors[node2comm.get(node, 0) % len(custom_colors)] for node in G.nodes()]

    if node_positions is None:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = node_positions

    nx.draw_networkx(
        G,
        pos=pos,
        node_color=node_colors,
        node_size=50,
        with_labels=False,
        edge_color='lightgray',
        alpha=0.8
    )

    legend_handles = []
    for i in range(len(communities)):
        color = custom_colors[i % len(custom_colors)]
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=10,
                                         label=f'Community {i + 1}'))

    plt.legend(handles=legend_handles, title='Communities',
               loc='upper right', bbox_to_anchor=(1.1, 1))

    plt.title(f"{filename}")
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

    for prr, group in prr_groups:
        if len(group) >= 3:
            max_comm = group['num_communities'].max()
            min_comm = group['num_communities'].min()
            if max_comm - min_comm > 0:
                print(f"prr={prr}: Range of communities: {min_comm} to {max_comm}")

    plt.figure(figsize=(12, 6))

    prr_values = sorted(results_df['prr'].unique())
    algorithms = ['Infomap', 'Louvain', 'Leiden']

    bar_width = 0.25
    r1 = np.arange(len(prr_values))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    for i, algo in enumerate(['Infomap', 'Louvain', 'Leiden']):
        algo_data = results_df[results_df['algorithm'] == algo]

        value_map = {}
        for _, row in algo_data.iterrows():
            value_map[row['prr']] = row['num_communities']

        values = [value_map.get(prr, 0) for prr in prr_values]

        r = [r1, r2, r3][i]
        plt.bar(r, values, width=bar_width, label=algo)

    plt.xlabel('prr value')
    plt.ylabel('Number of Communities')
    plt.title('Number of Communities Detected by Each Algorithm')
    plt.xticks([r + bar_width for r in range(len(prr_values))], [str(prr) for prr in prr_values])
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "algorithm_community_comparison.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))

    for algo in algorithms:
        algo_data = results_df[results_df['algorithm'] == algo]

        sorted_data = algo_data.sort_values('prr')

        plt.plot(sorted_data['prr'], sorted_data['modularity'], 'o-', label=algo)

    plt.xlabel('prr value')
    plt.ylabel('Modularity')
    plt.title('Modularity Values by Algorithm')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if all(x > 0 for x in prr_values):
        plt.xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "algorithm_modularity_comparison.png"), dpi=300)
    plt.close()
