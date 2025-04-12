import os
import pandas as pd
import networkx as nx
from tqdm import tqdm

from task_1.algos import run_infomap, run_louvain, run_leiden
from task_1.constants import PRR_VALUES, RESULTS_DIR
from task_1.task_1_utils import get_true_communities, load_network, save_communities_to_clu, calculate_modularity, \
    compare_partitions
from task_1.visuals import visualize_network, plot_results

def analyze_networks():
    true_communities = get_true_communities()

    results = {
        'prr': [],
        'algorithm': [],
        'num_communities': [],
        'modularity': [],
        'nmi': [],
        'variation_of_info': []
    }

    reference_positions = None

    for prr in tqdm(PRR_VALUES, desc="Processing networks"):
        # Format the filename with correct prr value
        filename = f"synthetic_network_N_300_blocks_5_prr_{prr:.2f}_prs_0.02.net"
        print(f"\nProcessing network with prr={prr}")

        try:
            G = load_network(filename)

            # If this is the most connected network, calculate positions for visualization reference
            if prr == 1.0:
                reference_positions = nx.kamada_kawai_layout(G)

            infomap_communities = run_infomap(G)
            louvain_communities = run_louvain(G)
            leiden_communities = run_leiden(G)

            for algo_name, communities in [
                ('Infomap', infomap_communities),
                ('Louvain', louvain_communities),
                ('Leiden', leiden_communities)
            ]:
                print(f"  Algorithm: {algo_name}, Found {len(communities)} communities")

                modularity = calculate_modularity(G, communities)

                comparison = compare_partitions(communities, true_communities, G)

                results['prr'].append(prr)
                results['algorithm'].append(algo_name)
                results['num_communities'].append(len(communities))
                results['modularity'].append(modularity)
                results['nmi'].append(comparison['nmi'])
                results['variation_of_info'].append(comparison['variation_of_info'])

                save_communities_to_clu(communities, f"{algo_name.lower()}_prr_{prr:.2f}.clu")

                if prr in [0.02, 0.16, 1.0]:
                    visualize_network(
                        G,
                        communities,
                        f"{algo_name.lower()}_prr_{prr:.2f}.png",
                        node_positions=reference_positions
                    )
        except Exception as e:
            print(f"Error processing network with prr={prr}: {str(e)}")
            continue

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "task1_results.csv"), index=False)

    return results_df


if __name__ == "__main__":
    print("Starting analysis of synthetic networks...")
    try:
        results_df = analyze_networks()
        print("Plotting results...")
        plot_results(results_df)
        print("Analysis complete. Results saved to the 'results' directory.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback

        traceback.print_exc()