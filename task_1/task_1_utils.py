from cdlib import evaluation
from community import community_louvain

from task_1.algos import create_node_clustering
import os
import networkx as nx

from task_1.constants import N_NODES, N_BLOCKS, DATA_DIR, RESULTS_DIR


def get_true_communities():
    communities = []
    nodes_per_block = N_NODES // N_BLOCKS

    for block in range(N_BLOCKS):
        start_node = block * nodes_per_block + 1  # +1 because nodes are 1-indexed
        end_node = (block + 1) * nodes_per_block
        communities.append([str(i) for i in range(start_node, end_node + 1)])

    return communities


def communities_to_node2comm(communities):
    node2comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node2comm[node] = i
    return node2comm


def load_network(filename):
    # Construct full path or look for files with matching prr value
    file_path = os.path.join(DATA_DIR, filename)

    # If file doesn't exist with exact name, try to find a file with matching prr
    if not os.path.exists(file_path):
        # Extract prr value from the filename
        if "prr_" in filename:
            prr_str = filename.split('prr_')[1].split('_')[0]

            # List all files in the directory
            for f in os.listdir(DATA_DIR):
                if f'prr_{prr_str}' in f and f.endswith('.net'):
                    file_path = os.path.join(DATA_DIR, f)
                    print(f"Found matching file: {f}")
                    break

    print(f"Loading network from: {file_path}")
    G = nx.read_pajek(file_path)
    return G

def calculate_modularity(G, communities):
    node2comm = communities_to_node2comm(communities)
    return community_louvain.modularity(node2comm, G)

def compare_partitions(detected_communities, true_communities, G):
    # Create NodeClustering objects
    detected_clustering = create_node_clustering(detected_communities, G)
    true_clustering = create_node_clustering(true_communities, G)

    # Calculate comparison metrics
    # Fixed: using correct metric names from cdlib
    try:
        # Note: adjusted_mutual_information is a replacement for jaccard since it's not available
        ami = evaluation.adjusted_mutual_information(detected_clustering, true_clustering)
        nmi = evaluation.normalized_mutual_information(detected_clustering, true_clustering)
        variation_of_info = evaluation.variation_of_information(detected_clustering, true_clustering)

        return {
            'ami': ami.score,  # Adjusted Mutual Information instead of Jaccard
            'nmi': nmi.score,
            'variation_of_info': variation_of_info.score
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        # Return default values if calculation fails
        return {
            'ami': 0.0,
            'nmi': 0.0,
            'variation_of_info': 0.0
        }

def save_communities_to_clu(communities, filename):
    # Count total number of nodes
    total_nodes = sum(len(comm) for comm in communities)

    # Create node to community mapping
    node2comm = communities_to_node2comm(communities)

    # Write to .clu file
    with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
        f.write(f"*Vertices {total_nodes}\n")

        # Sort nodes by their numeric value
        sorted_nodes = sorted(node2comm.keys(), key=lambda x: int(x))

        for node in sorted_nodes:
            f.write(f"{node2comm[node] + 1}\n")  # +1 because Pajek uses 1-indexed community IDs
