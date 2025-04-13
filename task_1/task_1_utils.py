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
        start_node = block * nodes_per_block + 1
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
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        if "prr_" in filename:
            prr_str = filename.split('prr_')[1].split('_')[0]

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
    detected_clustering = create_node_clustering(detected_communities, G)
    true_clustering = create_node_clustering(true_communities, G)

    try:
        jaccard = calculate_jaccard_index(detected_communities, true_communities)
        nmi = evaluation.normalized_mutual_information(detected_clustering, true_clustering)
        variation_of_info = evaluation.variation_of_information(detected_clustering, true_clustering)

        return {
            'jaccard': jaccard,
            'nmi': nmi.score,
            'variation_of_info': variation_of_info.score,
        }

    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            'jaccard': 0.0,
            'nmi': 0.0,
            'variation_of_info': 0.0,
        }

def save_communities_to_clu(communities, filename):
    total_nodes = sum(len(comm) for comm in communities)

    node2comm = communities_to_node2comm(communities)

    with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
        f.write(f"*Vertices {total_nodes}\n")

        sorted_nodes = sorted(node2comm.keys(), key=lambda x: int(x))

        for node in sorted_nodes:
            f.write(f"{node2comm[node] + 1}\n")


def calculate_jaccard_index(communities1, communities2):
    node_to_comm1 = {}
    for i, comm in enumerate(communities1):
        for node in comm:
            node_to_comm1[node] = i

    node_to_comm2 = {}
    for i, comm in enumerate(communities2):
        for node in comm:
            node_to_comm2[node] = i

    all_nodes = set(node_to_comm1.keys()).union(set(node_to_comm2.keys()))
    nodes_list = list(all_nodes)

    for node in all_nodes:
        if node not in node_to_comm1:
            node_to_comm1[node] = len(communities1)
        if node not in node_to_comm2:
            node_to_comm2[node] = len(communities2)

    tp = 0  # Same community in both partitions
    fp = 0  # Same community in partition1, different in partition2
    fn = 0  # Different community in partition1, same in partition2

    for i in range(len(nodes_list)):
        for j in range(i + 1, len(nodes_list)):
            node1 = nodes_list[i]
            node2 = nodes_list[j]

            same_comm1 = node_to_comm1[node1] == node_to_comm1[node2]
            same_comm2 = node_to_comm2[node1] == node_to_comm2[node2]

            if same_comm1 and same_comm2:
                tp += 1
            elif same_comm1 and not same_comm2:
                fp += 1
            elif not same_comm1 and same_comm2:
                fn += 1

    if tp + fp + fn == 0:
        return 0.0

    return tp / (tp + fp + fn)
