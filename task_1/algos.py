import infomap
from community import community_louvain
from cdlib import algorithms, NodeClustering


def run_infomap(G):
    im = infomap.Infomap("--two-level")

    for e in G.edges():
        # Convert node labels to integers (Infomap requires integer node IDs)
        source = int(e[0])
        target = int(e[1])
        im.add_link(source, target)

    im.run()

    communities = {}
    for node in im.tree:
        if node.is_leaf:
            if node.module_id not in communities:
                communities[node.module_id] = []
            communities[node.module_id].append(str(node.node_id))

    return list(communities.values())

def run_louvain(G):
    partition = community_louvain.best_partition(G)

    # Convert partition format
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)

    return list(communities.values())

def run_leiden(G):
    # Convert NetworkX graph to igraph for Leiden algorithm
    communities = algorithms.leiden(G).communities

    # Ensure all node IDs are strings for consistency
    return [[str(node) for node in comm] for comm in communities]


def create_node_clustering(communities, graph):
    communities_as_lists = [list(comm) for comm in communities]
    return NodeClustering(communities_as_lists, graph, "")