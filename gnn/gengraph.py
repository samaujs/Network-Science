##########################################################################################
# This file contains functions to create graph shapes with adjacency matrix and labels
# Filename    : gengraph.py
# Created by  : Au Jit Seah
# File owners : Au Jit Seah
##########################################################################################

import networkx as nx
import numpy as np
import matplotlib

# Own modules
from utils import syntheticSim
from utils import featureGen

# Clean up graph and create its adjacency matrix
"""
Perturb the list of (sparse) graphs by adding/removing edges.

Input parameters :
----------------------------------------------------------------------------------------
graph_list           :      the list of graphs to be perturbed
p                    :      proportion of added edges based on current number of edges.

Return values :
----------------------------------------------------------------------------------------
perturbed_graph_list :      the list of graphs that are perturbed from the original graphs.
"""
def perturb(graph_list, p):
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list

"""
Load an existing graph to be converted for the experiments.

Input parameters :
----------------------------------------------------------------------------------------
G                        :      Networkx graph is the input for preprocessing
labels                   :      corresponding node labels
normalize_adj            :      Boolean
                                returns a normalized adjacency matrix (True)

Return values :
----------------------------------------------------------------------------------------
{"adj", "feat" "labels"} :  dictionary containing adjacency, node features and labels
"""
def preprocess_input_graph(G, labels, normalize_adj=False):
    # Create the adjacency matrix for graph
    adj = np.array(nx.to_numpy_matrix(G))
    print("------ Preprocess Input graph ------")
    print("The shape of the adjacency matrix ('dxd') of input graph :", adj.shape)

    # If normalization is required
#     if normalize_adj:
#         # Create a diagonal array
#         sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
#         adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

    # last index from 0 - 34
    existing_node = list(G.nodes)[-1]
    # Dimension of features
    feat_dim = G.nodes[existing_node]["feat"].shape[0]
    print("Feature dimensions of the last node '" + str(existing_node) + "' : " + str(feat_dim))

    # Initialize feature ndarray (dimension of number_of_nodes x feat_dim)
    features = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for idx, node_id in enumerate(G.nodes()):
        features[idx, :] = G.nodes[node_id]["feat"]

    # add batch dim by expanding the shape horizontally
    adj = np.expand_dims(adj, axis=0)
    features = np.expand_dims(features, axis=0)
    labels = np.expand_dims(labels, axis=0)
    print("The shape of the adjacency matrix after expansion :", adj.shape)
    print("The shape of the features matrix after expansion :", features.shape)
    print("The shape of the labels matrix after expansion :", labels.shape)

    return {"adj": adj, "feat": features, "labels": labels}


"""
Generating Synthetic Graph for experimentation :
- Barabasi-Albert base graph and attach the no. of "house" motifs

Input parameters :
----------------------------------------------------------------------------------------
nb_shapes         :  the no. of shapes ('house' motifs) that should be added to the
                     base graph
width_basis       :  the no. of nodes of the basis graph (ie. 'BA' graph)
feature_generator :  a `Feature Generator` for node features
                     addition of constant features to nodes ('None')
m                 :  no. of edges to be attached to existing node (for 'BA' graph)

Return values :
----------------------------------------------------------------------------------------
G                 :  a generated networkx "ba" graph with attached "house" motifs
role_id           :  a list with total number of nodes in the entire graph (base graph
                     and  motifs).  role_id[i] is the ID of the role of node i.
                     It is also the label used for training and predictions
name              :  a graph identifier
"""
def gen_syn1(nb_shapes=3, width_basis=20, feature_generator=None, m=5):
    basis_type = "ba"
    list_shapes = [["house"]] * nb_shapes

    G, role_id, _ = syntheticSim.build_graph(width_basis, basis_type, list_shapes, start=0, m=5)
    G = perturb([G], 0.01)[0]

    if feature_generator is None:
        # feature generator
        feature_generator = featureGen.ConstFeatureGen(1)

    # Generate node features
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    print("------ Generated the Synthetic BA graph with 'House' motifs ------")
    print("Name of generated graph :", name)
    return G, role_id, name
