##########################################################################################
# This file contains functions to create synthetic BA graph and house motifs
# Filename    : syntheticSim.py
# Created by  : Au Jit Seah
# File owners : Au Jit Seah
##########################################################################################

import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Create the BA-shape with house motifs and setting roles that will be used as labels
"""
Builds a BA preferential attachment graph, with "node index" starting from "start"
parameter and "role_ids" from "role_start" parameter

Input parameters :
----------------------------------------------------------------------------------------
start       :    starting index of the shape
width       :    int size of the graph (no. of nodes)
role_start  :    starting index for the roles

Return values :
----------------------------------------------------------------------------------------
graph       :    a ba graph, with ids beginning from start
roles       :    list of the roles of the nodes (indexed starting from
                 role_start) that will be used as labels
"""
def ba(start, width, role_start=0, m=5):
    graph = nx.barabasi_albert_graph(width, m)
    graph.add_nodes_from(range(start, start + width))
    nids = sorted(graph)
    mapping = {nid: start + i for i, nid in enumerate(nids)}
    graph = nx.relabel_nodes(graph, mapping)
    roles = [role_start for i in range(width)]
    return graph, roles

"""
Builds a house-like graph/motif, with "node index" starting from "start"
parameter and "role_ids" from "role_start" parameter

Input parameters :
----------------------------------------------------------------------------------------
start       :    starting index for the shape
role_start  :    starting index for the roles

Return values :
----------------------------------------------------------------------------------------
graph       :    a house-like graph/motif, with ids beginning from start
roles       :    list of the roles of the nodes (indexed starting at
                 role_start) that will be used as labels
"""
def house(start, role_start=0):
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 5))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start),
        ]
    )
    # graph.add_edges_from([(start, start + 2), (start + 1, start + 3)])
    graph.add_edges_from([(start + 4, start), (start + 4, start + 1)])
    roles = [role_start, role_start, role_start + 1, role_start + 1, role_start + 2]
    return graph, roles

"""
Creates a basis graph and attaches elements of the type in the list randomly along the basis.
Possibility to add random edges afterwards.

Input parameters :
----------------------------------------------------------------------------------------
width_basis       :      width (in terms of number of nodes) of the basis
basis_type        :      "ba"
shapes            :      list of shape list
                         (1st arg  : type of shape,
                         next args : args for building the shape except for the start)
start             :      initial node label for the first node
rdm_basis_plugins :      Boolean
                         For the shapes to be attached randomly (True) or
                         regularly (False) to the basis graph
add_random_edges  :      no. of edges to randomly add on the structure
m                 :      no. of new edges to attach to existing node (for BA graph)

Return values :
----------------------------------------------------------------------------------------
basis             :      a networkx graph with the particular shape used as the base
role_id           :      label for each role (eg. representing basis or edges)
plugins           :      node ids whereby the motif graph will be attached to the basis
"""
# Eg. build_graph(20, "ba", [["house"]], start=0, m=5)
def build_graph(width_basis, basis_type, list_shapes, start=0,
                rdm_basis_plugins=False,add_random_edges=0, m=5):
    print("------ Building the Synthetic BA graph with 'House' motifs ------")
    # Build the BA graph start with 0 and number of nodes (width basis)
    if basis_type == "ba":
        # Drawing of a house motif
        basis, role_id = eval(basis_type)(start, width_basis, m=m)
        print("Role Id of the BA graph :\n", role_id)
#     else:
#         # Drawing other type of motif
#         basis, role_id = eval(basis_type)(start, width_basis)

    n_basis, n_shapes = nx.number_of_nodes(basis), len(list_shapes)
    start += n_basis  # indicator of the id of the next node
    print("Indicator of the id of the next node :", start)

    # role_id are '0's for all the nodes of the basis, BA graph
    print("Number of nodes in the BA graph : ", n_basis)
    print("Number of motifs : ", n_shapes)

    print("List of shapes :", list_shapes)
    print("No. of shapes :", len(list_shapes))

    # Sample (with replacement) where to attach the new motifs
    if rdm_basis_plugins is True:
        plugins = np.random.choice(n_basis, n_shapes, replace=False)
    else:
        spacing = math.floor(n_basis / n_shapes)
        print("Spacing : ", spacing)
        plugins = [int(k * spacing) for k in range(n_shapes)]
        print("Plugins : ", plugins)
    seen_shapes = {"basis": [0, n_basis]}
    print("seen_shapes : ", seen_shapes)

    for shape_index, shape in enumerate(list_shapes):
        shape_type = shape[0]
        print("\n-----------------------------------------")
        print("Shape_ID : " + str(shape_index) + " with shape type : " + str(shape_type))
        print(str(len(shape)) + " shapes with list of Shape :", shape)
        print("The shape starts from index 1 : ", shape[1:])

        args = [start]

        # More than one shape
        if len(shape) > 1:
            args += shape[1:]

        # Append 0 for the "role_start" in "house" function
        args += [0]
        print("\nThe list of arguments :", args)
        # *args parameter to send a non-keyworded variable-length argument list to function, 1-2 parameters in this case
        print("The first item in list of arguments :", args[0])
        print("The second item in list of arguments :", args[1])

        # Creating the "house" motif
        graph_s, roles_graph_s = eval(shape_type)(*args)
        n_s = nx.number_of_nodes(graph_s)

        try:
             # Get the last seen label from first index
            col_start = seen_shapes[shape_type][0]
        except:
            # Get the max label value 1
            col_start = np.max(role_id) + 1
            # Add the new shape_type to the seen_shapes dictionary
            seen_shapes[shape_type] = [col_start, n_s]
        print("Column start :", col_start)
        print("Observe seen_shapes : ", seen_shapes)


        # Attach the shape to the basis, BA graph
        basis.add_nodes_from(graph_s.nodes())
        basis.add_edges_from(graph_s.edges())
        # Connecting the motif to the BA graph from node 20 to 0, 25 to 6 and 30 to 12
        basis.add_edges_from([(start, plugins[shape_index])])
#         if shape_type == "cycle":
#             if np.random.random() > 0.5:
#                 a = np.random.randint(1, 4)
#                 b = np.random.randint(1, 4)
#                 basis.add_edges_from([(a + start, b + plugins[shape_id])])

        # start = 0; col_start = 1; roles_graph_s = [0, 0, 1, 1, 2]
        temp_labels = [r + col_start for r in roles_graph_s]
        # temp_labels increment roles_graph_s by col_start

        # temp_labels[0] += 100 * seen_shapes[shape_type][0]

        # role_id is [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Append labels of motif to the labels of BA graph
        role_id += temp_labels
        print("Labels of BA graph with attached motifs :\n", role_id)
        print("No. of nodes in attached graph : ", nx.number_of_nodes(basis))
        start += n_s
        print("With attached motif nodes, index starts from : ", start)

#     if add_random_edges > 0:
#         # add random edges between nodes:
#         for p in range(add_random_edges):
#             src, dest = np.random.choice(nx.number_of_nodes(basis), 2, replace=False)
#             print(src, dest)
#             basis.add_edges_from([(src, dest)])

    # Plotting the basis "BA" graph
    # plt.figure(figsize=(8, 6), dpi=300)
    plt.figure(figsize=(20, 10))
    plt.title('BA graph'.upper(), y=1.0, fontsize=14)
    nx.draw(basis, with_labels=True, font_weight='bold')

    # Plot the motif "house" graph
    plt.figure(figsize=(2, 2))
    plt.title('"House" motif', y=1.0, fontsize=12)
    nx.draw(graph_s, with_labels=True, font_weight='bold')
    print("\nInformation of the motif graph :\n", nx.info(graph_s))

    plt.show()

    return basis, role_id, plugins
