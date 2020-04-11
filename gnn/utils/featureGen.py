##########################################################################################
# This file contains functions to generate node features
# Filename    : featureGen.py
# Created by  : Au Jit Seah
# File owners : Au Jit Seah
##########################################################################################

import networkx as nx
import numpy as np
import abc

class FeatureGen(metaclass=abc.ABCMeta):
    # Feature Generator base class from Abstract Base Classes
    @abc.abstractmethod
    def gen_node_features(self, G):
        pass

class ConstFeatureGen(FeatureGen):
    # Generate constant node features in class
    def __init__(self, val):
        print("Values in Constant Feature Generator : ", val)
        self.val = val

    def gen_node_features(self, G):
        print("Generate node features for " + str(len(G.nodes())) + " nodes.")
        feat_dict = {i:{'feat': np.array(self.val, dtype=np.float32)} for i in G.nodes()}
        print('Values of feat_dict[0]["feat"]:', feat_dict[0]['feat'])
        
        # Set node attributes with values in feature dictionary of values '1's
        nx.set_node_attributes(G, feat_dict)
        print('Node attributes of node \'0\', G.nodes[0]["feat"]:', G.nodes[0]['feat'])
