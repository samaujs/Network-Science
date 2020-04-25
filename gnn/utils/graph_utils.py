##########################################################################################
# This file contains functions to get neighbours from an adjacency matrix.
# Filename    : graph_utils.py
# Created by  : Au Jit Seah
# File owners : Au Jit Seah
##########################################################################################

import torch

"""
Returns the n_hops degree adjacency matrix adj.
"""
def get_neighborhoods(adj, n_hops, use_cuda):
    adj = torch.tensor(adj, dtype=torch.float)
    if use_cuda:
        adj = adj.cuda()
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.cpu().numpy().astype(int)
