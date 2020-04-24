##########################################################################################
# This file contains functions to implement the GNNExplainer for node prediction.
# Filename    : explain.py
# Created by  : Au Jit Seah
# File owners : Au Jit Seah
##########################################################################################

# "--explain-node", dest="explain_node", type=int, help="Node to explain."
import math
import time
import os
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import statistics
import torch.optim as optim
import utils.io_utils as io_utils
import utils.graph_utils as graph_utils

# Global variables
PROJECT_ROOT_DIR = "."
SUBGRAPH_FOLDER = "explainSubgraphs"
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

class Explainer:
    def __init__(
        self,
        model,
        adj,
        feat,
        label,
        pred,
        train_idx,
        args,
        writer=None,
        print_training=True,
        graph_mode=False,
        graph_idx=False,
    ):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat
        self.label = label
        self.pred = pred
        self.train_idx = train_idx
        self.n_hops = args.num_gc_layers
        self.graph_mode = graph_mode
        self.graph_idx = graph_idx
        self.neighborhoods = None if self.graph_mode else graph_utils.get_neighborhoods(adj=self.adj, n_hops=self.n_hops, use_cuda=use_cuda)
        self.args = args
        self.writer = writer
        self.print_training = print_training


    # Main method
    """
    Explain a single node prediction
    """
    def explain(
        self, node_idx, graph_idx=0, graph_mode=False, unconstrained=False, exp_model="exp"
    ):
        print('************** Explaining node : {} **************'.format(node_idx))
        print('The label for graph index {} and node index {} : {}'.format(graph_idx, node_idx, self.label[graph_idx][node_idx]))
        print("Labels of all the nodes :\n", self.label)

        # Adjacency matrix of entire graph
        print("Shape of retrieved neighborhoods :", self.neighborhoods.shape)
        print("No. of neighborhoods :", len(self.neighborhoods[graph_idx][node_idx]))
        print('List of neighborhoods for explaining node {} :'.format(node_idx))
        print(self.neighborhoods[graph_idx][node_idx])

        # index of the query node in the new adj
        if graph_mode:
            node_idx_new = node_idx
            sub_adj = self.adj[graph_idx]
            sub_feat = self.feat[graph_idx, :]
            sub_label = self.label[graph_idx]
            neighbors = np.asarray(range(self.adj.shape[0]))
        else:
            print("Ground truth, node label :", self.label[graph_idx][node_idx])
            # Computational graph :
            # Extracting subgraph adjacency matrix, subgraph features, subgraph labels and the nodes neighbours
            node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                node_idx, graph_idx
            )
            sub_label = np.expand_dims(sub_label, axis=0)

        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)

        print("Neighbouring graph index for node " + str(node_idx) + " with new node index " + str(node_idx_new))
        #print("Expand dimension of Subgraph adjacency :\n", sub_adj)
        #print("Expand dimension of Subgraph features :\n", sub_feat)
        print("Expand dimension of Subgraph label :\n", sub_label)

        # All the nodes in the graph (eg. indexes from 0 to 34)
        print("Subgraph neighbors :\n", neighbors)
        tensor_adj   = torch.tensor(sub_adj, dtype=torch.float)
        tensor_x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        tensor_label = torch.tensor(sub_label, dtype=torch.long)

        if self.graph_mode:
            pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
            print("Graph predicted label: ", pred_label)
        else:
            pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)
            print("Neighbours of predicted node labels :", self.pred[graph_idx][neighbors])
            print('Predicted labels for all {} neighbours (includes itself) :\n{}'.format(len(pred_label), pred_label))
            print('Predicted label for node {} : {}'.format(node_idx, pred_label[node_idx_new]))

        # Have to use the tensor version of adj for Tensor computation
        explainerMod = ExplainModule(
            adj=tensor_adj,                # adj
            x=tensor_x,                    # x
            model=self.model,              # model
            label=tensor_label,            # label
            args=self.args,                # prog_args
            writer=self.writer,            # None
            graph_idx=self.graph_idx,      # graph_idx
            graph_mode=self.graph_mode     # graph_mode
        )

        self.model.eval()
        explainerMod.train()
        begin_time = time.time()

        # prog_args.num_epochs
        for epoch in range(self.args.num_epochs):
            explainerMod.zero_grad()
            explainerMod.optimizer.zero_grad()

            # node_idx_new is passed to explainerMod.forward to training with the new index
            ypred, adj_atts = explainerMod(node_idx_new, unconstrained=unconstrained)
            loss = explainerMod.loss(ypred, pred_label, node_idx_new, epoch)
            loss.backward()

            explainerMod.optimizer.step()
            mask_density = explainerMod.mask_density()

            print("epoch: ",
                  epoch,
                  "; loss: ",
                  loss.item(),
                  "; mask density: ",
                  mask_density.item(),
                  "; pred: ",
                  ypred)
            print("------------------------------------------------------------------")

            if exp_model != "exp":
                break


        print("\n--------------------------------------------")
        print("Final ypred after training : ", ypred)
        print("pred_label : ", pred_label)
        print("node_idx_new : ", node_idx_new)

        print("Completed training in ", time.time() - begin_time)

        if exp_model == "exp":
            masked_adj = (
                explainerMod.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
            )

            # Added for plotting node explanation subgraph
            # explainerMod.mask.shape, masked_edges.shape
            masked_edges = explainerMod.mask.cpu().detach().numpy()
            masked_features = explainerMod.feat_mask.cpu().detach().numpy()
            # explainerMod.feat_mask.shape, masked_features.shape

            ypred_detach = ypred.cpu().detach().numpy()
            ypred_node = np.argmax(ypred_detach, axis=0) # labels

            # ypred = tensor([0.0119, 0.6456, 0.3307, 0.0118]
            print('Detach ypred : {} and Argmax node : {}'.format(ypred_detach, ypred_node))

        # Trained masked, edges and features adjacency matrices
        print("Shape of masked adjacency matrix : ", masked_adj.shape)
        print("The masked adjacency matrix at index [0] :\n", masked_adj[0])
        print("Shape of masked edges matrix : ", masked_edges.shape)
        print("The masked edges adjacency matrix at index [0] :\n", masked_edges[0])
        print("Shape of masked features matrix : ", masked_features.shape)
        print("The masked features adjacency matrix at index [0] :\n", masked_features[0])

        fname = 'masked_adj_' + io_utils.gen_explainer_prefix(self.args) + (
                '_node_idx_'+str(node_idx)+'_graph_idx_'+str(self.graph_idx)+'.npy')
        with open(os.path.join(self.args.logdir, fname), 'wb') as outfile:
            np.save(outfile, np.asarray(masked_adj.copy()))
            print("Saved adjacency matrix to \"" + fname + "\".")


        # PlotSubGraph (sub_edge_index not used)
        self.PlotSubGraph(masked_adj, masked_edges, node_idx_new, node_idx,
                          feats=sub_feat.squeeze(), labels=tensor_label.cpu().detach().numpy().squeeze(),
                          threshold_num=12, adj_mode=True)

        # Shape of masked adjacency matrix : (27, 27)
        # Shape of masked edges matrix : (27, 27)
        # Shape of masked features matrix : (10,)
        return masked_adj, masked_edges, masked_features

    # Utilities
    """
    Returns the neighborhood of a given node.
    """
    def extract_neighborhood(self, node_idx, graph_idx=0):
        print('node_idx : {} and graph_idx : {}'.format(node_idx, graph_idx))
        # neighborhoods[graph_idx][node_idx] gives same result
        neighbors_adj_row = self.neighborhoods[graph_idx][node_idx, :]
        print("Total no. of neighborhoods adj in row :", len(neighbors_adj_row))
        print("No. of neighborhoods adj in row up to node index :", len(neighbors_adj_row[:node_idx]))
        print("List of neighborhoods up to node index", node_idx)
        print(neighbors_adj_row[:node_idx])

        # Gives the index of the query node in the subgraph adjacency matrix
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        print("List of neighborhoods :\n", neighbors)
        sub_adj = self.adj[graph_idx][neighbors][:, neighbors]
        sub_feat = self.feat[graph_idx, neighbors]
        sub_label = self.label[graph_idx][neighbors]
        print("Returning new node index :", node_idx_new)

        # Returns new node index, adjacency, features abd neighbors of the subgraph
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors

    """
    Plot the subgraph for explaining the given node.
    """
    # edge_index is omitted
    # Standalone : Need to remove self or it will be initialized with zero values of the variables due to __init__
    def PlotSubGraph(self, adj, masked_edges, new_node_idx, node_idx, feats=None, labels=None, threshold_num=None, adj_mode=True):
        G = nx.Graph()
        G1 = nx.Graph()

        weighted_edge_list = []
        weighted_edge_list1 = []
        if threshold_num is not None:
            if adj_mode:
                # this is for symmetric graphs: edges are repeated twice in adj
                adj_threshold_num = threshold_num * 2
                neigh_size = len(adj[adj > 0])
                threshold_num = min(neigh_size, adj_threshold_num)
                # sort in ascending and retrieve last threshold_num from the back
                threshold = np.sort(adj[adj > 0])[-threshold_num]

                num_nodes = len(adj[0])
                weighted_edge_list = [
                    (i, j, adj[i, j])
                    for i in range(num_nodes)
                    for j in range(num_nodes)
                    # if adj[i, j] >= threshold
                    if adj[i,j] > 0
                ]

                weighted_edge_list1 = [
                    (i, j, adj[i, j])
                    for i in range(num_nodes)
                    for j in range(num_nodes)
                    if adj[i, j] >= threshold
                ]
            else:
                threshold = np.sort(masked_edges)[-threshold_num]
                row, col = edge_index

                for i in range(len(row)):
                    if masked_edges[i] >= threshold:
                        weighted_edge_list.append((row[i], col[i], masked_edges[i]))


        G.add_weighted_edges_from(weighted_edge_list)

        G1.add_weighted_edges_from(weighted_edge_list1)

        if feats is not None:
            for node in G.nodes():
                G.nodes[node]["feat"] = feats[node]
            for node in G1.nodes():
                G1.nodes[node]["feat"] = feats[node]

        node_labels = {}
        node_labels1 = {}
        if labels is not None:
            for node in G.nodes():
                G.nodes[node]["label"] = labels[node]
                node_labels.update({node: labels[node]})
            for node in G1.nodes():
                G1.nodes[node]["label"] = labels[node]
                node_labels1.update({node: labels[node]})

        node_color = []
        for node in G.nodes.data():
            if node[0] == new_node_idx:
                node_color.append('green')
            else:
                node_color.append('lightgrey')

        node_color1 = []
        for node in G1.nodes.data():
            if node[0] == new_node_idx:
                node_color1.append('orange')
            else:
                node_color1.append('grey')


        edge_colors = [w for (u, v, w) in G.edges.data("weight", default=1)]
        edge_vmax = statistics.median_high(
            [d for (u, v, d) in G.edges(data="weight", default=1)]
        )
        min_color = min([d for (u, v, d) in G.edges(data="weight", default=1)])
        # color range: gray to black
        edge_vmin = 2 * min_color - edge_vmax

        edge_colors1 = [w for (u, v, w) in G1.edges.data("weight", default=1)]
        edge_vmax1 = statistics.median_high(
            [d for (u, v, d) in G1.edges(data="weight", default=1)]
        )
        min_color1 = min([d for (u, v, d) in G1.edges(data="weight", default=1)])
        # color range: gray to black
        edge_vmin1 = 2 * min_color1 - edge_vmax1

        #plt.close()
        plt.figure()
        pos = nx.spring_layout(G, k=0.5)
        nx.draw_networkx_nodes(G, pos, node_color=node_color)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=plt.get_cmap("Greys"),
                               edge_vmin=edge_vmin, edge_vmax=edge_vmax)
        nx.draw_networkx_labels(G, pos, labels=node_labels)
        plt.axis('off')

        # Create subdirectory if does not already exist
        data_path = os.path.join(PROJECT_ROOT_DIR, SUBGRAPH_FOLDER)
        if not os.path.isdir(data_path):
            os.makedirs(data_path)
        # plot G into plt and highlist node_idx in the explanation
        plt.savefig('{}/subgraph_node_non_mask{}.png'.format(data_path, node_idx))

        #plt.close()
        plt.figure()
        pos = nx.spring_layout(G1, k=0.5)
        nx.draw_networkx_nodes(G1, pos, node_color=node_color1)
        nx.draw_networkx_edges(G1, pos, edge_color=edge_colors1, edge_cmap=plt.get_cmap("Greys"),
                               edge_vmin=edge_vmin1, edge_vmax=edge_vmax1)
        nx.draw_networkx_labels(G1, pos, labels=node_labels1)
        plt.axis('off')
        # plot G into plt and highlist node_idx to be explained as red
        plt.savefig('{}/subgraph_node_masked_edges{}.png'.format(data_path, node_idx))

        return None


class ExplainModule(nn.Module):
    def __init__(
        self,
        adj,
        x,
        model,
        label,
        args,
        graph_idx=0,
        writer=None,
        use_sigmoid=True,
        graph_mode=False,
    ):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.graph_idx = graph_idx
        self.args = args
        self.writer = writer
        self.mask_act = args.mask_act
        self.use_sigmoid = use_sigmoid
        self.graph_mode = graph_mode

        init_strategy = "normal"
        num_nodes = adj.size()[1]

        print("Explain module for {} nodes.".format(num_nodes))
        print("Explain module with model :\n", model)

        # Constructs edge mask
        self.mask, self.mask_bias = self.construct_edge_mask(
            num_nodes, init_strategy=init_strategy
        )

        # Constructs feature mask
        self.feat_mask = self.construct_feat_mask(x.size(-1), init_strategy="constant")
        params = [self.mask, self.feat_mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)

        # For masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        if args.gpu:
            self.diag_mask = self.diag_mask.cuda()

        # train_utils.build_optimizer
        #self.scheduler, self.optimizer = build_optimizer(args, params)

        # Insert for optimizer and scheduler
        filter_fn = filter(lambda p : p.requires_grad, params)
        self.optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=0.0)
        self.scheduler = None

        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,
        }

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
                # mask[0] = 2
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def _masked_adj(self):
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.args.gpu else self.adj
        masked_adj = adj * sym_mask
        if self.args.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        return masked_adj * self.diag_mask

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def forward(self, node_idx, unconstrained=False, mask_features=True, marginalize=False):
        print("Started Training ... for :", node_idx)

        x = self.x.cuda() if self.args.gpu else self.x

        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = (
                torch.unsqueeze((sym_mask + sym_mask.t()) / 2, 0) * self.diag_mask
            )
        else:
            self.masked_adj = self._masked_adj()
            if mask_features:
                feat_mask = (
                    torch.sigmoid(self.feat_mask)
                    if self.use_sigmoid
                    else self.feat_mask
                )
                if marginalize:
                    std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                    mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                    z = torch.normal(mean=mean_tensor, std=std_tensor)
                    x = x + z * (1 - feat_mask)
                else:
                    x = x * feat_mask

        ypred, adj_att = self.model(x, self.masked_adj)
        if self.graph_mode:
            res = nn.Softmax(dim=0)(ypred[0])
        else:
            node_pred = ypred[self.graph_idx, node_idx, :]
            res = nn.Softmax(dim=0)(node_pred)

        print("Node prediction with softmax after one epoch :", res)
        return res, adj_att

    def adj_feat_grad(self, node_idx, pred_label_node):
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
            self.x.grad.zero_()
        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
            label = self.label.cuda()
        else:
            x, adj = self.x, self.adj
        ypred, _ = self.model(x, adj)
        if self.graph_mode:
            logit = nn.Softmax(dim=0)(ypred[0])
        else:
            logit = nn.Softmax(dim=0)(ypred[self.graph_idx, node_idx, :])
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        return self.adj.grad, self.x.grad

    def loss(self, pred, pred_label, node_idx, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        mi_obj = False
        if mi_obj:
            pred_loss = -torch.sum(pred * torch.log(pred))
        else:
            pred_label_node = pred_label if self.graph_mode else pred_label[node_idx]
            gt_label_node = self.label if self.graph_mode else self.label[0][node_idx]
            logit = pred[gt_label_node]
            pred_loss = -torch.log(logit)
        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs["size"] * torch.sum(mask)

        # pre_mask_sum = torch.sum(self.feat_mask)
        feat_mask = (
            torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        )
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # entropy
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        feat_mask_ent = - feat_mask             \
                        * torch.log(feat_mask)  \
                        - (1 - feat_mask)       \
                        * torch.log(1 - feat_mask)

        feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.mean(feat_mask_ent)

        # laplacian
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        m_adj = self.masked_adj if self.graph_mode else self.masked_adj[self.graph_idx]
        L = D - m_adj
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            L = L.cuda()
        if self.graph_mode:
            lap_loss = 0
        else:
            lap_loss = (self.coeffs["lap"]
                * (pred_label_t @ L @ pred_label_t)
                / self.adj.numel()
            )

        # grad
        # adj
        # adj_grad, x_grad = self.adj_feat_grad(node_idx, pred_label_node)
        # adj_grad = adj_grad[self.graph_idx]
        # x_grad = x_grad[self.graph_idx]
        # if self.args.gpu:
        #    adj_grad = adj_grad.cuda()
        # grad_loss = self.coeffs['grad'] * -torch.mean(torch.abs(adj_grad) * mask)

        # feat
        # x_grad_sum = torch.sum(x_grad, 1)
        # grad_feat_loss = self.coeffs['featgrad'] * -torch.mean(x_grad_sum * mask)

        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss
        if self.writer is not None:
            self.writer.add_scalar("optimization/size_loss", size_loss, epoch)
            self.writer.add_scalar("optimization/feat_size_loss", feat_size_loss, epoch)
            self.writer.add_scalar("optimization/mask_ent_loss", mask_ent_loss, epoch)
            self.writer.add_scalar(
                "optimization/feat_mask_ent_loss", mask_ent_loss, epoch
            )
            # self.writer.add_scalar('optimization/grad_loss', grad_loss, epoch)
            self.writer.add_scalar("optimization/pred_loss", pred_loss, epoch)
            self.writer.add_scalar("optimization/lap_loss", lap_loss, epoch)
            self.writer.add_scalar("optimization/overall_loss", loss, epoch)
        return loss

    def log_mask(self, epoch):
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(4, 3), dpi=400)
        plt.imshow(self.mask.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image(
            "mask/mask", tensorboardX.utils.figure_to_image(fig), epoch
        )

        # fig = plt.figure(figsize=(4,3), dpi=400)
        # plt.imshow(self.feat_mask.cpu().detach().numpy()[:,np.newaxis], cmap=plt.get_cmap('BuPu'))
        # cbar = plt.colorbar()
        # cbar.solids.set_edgecolor("face")

        # plt.tight_layout()
        # fig.canvas.draw()
        # self.writer.add_image('mask/feat_mask', tensorboardX.utils.figure_to_image(fig), epoch)
        io_utils.log_matrix(
            self.writer, torch.sigmoid(self.feat_mask), "mask/feat_mask", epoch
        )

        fig = plt.figure(figsize=(4, 3), dpi=400)
        # use [0] to remove the batch dim
        plt.imshow(self.masked_adj[0].cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image(
            "mask/adj", tensorboardX.utils.figure_to_image(fig), epoch
        )

        if self.args.mask_bias:
            fig = plt.figure(figsize=(4, 3), dpi=400)
            # use [0] to remove the batch dim
            plt.imshow(self.mask_bias.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
            cbar = plt.colorbar()
            cbar.solids.set_edgecolor("face")

            plt.tight_layout()
            fig.canvas.draw()
            self.writer.add_image(
                "mask/bias", tensorboardX.utils.figure_to_image(fig), epoch
            )

    def log_adj_grad(self, node_idx, pred_label, epoch, label=None):
        log_adj = False

        if self.graph_mode:
            predicted_label = pred_label
            # adj_grad, x_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[0]
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[0]
            x_grad = torch.sum(x_grad[0], 0, keepdim=True).t()
        else:
            predicted_label = pred_label[node_idx]
            # adj_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[self.graph_idx]
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[self.graph_idx]
            x_grad = x_grad[self.graph_idx][node_idx][:, np.newaxis]
            # x_grad = torch.sum(x_grad[self.graph_idx], 0, keepdim=True).t()
        adj_grad = (adj_grad + adj_grad.t()) / 2
        adj_grad = (adj_grad * self.adj).squeeze()
        if log_adj:
            io_utils.log_matrix(self.writer, adj_grad, "grad/adj_masked", epoch)
            self.adj.requires_grad = False
            io_utils.log_matrix(self.writer, self.adj.squeeze(), "grad/adj_orig", epoch)

        masked_adj = self.masked_adj[0].cpu().detach().numpy()

        # only for graph mode since many node neighborhoods for syn tasks are relatively large for
        # visualization
        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj, node_idx, feat=self.x[0], threshold=None, max_component=False
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph_orig",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        io_utils.log_matrix(self.writer, x_grad, "grad/feat", epoch)

        adj_grad = adj_grad.detach().numpy()
        if self.graph_mode:
            print("GRAPH model")
            G = io_utils.denoise_graph(
                adj_grad,
                node_idx,
                feat=self.x[0],
                threshold=0.0003,  # threshold_num=20,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        else:
            # G = io_utils.denoise_graph(adj_grad, node_idx, label=label, threshold=0.5)
            G = io_utils.denoise_graph(adj_grad, node_idx, threshold_num=12)
            io_utils.log_graph(
                self.writer, G, name="grad/graph", epoch=epoch, args=self.args
            )

        # if graph attention, also visualize att

    def log_masked_adj(self, node_idx, epoch, name="mask/graph", label=None):
        # use [0] to remove the batch dim
        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj,
                node_idx,
                feat=self.x[0],
                threshold=0.2,  # threshold_num=20,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=False,
                nodecolor="feat",
                epoch=epoch,
                label_node_feat=True,
                edge_vmax=None,
                args=self.args,
            )
        else:
            G = io_utils.denoise_graph(
                masked_adj, node_idx, threshold_num=12, max_component=True
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=True,
                nodecolor="label",
                epoch=epoch,
                edge_vmax=None,
                args=self.args,
            )
