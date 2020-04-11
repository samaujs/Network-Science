##########################################################################################
# This file trains the GNNs for classifications, save the checkpoint to be loaded for
# explanations (node, edge, graph)
# Filename    : train.py
# Created by  : Au Jit Seah
# File owners : Au Jit Seah
##########################################################################################

import time
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim

# Own modules
import gengraph
import utils.io_utils as io_utils
import utils.featureGen as featureGen
import models

# Evaluate node classifications
def evaluate_node(ypred, labels, train_idx, test_idx):
    _, pred_labels = torch.max(ypred, 2)
    pred_labels = pred_labels.numpy()

    pred_train = np.ravel(pred_labels[:, train_idx])
    pred_test = np.ravel(pred_labels[:, test_idx])
    labels_train = np.ravel(labels[:, train_idx])
    labels_test = np.ravel(labels[:, test_idx])

    result_train = {
        "prec": metrics.precision_score(labels_train, pred_train, average="macro"),
        "recall": metrics.recall_score(labels_train, pred_train, average="macro"),
        "acc": metrics.accuracy_score(labels_train, pred_train),
        "conf_mat": metrics.confusion_matrix(labels_train, pred_train),
    }
    result_test = {
        "prec": metrics.precision_score(labels_test, pred_test, average="macro"),
        "recall": metrics.recall_score(labels_test, pred_test, average="macro"),
        "acc": metrics.accuracy_score(labels_test, pred_test),
        "conf_mat": metrics.confusion_matrix(labels_test, pred_test),
    }
    return result_train, result_test

# Train node classifier and save the prediction results
def train_node_classifier(G, labels, model, args, writer=None):
    # train/test split only for nodes
    num_nodes = G.number_of_nodes()

    # Training data with 80% ratio, labels_train.size()
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]

    # Shuffle for training
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    data = gengraph.preprocess_input_graph(G, labels)
    labels_train = torch.tensor(data["labels"][:, train_idx], dtype=torch.long)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)


#     scheduler, optimizer = train_utils.build_optimizer(
#         args, model.parameters(), weight_decay=args.weight_decay
#     )
    # list(testModel.parameters()) and list(filter_fn) to show contents
    # train_utils.build_optimizer
    filter_fn = filter(lambda p : p.requires_grad, model.parameters())

    # args.opt == 'adam':
    optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=0.0)
    scheduler = None

    # Sets the module in training mode
    model.train()
    ypred = None
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()

        if args.gpu:
            ypred, adj_att = model(x.cuda(), adj.cuda())
        else:
            ypred, adj_att = model(x, adj)
        ypred_train = ypred[:, train_idx, :]
        if args.gpu:
            loss = model.loss(ypred_train, labels_train.cuda())
        else:
            loss = model.loss(ypred_train, labels_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        #for param_group in optimizer.param_groups:
        #    print(param_group["lr"])
        elapsed = time.time() - begin_time

        # Obtain with Confusion matrices for Train and Test results
        result_train, result_test = evaluate_node(
            ypred.cpu(), data["labels"], train_idx, test_idx
        )

        if writer is not None:
            writer.add_scalar("loss/avg_loss", loss, epoch)
            writer.add_scalars(
                "prec",
                {"train": result_train["prec"], "test": result_test["prec"]},
                epoch,
            )
            writer.add_scalars(
                "recall",
                {"train": result_train["recall"], "test": result_test["recall"]},
                epoch,
            )
            writer.add_scalars(
                "acc", {"train": result_train["acc"], "test": result_test["acc"]}, epoch
            )

        if epoch % 10 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; train_acc: ",
                result_train["acc"],
                "; test_acc: ",
                result_test["acc"],
                "; train_prec: ",
                result_train["prec"],
                "; test_prec: ",
                result_test["prec"],
                "; epoch time: ",
                "{0:0.2f}".format(elapsed),
            )

        if scheduler is not None:
            scheduler.step()

    print("Confusion Matrix of train result :\n", result_train["conf_mat"])
    print("Confusion Matrix of test result :\n", result_test["conf_mat"])

    # Sets the module in evaluation mode for computational graph
    model.eval()
    if args.gpu:
        ypred, _ = model(x.cuda(), adj.cuda())
    else:
        ypred, _ = model(x, adj)

    cg_data = {
        "adj": data["adj"],
        "feat": data["feat"],
        "label": data["labels"],
        "pred": ypred.cpu().detach().numpy(),
        "train_idx": train_idx,
    }

    print("Labels of the Computational graph :\n", cg_data['label'])
    print("Prediction result of the Computational graph :\n", cg_data['pred'])
    print("Train index of the Computational graph data :\n", cg_data['train_idx'])
    # import pdb
    # pdb.set_trace()

    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)

#####################################################################################
# (1) GcnEncoderNode(GcnEncoderGraph) -> build_conv_layers -> GraphConv
# build_conv_layers return conv_first, conv_block, conv_last
# (2) GcnEncoderNode(GcnEncoderGraph) -> self.pred_model = self.build_pred_layers
# build_pred_layers return pred_model
# (3) syn_task1 -> train_node_classifier -> save_checkpoint
#####################################################################################
# Create the GCN model and encoding the nodes
def syn_task1(args, writer=None):
    # np.ones(input_dim, dtype=float) = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    constant_feature = featureGen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    print("Constant feature generator : ", constant_feature.val)

    #feat_dict = {i:{'feat': np.array(constant_feature.val, dtype=np.float32)} for i in G.nodes()}
    #print ('Values of feat_dict[0]["feat"]:', feat_dict[0]['feat'])

    #nx.set_node_attributes(G, feat_dict)
    #print('Node attributes of node \'0\', G.nodes[0]["feat"]:', G.nodes[0]['feat'])

    # Create the BA graph with the "house" motifs
    G, labels, name = gengraph.gen_syn1(feature_generator=constant_feature)

    # No .of classes from [0-3] for BA graph with house motifs
    num_classes = max(labels) + 1
    # Update number of classes in argument for training (Out of bounds error)
    args.num_classes = num_classes

    # GcnEncoderNode model
    print("------------ GCNEncoderNode Model ------------")
    print("Input dimensions :", args.input_dim)
    print("Hidden dimensions :", args.hidden_dim)
    print("Output dimensions :", args.output_dim)
    print("Number of classes in args :", args.num_classes)
    print("Number of GCN layers :", args.num_gc_layers)
    print("Method : ", args.method)

    model = models.GcnEncoderNode(args.input_dim, args.hidden_dim, args.output_dim,
                                  args.num_classes, args.num_gc_layers, bn=args.bn, args=args)

    print("GcnEncoderNode model :\n", model)


#     if args.method == "att":
#         print("Method: att")
#         model = models.GcnEncoderNode(
#             args.input_dim,
#             args.hidden_dim,
#             args.output_dim,
#             num_classes,
#             args.num_gc_layers,
#             bn=args.bn,
#             args=args,
#         )
#     else:
#         print("Method:", args.method)
#         model = models.GcnEncoderNode(
#             args.input_dim,
#             args.hidden_dim,
#             args.output_dim,
#             num_classes,
#             args.num_gc_layers,
#             bn=args.bn,
#             args=args,
#         )
    if args.gpu:
        model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)

    # To be removed after testing
    return model
