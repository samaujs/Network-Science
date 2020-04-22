##########################################################################################
# This file provides the utilities for reading and writing checkpoints after training
# explanations (node, edge, graph)
# Filename    : io_utils.py
# Created by  : Au Jit Seah
# File owners : Au Jit Seah
##########################################################################################

import os
import torch

'''
Generate label prefix for a graph model.
'''
def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += "_" + args.method

    name += "_hdim" + str(args.hidden_dim) + "_odim" + str(args.output_dim)
    if not args.bias:
        name += "_nobias"
    if len(args.name_suffix) > 0:
        name += "_" + args.name_suffix
    return name


"""
Create filename for saving.

Args:
    args        :  the arguments parsed in the parser
    isbest      :  whether the saved model is the best-performing one
    num_epochs  :  epoch number of the model (when isbest=False)
"""
def create_filename(save_dir, args, isbest=False, num_epochs=-1):
    filename = os.path.join("./", save_dir, gen_prefix(args))
    os.makedirs(filename, exist_ok=True)

    if isbest:
        filename = os.path.join(filename, "best")
    elif num_epochs > 0:
        filename = os.path.join(filename, str(num_epochs))
    else:
        filename = os.path.join(filename, "BA_graph")

    path_filename = filename + "_model_dict.pth" # ".pth.tar"
    print("Created filename with path : ", path_filename)
    return path_filename

"""
Save pytorch model checkpoint.

Input parameters :
----------------------------------------------------------------------------------------
model         : The PyTorch model to save.
optimizer     : The optimizer used to train the model.
args          : A dict of meta-data about the model.
num_epochs    : Number of training epochs.
isbest        : True if the model has the highest accuracy so far.
cg_dict       : A dictionary of the sampled computation graphs.

Output :
----------------------------------------------------------------------------------------
filename      : File saved in "ckpt" subdirectory
"""
def save_checkpoint(model, optimizer, args, num_epochs=-1, isbest=False, cg_dict=None):
    filename = create_filename(args.ckptdir, args, isbest, num_epochs=num_epochs)
    torch.save(
        {
            "epoch": num_epochs,
            "model_type": args.method,
            "optimizer": optimizer,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cg": cg_dict,
        },
        filename
    )

'''
Load a pre-trained pytorch model from checkpoint.
'''
def load_ckpt(args, isbest=False):

    print("Attempt to load model...")
    filename = create_filename(args.ckptdir, args, isbest)
    print("Loading file : ", filename)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = torch.load(filename)
    else:
        print("Checkpoint does not exist!")
        print("Check correct path for : {}".format(filename))
        print("Make sure you have provided the correct path!")
        print("Or you may have forgotten to train a model for this dataset.")
        print()
        print("To train one of the models, run the following")
        print(">> python train.py --dataset=DATASET_NAME")
        print()
        raise Exception("File is not found.")
    return ckpt

'''
Generate label prefix for a graph explainer model.
'''
def gen_explainer_prefix(args):

    name = gen_prefix(args) + "_explain"
    if len(args.explainer_suffix) > 0:
        name += "_" + args.explainer_suffix

    print("GNN Explainer prefix :\"" + name + "\".")
    return name
