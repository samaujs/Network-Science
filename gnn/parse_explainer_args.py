##########################################################################################
# This file performs arguments parsing for the main program, explainer_main.py
# Filename    : parse_explainer_args.py
# Created by  : Au Jit Seah
# File owners : Au Jit Seah
##########################################################################################

import argparse
import utils.parser_utils as parser_utils


# Set training parameters for GNN Explainer
def arg_parse():
    print("Attempt to parse arguments for GNN Explainer ...")
    parser = argparse.ArgumentParser(description='GNN program arguments.')
    
    # Set optimizer parameters
    parser_utils.parse_optimizer(parser)

    ## Add parsing arguments
    parser.add_argument('--logdir', dest='logdir',
                        help='Tensorboard log directory')
    parser.add_argument('--ckptdir', dest='ckptdir',
                        help='Model checkpoint directory')

    # dataset is set by defaults below
    # opt and opt_scheduler defined in parse_optimizer
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    # lr and clip defined in parse_optimizer

    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.')

    # Cannot use "parse_prog.py --h" causes "error: ambiguous" (Usage: parse_prog.py --hidden_dim=30)
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output_dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--num_classes', dest='num_classes', type=int,
                        help='Number of label classes')
    parser.add_argument('--num_gc_layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling')

    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--method', dest='method',
                        help='Method. Possible values: base, att.')
    parser.add_argument('--name-suffix', dest='name_suffix',
                        help='suffix added to the output filename')

    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--gpu', dest='gpu', action='store_const',
                        const=True, default=False, help='Whether to use GPU.')
    parser.add_argument('--bn', dest='bn', action='store_const',
                        const=True, default=False,
                        help='Whether batch normalization is used')

    # bmname is set by defaults below

    parser.add_argument("--no-writer", dest="writer", action="store_const",
                        const=False, default=True,
                        help="Whether to add bias. Default to True.")

    # Parameters for Explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, 
                        help="sigmoid, ReLU.")
    parser.add_argument("--mask-bias", dest="mask_bias", action="store_const",
                        const=True, default=False,
                        help="Whether to add bias. Default to True.")
    parser.add_argument("--explain-node", dest="explain_node", type=int,
                        help="Node to explain.")
    parser.add_argument("--graph-idx", dest="graph_idx", type=int,
                        help="Graph to explain.")
    parser.add_argument("--graph-mode", dest="graph_mode", action="store_const",
                        const=True, default=False,
                        help="Whether to run Explainer on Graph Classification task.")
    parser.add_argument("--multigraph-class", dest="multigraph_class", type=int,
                        help="Whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.")
    parser.add_argument("--multinode-class", dest="multinode_class", type=int,
                        help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.")

    parser.add_argument("--explainer-suffix", dest="explainer_suffix",
                        help="suffix added to the explainer log")

    # Set defaults for all program parameters unless provided by user
    parser.set_defaults(logdir = "log",          # Tensorboard log directory
                        ckptdir = "ckpt",        # Model checkpoint directory
                        dataset = "BAGraph",     # Synthetic dataset, syn1
                        opt = "adam",            # Opt parser
                        opt_scheduler = "none",  # Optimizer scheduler
                        cuda = "0",              # CUDA value
                        lr = 0.1,                # Learning rate (train : 0.001)
                        clip = 2.0,              # Gradient clipping

                        batch_size = 20,         # Batch size
                        num_epochs = 100,        # Number of epochs for explainer training (train : 1000)
                        hidden_dim = 20,         # Hidden layer dimension
                        output_dim = 20,         # Output layer dimension
                        num_classes = 2,         # Number of label classes
                        num_gc_layers = 3,       # Number of graph convolution layers before each pooling

                        dropout = 0.0,           # Dropout rate
                        method = "base",         # Method used with possible values : base
                        name_suffix = "",        # Suffix added to the output filename

                        bias = True,             # "Whether to add bias
                        gpu = False,             # Whether to use GPU
                        bn = False,              # Whether batch normalization is used
                        bmname = None,           # Name of the benchmark datase

                        explainer_suffix="",     # Suffix added to the explainer log
                        explain_node = 25,       # Node to explain (eg. Check is on None)
                        graph_idx = -1,          # Graph to explain
                        mask_act="sigmoid",      # Type of activation for mask (sigmoid, ReLU)
                        multigraph_class = -1,   # Whether to run Explainer on multiple Graphs from the Classification task for examples in the same class
                        multinode_class=-1)      # Whether to run Explainer on multiple nodes from the Classification task for examples in the same class

    return parser.parse_args()
