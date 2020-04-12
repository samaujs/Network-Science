##########################################################################################
# This file performs arguments parsing for the main program, train.py
# Filename    : parse_prog_args.py
# Created by  : Au Jit Seah
# File owners : Au Jit Seah
##########################################################################################

import argparse

# Set optimizer parameters
def parse_optimizer(parser):
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
                            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
                            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
                            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
                            help='Number of epochs before decay')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
                            help='Learning rate decay ratio')
    opt_parser.add_argument('--lr', dest='lr', type=float,
                            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
                            help='Gradient clipping.')

# Set training parameters
def arg_parse():
    print("Attempt to parse arguments...")
    parser = argparse.ArgumentParser(description='GNN program arguments.')
    
    parse_optimizer(parser) # parser_utils.py in utils

    ## Add parsing arguments
    parser.add_argument('--datadir', dest='datadir',
                        help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
                        help='Tensorboard log directory')
    parser.add_argument('--ckptdir', dest='ckptdir',
                        help='Model checkpoint directory')

    # dataset is set by defaults below
    # opt and opt_scheduler defined in parse_optimizer
    parser.add_argument('--max_nodes', dest='max_nodes', type=int,
                        help='Maximum number of nodes (ignore graphs with nodes exceeding the number.')
    parser.add_argument('--cuda', dest='cuda',
                        help='CUDA.')
    parser.add_argument('--feature', dest='feature_type',
                        help='Feature used for encoder. Can be : id, deg')

    # lr and clip defined in parse_optimizer
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--train_ratio', dest='train_ratio', type=float,
                        help='Ratio of number of graphs training set to all graphs.')

    # test_ratio is set by defaults below
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--input_dim', dest='input_dim', type=int,
                        help='Input feature dimension')

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
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        help='Weight decay regularization constant.')
    parser.add_argument('--method', dest='method',
                        help='Method. Possible values: base, ')
    parser.add_argument('--name-suffix', dest='name_suffix',
                        help='suffix added to the output filename')

    # assign_ratio is set by defaults below
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--gpu', dest='gpu', action='store_const',
                        const=True, default=False, help='Whether to use GPU.')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
                        const=True, default=False,
                        help='Whether link prediction side objective is used')
    parser.add_argument('--bn', dest='bn', action='store_const',
                        const=True, default=False,
                        help='Whether batch normalization is used')

    # bmname is set by defaults below

    # Set defaults for all program parameters unless provided by user
    parser.set_defaults(datadir = "data",        # Directory where benchmark is stored (io_parser)
                        logdir = "log",          # Tensorboard log directory
                        ckptdir = "ckpt",        # Model checkpoint directory
                        dataset = "BAGraph",     # Synthetic dataset, syn1
                        opt = "adam",            # Opt parser
                        opt_scheduler = "none",  # Optimizer scheduler
                        max_nodes = 100,         # Maximum number of nodes
                                                 # (ignore graphs with nodes exceeding the number)
                        cuda = "0",              # CUDA value
                        feature_type = "default",# Feature used for encoder with possible values : id, deg
                        lr = 0.001,              # Learning rate
                        clip = 2.0,              # Gradient clipping

                        batch_size = 20,         # Batch size
                        num_epochs = 1000,       # Number of epochs to train data
                        train_ratio = 0.8,       # Ratio of number of training set to all graphs
                        test_ratio = 0.1,
                        num_workers = 1,         # Number of workers to load data
                        input_dim = 10,          # Input feature dimension
                        hidden_dim = 20,         # Hidden layer dimension
                        output_dim = 20,         # Output layer dimension
                        num_classes = 2,         # Number of label classes
                        num_gc_layers = 3,       # Number of graph convolution layers before each pooling

                        dropout = 0.0,           # Dropout rate
                        weight_decay = 0.005,    # Weight decay regularization constant
                        method = "base",         # Method used with possible values : base
                        name_suffix = "",        # Suffix added to the output filename
                        assign_ratio = 0.1,      # Ratio of number of nodes in consecutive layers

                        bias = True,             # "Whether to add bias
                        gpu = False,             # Whether to use GPU
                        linkpred = False,        # Whether link prediction side objective is used
                        bn = False,              # Whether batch normalization is used
                        bmname = None)           # Name of the benchmark datase

    return parser.parse_args()
