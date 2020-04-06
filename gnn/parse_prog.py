# configs.py - Parsing arguments
import argparse
import os

def parse_optimizer(parser):
    '''Set optimizer parameters'''
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

def arg_parse():
    print("Attempt to parse arguments...")
    parser = argparse.ArgumentParser(description='GNN project arguments.')
    
    parse_optimizer(parser) # parser_utils.py in utils

    parser.add_argument('--gpu', dest='gpu', action='store_const',
                        const=True, default=False, help='whether to use GPU.')
    
    parser.set_defaults(datadir='data',       # io_parser
                        logdir='log',
                        ckptdir='ckpt',
                        dataset='syn1',
                        opt='adam',           # opt_parser
                        opt_scheduler='none',
                        max_nodes=100,
                        cuda='0',             # 1
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=20,
                        num_epochs=1000,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=2,
                        num_gc_layers=3,
                        dropout=0.0,
                        weight_decay=0.005,
                        method='base',
                        name_suffix='',
                        assign_ratio=0.1)
    
    return parser.parse_args()


def main():
    prog_args = arg_parse()
    print(prog_args)

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        env = os.environ.get('CUDA_VISIBLE_DEVICES')
        print("ENV set", env)
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")


if __name__ == "__main__":
    main()
