##########################################################################################
# This is the main interface for explaining the node predictions.
# Filename    : explainer_main.py
# Created by  : Au Jit Seah
# File owners : Au Jit Seah
##########################################################################################

import os

# Own modules
import models
import utils.io_utils as io_utils
import parse_prog_args
from explainer import explain

# Start of Program from command line
def main():
    # Parsing defaults for all program parameters unless provided by user
    prog_args = parse_prog_args.arg_parse()

    # More params on top of train.py
    prog_args.graph_mode = False
    prog_args.multigraph_class=-1
    prog_args.graph_idx=-1
    prog_args.explain_node = None

    prog_args.mask_act="sigmoid"
    prog_args.mask_bias = False
    prog_args.explainer_suffix=""
    prog_args.num_epochs = 100

    path = os.path.join(prog_args.logdir, io_utils.gen_prefix(prog_args))
    print("Tensorboard writer path :\n", path)
    print("No. of epochs :", prog_args.num_epochs)

    # writer = SummaryWriter(path)

    if prog_args.gpu:
    #    os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
    #    env = os.environ.get('CUDA_VISIBLE_DEVICES')
    #    print("Environment is set :", env)
        print('\nCUDA_VISIBLE_DEVICES')
        print('------------------------------------------')
        print("CUDA", prog_args.cuda)
    else:
        print('\n------------------------------------------')
        print("Using CPU")
 
    # Loading previously saved computational graph data (model checkpoint)
    model_dict = io_utils.load_ckpt(prog_args)
    model_optimizer = model_dict['optimizer']

    print("Model optimizer :", model_optimizer)
    print("Model optimizer state dictionary :\n", model_optimizer.state_dict()['param_groups'])
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    print('------------------------------------------------------------------------------------')
    print("Keys in loaded model dictionary :", list(model_dict))
    print("Keys in loaded model optimizer dictionary:", list(model_optimizer.state_dict()))
    print("All loaded labels :\n", model_dict['cg']['label'])

    print("Default graph mode :", prog_args.graph_mode)

    # Determine explainer mode
    graph_mode = (
        prog_args.graph_mode
        or prog_args.multigraph_class >= 0
        or prog_args.graph_idx >= 0
    )

    cg_dict = model_dict['cg']
    input_dim = cg_dict['feat'].shape[2]
    num_classes = cg_dict['pred'].shape[2]
    print("Loaded model from subdirectory \"{}\" ...".format(prog_args.ckptdir))
    print("input dim: ", input_dim, "; num classes: ", num_classes)

    print('------------------------------------------------------------------------------------')
    print("Multigraph class :", prog_args.multigraph_class)
    print("Graph Index :", prog_args.graph_idx)
    print("Explainer graph mode :", graph_mode)
    print("Input dimension :", input_dim)
    print("Hidden dimension :", prog_args.hidden_dim)
    print("Output dimension :", prog_args.output_dim)
    print("Number of classes :", num_classes)
    print("Number of GCN layers :", prog_args.num_gc_layers)
    print("Batch Normalization :", prog_args.bn)

    model = models.GcnEncoderNode(input_dim=input_dim,
                                  hidden_dim=prog_args.hidden_dim,
                                  embedding_dim=prog_args.output_dim,
                                  label_dim=num_classes,
                                  num_layers=prog_args.num_gc_layers,
                                  bn=prog_args.bn,
                                  args=prog_args)

    print("\nGcnEncoderNode model :\n", model)

    # load state_dict (obtained by model.state_dict() when saving checkpoint)
    # Loading Model for Inference
    print("Model checked result :", model.load_state_dict(model_dict['model_state']))
    print('------------------------------------------------------------------------------------')

    # writer.close()


if __name__ == "__main__":
    main()

