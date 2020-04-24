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
import parse_explainer_args
from explainer import explain

# Start of Program from command line
def main():
    # Parsing defaults for all program parameters unless provided by user
    prog_args = parse_explainer_args.arg_parse()

    # More params on top of train.py
    prog_args.writer = False
    prog_args.graph_mode = False
    prog_args.multigraph_class=-1
    prog_args.graph_idx=-1
    prog_args.explain_node = None

    prog_args.mask_act="sigmoid"
    prog_args.mask_bias = False
    prog_args.explainer_suffix=""

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

    # Trained data stored in computational graph dictionary
    cg_dict = model_dict['cg']
    input_dim = cg_dict['feat'].shape[2]
    num_classes = cg_dict['pred'].shape[2]
    print("Loaded model from subdirectory \"{}\" ...".format(prog_args.ckptdir))
    print("input dim : ", input_dim, "; num classes : ", num_classes)
    print("Labels of retrieved data :\n", cg_dict['label'])

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
    print('------------------------------------------------------------------------------------\n')

    # Explaining single node prediction 
    #prog_args.explain_node = 25
    # The number of epochs used for explanation training is much smaller than the 1K epochs used for node label
    # trainings and predictions in the GCN.  The former is trained only based on the k-hop labels which depends
    # on the number GCN layers (at a smaller scale, so the number of epochs can be lower without reducing the
    # accuracy). Whereas, the latter will affect the node predictions and thus, it will affect the accuracy of
    # the node explanations.
    #prog_args.num_epochs = 100
    print('GNN Explainer is trained based on {} epochs.'.format(prog_args.num_epochs))

    # Create explainer
    explainer = explain.Explainer(
        model=model,
        adj=cg_dict["adj"],
        feat=cg_dict["feat"],
        label=cg_dict["label"],
        pred=cg_dict["pred"],
        train_idx=cg_dict["train_idx"],
        args=prog_args,
        writer=prog_args.writer,
        print_training=True,
        graph_mode=graph_mode,
        graph_idx=prog_args.graph_idx,
    )
    
    if prog_args.explain_node is not None:
        # Returned masked adjacency, edges and features of the subgraph
        masked_adj, masked_edges, masked_features = explainer.explain(prog_args.explain_node, unconstrained=False)

        print("Returned masked adjacency matrix :\n", masked_adj)
        print("Returned masked edges matrix :\n", masked_edges)
        print("Returned masked features matrix :\n", masked_features)
    else:
        print("Please provide node for explanation.")

    # writer.close()


if __name__ == "__main__":
    main()

