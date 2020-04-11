# Network Science
## Project 2 :
This project attempts to recreate some of the experiments carried out for explaining Graph Neural Network prediction on node/link/graph classifications.<br>
References and adaptations:<br>
[1][a link](https://cs.stanford.edu/people/jure/pubs/gnnexplainer-neurips19.pdf)<br>
[2][a link](https://github.com/RexYing/gnn-model-explainer)

(A) Run ***"train.py --gpu --input_dim --output_dim"*** will create a BA graph with attached "house" motifs that are used for node classifications.<br>
Provided arguments : <br>
(1) Provide the input parameters (eg. refer to ./train.py)<br>

Outputs : <br>
(1) Save checkpoint data file (eg. ./ckpt/BAGraph_base_hdim20_odim20/BA_graph_model_dict.pth) after training model.<br>
(2) Save log file for displaying the training informtion in tensorboard (eg. ./log/20200405-151553/events.out.tfevents.1586099753.gnn) after training model.<br>

## Approach 1 :
- Exploring Pytorch geometric GNNExplainer with Cora dataset on an GCP instance (with CUDA 1.0) using k-hop computation subgraph.
- Pytorch geometric expects the input to be based on edge index format instead of adjacency matrix.
- Understanding Message Passing with Aggregation on node classification.

## Approach 2 :
- Constructs the input graph and creates the adjacency matrix
- Creates the Convolution layers of the GNN for training
- Produces the computational graph
- Performs training for node classification
