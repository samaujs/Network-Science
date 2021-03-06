# Network Science
## Project 2 :
This project attempts to recreate some of the experiments carried out for explaining Graph Neural Network prediction on node/link/graph classifications.<br>
References and adaptations:<br>
[1]["GNNExplainer: Generating Explanations for Graph Neural Networks", Rex Ying et. al](https://cs.stanford.edu/people/jure/pubs/gnnexplainer-neurips19.pdf)<br>
[2][GNNExplainer original code](https://github.com/RexYing/gnn-model-explainer)

(A) Run ***"python3 train.py --batch_size=25 --dropout=0.001"*** will create a BA graph with attached "house" motifs that are used for node classifications.<br>
Provided arguments : <br>
(1) Provide the input parameters (eg. refer to 'python3 ./train.py -h')<br>

Outputs : <br>
(1) Save checkpoint data file (eg. ./ckpt/BAGraph_base_hdim20_odim20/BA_graph_model_dict.pth) after training model.<br>
(2) Save log file for displaying the training informtion in tensorboard (eg. ./log/20200405-151553/events.out.tfevents.1586099753.gnn) after training model.<br>

(B) Run ***"python3 explainer_main.py --explain-node=25 --epochs=101"*** will load the stored BA graph with attached "house" motifs for explaining node prediction.<br>
Provided arguments : <br>
(1) Provide the input parameters (eg. refer to 'python3 ./explainer_main.py -h')<br>

Outputs : <br>
(1) Save explanation non-masked files (eg. ./explainSubgraphs/subgraph_node_non_mask25.png) after explaining node prediction.<br>


## Approach 1 :
- Exploring Pytorch geometric GNNExplainer with Cora dataset on an GCP instance (with CUDA 1.0) using k-hop computation subgraph.
- Pytorch geometric expects the input to be based on edge index format instead of adjacency matrix.
- Understanding Message Passing with Aggregation on node classification.

## Approach 2 :
- Constructs the input graph and creates the adjacency matrix
- Creates the Convolution layers of the GNN for training
- Produces the computational graph
- Performs training for node classification
- Load saved model information
- Explains the node prediction with adjacency, edges and features maskings
- Save node explanation files
