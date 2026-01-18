import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from collections import Counter
from sklearn.model_selection import train_test_split
import time
import sys
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
import pickle
import networkx as nx
from torch_geometric.utils import from_networkx

#### Process data files ####
def get_gene_idx_dict_from_file(node_file_name):
    f = open(node_file_name, "r")
    gene_idx_dict = {}
    idx = 0
    for line in f:
        node = line.strip()
        gene_idx_dict[node] = idx
        idx += 1
    f.close()
    return gene_idx_dict

def get_disease_sets(file_path):
    dis_pairs = []   #[(disA, disB), ...]
    labels = []      # [label, ...]
    disease_genes_dict = {}     #{disease: [gene_1, gene_2, ...]}

    f = open(file_path, "r")
    head = True
    for line in f:
        if head:
            head = False
            continue

        row = line.strip().split("\t")
        dis_pair, disease_a_genes, disease_b_genes, all_genes, rr = row

        disease_a, disease_b = dis_pair.split("&")

        dis_pairs.append((disease_a, disease_b))
        labels.append(int(rr))

        disease_genes_dict[disease_a] = [int(gene) for gene in disease_a_genes.split(",")]
        disease_genes_dict[disease_b] = [int(gene) for gene in disease_b_genes.split(",")]


    f.close()

    return dis_pairs, labels, disease_genes_dict

def get_disease_pair_rr_list(dis_pairs, labels, disease_genes_dict, node_idx_dict):
    disease_pair_rr = []
    for idx in range(len(dis_pairs)):
        disease_a, disease_b = dis_pairs[idx]
        gene_list = disease_genes_dict[disease_a] + disease_genes_dict[disease_b]
        gene_list = [node_idx_dict[str(node)] for node in gene_list if str(node) in node_idx_dict]
        RR = labels[idx]
        disease_pair_rr.append([gene_list, RR])
    return disease_pair_rr

def get_graph_from_file_and_map_ids(network_file, node_dict, **kwargs):
    """
        generate a graph based on the input file
        The input file is provided by Joerg Menche et al. in their paper's supplementary
        Thus modify their function to parse the file and get the graph
        The function returns:
        G: the graph with self loop removed
    """

    defaultKwargs = {'self_link': True}
    kwargs = { **defaultKwargs, **kwargs}

    G = nx.Graph()
    network_file = open(network_file,'r')
    for line in network_file:
        # lines starting with '#' will be ignored
        if line[0]=='#':
            continue
        line_data   = line.strip().split('\t')
        gene1 = line_data[0]
        gene2 = line_data[1]

        G.add_edge(node_dict[gene1],node_dict[gene2])

    # remove self links
    if not kwargs['self_link']:
        remove_self_links(G)
    return G
#------------------------------------------------------------------------------#
def remove_self_links(G):
    sl = nx.selfloop_edges(G)
    G.remove_edges_from(sl)

#------------------------------------------------------------------------------#
def file_to_matrix(file):
    matrix = np.loadtxt(file, delimiter='\t')
    return matrix

####------- Model ------####
# GGAT + GRU for Disease Pair Prediction (RR labels) with Attention Pooling + Validation (yes/no)
"""
  Node Features + Graph → GGATGRU → Node Embeddings
                                  ↓
                            AttentionPooling
                                  ↓
                       Fully-Connected Predictor → Binary label
"""


# GRU
class GRUSubLayer(nn.Module):
    """
      define layer GRUSub:
        h_dim: dimensions of the hidden state vector h
        h_in_dim: dimensions of the input vector h_in
    """
    def __init__(self, h_dim, h_in_dim):
        super().__init__() # Initializes the parent class nn.Module, allows to use .to(), .eval(), etc. those build-in functions
        self.reset_gate = nn.Sequential(nn.Linear(h_dim + h_in_dim, h_in_dim), nn.Sigmoid())
        self.update_gate = nn.Sequential(nn.Linear(h_dim + h_in_dim, h_in_dim), nn.Sigmoid())
        self.transform = nn.Sequential(nn.Linear(h_dim + h_in_dim, h_in_dim), nn.Tanh())  # GRU uses Tanh

    def forward(self, h, h_in):
        # Concatenates h and h_in then used as input for the gates
        # dim = 0, concat batch dim; dim = 1, concat features
        a = torch.cat((h, h_in), dim=1)
        r = self.reset_gate(a) # apply reset gate to attentions a
        z = self.update_gate(a) # apply update gate to attentions a
        joined = torch.cat((h, r * h_in), dim=1) # Element-wise multiplication to get the weighted h_in and concat with h
        h_hat = self.transform(joined)  # apply Tanh
        return (1 - z) * h_in + z * h_hat # GRU

# GGAT + GRU
class GGATGRU(nn.Module):
    """
      defube module GGATGRU:
        input -> input_droupout -> GATConv1 (with gat_dropout) -> ELU activation -> out1
              |                                             |
              --------------------------------------------------> GRU1 -> x1

          x1  -> input_droupout -> GATConv2 (with gat_dropout) -> ELU activation -> Linear(dimTrans) -> out2
              |                                                                 |
              ----------------------------------------------------------------------> GRU2 -> x2

          x2  -> input_droupout -> GATConv3 (with gat_dropout) -> ELU activation -> Linear(dimTrans) -> out3
              |                                                                 |
              ----------------------------------------------------------------------> GRU3 -> x3
      features:
        input_dropout:
          For rich input features to prevents the model from overly relying on specific input feature dimensions and help generalize better.
          Set to 0 when the input is already highly sparse, doing so may discard too much information by ignoring critical ones for the task
        heads: higher is better to learn diverse attention but higher memory.
        gat_dropout: Encourages to consider a wider variety of neighbors rather than fixating on a few strong edges. Used when the task is binary classification and sensitive to overconfidence.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, input_dropout = 0.2, gat_dropout=0.4):
        super().__init__()
        self.heads1 = heads                          # original
        self.heads2 = round(heads / 2)              # reduced
        self.input_dropout = input_dropout
        self.gat_dropout = gat_dropout
        self.heads = heads
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels

        self.transform = nn.Linear(in_channels, hidden_channels * heads) if in_channels != hidden_channels * heads else nn.Identity()

        self.gat1 = GATConv(hidden_channels * self.heads1, hidden_channels, heads=self.heads1, dropout=gat_dropout)       # GAT1 input dim: hidden_channels * heads1, output dim: hidden_channels * heads1
        self.gat2 = GATConv(hidden_channels * self.heads1, hidden_channels, heads=self.heads2, dropout=gat_dropout)       # GAT2 input dim: hidden_channels * heads2, output dim: hidden_channels * heads2
        self.gat3 = GATConv(hidden_channels * self.heads2, out_channels, heads=1, dropout=gat_dropout)

        # GRU input/output alignment
        self.gru1 = GRUSubLayer(h_dim=hidden_channels * self.heads1, h_in_dim=hidden_channels * self.heads1)
        # change dimensions: [N, hidden_channels * heads] -> [N, hidden_channels]
        self.pre_gat2 = nn.Linear(hidden_channels * self.heads1, hidden_channels * self.heads2)
        self.gru2 = GRUSubLayer(h_dim=hidden_channels * self.heads2, h_in_dim=hidden_channels * self.heads2)

        # self.trans3 = nn.Linear(hidden_channels, hidden_channels)
        self.pre_gat3 = nn.Linear(hidden_channels * self.heads2, out_channels)
        self.gru3 = GRUSubLayer(h_dim=out_channels, h_in_dim=out_channels)

    def forward(self, x, edge_index):
        x = self.transform(x)

        h1 = x
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = self.gru1(x, h1)  # after gru1: [N, hidden_channels * heads]

        h2 = x # h2: [N, hidden_channels * heads]
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index)) # now, x after gat2: [N, hidden_channels]
        x = self.gru2(x, self.pre_gat2(h2)) # need to transfer h2's dimentions to the same as x

        # h3 = self.trans3(x)
        # x = self.pre_gat3(x)
        h3 = x
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.gat3(x, edge_index)
        x = self.gru3(x, self.pre_gat3(h3))
        return x

# RELUGate
class RELUSubLayer(nn.Module):
    """
      define layer GRUSub:
        h_dim: dimensions of the hidden state vector h
        h_in_dim: dimensions of the input vector h_in
    """
    def __init__(self, h_dim, h_in_dim):
        super().__init__() # Initializes the parent class nn.Module, allows to use .to(), .eval(), etc. those build-in functions
        self.reset_gate = nn.Sequential(nn.Linear(h_dim + h_in_dim, h_in_dim), nn.Sigmoid())
        self.update_gate = nn.Sequential(nn.Linear(h_dim + h_in_dim, h_in_dim), nn.Sigmoid())
        self.transform = nn.Sequential(nn.Linear(h_dim + h_in_dim, h_in_dim), nn.ReLU())  # GRU uses Tanh, here use RELU

    def forward(self, h, h_in):
        # Concatenates h and h_in then used as input for the gates
        # dim = 0, concat batch dim; dim = 1, concat features
        a = torch.cat((h, h_in), dim=1)
        r = self.reset_gate(a) # apply reset gate to attentions a
        z = self.update_gate(a) # apply update gate to attentions a
        joined = torch.cat((h, r * h_in), dim=1) # Element-wise multiplication to get the weighted h_in and concat with h
        h_hat = self.transform(joined)  # apply Tanh
        return (1 - z) * h_in + z * h_hat # GRU

# GGAT + RELUGate
class GGATRELU(nn.Module):
    """
      defube module GGATGRU:
        input -> input_droupout -> GATConv1 (with gat_dropout) -> ELU activation -> out1
              |                                             |
              --------------------------------------------------> GRU1 -> x1

          x1  -> input_droupout -> GATConv2 (with gat_dropout) -> ELU activation -> Linear(dimTrans) -> out2
              |                                                                 |
              ----------------------------------------------------------------------> GRU2 -> x2

          x2  -> input_droupout -> GATConv3 (with gat_dropout) -> ELU activation -> Linear(dimTrans) -> out3
              |                                                                 |
              ----------------------------------------------------------------------> GRU3 -> x3
      features:
        input_dropout:
          For rich input features to prevents the model from overly relying on specific input feature dimensions and help generalize better.
          Set to 0 when the input is already highly sparse, doing so may discard too much information by ignoring critical ones for the task
        heads: higher is better to learn diverse attention but higher memory.
        gat_dropout: Encourages to consider a wider variety of neighbors rather than fixating on a few strong edges. Used when the task is binary classification and sensitive to overconfidence.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, input_dropout = 0.2, gat_dropout=0.4):
        super().__init__()
        self.input_dropout = input_dropout
        self.gat_dropout = gat_dropout
        self.heads = heads
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels

        self.transform = nn.Linear(in_channels, hidden_channels * heads) if in_channels != hidden_channels * heads else nn.Identity()

        self.gat1 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=gat_dropout)   # GAT output dim: hidden_channels * heads
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=gat_dropout)
        self.gat3 = GATConv(hidden_channels, out_channels, heads=1, dropout=gat_dropout)


        self.relu1 = RELUSubLayer(h_dim=hidden_channels * heads, h_in_dim=hidden_channels * heads)
        # change dimensions: [N, hidden_channels * heads] -> [N, hidden_channels]
        self.pre_gat2 = nn.Linear(hidden_channels * heads, hidden_channels)
        self.relu2 = RELUSubLayer(h_dim=hidden_channels, h_in_dim=hidden_channels)

        self.trans3 = nn.Linear(hidden_channels, hidden_channels)
        self.pre_gat3 = nn.Linear(hidden_channels, hidden_channels)
        self.relu3 = RELUSubLayer(h_dim=out_channels, h_in_dim=hidden_channels)

    def forward(self, x, edge_index):
        x = self.transform(x)

        h1 = x
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = self.relu1(x, h1)  # after gru1: [N, hidden_channels * heads]

        h2 = x # h2: [N, hidden_channels * heads]
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index)) # now, x after gat2: [N, hidden_channels]
        x = self.relu2(x, self.pre_gat2(h2)) # need to transfer h2's dimentions to the same as x

        h3 = self.trans3(x)
        x = self.pre_gat3(x)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.gat3(x, edge_index)
        x = self.relu3(x, h3)
        return x

class AttentionPooling(nn.Module):
    """
      define AttentionPooling:
        att_mlp: A 2-layer MLP (Multi-Layer Perceptron) that computes a scalar attention score for each node (which nodes are important overall)
        softmax: Normalizes scores into probabilities

    """
    def __init__(self, input_dim):
        super().__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),  # expands the representation to a higher-dimensional attention space, allowing the MLP to learn richer interactions between features.
            nn.Tanh(),
            nn.Linear(input_dim*2, 1)
        )

    def forward(self, node_embs):
        attn_weights = self.att_mlp(node_embs)
        attn_weights = torch.softmax(attn_weights, dim=0) # Ensures attention scores are positive and sum to 1.
        pooled = (attn_weights * node_embs).sum(dim=0)
        # Output is a single vector: the weighted average of all nodes based on learned attention.
        return pooled # shape: [num_nodes, 1] → score for each node.

def build_rr_predictor(input_dim, hidden_dim=16, output_dim=1):
    """
      No custom logic, no need to define init and forward.
      just use nn.Sequential to stack layers.
    """
    return nn.Sequential(
        nn.ReLU(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )

####Helper functions for Training and Testing####
"""
Adapted from the GGAT-cancer to follow a similar loading and normalization procedure
link: https://github.com/lhanlhanlhan/ggat

Note that GGAT-cancer is designed for node classification, whereas our task is disease pair prediction. Moreover, the GGAT-cancer framework is not scalable to datasets of the size used in our study. As a result, we do not adapt our model to a node classification setting nor perform direct comparisons with GGAT-cancer.
"""
def normalize_features(mx):
    """
      Row-normalize sparse matrix
    """
    rowsum = np.array(mx.sum(1)) # sum of each row

    # for rowsum = 0, replace with epsilon to avoid division by zero
    epsilon = 1e-10
    rowsum_safe = np.where(rowsum == 0, epsilon, rowsum)

    r_inv = np.power(rowsum_safe, -1).flatten() # 1 / rowsum, then flattern to 1d array
    r_inv[np.isinf(r_inv)] = 0.   # in case there is still infinities due to division , set to 0.

    r_mat_inv = sp.diags(r_inv)   # diagonal matrix with 1/rowsum
    mx = r_mat_inv.dot(mx)        # multiplicates the matrix itself, so that the matrix is scaled by 1/rowsum
    return mx

def load_data(path="./data/dis/", dataset="dis"):
    """
      [modified for our data] from GGAT cancer to make sure fair comparison
    """
    print('Loading {} dataset...'.format(dataset))

    # [node id, embedding, label] ... for dis set, there is no label, all are place holder -1
    idx_features_labels = np.genfromtxt(f"{path}{dataset}.content", dtype=np.dtype(str))
    # extract features: from columns idx 1 to -1, use sparse CSR matrix
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32) # node ids, cloumn idx 0
    idx_map = {j: i for i, j in enumerate(idx)}   # {node id: in file row id}

    # edges from file: [[nodei,nodej],...]
    edges_unordered = np.genfromtxt(f"{path}{dataset}.cites", dtype=np.int32)
    # falttern the list, then map the node ids with in file row ids, then reshape back to [[rowi,rowj],...]
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    # coo_matrix((list-of-data, (row_indices, col_indices)), shape=(N, N))
    # adj is a sparse matrix with [i,j] = 1
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    # build symmetric adjacency matrix, adds [j,i] = 1
    # (adj.T > adj) marks the edges that reverse edges exist but forward ones don’t (if has weights, then check larger weights and mark True)
    # adj.T.multiply(adj.T > adj) does element-wise multiplication to add those edges.
    # keep "- adj.multiply(adj.T > adj)" for future use when edge has different weights, so for each edge-pair (i, j) and (j, i), keep the higher weight
    # this will not affect unweighted edges, adj.multiply(adj.T > adj) will get all 0s.
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    # converts the sparse matrix into a dense NumPy matrix
    # then np.matrix into a standard np.ndarray, finally convert to torch tensors with float32
    features = torch.FloatTensor(np.array(features.todense()))

    # from_scipy_sparse_matrix => PyTorch Geometric (PyG) format: edge_index, edge_weight
    # edge_index: tensor of shape [2, num_edges], represents the edge list in COO format.
    # edge_weight: tensor of the weights on the edges, not used for unweighted graph
    edge_index, _ = from_scipy_sparse_matrix(adj)
    return features, edge_index

"""
Mine
"""
def log_print(message, log_file):
    """
      used to print the training log and write into files
    """
    print(message)
    log_file.write(message + "\n")
    log_file.flush()  # flush the buffer, writes to the disk immediately.


def prepare_data(path, dataset_name):
    features, edge_index = load_data(path, dataset_name)
    x = features.clone().detach().float()
    edge_index = edge_index.clone().detach().long()
    return Data(x=x, edge_index=edge_index), x

def prepare_models_and_data(args):
    """
      define models and prep data based on the args
      1. label: label_model, label_data
      2. n2v: n2v_model, n2v_data
    """
    # init the objs, they are needed for defined functions evenif the current mode doesn't need it, still need to set to None.
    # None is immutable in Python, this does not cause problems when later assign new values to any of them individually, they will point to others instead of None
    n2v_data = label_data = n2v_model = label_model = None

    if args.model_type == "n2v":
        n2v_data, n2v_x = prepare_data('./data/dis/', 'dis')
        n2v_model = build_ggat(n2v_x.shape[1], args)

    if args.model_type == "label":
        label_data, label_x = prepare_data('./data/dis/', 'label2vec')  # 'dishot', 'label2vec', 'label2vec_autoencoder'
        label_model = build_ggat(label_x.shape[1], args)

    return label_data, label_model, n2v_data, n2v_model

def build_ggat(input_dim, args):
  if args.model == "GGATGRU":
    return build_ggatgru(input_dim, args)

def build_ggatgru(input_dim, args):
    """
      used to build 2 models: label_model and n2v_model
    """
    return GGATGRU(
        in_channels=input_dim,
        hidden_channels=args.hidden,
        out_channels=args.inter_dim,
        heads=args.nb_heads,
        input_dropout=args.input_dropout,
        gat_dropout=args.gat_dropout
    )

def to_device(*objs, device):
    """
      send the obj to the given device
    """
    return [t.to(device) if t is not None else t for t in objs]

def get_model_params(args, label_model=None, n2v_model=None,
                     attention_pooler=None, rr_predictor=None):
    """
      add model parameters for optimizer based on the model option
    """
    model_params = []

    if args.model_type == "label":
        model_params += list(label_model.parameters())
    if args.model_type == "n2v":
        model_params += list(n2v_model.parameters())

    # Always include attention pooler and predictor
    model_params += list(attention_pooler.parameters())
    model_params += list(rr_predictor.parameters())
    return model_params

def split_train_val(gene_lists, labels, use_valid=True, test_size=0.1, seed=42, log_fn=print):
    """
      If choose to include the validation set:
      stratified train/val split, default split: 0.9 train, 0.1 valid, seed: 42
      If not:
      returns full data as train_set and empty val_set.

      Parameters:
          gene_lists: List of gene sets
          use_valid: validation set or not
          test_size: size of validation set
          seed: Random seed
          log_fn: defined logging function, print log and write to file

      Returns:
          train_set (tuples): (gene_list, label)
          val_set (tuples): (gene_list, label)
    """
    if use_valid:
        train_gene, val_gene, train_label, val_label = train_test_split(
            gene_lists, labels, test_size=test_size, stratify=labels, random_state=seed
        )
        # zip genes and labels into paired lists for iteration.
        train_set = list(zip(train_gene, train_label))
        val_set = list(zip(val_gene, val_label))
        # check for balance: how many 0 and 1
        log_fn(f"Train Label Distribution: {Counter(train_label)}")
        log_fn(f"Val Label Distribution: {Counter(val_label)}")
    else:
        train_set = list(zip(gene_lists, labels))
        val_set = []
        log_fn(f"Train Label Distribution: {Counter(labels)}")

    return train_set, val_set

def set_model_mode(mode, args, label_model, n2v_model, rr_predictor, attention_pooler):
    """
      helps to set all the models into the same mode: train or eval
      models include:
      label_model
      n2v_model
      rr_predictor
      attention_pooler
    """
    if args.model_type == "label" and label_model:
        getattr(label_model, mode)()  # use () at the end to call the method, without () will just get the method not runing it.
    if args.model_type == "n2v" and n2v_model:
        getattr(n2v_model, mode)()
    getattr(rr_predictor, mode)()
    getattr(attention_pooler, mode)()

def compute_node_embeddings(args, label_model, n2v_model, label_data, n2v_data):
    """
      construct the embedding:
      1. label mode: only get the embedding output from label_model
      2. n2v mode: only get the embedding output from n2v_model
    """
    if args.model_type == "label":
        return label_model(label_data.x, label_data.edge_index)
    elif args.model_type == "n2v":
        return n2v_model(n2v_data.x, n2v_data.edge_index)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

def embed_disease_pairs(node_embeddings, pair_set, attention_pooler):
    """
      each gene_list for each disease pair -> gene_embeddings -> attention score
      embs: list of attention scores for each disease pair
      labels: list of rr labels
    """
    embs, labels = [], []
    for gene_list, rr_label in pair_set:
        try:
            gene_embs = node_embeddings[gene_list]
            embs.append(attention_pooler(gene_embs)) # get attention score
            labels.append(rr_label)
        except Exception as e:
            print("!", e)
    # Stacks disease embeddings into a batch: [batch_size, hidden_dim]
    # Converts labels to a tensor and reshapes to [batch_size, 1]
    return torch.stack(embs), torch.tensor(labels, dtype=torch.float32).view(-1, 1)

def compute_metrics(logits, labels_tensor):
    """
      Converts logits to probabilities and computes accuracy and AUC.
      Assumes binary classification (sigmoid + threshold 0.5).
    """
    # Binary classification: outputs the probability of class 1.
    # Applies the sigmoid activation function to convert the raw logits into probabilities in the range (0, 1).
    # .view(-1): Flattens the tensor from [N,1] into a 1D array of shape [N], E.g., turns [[0.7], [0.3]] into [0.7, 0.3]
    # .cpu(): Moves the tensor from GPU to CPU memory. NumPy arrays can only live on the CPU.
    # .numpy(): converts the PyTorch tensor into a NumPy array
    probs = torch.sigmoid(logits).view(-1).cpu().numpy()
    labels_np = labels_tensor.view(-1).cpu().numpy()  # reshape the tensor from [N, 1] to [N] send to cpu and turn into numpy array
    # True for values > 0.5, and False otherwise, then transfer boolean type to fload type
    preds = (probs > 0.5).astype(np.float32)
    # val_probs, val_pred vs ground truth labels (val_labels_np)
    acc = (preds == labels_np).mean() # acc
    # if len(np.unique(val_labels_np)) == 2: checks if both classes (0 and 1) are present.
    # If only one class is present (e.g., all labels are 1s), AUC is undefined since AUC requires both positive and negative samples to compute the trade-off between TPR and FPR.
    # --> then sets AUC to NaN (Not a Number).
    auc = roc_auc_score(labels_np, probs) if len(np.unique(labels_np)) == 2 else float('nan') # AUC
    return acc, auc

def evaluate_on_validation(args, label_model, n2v_model, rr_predictor, attention_pooler,
                           label_data, n2v_data, val_set, device):
    """
      evaluation
    """
    # set to validation mode so that:
    # 1. Dropout won't randomly zeroes out parts of the embeddings during validation (through nn.Dropout)
    # 2. BatchNorm doesn't update its running mean and variance (nn.BatchNorm) -- BatchNorm is not used in this script yet
    set_model_mode('eval', args, label_model, n2v_model, rr_predictor, attention_pooler)
    # disables gradient tracking globally inside the block
    # if not gradients will be computed but not used, this wastes memory and compute
    with torch.no_grad():
        node_embeddings = compute_node_embeddings(args, label_model, n2v_model, label_data, n2v_data)

        val_embs, val_labels = embed_disease_pairs(node_embeddings, val_set, attention_pooler)
        val_embs, val_labels = to_device(val_embs, val_labels, device = device)

        val_logits = rr_predictor(val_embs)
        val_acc, val_auc = compute_metrics(val_logits, val_labels)
        return val_acc, val_auc

def save_best_states(args, rr_predictor, attention_pooler, label_model=None, n2v_model=None):
    """
      construct the dictionary of best model state_dicts based on model_type.
    """
    best_states = {
        'predictor_state_dict': rr_predictor.state_dict(),
        'pooler_state_dict': attention_pooler.state_dict()
    }
    if args.model_type == "label" and label_model:
        best_states['label_model_state_dict'] = label_model.state_dict()
    if args.model_type == "n2v" and n2v_model:
        best_states['n2v_model_state_dict'] = n2v_model.state_dict()
    return best_states

####---- Test with Test Set ----####
import csv
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score, roc_curve
import torch.nn.functional as F

def predict_disease_pairs(args, label_model, n2v_model, attention_pooler, rr_predictor,
                          label_data, n2v_data, disease_pair_rr, dis_pairs, device):
    """
      Computes probabilities for disease pairs using the provided models and attention pooling.
      Returns:
      1. rows for tsv file record
      2. ground labels
      3. predicted probabilities.
    """
    rows = []
    labels = []
    probs = []

    with torch.no_grad():
        # get embeddings from label_model / n2v_model / fusion_model
        node_embeddings = compute_node_embeddings(args, label_model, n2v_model, label_data, n2v_data)

        for i, (gene_list, rr_label) in enumerate(disease_pair_rr):
            # skips any gene indices that are out of bounds for the available node_embeddings
            if max(gene_list) >= node_embeddings.shape[0]:
                continue
            try:
                # disease pair prob for rr
                gene_embs = node_embeddings[gene_list]
                pooled_emb = attention_pooler(gene_embs)
                logit = rr_predictor(pooled_emb.unsqueeze(0))
                prob = torch.sigmoid(logit).item()
                # record results
                probs.append(prob)
                # Using labels only for logging and scikit-learn, not fot loss function or model calculation, no need to move to device
                labels.append(rr_label)
                rows.append({
                    "pair_id": i,
                    "disease_pair": "&".join(dis_pairs[i]),
                    "label": int(rr_label),
                    "prob": prob
                })
            except:
                continue

    return rows, labels, probs

def calculate_metrics_and_update_rows(probs, labels, rows):
    """
      rows are updated with calculated acc, auc, auprc, mcc and best_thresh
    """
    labels_np = np.array(labels)
    probs_np = np.array(probs)

    # calculate metrics
    try:
        fpr, tpr, thresholds = roc_curve(labels_np, probs_np)
        j_scores = tpr - fpr
        best_thresh = thresholds[j_scores.argmax()]
        preds = (probs_np > best_thresh).astype(int)

        acc = (preds == labels_np).mean()
        mcc = matthews_corrcoef(labels_np, preds)
        auprc = average_precision_score(labels_np, probs_np)
        auc = roc_auc_score(labels_np, probs_np)
    except:
        acc = mcc = auc = auprc = best_thresh = float("nan")
        preds = np.zeros_like(labels_np)

    # update rows
    for i, row in enumerate(rows):
        row["pred"] = int(probs[i] > best_thresh)
        row["acc"] = f"{acc:.4f}"
        row["mcc"] = f"{mcc:.4f}"
        row["auprc"] = f"{auprc:.4f}"
        row["roc_auc"] = f"{auc:.4f}"
        row["best_thresh"] = f"{best_thresh:.4f}"

    msg = (f"\nTest — Best Threshold (J): {best_thresh:.4f} | "
      f"Acc: {acc:.4f} | ROC AUC: {auc:.4f} | MCC: {mcc:.4f} | AUPRC: {auprc:.4f}")
    log_print(msg, log_file)

def save_predictions_to_tsv(rows, output_file):
    """
      save prediction results and metrics to a tsv file.

      Args:
          rows (list of dict): Prediction records, each with keys like 'prob', 'label', 'acc', etc.
          output_file (str): Path to the output .tsv file
    """
    fieldnames = list(rows[0].keys())
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    log_print(f"Saved test predictions and metrics to {output_file}", log_file)

def load_model_from_checkpoint(save_path, device, args,
                               rr_predictor, attention_pooler,
                               label_model=None, n2v_model=None):
    """
      loads model weights from a saved checkpoint.
    """
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    print("Checkpoint keys:", checkpoint.keys())

    rr_predictor.load_state_dict(checkpoint['predictor_state_dict'])
    attention_pooler.load_state_dict(checkpoint['pooler_state_dict'])

    if args.model_type =='label' and label_model:
        label_model.load_state_dict(checkpoint['label_model_state_dict'])
    if args.model_type =='n2v' and n2v_model:
        n2v_model.load_state_dict(checkpoint['n2v_model_state_dict'])

    log_print(f"Model loaded from checkpoint: {save_path}", log_file)

def evaluate_on_test_set_with_best_model_and_save(args, label_model, n2v_model, attention_pooler,
                                                  rr_predictor, label_data, n2v_data, test_disease_pair_rr,
                                                  test_dis_pairs, device, model_type, val_set, save_path,
                                                  output_file):
    """
      the data and models are moved to the device already in the prep data and model step
    """

    # load best model checkpoint, so that no longer using the model in the last epoch
    load_model_from_checkpoint(save_path, device, args, rr_predictor, attention_pooler,
                                label_model, n2v_model)
    # set models to eval mode
    set_model_mode('eval', args, label_model, n2v_model, rr_predictor, attention_pooler)

    # check after loading best model, make sure valid roc_auc match the best one in log
    val_acc, val_auc = evaluate_on_validation(args, label_model, n2v_model,
                  rr_predictor, attention_pooler, label_data, n2v_data, val_set, device)

    log_print(f"Reloaded model Val ROC AUC: {val_auc:.4f}", log_file)

    # make predictions
    rows, labels, probs = predict_disease_pairs(args, label_model, n2v_model, attention_pooler,
                            rr_predictor, label_data, n2v_data, test_disease_pair_rr, test_dis_pairs, device)

    if not labels:      # empty label list
        print("!! No valid predictions.")
        return

    # calcualte acc, mcc, rocauc, auprc, and update rows
    calculate_metrics_and_update_rows(probs, labels, rows)

    # write to file
    save_predictions_to_tsv(rows, output_file)