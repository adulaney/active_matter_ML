import time

### Torch Packages ###
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import tensorflow as tf
import tensorflow.keras as keras
### Graph packages ###
import networkx as nx
import dgl

from sklearn.metrics import f1_score
import numpy as np


class TorchGAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop=0,
                 attn_drop=0,
                 negative_slope=0.2,
                 residual=True,
                 ):
        """Torch implementation of the Gated Attention Network (GAT) class.
        Arguments:
            num_layers (int): Number of layers in the network.
            in_dim (int): Input feature dimension.
            num_hidden (int): Number of hidden features.
            num_classes (int): Number of possible classes for classification.
            heads (int): Number of attention heads
            activation (func): Activation function.
            feat_drop (float): Feature dropout rate.
            attn_drop (float): Attention head dropout rate.
            negative_slope (float): Slope of leaky ReLU activation function
            residual (bool): If true use residual connection. 
        """
        super(TorchGAT, self).__init__()
        # self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(
            GATConv(
                in_dim,
                num_hidden[0],
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )

        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                GATConv(
                    num_hidden[l - 1] * heads[l - 1],
                    num_hidden[l],
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )

        # output projection
        self.gat_layers.append(
            GATConv(
                num_hidden[-1] * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, inputs, graph):
        """ Update GAT weights.
        Arguments:
            inputs (floats): Training data.
        Returns:
            output (float): Loss.
        """
        h = inputs
        g2 = graph
        for l in range(self.num_layers):
            h = self.gat_layers[l](g2, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g2, h).mean(1)
        return logits

    def predict(self, feats, subgraph):
        with torch.no_grad():
            self.eval()
            self.g = subgraph
            for layer in self.gat_layers:
                layer.g = subgraph
            output = self(feats.float())

            return output


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATConv(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):

        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, graphs, labels, mask, gpu=-1):
    with torch.no_grad():
        model.eval()
        if gpu >= 0:
            full_logits = torch.empty(len(features), 2).cuda()
        else:
            full_logits = torch.empty(len(features), 2)
        for i, file in enumerate(graphs):
            g_ind = [a for a in range(i*40000, (i+1)*40000)]
            g_network = pd.read_pickle(file)[-1:]
            g_network = g_network['Features'].iloc[0]

            # add self loop
            g_network.remove_edges_from(nx.selfloop_edges(g_network))
            g = dgl.DGLGraph()
            g.from_networkx(g_network)

            # forward
            full_logits[g_ind] = model(features[g_ind].float(), g)

        logits = full_logits[mask]
        labels = labels[mask]

        return accuracy(logits, labels)


def single_evaluate(feats, model, subgraph, labels, loss_fcn, mask):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(feats.float(), subgraph)
        loss_data = loss_fcn(output[mask], labels[mask].float())
        predict = np.where(output[mask].data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels[mask].data.cpu().numpy(),
                         predict, average='micro')
        return score, loss_data.item()


def predict(feats, model, subgraph):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.gat_layers():
            layer.g = subgraph
        output = model(feats.float())

        return output


def train_GAT(x_train_norm, train_mask, val_mask, preds, graph):
    n_classes = 2
    device = torch.device("cpu")

    features = torch.FloatTensor(x_train_norm)
    labels = torch.LongTensor(keras.utils.to_categorical(preds, n_classes))

    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(train_mask)
        val_mask = torch.BoolTensor(val_mask)
    else:
        train_mask = torch.ByteTensor(train_mask)
        val_mask = torch.ByteTensor(val_mask)

    num_feats = features.shape[1]

    # Initialize Graphs
    graph.remove_edges_from(nx.selfloop_edges(graph))
    G = dgl.DGLGraph()
    G.from_networkx(graph)

    model = TorchGAT(num_layers=2,
                     in_dim=num_feats,
                     num_hidden=[8, 8],
                     num_classes=2,
                     heads=[2, 2],
                     activation=F.elu)

    m = torch.nn.Softmax(dim=1)

    loss_fcn = torch.nn.BCEWithLogitsLoss()

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    # Early Stopping
    cur_step = 0
    patience = 10
    best_acc = 0.01
    best_loss = 10000
    monitor_bests = [best_acc, best_loss]
    waits = [0, 0]
    # main loop
    dur = []
    for epoch in range(50):
        model.train()
        if epoch >= 3:
            t0 = time.time()
#         logits = torch.empty(len(x_train_norm),2).cuda()
        logits = torch.empty(len(x_train_norm), 2)

        # forward
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features.float(), G)
        loss = loss_fcn(m(logits[train_mask]), labels[train_mask].float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        predict = np.where(logits[train_mask].data.cpu().numpy() >= 0.5, 1, 0)
        train_acc = f1_score(labels[train_mask].data.cpu().numpy(),
                             predict, average='micro')

        fastmode = False
        if fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc, val_loss = single_evaluate(
                features.float(), model, G, labels.float(), loss_fcn, val_mask)
            monitors = [val_acc, val_loss]
            early_stop = False
            reset_waits = False
            for i, monitor in enumerate(monitors):
                if (-1)**(i)*(monitor - monitor_bests[i])/monitor_bests[i] > 1e-3:
                    monitor_bests[i] = monitor
                    waits[i] = 0
                    reset_waits = True
                else:
                    waits[i] += 1
            if reset_waits:
                for i in range(len(waits)):
                    waits[i] = 0
            else:
                num_sat = 0
                for wait in waits:
                    if wait >= patience:
                        num_sat += 1
                if num_sat == len(waits):
                    break

        # Only print every 10 epochs
        if epoch % 10 == 0:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                  " ValAcc {:.4f} | ValLoss {:.4f}".
                  format(epoch, np.mean(dur), loss.item(), train_acc,
                         val_acc, val_loss))
    # Print last epoch
    print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
          " ValAcc {:.4f} | ValLoss {:.4f}".
          format(epoch, np.mean(dur), loss.item(), train_acc,
                 val_acc, val_loss))
    print()
    if early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))

    return model
