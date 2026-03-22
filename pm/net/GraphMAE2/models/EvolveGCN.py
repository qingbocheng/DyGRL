import tqdm

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import GRU
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptTensor,
)

from torch_geometric.nn.inits import glorot, zeros
import torch
import torch.nn as nn

import dgl

def identity_norm(x):
    def func(x):
        return x
    return func

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "identity":
        return identity_norm
    else:
        # print("Identity norm")
        return None

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "elu":
        return nn.ELU()
    elif name == "silu":
        return nn.SiLU()
    elif name is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

class EvolveGCNO(nn.Module):
    def __init__(self,
                 in_dim,
                 num_layers,
                 activation,
                 norm,
                 encoding=False,
                 ):
        super(EvolveGCNO, self).__init__()

        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        last_activation = create_activation(activation) if encoding else None
        last_norm = norm if encoding else None

        hidden_in = in_dim

        if num_layers == 1:
            self.gat_layers.append(EvolveGCNOConv(
                in_channels=hidden_in,
                norm=last_norm))
        else:
            # input projection (no residual)
            self.gat_layers.append(EvolveGCNOConv(
                in_channels=hidden_in,
                activation=create_activation(activation), norm=norm))
            # hidden layers

            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(EvolveGCNOConv(
                    in_channels=hidden_in,
                    activation=create_activation(activation), norm=norm))

            # output projection
            self.gat_layers.append(EvolveGCNOConv(
                in_channels=hidden_in,
                activation=last_activation, norm=last_norm))
        self.head = nn.Identity()
        
    def forward(self, edge, inputs):
        h = inputs

        for l in range(self.num_layers):
            h = self.gat_layers[l](edge, h)

        if self.head is not None:
            return self.head(h)
        else:
            return h

    def inference(self, g, x, batch_size, device, emb=False):
        """
        Inference with the GAT model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        num_heads = self.num_heads
        num_heads_out = self.num_heads_out
        for l, layer in enumerate(self.gat_layers):
            if l < self.num_layers - 1:
                y = torch.zeros(g.num_nodes(), self.num_hidden * num_heads if l != len(self.gat_layers) - 1 else self.num_classes)
            else:
                if emb == False:
                    y = torch.zeros(g.num_nodes(), self.num_hidden if l != len(self.gat_layers) - 1 else self.num_classes)
                else:
                    y = torch.zeros(g.num_nodes(), self.out_dim * num_heads_out)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                    g,
                    torch.arange(g.num_nodes()),
                    sampler,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=8)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)
                h = x[input_nodes].to(device)
                if l < self.num_layers - 1:
                    h = layer(block, h)
                else:
                    h = layer(block, h)

                if l == len(self.gat_layers) - 1 and (emb == False):
                    h = self.head(h)
                y[output_nodes] = h.cpu()
            x = y
        return y
    
    def reweight(self):
        for l in range(self.num_layers):
            self.gat_layers[l].reinitialize_weight()


class GCNConv_Fixed_W(MessagePassing):
    r"""The graph convolutional operator adapted from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, with weights not trainable.
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.
    Its node-wise formulation is given by:
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup
        \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j
    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)
    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv_Fixed_W, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, W: torch.FloatTensor, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops)

        x = torch.matmul(x, W)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j



class EvolveGCNOConv(torch.nn.Module):
    r"""An implementation of the Evolving Graph Convolutional without Hidden Layer.
    For details see this paper: `"EvolveGCN: Evolving Graph Convolutional
    Networks for Dynamic Graph." <https://arxiv.org/abs/1902.10191>`_
    Args:
        in_channels (int): Number of filters.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        improved: bool = False,
        cached: bool = False,
        normalize: bool = True,
        add_self_loops: bool = True,
        norm = None,
        activation = None,
    ):
        super(EvolveGCNOConv, self).__init__()

        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.initial_weight = torch.nn.Parameter(torch.Tensor(1, in_channels, in_channels))
        self.weight = None
        self._create_layers()
        self.reset_parameters()
        self.norm = create_norm(norm)(in_channels) if norm is not None else None
        self.active = activation
    
    def reset_parameters(self):
        glorot(self.initial_weight)

    def reinitialize_weight(self):
        self.weight = None

    def _create_layers(self):

        self.recurrent_layer = GRU(
            input_size=self.in_channels, hidden_size=self.in_channels, num_layers=1
        )
        for param in self.recurrent_layer.parameters():
            param.requires_grad = True
            param.retain_grad()

        self.conv_layer = GCNConv_Fixed_W(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            improved=self.improved,
            cached=self.cached,
            normalize=self.normalize,
            add_self_loops=self.add_self_loops
        )

    def forward(
        self,
        edge_index: torch.LongTensor,
        X: torch.FloatTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass.
        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node embedding.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.
        Return types:
            * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
        """
        
        if self.weight is None:
            _, self.weight = self.recurrent_layer(self.initial_weight, self.initial_weight)
        else:
            _, self.weight = self.recurrent_layer(self.weight, self.weight)
        X = self.conv_layer(self.weight.squeeze(dim=0), X, edge_index, edge_weight)
        if self.norm is not None:
            X = self.norm(X)

        # activation
        if self.active is not None:
            X = self.active(X)

        return X
