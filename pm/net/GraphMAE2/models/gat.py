import tqdm
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

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

class GAT(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 nhead_out,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False,
                 ):
        super(GAT, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_heads_out = nhead_out
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.concat_out = concat_out

        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        hidden_in = in_dim
        hidden_out = out_dim

        if num_layers == 1:
            self.gat_layers.append(GATConv(
                in_channels=hidden_in, out_channels=hidden_out, heads=nhead_out,
                dropout=attn_drop, negative_slope=negative_slope, residual=last_residual, norm=last_norm, concat=concat_out))
        else:
            # input projection (no residual)
            self.gat_layers.append(GATConv(
                in_channels=hidden_in, out_channels=num_hidden, heads=nhead,
                dropout=attn_drop, negative_slope=negative_slope, residual=residual, active=create_activation(activation), norm=norm, concat=concat_out))
            # hidden layers

            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    in_channels=num_hidden * nhead, out_channels=num_hidden, heads=nhead,
                    ropout=attn_drop, negative_slope=negative_slope, residual=residual, active=create_activation(activation), norm=norm, concat=concat_out))

            # output projection
            self.gat_layers.append(GATConv(
                in_channels=num_hidden * nhead, out_channels=hidden_out, heads=nhead_out,
                ropout=attn_drop, negative_slope=negative_slope, residual=last_residual, activation=last_activation, norm=last_norm, concat=concat_out))
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

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.is_pretraining = False
        self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)



# class GATConv(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  out_feats,
#                  num_heads,
#                  feat_drop=0.,
#                  attn_drop=0.,
#                  negative_slope=0.2,
#                  residual=False,
#                  activation=None,
#                  allow_zero_in_degree=False,
#                  bias=True,
#                  norm=None,
#                  concat_out=True):
#         super(GATConv, self).__init__()
#         self._num_heads = num_heads
#         self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
#         self._out_feats = out_feats
#         self._allow_zero_in_degree = allow_zero_in_degree
#         self._concat_out = concat_out

#         if isinstance(in_feats, tuple):
#             self.fc_src = nn.Linear(
#                 self._in_src_feats, out_feats * num_heads, bias=False)
#             self.fc_dst = nn.Linear(
#                 self._in_dst_feats, out_feats * num_heads, bias=False)
#         else:
#             self.fc = nn.Linear(
#                 self._in_src_feats, out_feats * num_heads, bias=False)
#         self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
#         self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
#         self.feat_drop = nn.Dropout(feat_drop)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.leaky_relu = nn.LeakyReLU(negative_slope)
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
#         else:
#             self.register_buffer('bias', None)
#         if residual:
#             if self._in_dst_feats != out_feats * num_heads:
#                 self.res_fc = nn.Linear(
#                     self._in_dst_feats, num_heads * out_feats, bias=False)
#             else:
#                 self.res_fc = None
#         else:
#             self.register_buffer('res_fc', None)
#         self.reset_parameters()
#         self.activation = activation
    
#         self.norm = norm
#         if norm is not None:
#             self.norm = create_norm(norm)(num_heads * out_feats)
#         self.set_allow_zero_in_degree(False)

#     def reset_parameters(self):
#         """

#         Description
#         -----------
#         Reinitialize learnable parameters.

#         Note
#         ----
#         The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
#         The attention weights are using xavier initialization method.
#         """
#         gain = nn.init.calculate_gain('relu')
#         if hasattr(self, 'fc'):
#             nn.init.xavier_normal_(self.fc.weight, gain=gain)
#         else:
#             nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
#             nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
#         nn.init.xavier_normal_(self.attn_l, gain=gain)
#         nn.init.xavier_normal_(self.attn_r, gain=gain)
#         if self.bias is not None:
#             nn.init.constant_(self.bias, 0)
#         if isinstance(self.res_fc, nn.Linear):
#             nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

#     def set_allow_zero_in_degree(self, set_value):
#         self._allow_zero_in_degree = set_value

#     def forward(self, graph, feat, get_attention=False):
#         with graph.local_scope():
#             if not self._allow_zero_in_degree:
#                 if (graph.in_degrees() == 0).any():
#                     raise RuntimeError('There are 0-in-degree nodes in the graph, '
#                                    'output for those nodes will be invalid. '
#                                    'This is harmful for some applications, '
#                                    'causing silent performance regression. '
#                                    'Adding self-loop on the input graph by '
#                                    'calling `g = dgl.add_self_loop(g)` will resolve '
#                                    'the issue. Setting ``allow_zero_in_degree`` '
#                                    'to be `True` when constructing this module will '
#                                    'suppress the check and let the code run.')

#             if isinstance(feat, tuple):
#                 src_prefix_shape = feat[0].shape[:-1]
#                 dst_prefix_shape = feat[1].shape[:-1]
#                 h_src = self.feat_drop(feat[0])
#                 # h_dst = self.feat_drop(feat[1])
#                 h_dst = feat[1]

#                 if not hasattr(self, 'fc_src'):
#                     feat_src = self.fc(h_src).view(
#                         *src_prefix_shape, self._num_heads, self._out_feats)
#                     feat_dst = self.fc(h_dst).view(
#                         *dst_prefix_shape, self._num_heads, self._out_feats)
#                 else:
#                     feat_src = self.fc_src(h_src).view(
#                         *src_prefix_shape, self._num_heads, self._out_feats)
#                     feat_dst = self.fc_dst(h_dst).view(
#                         *dst_prefix_shape, self._num_heads, self._out_feats)
#             else:
#                 src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
#                 h_src = h_dst = self.feat_drop(feat)
#                 feat_src = feat_dst = self.fc(h_src).view(
#                     *src_prefix_shape, self._num_heads, self._out_feats)
#                 if graph.is_block:
#                     feat_dst = feat_src[:graph.number_of_dst_nodes()]
#                     h_dst = h_dst[:graph.number_of_dst_nodes()]
#                     dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
#             # NOTE: GAT paper uses "first concatenation then linear projection"
#             # to compute attention scores, while ours is "first projection then
#             # addition", the two approaches are mathematically equivalent:
#             # We decompose the weight vector a mentioned in the paper into
#             # [a_l || a_r], then
#             # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
#             # Our implementation is much efficient because we do not need to
#             # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
#             # addition could be optimized with DGL's built-in function u_add_v,
#             # which further speeds up computation and saves memory footprint.
#             el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
#             er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
#             graph.srcdata.update({'ft': feat_src, 'el': el})
#             graph.dstdata.update({'er': er})
#             # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
#             graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
#             e = self.leaky_relu(graph.edata.pop('e'))
#             # e[e == 0] = -1e3
#             # e = graph.edata.pop('e')
#             # compute softmax
#             graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
#             # message passing
#             graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
#                              fn.sum('m', 'ft'))
#             rst = graph.dstdata['ft']

#             # bias
#             if self.bias is not None:
#                 rst = rst + self.bias.view(
#                     *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)

#             # residual
#             if self.res_fc is not None:
#                 # Use -1 rather than self._num_heads to handle broadcasting
#                 resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
#                 rst = rst + resval

#             if self._concat_out:
#                 rst = rst.flatten(1)
#             else:
#                 rst = torch.mean(rst, dim=1)

#             if self.norm is not None:
#                 rst = self.norm(rst)

#             # activation
#             if self.activation:
#                 rst = self.activation(rst)

#             if get_attention:
#                 return rst, graph.edata['a']
#             else:
#                 return rst



class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,j}]\right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]\right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        norm = None,
        residual = None,
        activation = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.norm = create_norm(norm)(heads*out_channels) if norm is not None else None
        self.active = activation
        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        if residual:
            self.residual = nn.Linear(
                    heads * out_channels, heads * out_channels, bias=False)
        else:
            self.residual = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, edge_index: Adj, x: Union[Tensor, OptPairTensor], 
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias
        
        if self.residual is not None:
            resval = self.residual(out)
            out = out + resval

        if self.norm is not None:
            out = self.norm(out)

        # activation
        if self.active is not None:
            out = self.active(out)

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
