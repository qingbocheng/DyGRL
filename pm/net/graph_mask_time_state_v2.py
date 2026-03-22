import dgl
from pm.registry import NET
from pm.net import PreModel_V2
import torch
from torch import optim
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from typing import List, Final, Optional
from functools import partial
from timm.layers import Mlp, DropPath, use_fused_attn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from mmengine.config import Config,DictAction
import argparse
import os
from tqdm import tqdm
import numpy as np

@NET.register_module(force=True)
class GraphMaskTimeState_V2(PreModel_V2):
    def __init__(self,
                in_dim: int,
                num_hidden: int,
                num_layers: int,
                nhead: int,
                nhead_out: int,
                activation: str,
                feat_drop: float,
                attn_drop: float,
                negative_slope: float,
                residual: bool,
                norm: Optional[str],
                mask_rate: float = 0.3,
                encoder_type: str = "gat",
                decoder_type: str = "gat",
                loss_fn: str = "sce",
                drop_edge_rate: float = 0.0,
                replace_rate: float = 0.1,
                alpha_l: float = 2,
                ** kwargs
                ):
        super(GraphMaskTimeState_V2, self).__init__(
            in_dim=in_dim,
            num_hidden = num_hidden,
            num_layers = num_layers,
            nhead = nhead,
            nhead_out = nhead_out,
            activation = activation,
            feat_drop = feat_drop,
            attn_drop = attn_drop,
            negative_slope = negative_slope,
            residual = residual,
            norm = norm,
            mask_rate = mask_rate,
            encoder_type = encoder_type,
            decoder_type = decoder_type,
            loss_fn = loss_fn,
            drop_edge_rate = drop_edge_rate,
            replace_rate = replace_rate,
            alpha_l = alpha_l,
            **kwargs,
        )

        self.decoder_blocks = None
        self.loss_list = []
        self.graph_data = {}
        self.graph = {}
        self.train_mae = True
        # self.initialize_weights()

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        if getattr(self.patch_embed, "proj", None) is not None:
            w = self.patch_embed.proj.weight.data
            if self.trunc_init:
                torch.nn.init.trunc_normal_(w)
                torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, 1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, g, x, mask):
        """
        b, c, n, d, f = x.shape # batch size, num stocks, days, features
        """
        if mask is not None:
            pre_use_g, use_x = self.encoding_mask_noise(g, x, mask=mask)
            nodes = mask
            use_g = pre_use_g
            enc_rep = self.encoder(use_g.edge_index, use_x)
        else:
            loss, enc_rep, (mask_nodes, keep_nodes) = self.forward(g, x)
            if self.train_mae:
                self.loss_list.append(loss)
            nodes = torch.zeros(1, x.shape[0], device=x.device)   # 全 0
            # pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
            nodes[0, mask_nodes] = 1  

        # append cls token
        # if self.cls_embed:
        #     cls_token = self.cls_token
        #     cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        #     x = torch.cat((cls_tokens, x), dim=1)

        return enc_rep, nodes

    def cross_sectional_normalize(self, feat):
        """
        对节点特征矩阵进行截面标准化
        
        参数:
        feat: 节点特征矩阵，形状为 [num_nodes, num_features]
        
        返回:
        标准化后的特征矩阵
        """
        # 计算每个特征维度的均值和标准差
        mean = torch.mean(feat, dim=0, keepdim=True)  # [1, num_features]
        std = torch.std(feat, dim=0, keepdim=True)    # [1, num_features]
        
        # 避免除零错误
        std = torch.clamp(std, min=1e-6)
        
        # 标准化
        normalized_feat = (feat - mean) / std
        
        return normalized_feat

    def forward_state(self, market='ndx', date=None, mask = None, ids_restore = None, device='cpu'):
        
        if self.graph.get(date) is None:
            g = torch.load(f'/root/quant-ml-qlib/Graph-EarnMore/datasets/{market}/distance_graph/{date}.pt')
            # g = g[0]
            # g = g.to(device)
            # in_degrees = g.in_degrees()
            # zero_in_degree_nodes = (in_degrees == 0).sum().item()
            # if zero_in_degree_nodes > 0:
            #     g = dgl.add_self_loop(g)
            # data = self.cross_sectional_normalize(g.ndata['feat'])
            # data = data.to(device)
            # g.ndata['feat'] = data
            # self.graph_data[date] = data
            self.graph[date] = g
        else:
            # data = self.graph_data[date]
            g = self.graph[date]
        #self.dgl_to_pyg(g, date)
        x, masks = self.forward_encoder(g.to(device), g.x.to(device), mask = mask)

        # embed tokens
        # x = self.decoder_embed(x)

        # append cls token
        # if self.cls_embed:
        #     decoder_cls_token = self.decoder_cls_token
        #     decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
        #     x = torch.cat((decoder_cls_tokens, x), dim=1)

        return x.unsqueeze(0), masks

    def forward_state_val(self, market='ndx', date=None, mask=None, ids_restore=None, device='cpu'):

        if self.graph_data.get(date) is None:
            g, _ = dgl.load_graphs(f'/root/quant-ml-qlib/Graph-EarnMore/datasets/{market}/distance_graph/{date}.bin')
            g = g[0]
            g = g.to(device)
            in_degrees = g.in_degrees()
            zero_in_degree_nodes = (in_degrees == 0).sum().item()
            if zero_in_degree_nodes > 0:
                g = dgl.add_self_loop(g)
            data = self.cross_sectional_normalize(g.ndata['feat'])
            data = data.to(device)
            g.ndata['feat'] = data
            self.graph_data[date] = data
            self.graph[date] = g
        else:
            data = self.graph_data[date].to(device)
            g = self.graph[date].to(device)

        #self.dgl_to_pyg(g, date)
        x, masks = self.forward_encoder(g, data)
    
    def forward_loss(self):
        mean_loss = torch.stack(self.loss_list).mean()  # shape: [], 仍在计算图中
        print(f"-------------graphmae: {mean_loss}-------------")
        self.loss_list = []
        return mean_loss

    def dgl_to_pyg(self, dgl_graph,date):
        """
        将 DGL 图（无向）转换为 PyG 的 Data 对象
        """
        # 获取边索引（u, v）
        u, v = dgl_graph.edges()

        edge_index = torch.stack([u, v], dim=0)

        # 构建 PyG 的 Data 对象
        data = Data(edge_index=edge_index)
        data.num_nodes = dgl_graph.num_nodes()

        # 拷贝节点特征（如果有的话）
        if 'feat' in dgl_graph.ndata:
            data.x = dgl_graph.ndata['feat']

        # 拷贝节点标签（如果有的话）
        if 'label' in dgl_graph.ndata:
            data.y = dgl_graph.ndata['label']

        # 拷贝边特征（如果有的话）
        if 'feat' in dgl_graph.edata:
            data.edge_attr = dgl_graph.edata['feat']

        # 拷贝图级标签（如果有的话）
        # if 'label' in dgl_graph.graph:
        #     data.y = dgl_graph.graph['label']

        torch.save(data, f'/root/quant-ml-qlib/EarnMore/datasets/ndx/distance_graph/{date}.pt')


def parse_args():
    ROOT = '/root/quant-ml-qlib/EarnMore'
    parser = argparse.ArgumentParser(description='PM train script')
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "mask_sac_portfolio_management_cap.py"),
                        help="config file path")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument("--root", type=str, default=ROOT)
    parser.add_argument("--workdir", type=str, default="workdir")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--if_remove", action="store_true", default=True)
    args = parser.parse_args()
    return args

def min_max_normalize_node_features(graph):
    """对图的节点特征进行Min-Max标准化"""
#     features = graph.ndata['feat']
    
#     # 计算全局最小值和最大值
#     min_val = features.min(0, keepdim=True)[0]
#     max_val = features.max(0, keepdim=True)[0]
    
#     # 防止除零错误
#     range_val = max_val - min_val
#     range_val = torch.where(range_val > 1e-8, range_val, torch.ones_like(range_val))
    
#     # 标准化
#     normalized_features = (features - min_val) / range_val
    
#     # 放回图中
#     graph.ndata['feat'] = normalized_features
#     return graph

if __name__ == "__main__":
    args = parse_args()
    # print(args)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is None:
        args.cfg_options = dict()
    if args.root is not None:
        args.cfg_options["root"] = args.root
    if args.workdir is not None:
        args.cfg_options["workdir"] = args.workdir
    if args.tag is not None:
        args.cfg_options["tag"] = args.tag
    cfg.merge_from_dict(args.cfg_options)
#     # print(cfg)
#     num_nodes = 300
#     num_edges = 90000
#     node_feat_dim = 104

#     # 随机生成节点特征和标签
#     node_attr = torch.randn(num_nodes, node_feat_dim)            # float32 特征
#     node_labels = torch.randint(0, 5, (num_nodes, 1))             # 假设标签为 0~4
#     node_ids = torch.arange(num_nodes)

#     # 随机生成边（避免自环）
#     src = torch.randint(0, num_nodes, (num_edges,))
#     dst = torch.randint(0, num_nodes, (num_edges,))
#     mask = src != dst
#     src, dst = src[mask], dst[mask]

#     # 裁剪回到边数限制
#     src, dst = src[:num_edges], dst[:num_edges]

#     # 随机生成边标签
#     edge_labels = torch.randint(0, 3, (len(src), 1))              # 假设边标签为 0~2
#     edge_ids = torch.arange(len(src))

#     # 构图
#     g = dgl.graph((src, dst), num_nodes=num_nodes)

#     # 设置节点属性
    
#     g.ndata['_ID'] = node_ids
#     g.ndata['node_labels'] = node_labels
#     g.ndata['attr'] = node_attr

#     # 设置边属性
#     g.edata['_ID'] = edge_ids
#     g.edata['edge_labels'] = edge_labels

#     # 输出检查
#     # print(g)
#     # print("attr:\n", g.ndata['attr'])

    model = GraphMaskTimeState_V2(
                in_dim=cfg.graph_rep_net.num_features,
                num_hidden=cfg.graph_rep_net.num_hidden,
                num_layers=cfg.graph_rep_net.num_layers,
                nhead=cfg.graph_rep_net.num_heads,
                nhead_out=cfg.graph_rep_net.num_out_heads,
                activation=cfg.graph_rep_net.activation,
                feat_drop=cfg.graph_rep_net.in_drop,
                attn_drop=cfg.graph_rep_net.attn_drop,
                negative_slope=cfg.graph_rep_net.negative_slope,
                residual=cfg.graph_rep_net.residual,
                encoder_type=cfg.graph_rep_net.encoder,
                decoder_type=cfg.graph_rep_net.decoder,
                mask_rate=cfg.graph_rep_net.mask_rate,
                norm=cfg.graph_rep_net.norm,
                loss_fn=cfg.graph_rep_net.loss_fn,
                drop_edge_rate=cfg.graph_rep_net.drop_edge_rate,
                replace_rate=cfg.graph_rep_net.replace_rate,
                alpha_l=cfg.graph_rep_net.alpha_l,
                concat_hidden=cfg.graph_rep_net.concat_hidden,
            )
    # g = dgl.add_self_loop(g)
    # loss = model(g,node_attr)
    # print(loss)
    # 加载单个图
    loaded_graphs, labels_dict = dgl.load_graphs('/root/quant-ml-qlib/EarnMore/datasets/csi300/graph/CSI300_2015-11-02.bin')
    loaded_graph = loaded_graphs[0]  # 获取第一个图
    # print(loaded_graph,loaded_graph.ndata['feat'])

    # graph = min_max_normalize_node_features(loaded_graph)
    # graph.edata['feat'] = torch.nan_to_num(graph.edata['feat'], nan=0.0, posinf=0.0, neginf=0.0)

    loss = model(loaded_graph,loaded_graph.ndata['feat'])
    print(loss)
    # loss, mask, ids_restore = model(batch)
    # print(loss, mask.shape, ids_restore.shape)

    # x, mask, ids_restore = model.forward_state(batch)
    # print(x.shape, mask.shape, ids_restore.shape)