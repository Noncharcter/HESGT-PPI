import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv


def _get_edge_index(g, device):
    if hasattr(g, "data") and hasattr(g.data, "edge_index"):
        ei = g.data.edge_index
    elif hasattr(g, "edge_index"):
        ei = g.edge_index
    elif torch.is_tensor(g):
        ei = g
    else:
        raise TypeError(f"Unsupported graph type for PyG: {type(g)}")
    return ei.to(device)


class LinkPredGCN(nn.Module):
    def __init__( self, n_layers=3, in_dim=512, hidden=128, out_dim=64, class_num=7, dropout=0.0):
        super().__init__()
        self.n_layers = int(n_layers)
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.out_dim = int(out_dim)
        self.class_num = int(class_num)
        self.dropout = float(dropout)

        self.in_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([
            GCN2Conv(
                channels=hidden,
                alpha=0.1,
                theta=1.0,
                layer=l + 1,
                shared_weights=True,
                add_self_loops=True,
                normalize=True,
            )
            for l in range(n_layers)
        ])
        self.extra_bns = nn.ModuleList()

        for _ in range(self.n_layers):
            self.extra_bns.append(nn.BatchNorm1d(hidden))

        if self.class_num > 1:
            self.lin = nn.Linear(hidden, out_dim)
            self.edge_cls = nn.Linear(out_dim, class_num)

    def encode(self, x, edge_index):
        x = self.in_proj(x)
        x = F.relu(x)
        x0 = x
        for conv, bn in zip(self.convs, self.extra_bns):
            x = conv(x, x0, edge_index)
            x = bn(F.relu(x))
        return x

    def _gather_edge_pairs(self, edge_list_or_tensor, batch_edge_idx, device):
        if isinstance(edge_list_or_tensor, torch.Tensor):
            edges_t = edge_list_or_tensor.to(device)
        else:
            edges_t = torch.as_tensor(edge_list_or_tensor, dtype=torch.long, device=device)

        if isinstance(batch_edge_idx, torch.Tensor):
            bidx = batch_edge_idx.to(device)
        else:
            bidx = torch.as_tensor(batch_edge_idx, dtype=torch.long, device=device)


        node_id = edges_t[bidx]  # [B, 2]
        u = node_id[:, 0]
        v = node_id[:, 1]
        edge_label_index = torch.stack([u, v], dim=0)  # [2, B]
        return edge_label_index

    def decode(self, z, edge_label_index):
        src = edge_label_index[0]
        dst = edge_label_index[1]

        z_src = z[src]
        z_dst = z[dst]

        # pair_feat = z_src * z_dst  # [B, D]
        pair_feat = torch.mul(z_src, z_dst)

        pair_feat = F.relu(self.lin(pair_feat))
        pair_feat = F.dropout(pair_feat, p=self.dropout, training=self.training)
        return self.edge_cls(pair_feat)  # [B, class_num]

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self, g, x_node, edge_list_or_tensor, batch_edge_idx):
        device = x_node.device
        edge_index = _get_edge_index(g, device)

        z = self.encode(x_node, edge_index)
        edge_label_index = self._gather_edge_pairs(
            edge_list_or_tensor=edge_list_or_tensor,
            batch_edge_idx=batch_edge_idx,
            device=device,
        )
        out = self.decode(z, edge_label_index)
        return out
