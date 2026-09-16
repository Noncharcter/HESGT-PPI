import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_add


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, act=nn.ReLU(inplace=True), dropout=0.0):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.BatchNorm1d(dims[i + 1]), act]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def rbf_expand(dist, num_k=8, gamma=2.0):
    centers = torch.linspace(0, 1.0, steps=num_k, device=dist.device, dtype=dist.dtype)
    dist = dist.unsqueeze(-1)
    d2 = (dist - centers) ** 2
    return torch.exp(-gamma * d2)

def round_ste(z: torch.Tensor) -> torch.Tensor:
    zhat = z.round()
    return z + (zhat - z).detach()

class FSQBranch(nn.Module):
    def __init__(self, in_dim, out_dim, levels=(8, 8, 8, 2), vq_dim=None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.levels = [int(x) for x in levels]
        self.vq_dim = int(len(self.levels) if vq_dim is None else vq_dim)

        self.codebook_size = 1
        for l in self.levels:
            self.codebook_size *= l

        self.proj = nn.Linear(self.in_dim, self.vq_dim)
        self.proj_inv = nn.Linear(self.vq_dim, self.out_dim)

        levels_t = torch.tensor(self.levels, dtype=torch.long)
        basis = torch.cumprod(torch.tensor([1] + self.levels[:-1], dtype=torch.long), dim=0)

        self.register_buffer("_levels", levels_t, persistent=False)
        self.register_buffer("_basis", basis, persistent=False)

    def bound(self, z):
        levels_f = self._levels.to(device=z.device, dtype=z.dtype)
        half_l = torch.floor(levels_f / 2.0)
        even_mask = (self._levels % 2 == 0).to(device=z.device)
        offset = torch.where(
            even_mask,
            torch.tensor(0.5, device=z.device, dtype=z.dtype),
            torch.tensor(0.0, device=z.device, dtype=z.dtype),
        )
        shift = torch.atanh(offset / half_l.clamp_min(1.0))
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z):
        z_bounded = self.bound(z)
        zhat = round_ste(z_bounded)
        levels_f = self._levels.to(device=z.device, dtype=z.dtype)
        half_width = torch.floor(levels_f / 2.0).clamp_min(1.0)
        codes = zhat / half_width
        return codes

    def forward(self, z):
        z_low = self.proj(z)
        codes = self.quantize(z_low)
        z_q = self.proj_inv(codes)
        return z_q


class MultiBranchFSQQuantizer(nn.Module):
    def __init__(self, embedding_dim, mm_dim=15, esp_levels=(8, 4, 4), geo_levels=(2, 2), ctx_levels=(2,), esp_vq_dim=3,
                 geo_vq_dim=2, ctx_vq_dim=1, dropout=0.0, cond_hidden=64, weight_hidden=32):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.mm_dim = int(mm_dim)

        self.esp_idx = [0, 1, 2, 3, 4, 9, 10]     # phi / gradn / logp
        self.geo_idx = [5, 6, 7, 11, 12, 13]      # lscale / kH / kG
        self.ctx_idx = [8, 14]                    # s_pers

        self.esp_split = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.geo_split = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.ctx_split = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        self.esp_scale = nn.Sequential(
            nn.Linear(len(self.esp_idx), cond_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cond_hidden, self.embedding_dim),
        )
        self.esp_bias = nn.Sequential(
            nn.Linear(len(self.esp_idx), cond_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cond_hidden, self.embedding_dim),
        )

        self.geo_scale = nn.Sequential(
            nn.Linear(len(self.geo_idx), cond_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cond_hidden, self.embedding_dim),
        )
        self.geo_bias = nn.Sequential(
            nn.Linear(len(self.geo_idx), cond_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cond_hidden, self.embedding_dim),
        )

        self.ctx_scale = nn.Sequential(
            nn.Linear(len(self.ctx_idx), cond_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cond_hidden, self.embedding_dim),
        )
        self.ctx_bias = nn.Sequential(
            nn.Linear(len(self.ctx_idx), cond_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cond_hidden, self.embedding_dim),
        )

        self.mm_weight = nn.Sequential(
            nn.Linear(3, weight_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(weight_hidden, 1),
        )

        self.fsq_esp = FSQBranch(
            in_dim=self.embedding_dim,
            out_dim=self.embedding_dim,
            levels=esp_levels,
            vq_dim=esp_vq_dim,
        )
        self.fsq_geo = FSQBranch(
            in_dim=self.embedding_dim,
            out_dim=self.embedding_dim,
            levels=geo_levels,
            vq_dim=geo_vq_dim,
        )
        self.fsq_ctx = FSQBranch(
            in_dim=self.embedding_dim,
            out_dim=self.embedding_dim,
            levels=ctx_levels,
            vq_dim=ctx_vq_dim,
        )

        self.aux_num_classes = 5
        self.esp_aux_head = nn.Linear(self.embedding_dim, self.aux_num_classes)
        self.geo_aux_head = nn.Linear(self.embedding_dim, self.aux_num_classes)
        self.ctx_aux_head = nn.Linear(self.embedding_dim, self.aux_num_classes)
        self.last_aux_logits = None

        self.fuse = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.norm = nn.LayerNorm(self.embedding_dim)

    def _split_mm_groups(self, mm_patch):
        esp_mm = mm_patch[:, self.esp_idx]
        geo_mm = mm_patch[:, self.geo_idx]
        ctx_mm = mm_patch[:, self.ctx_idx]
        return esp_mm, geo_mm, ctx_mm

    def _group_summary(self, x):
        # x: [B, D]
        x_mean = x.mean(dim=-1, keepdim=True)
        x_max = x.max(dim=-1, keepdim=True).values
        x_min = x.min(dim=-1, keepdim=True).values
        return torch.cat([x_mean, x_max, x_min], dim=-1)

    def _cond_modulate(self, z, mm_group, scale_net, bias_net):
        scale = scale_net(mm_group)
        bias = bias_net(mm_group)

        z_mod = z * scale + bias
        return z_mod

    def forward(self, z, mm_patch):
        esp_mm, geo_mm, ctx_mm = self._split_mm_groups(mm_patch)

        z_esp = self._cond_modulate(z, esp_mm, self.esp_scale, self.esp_bias)
        z_geo = self._cond_modulate(z, geo_mm, self.geo_scale, self.geo_bias)
        z_ctx = self._cond_modulate(z, ctx_mm, self.ctx_scale, self.ctx_bias)

        h_esp = self.esp_split(z_esp)
        h_geo = self.geo_split(z_geo)
        h_ctx = self.ctx_split(z_ctx)

        self.last_aux_logits = {
            "esp": self.esp_aux_head(h_esp),
            "geo": self.geo_aux_head(h_geo),
            "ctx": self.ctx_aux_head(h_ctx),
        }

        zq_esp = self.fsq_esp(h_esp)
        zq_geo = self.fsq_geo(h_geo)
        zq_ctx = self.fsq_ctx(h_ctx)

        weight_feat = torch.cat([
            self.mm_weight(self._group_summary(esp_mm)),
            self.mm_weight(self._group_summary(geo_mm)),
            self.mm_weight(self._group_summary(ctx_mm)),
        ], dim=-1)
        weights = torch.softmax(weight_feat, dim=-1)
        g_esp = weights[:, 0:1]
        g_geo = weights[:, 1:2]
        g_ctx = weights[:, 2:3]

        z_fused = torch.cat([g_esp * zq_esp, g_geo * zq_geo, g_ctx * zq_ctx], dim=-1)
        z_q = self.fuse(z_fused)
        z_q = self.norm(z + z_q)

        return z_q


class GraphAttentionLayer(nn.Module):
    def __init__(self, hid, heads=4, rbf_k=8, dropout=0.1):
        super().__init__()
        assert hid % heads == 0
        self.hid = hid
        self.heads = heads
        self.dk = hid // heads
        self.rbf_k = rbf_k

        self.q = nn.Linear(hid, hid, bias=False)
        self.k = nn.Linear(hid, hid, bias=False)
        self.v = nn.Linear(hid, hid, bias=False)

        self.map_geo = nn.Linear(rbf_k + 1, hid, bias=False)
        self.proj = nn.Linear(hid, hid)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hid)

    def _bias(self, src_pos, dst_pos, src_phi_mean, dst_phi_mean):
        dist = torch.norm(src_pos - dst_pos, dim=-1)
        dist_n = (dist / (dist.max() + 1e-6)).clamp(0, 1)
        rbf = rbf_expand(dist_n, num_k=self.rbf_k, gamma=2.0)
        dphi = (src_phi_mean - dst_phi_mean).abs()
        feat = torch.cat([rbf, dphi], dim=-1)
        b = self.map_geo(feat)
        return b.view(-1, self.heads, self.dk)

    def forward(self, g, h_patch):
        data = g.data
        edge_index = data.edge_index
        u = edge_index[0]
        v = edge_index[1]
        N = h_patch.shape[0]

        pos = g.nodes["patch"].data.get("pos")
        phi_mean = g.nodes["patch"].data.get("phi_mean")

        Q = self.q(h_patch).view(-1, self.heads, self.dk)
        K = self.k(h_patch).view(-1, self.heads, self.dk)
        V = self.v(h_patch).view(-1, self.heads, self.dk)

        bias = self._bias(pos[u], pos[v], phi_mean[u], phi_mean[v])
        att_logits = (Q[v] * (K[u] + bias)).sum(dim=-1) / math.sqrt(self.dk)
        att = pyg_softmax(att_logits, index=v, num_nodes=N)

        msg = V[u] * att.unsqueeze(-1)
        out = scatter_add(msg, v, dim=0, dim_size=N)

        out = out.reshape(N, self.hid)
        out = self.proj(out)
        out = self.drop(F.relu(out))
        # h_new = self.norm(h_patch + out)
        return out


# Heterogeneous Electrostatic Surface Graph Transformer
class HESGT(nn.Module):
    def __init__(self, patch_in_dim, hid=256, depth=6, heads=4, rbf_k=8, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(patch_in_dim, hid)
        self.layers = nn.ModuleList([
            GraphAttentionLayer(hid=hid, heads=heads, rbf_k=rbf_k, dropout=dropout) for _ in range(depth)
        ])
        self.dense_layer = nn.Linear(hid * depth, hid)

    def forward(self, g, x_patch):
        tem_h = []
        h = self.embed(x_patch)
        for layer in self.layers:
            h = layer(g, h)
            tem_h.append(h)
        h_dense = torch.cat(tem_h, dim=-1)
        h = h + self.dense_layer(h_dense)
        return h


# Heterogeneous Electrostatic Surface Quantization Network
class HESQ_Net(nn.Module):
    def __init__(self, mm_dim=15, patch_spe_dim=16, hid=256, depth=6, heads=4, rbf_k=8, mask_ratio_patch=0.6, dropout=0.1, res_in_dim=7,
                 esp_levels=(7, 7, 7, 7), geo_levels=(5, 5), ctx_levels=(5, 5),esp_vq_dim=4, geo_vq_dim=2, ctx_vq_dim=2):
        super().__init__()
        self.hid = int(hid)
        self.patch_spe_dim = int(patch_spe_dim)
        self.mm_dim = int(mm_dim)
        self.res_in_dim = int(res_in_dim)

        self.in_proj = MLP(self.mm_dim + self.patch_spe_dim, self.hid, self.hid, num_layers=2, dropout=dropout)
        self.res_proj = MLP(self.res_in_dim, self.hid, self.hid, num_layers=2, dropout=dropout)

        self.patch_encoder_rp = HESGT(
            patch_in_dim=self.hid,
            hid=self.hid,
            depth=depth,
            heads=heads,
            rbf_k=rbf_k,
            dropout=dropout,
        )
        self.patch_encoder_pp = HESGT(
            patch_in_dim=self.hid,
            hid=self.hid,
            depth=depth,
            heads=heads,
            rbf_k=rbf_k,
            dropout=dropout,
        )

        self.vq_patch = MultiBranchFSQQuantizer(
            embedding_dim=self.hid,
            mm_dim=self.mm_dim,
            esp_levels=esp_levels,
            geo_levels=geo_levels,
            ctx_levels=ctx_levels,
            esp_vq_dim=esp_vq_dim,
            geo_vq_dim=geo_vq_dim,
            ctx_vq_dim=ctx_vq_dim,
            dropout=dropout,
        )

        self.decoder = HESGT(
            patch_in_dim=self.hid,
            hid=self.hid,
            depth=depth,
            heads=heads,
            rbf_k=rbf_k,
            dropout=dropout,
        )

        self.proj = MLP(self.hid, self.hid, self.mm_dim, num_layers=2, dropout=dropout)
        self.pos_head = MLP(self.hid, self.hid, 3, num_layers=2, dropout=dropout)
        self.mask_ratio_patch = float(mask_ratio_patch)

    def _sample_mask_patch(self, g, patch_idx, r):
        dev = patch_idx.device
        P = int(patch_idx.numel())
        r = float(r)

        if r <= 0:
            return torch.zeros((P,), dtype=torch.bool, device=dev)

        mask = torch.zeros((P,), dtype=torch.bool, device=dev)

        data = getattr(g, "data")
        batch = getattr(data, "batch")

        batch = batch.to(dev).view(-1)
        batch_patch = batch[patch_idx]
        B = int(batch.max().item()) + 1

        patch_counts = torch.bincount(batch_patch, minlength=B).to(torch.long)

        offset = 0
        for cnt in patch_counts.detach().cpu().tolist():
            cnt = int(cnt)
            n_mask = int(cnt * r)
            if n_mask > 0:
                perm = torch.randperm(cnt, device=dev)[:n_mask] + offset
                mask[perm] = True
            offset += cnt

        return mask

    def _run_vq_patch(self, z, mm_patch=None):
        return self.vq_patch(z, mm_patch=mm_patch)

    def _make_group_label(self, x_group):
        score = x_group.detach().mean(dim=-1)
        qs = torch.linspace(0, 1, steps=6, device=score.device)
        bins = torch.quantile(score.float(), qs[1:-1])
        return torch.bucketize(score.float(), bins).long()

    def _aux_ce_from_mm(self, mm_patch):
        aux_logits = getattr(self.vq_patch, "last_aux_logits", None)
        if aux_logits is None:
            return mm_patch.new_tensor(0.0)

        esp_mm = mm_patch[:, self.vq_patch.esp_idx]
        geo_mm = mm_patch[:, self.vq_patch.geo_idx]
        ctx_mm = mm_patch[:, self.vq_patch.ctx_idx]

        y_esp = self._make_group_label(esp_mm)
        y_geo = self._make_group_label(geo_mm)
        y_ctx = self._make_group_label(ctx_mm)

        aux_ce = (F.cross_entropy(aux_logits["esp"], y_esp) + F.cross_entropy(aux_logits["geo"], y_geo)
                  + F.cross_entropy(aux_logits["ctx"], y_ctx)) / 3.0

        return aux_ce


    def forward(self, g):
        dev = next(self.parameters()).device

        N = g.num_nodes("patch")
        is_patch = g.nodes["patch"].data.get("is_patch")
        is_patch = is_patch.to(dev).view(-1)
        patch_idx = torch.nonzero(is_patch > 0, as_tuple=False).view(-1)
        P = int(patch_idx.numel())

        mm_all = g.nodes["patch"].data.get("mm")
        spe_all = g.nodes["patch"].data.get("spe")
        xyz_all = g.nodes["patch"].data.get("xyz")
        x_res_all = g.nodes["patch"].data.get("x_res")

        mm_all = mm_all.to(dev)
        spe_all = spe_all.to(dev)
        xyz_all = xyz_all.to(dev)
        x_res_all = x_res_all.to(dev)

        x_patch_in = torch.cat([mm_all, spe_all], dim=-1)
        x0 = self.in_proj(x_patch_in)
        x0 = x0 + self.res_proj(x_res_all)

        edge_index_full = g.data.edge_index.to(dev)
        edge_type = g.data.edge_type.to(dev).view(-1)
        edge_rp = edge_index_full[:, edge_type == 0]
        edge_pp = edge_index_full[:, edge_type == 1]

        mask_patch_full = self._sample_mask_patch(g, patch_idx, self.mask_ratio_patch)

        old_edge_index = g.data.edge_index

        g.data.edge_index = edge_rp
        h1 = self.patch_encoder_rp(g, x0)

        g.data.edge_index = edge_pp
        h2 = self.patch_encoder_pp(g, h1)

        g.data.edge_index = old_edge_index

        h_patch = h2[patch_idx]

        if float(self.mask_ratio_patch) <= 0.0:
            mm_patch = mm_all[patch_idx]
            zq_patch = self._run_vq_patch(h_patch, mm_patch=mm_patch)

            aux_ce_loss = self._aux_ce_from_mm(mm_patch)

            z_pre = h2.clone()
            z_pre[patch_idx] = zq_patch

            old_edge_index = g.data.edge_index
            try:
                g.data.edge_index = edge_pp
                h_dec = self.decoder(g, z_pre)
            finally:
                g.data.edge_index = old_edge_index

            recon_mm = self.proj(h_dec[patch_idx])
            recon_xyz = self.pos_head(h_dec[patch_idx])

            recon_gt = mm_all[patch_idx].detach()
            xyz_gt = xyz_all[patch_idx].detach()

            preds = {"recon": recon_mm, "xyz": recon_xyz}
            vq = zq_patch
            hdict = {
                "mask_patch": mask_patch_full,
                "recon_gt": recon_gt,
                "xyz_gt": xyz_gt,
                "aux_ce_loss": aux_ce_loss,
            }
            return preds, vq, hdict

        sel = mask_patch_full
        M = int(sel.sum().item())

        h_mask = h_patch[sel]
        mm_mask = mm_all[patch_idx[sel]]
        zq_m = self._run_vq_patch(h_mask, mm_patch=mm_mask)

        aux_ce_loss = self._aux_ce_from_mm(mm_mask)

        z_pre = h2.clone()
        z_pre[patch_idx[sel]] = zq_m

        old_edge_index = g.data.edge_index
        try:
            g.data.edge_index = edge_pp
            h_dec = self.decoder(g, z_pre)
        finally:
            g.data.edge_index = old_edge_index

        recon_mm_m = self.proj(h_dec[patch_idx[sel]])
        recon_xyz_m = self.pos_head(h_dec[patch_idx[sel]])

        recon_gt_m = mm_all[patch_idx[sel]].detach()
        xyz_gt_m = xyz_all[patch_idx[sel]].detach()

        preds = {"recon": recon_mm_m, "xyz": recon_xyz_m}
        vq = zq_m

        hdict = {
            "mask_patch": torch.ones((M,), dtype=torch.bool, device=dev),
            "recon_gt": recon_gt_m,
            "xyz_gt": xyz_gt_m,
            "aux_ce_loss": aux_ce_loss,
        }
        return preds, vq, hdict


def pretrain_losses_patch(g, preds, vq, hdict, w_recon=1.0):
    recon_gt = hdict.get("recon_gt")
    xyz_gt = hdict.get("xyz_gt")
    recon_pred = preds.get("recon")
    xyz_pred = preds.get("xyz")

    mask_patch = hdict.get("mask_patch")
    mask_patch = mask_patch.to(recon_gt.device).view(-1)

    sel = mask_patch
    recon_gt_sel = recon_gt[sel]
    recon_pred_sel = recon_pred[sel]
    xyz_gt_sel = xyz_gt[sel]
    xyz_pred_sel = xyz_pred[sel]

    L_recon = F.mse_loss(recon_pred_sel, recon_gt_sel)
    L_xyz = F.mse_loss(xyz_pred_sel, xyz_gt_sel)

    aux_ce_loss = hdict.get("aux_ce_loss", None)

    loss = (w_recon * L_recon) + L_xyz + aux_ce_loss

    return loss
