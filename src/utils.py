import os
import csv
import json
import math
import random
import shutil
import subprocess

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected, coalesce


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(p, overwrite=False):
    if not os.path.exists(p):
        os.makedirs(p)
    elif overwrite:
        shutil.rmtree(p)
        os.makedirs(p)


@torch.no_grad()
def micro_f1_from_logits(logits, labels, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).cpu().numpy()
    truth = labels.cpu().numpy()

    tp = ((pred == 1) & (truth == 1)).sum()
    fp = ((pred == 1) & (truth == 0)).sum()
    fn = ((pred == 0) & (truth == 1)).sum()

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return float(f1)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def evaluat_metrics(output, label):
    return micro_f1_from_logits(output, label)


def default_path(sub_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", sub_path))


def require_prebuilt_root(args):
    return getattr(args, "prebuilt_root", "") or (args.esp_root + "_patch_graphs")


def build_node_to_edge_index(edges, num_nodes=None):
    if not edges:
        return {} if num_nodes is None else {i: [] for i in range(num_nodes)}

    if num_nodes is None:
        num_nodes = max(max(int(u), int(v)) for u, v in edges) + 1

    m = {i: [] for i in range(num_nodes)}
    for ei, (u, v) in enumerate(edges):
        m[int(u)].append(ei)
        m[int(v)].append(ei)
    return m


def pick_start_node(node_to_edge_index, node_num, max_deg=20):
    if not node_to_edge_index or node_num <= 0:
        raise ValueError("Cannot pick start node from empty node_to_edge_index")

    valid_nodes = list(node_to_edge_index.keys())
    if not valid_nodes:
        raise ValueError("No valid nodes in node_to_edge_index")

    potential_starts = [
        n for n in valid_nodes
        if node_to_edge_index.get(n) and len(node_to_edge_index[n]) <= max_deg
    ]
    if potential_starts:
        return random.choice(potential_starts)
    return random.choice(valid_nodes)


def bfs_select_edges(edges, node_to_edge_index, sub_graph_size):
    candiate_node = []
    selected_edge_index = []
    visited_nodes = set()
    processed_edges = set()

    if not node_to_edge_index or sub_graph_size <= 0:
        return []

    node_num = len(node_to_edge_index)
    start_node = pick_start_node(node_to_edge_index, node_num, max_deg=20)
    candiate_node.append(start_node)
    visited_nodes.add(start_node)

    while candiate_node and len(selected_edge_index) < sub_graph_size:
        cur_node = candiate_node.pop(0)
        if cur_node not in node_to_edge_index:
            continue

        edges_to_process = list(node_to_edge_index[cur_node])
        random.shuffle(edges_to_process)

        for edge_index in edges_to_process:
            if len(selected_edge_index) >= sub_graph_size:
                break
            if edge_index in processed_edges:
                continue
            if not (isinstance(edge_index, int) and 0 <= edge_index < len(edges)):
                continue

            selected_edge_index.append(edge_index)
            processed_edges.add(edge_index)

            edge = edges[edge_index]
            if not (isinstance(edge, (list, tuple)) and len(edge) == 2):
                continue

            if edge[0] == cur_node:
                neighbor_node = edge[1]
            elif edge[1] == cur_node:
                neighbor_node = edge[0]
            else:
                neighbor_node = None

            if neighbor_node is not None and neighbor_node not in visited_nodes:
                visited_nodes.add(neighbor_node)
                candiate_node.append(neighbor_node)

    return selected_edge_index


def dfs_select_edges(edges, node_to_edge_index, sub_graph_size):
    stack = []
    selected_edge_index = []
    visited_nodes = set()
    visited_edges = set()

    if not node_to_edge_index or sub_graph_size <= 0:
        return []

    node_num = len(node_to_edge_index)
    start_node = pick_start_node(node_to_edge_index, node_num, max_deg=20)
    stack.append(start_node)

    while stack and len(selected_edge_index) < sub_graph_size:
        cur_node = stack[-1]
        if cur_node not in visited_nodes:
            visited_nodes.add(cur_node)

        found_neighbor = False
        if cur_node in node_to_edge_index:
            edges_to_explore = list(node_to_edge_index[cur_node])
            random.shuffle(edges_to_explore)

            for edge_index in edges_to_explore:
                if len(selected_edge_index) >= sub_graph_size:
                    break
                if not (isinstance(edge_index, int) and 0 <= edge_index < len(edges)):
                    continue

                edge = edges[edge_index]
                if not (isinstance(edge, (list, tuple)) and len(edge) == 2):
                    continue

                if edge[0] == cur_node:
                    neighbor_node = edge[1]
                elif edge[1] == cur_node:
                    neighbor_node = edge[0]
                else:
                    neighbor_node = None

                if edge_index not in visited_edges and len(selected_edge_index) < sub_graph_size:
                    selected_edge_index.append(edge_index)
                    visited_edges.add(edge_index)
                    if len(selected_edge_index) >= sub_graph_size:
                        break

                if neighbor_node is not None and neighbor_node not in visited_nodes:
                    stack.append(neighbor_node)
                    found_neighbor = True
                    break

        if not found_neighbor:
            stack.pop()

    return selected_edge_index


def split_edges_with_mode(edges, mode, seed=0):
    random.seed(seed)
    e_num = len(edges)

    if e_num <= 0:
        raise ValueError("Cannot split empty edge list")

    if mode == "random":
        random_list = list(range(e_num))
        random.shuffle(random_list)
        return (
            random_list[: int(e_num * 0.6)],
            random_list[int(e_num * 0.6): int(e_num * 0.8)],
            random_list[int(e_num * 0.8):],
        )

    if mode in ("bfs", "dfs"):
        node_to_edge_index = {}
        for i, edge in enumerate(edges):
            if not (isinstance(edge, (list, tuple)) and len(edge) == 2):
                continue
            node_to_edge_index.setdefault(edge[0], []).append(i)
            node_to_edge_index.setdefault(edge[1], []).append(i)

        node_num = len(node_to_edge_index)
        sub_graph_size = int(e_num * 0.4)

        if mode == "bfs":
            selected_edge_index = bfs_select_edges(edges, node_to_edge_index, sub_graph_size)
        else:
            selected_edge_index = dfs_select_edges(edges, node_to_edge_index, sub_graph_size)

        all_edge_index = list(range(e_num))
        unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

        random_list = list(range(len(selected_edge_index)))
        random.shuffle(random_list)

        train_index = unselected_edge_index
        val_index = [selected_edge_index[i] for i in random_list[: int(e_num * 0.2)]]
        test_index = [selected_edge_index[i] for i in random_list[int(e_num * 0.2):]]
        return train_index, val_index, test_index

    raise ValueError(f"Unsupported split mode: {mode}")


class PatchReadout(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = float(eps)

    @torch.no_grad()
    def forward(self, g, zq_patch):
        device = zq_patch.device
        dtype = zq_patch.dtype

        node_data = g.nodes["patch"].data

        mm_all = node_data.get("mm")
        mm_all = mm_all.to(device)

        is_patch = node_data.get("is_patch")
        is_patch = is_patch.to(device).view(-1)

        patch_idx = torch.nonzero(is_patch > 0, as_tuple=False).view(-1)
        mm_patch = mm_all[patch_idx]

        phi_max = mm_patch[:, 1].to(dtype)
        phi_min = mm_patch[:, 2].to(dtype)
        c_pos = torch.relu(phi_max)
        c_neg = torch.relu(-phi_min)

        sum_pos = c_pos.sum()
        sum_neg = c_neg.sum()

        if float(sum_pos.item()) > self.eps:
            w_pos = c_pos / sum_pos
            e_pos = (zq_patch * w_pos.unsqueeze(-1)).sum(dim=0)
        else:
            e_pos = torch.zeros((zq_patch.shape[1],), device=device, dtype=dtype)

        if float(sum_neg.item()) > self.eps:
            w_neg = c_neg / sum_neg
            e_neg = (zq_patch * w_neg.unsqueeze(-1)).sum(dim=0)
        else:
            e_neg = torch.zeros((zq_patch.shape[1],), device=device, dtype=dtype)

        total = sum_pos + sum_neg
        alpha_pos = sum_pos / total if float(sum_pos.item()) > self.eps else torch.zeros((), device=device, dtype=dtype)
        alpha_neg = sum_neg / total if float(sum_neg.item()) > self.eps else torch.zeros((), device=device, dtype=dtype)

        emb = alpha_pos * e_pos + alpha_neg * e_neg
        return emb


CLS_MAP = {
    "reaction": 0,
    "binding": 1,
    "ptmod": 2,
    "activation": 3,
    "inhibition": 4,
    "catalysis": 5,
    "expression": 6,
}


def load_embeddings(embed_pt):
    emb = torch.load(embed_pt, map_location="cpu")
    return emb


def read_seq_dict_ordered(path):
    protein_name_to_id = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] and row[0] not in protein_name_to_id:
                protein_name_to_id[row[0]] = len(protein_name_to_id)

    if not protein_name_to_id:
        raise ValueError(f"No protein IDs found in sequence dictionary: {path}")
    return protein_name_to_id


def read_seq_dict(path):
    return set(read_seq_dict_ordered(path).keys())


def build_ppi_and_labels(ppi_actions_path, protein_name_to_id):
    ppi_pair_to_edge_idx = {}
    edges = []
    labels = []

    with open(ppi_actions_path, "r", encoding="utf-8") as f:
        head = True
        for line in f:
            if head:
                head = False
                continue

            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            p1_name, p2_name, interaction_type = parts[0], parts[1], parts[2]

            if p1_name not in protein_name_to_id or p2_name not in protein_name_to_id:
                continue
            if interaction_type not in CLS_MAP:
                continue

            p1_id = protein_name_to_id[p1_name]
            p2_id = protein_name_to_id[p2_name]
            if p1_id == p2_id:
                continue

            pair_key = tuple(sorted((p1_id, p2_id)))
            if pair_key not in ppi_pair_to_edge_idx:
                ppi_pair_to_edge_idx[pair_key] = len(edges)
                edges.append(list(pair_key))

                lab = [0] * len(CLS_MAP)
                lab[CLS_MAP[interaction_type]] = 1
                labels.append(lab)
            else:
                labels[ppi_pair_to_edge_idx[pair_key]][CLS_MAP[interaction_type]] = 1

    if not edges:
        raise ValueError(f"No valid PPI pairs found from: {ppi_actions_path}")

    nodes = [None] * len(protein_name_to_id)
    for pid, idx in protein_name_to_id.items():
        nodes[idx] = pid

    labels = torch.tensor(np.asarray(labels, dtype=np.float32))
    return nodes, edges, labels


def build_mp_edge_index(edge_pairs, num_nodes):
    edge_index = torch.as_tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index


def run_epoch(model, graph, X, edges_t, labels, indices, batch_size, device, loss_fn, opt=None, train=True):
    model.train() if train else model.eval()
    n_batch = math.ceil(len(indices) / batch_size)
    loss_sum, f1_sum = 0.0, 0.0

    if train:
        indices = list(indices)
        random.shuffle(indices)

    with torch.set_grad_enabled(train):
        for bi in range(n_batch):
            bidx_list = indices[bi * batch_size: (bi + 1) * batch_size]
            bidx = torch.as_tensor(bidx_list, dtype=torch.long, device=device)

            out = model(graph, X, edges_t, bidx)
            gt = labels[bidx]
            loss = loss_fn(out, gt)

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()

            loss_sum += float(loss.item())
            f1_sum += float(evaluat_metrics(out.detach().cpu(), gt.detach().cpu()))

    return loss_sum / max(1, n_batch), f1_sum / max(1, n_batch)


@torch.no_grad()
def eval_epoch_full_micro(model, graph, X, edges_t, labels, indices, batch_size, device, loss_fn):
    model.eval()
    n_batch = math.ceil(len(indices) / batch_size)

    outs = []
    gts = []
    loss_sum = 0.0

    for bi in range(n_batch):
        bidx_list = indices[bi * batch_size: (bi + 1) * batch_size]
        bidx = torch.as_tensor(bidx_list, dtype=torch.long, device=device)

        out = model(graph, X, edges_t, bidx)
        gt = labels[bidx]
        loss = loss_fn(out, gt)

        loss_sum += float(loss.item())
        outs.append(out.detach())
        gts.append(gt.detach())

    out_all = torch.cat(outs, dim=0) if len(outs) else torch.empty((0, labels.shape[1]), device=device)
    gt_all = torch.cat(gts, dim=0) if len(gts) else torch.empty((0, labels.shape[1]), device=device)

    f1 = float(evaluat_metrics(out_all.detach().cpu(), gt_all.detach().cpu()))
    return loss_sum / max(1, n_batch), f1
