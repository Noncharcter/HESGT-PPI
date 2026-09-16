import os, glob
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from torch_geometric.data import Data, Batch

AA3_ASSIGN_ORDER = [
    "ALA", "CYS", "ASP", "GLU", "PHE",
    "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG",
    "SER", "THR", "VAL", "TRP", "TYR",
]
AA3_TO_ASSIGN_IDX = {aa3: i for i, aa3 in enumerate(AA3_ASSIGN_ORDER)}


def load_all_assign_features(path):
    if path is None:
        raise ValueError("all_assign_path is required when backbone residue features use all_assign.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"all_assign.txt not found: {path}")

    arr = np.loadtxt(path).astype(np.float32)
    if arr.shape != (20, 7):
        raise ValueError(f"all_assign.txt must have shape (20, 7), got {arr.shape}: {path}")
    return arr

def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))

def _robust_clip(x, q=0.995):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    lo = np.quantile(x, 1.0 - q)
    hi = np.quantile(x, q)
    return np.clip(x, lo, hi).astype(np.float32)


def stabilize_aux_features(gradn, logp, kH, kG, lscale, clip_q=0.995,):
    gradn = _robust_clip(gradn, q=clip_q)
    logp = _robust_clip(logp, q=clip_q)
    kH = _robust_clip(kH, q=clip_q)
    kG = _robust_clip(kG, q=clip_q)
    lscale = _robust_clip(lscale, q=clip_q)

    logp = _signed_log1p(logp).astype(np.float32)
    kH = _signed_log1p(kH).astype(np.float32)
    kG = _signed_log1p(kG).astype(np.float32)

    gradn = np.log1p(np.maximum(gradn, 0.0)).astype(np.float32)
    lscale = np.log1p(np.maximum(lscale, 0.0)).astype(np.float32)

    logp = np.tanh(logp).astype(np.float32)
    kH = np.tanh(kH).astype(np.float32)
    kG = np.tanh(kG).astype(np.float32)

    return gradn, logp, kH, kG, lscale


def read_point_txt(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.replace(',', ' ').split()
            phi, x, y, z = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            rows.append([phi, x, y, z])
    return np.asarray(rows, dtype=np.float32)


def _pairwise_d2(xyz):
    xx = np.sum(xyz ** 2, axis=1, keepdims=True)
    d2 = xx + xx.T - 2.0 * (xyz @ xyz.T)
    d2 = np.maximum(d2, 0.0)
    return d2.astype(np.float32)


def knn_indices(xyz, k):
    d2 = _pairwise_d2(xyz)
    order = np.argsort(d2, axis=1)
    idxs = order[:, 1:k + 1]
    dists = np.sqrt(np.take_along_axis(d2, idxs, axis=1) + 1e-8).astype(np.float32)
    return idxs.astype(np.int64), dists


def laplacian_pe(adj, k):
    N = adj.shape[0]

    deg = np.sum(adj, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
    L = np.eye(N) - D_inv_sqrt @ adj @ D_inv_sqrt

    w, v = np.linalg.eigh(L)
    order = np.argsort(w)
    v = v[:, order]

    start = 1 if N > 1 else 0
    k_eff = min(k, max(0, N - start))

    out = np.zeros((N, k), dtype=np.float32)
    if k_eff > 0:
        out[:, :k_eff] = v[:, start:start + k_eff].astype(np.float32)
    return out


def build_adjacency_from_edges(N, edges, sym):
    A = np.zeros((N, N), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        if sym:
            A[j, i] = 1.0
    return A


def estimate_normals_pca_from_knn(xyz, idxs):
    N = xyz.shape[0]
    normals = np.zeros((N, 3), dtype=np.float32)
    kH = np.zeros((N,), dtype=np.float32)
    kG = np.zeros((N,), dtype=np.float32)

    for i in range(N):
        nbr = xyz[idxs[i]]
        mu = nbr.mean(axis=0, keepdims=True)
        M = (nbr - mu)
        C = (M.T @ M) / max(1, nbr.shape[0] - 1)
        w, v = np.linalg.eigh(C)
        order = np.argsort(w)[::-1]
        v = v[:, order]
        w = w[order]
        n = v[:, -1]
        normals[i] = n / (np.linalg.norm(n) + 1e-8)

        w = np.maximum(w, 1e-12)
        kH[i] = float((w[0] + w[1] - w[2]))
        kG[i] = float((w[0] * w[1] - w[2]))
    return normals, kH, kG


def discrete_gradient_features_from_knn(phi, xyz, idxs, dists):
    N = xyz.shape[0]
    gradn = np.zeros((N,), dtype=np.float32)
    logp = np.zeros((N,), dtype=np.float32)

    DENOM_FLOOR = 1e-2

    for i in range(N):
        nbr = idxs[i]
        vi = xyz[i]
        vn = xyz[nbr]
        dp = phi[nbr] - phi[i]
        vec = vn - vi
        denom = (np.linalg.norm(vec, axis=1) + 1e-8)
        denom = np.maximum(denom, DENOM_FLOOR)
        unit = vec / denom[:, None]
        gvec = (dp[:, None] * unit).mean(axis=0)
        gradn[i] = float(np.linalg.norm(gvec))
        logp[i] = float((dp / (denom + 1e-6)).mean())
    return gradn, logp


def knn_density_at_scale(dists, k=24):
    k = int(k)
    dens = 1.0 / (dists[:, :k].mean(axis=1) + 1e-6)
    s = dens.astype(np.float32)
    s = (s - s.mean()) / (s.std() + 1e-8)
    return s.astype(np.float32)


def local_scale_feature_from_knn(dists, k):
    k = min(k, dists.shape[1])
    return dists[:, :k].mean(axis=1).astype(np.float32)


def fps_indices(xyz, P, seed):
    N = xyz.shape[0]
    P = int(min(P, N))
    rng = np.random.RandomState(seed)
    start = int(rng.randint(0, N))
    centers = np.zeros((P,), dtype=np.int64)
    centers[0] = start
    dist2 = np.full((N,), np.inf, dtype=np.float32)

    for i in range(1, P):
        c = centers[i - 1]
        d = np.sum((xyz - xyz[c]) ** 2, axis=1).astype(np.float32)
        dist2 = np.minimum(dist2, d)
        centers[i] = int(np.argmax(dist2))
    return centers


def assign_points_to_centers(xyz, center_xyz):
    xx = np.sum(xyz ** 2, axis=1, keepdims=True)          # (N,1)
    cc = np.sum(center_xyz ** 2, axis=1, keepdims=True).T # (1,P)
    d2 = xx + cc - 2.0 * (xyz @ center_xyz.T)
    d2 = np.maximum(d2, 0.0)
    assign = np.argmin(d2, axis=1).astype(np.int64)
    return assign


def build_patches_no_overlap(xyz, centers_idx, assign, K, seed=0):
    rng = np.random.RandomState(seed)
    P = centers_idx.shape[0]
    patch_idx = np.zeros((P, K), dtype=np.int64)
    point_mask = np.zeros((P, K), dtype=np.float32)

    for pid in range(P):
        members = np.where(assign == pid)[0]

        cxyz = xyz[int(centers_idx[pid])]
        d2 = np.sum((xyz[members] - cxyz) ** 2, axis=1)
        order = np.argsort(d2)
        members = members[order]

        take = members[:min(K, members.size)]
        patch_idx[pid, :take.size] = take
        point_mask[pid, :take.size] = 1.0

        if take.size < K:
            fill = rng.choice(take if take.size > 0 else members, size=(K - take.size), replace=True)
            patch_idx[pid, take.size:] = fill
            point_mask[pid, take.size:] = 0.0

    return patch_idx, point_mask


def patch_knn_edges(nodes_xyz, k=8):
    P = nodes_xyz.shape[0]
    d2 = _pairwise_d2(nodes_xyz)
    edges = []
    order = np.argsort(d2, axis=1)
    for i in range(P):
        nbr = order[i, 1:k + 1]
        for j in nbr:
            edges.append((int(j), int(i)))
    return edges


def compute_patch_meanmax(x_surf, patch_idx, point_mask, invalid_fill=-1e9):
    k = 10

    pts = x_surf[patch_idx]              # (P,K,F)
    valid = point_mask.astype(bool)      # (P,K)
    m0 = valid.astype(np.float32)        # (P,K)
    m = m0[..., None]                    # (P,K,1)

    denom0 = np.maximum(m0.sum(axis=1), 1e-6)   # (P,)

    phi_pts = pts[..., 0]                         # (P,K)
    phi_mean = (phi_pts * m0).sum(axis=1) / denom0

    phi_for_max = np.where(valid, phi_pts, invalid_fill)
    phi_max = phi_for_max.max(axis=1)

    pos_inf = -invalid_fill
    phi_for_min = np.where(valid, phi_pts, pos_inf)
    phi_min = phi_for_min.min(axis=1)

    denom = np.maximum(m.sum(axis=1), 1e-6)       # (P,1)
    mean_all = (pts * m).sum(axis=1) / denom      # (P,F)
    surf_f_mean = mean_all[:, 1:]                 # (P,F-1)
    surf_f_mean = k * surf_f_mean

    pts_for_max = np.where(m > 0.0, pts, invalid_fill)  # pad -> -1e9
    mx_all = pts_for_max.max(axis=1)                    # (P,F)
    surf_f_max = mx_all[:, 1:]    # (P,F-1)
    surf_f_max = k * surf_f_max

    mm = np.concatenate(
        [phi_mean[:, None], phi_max[:, None], phi_min[:, None], surf_f_mean, surf_f_max],
        axis=1
    ).astype(np.float32)
    return mm


class _NodeDataView:
    def __init__(self, pyg_data):
        self._d = pyg_data

    def __getitem__(self, key):
        return getattr(self._d, key)

    def __setitem__(self, key, value):
        setattr(self._d, key, value)

    def get(self, key, default=None):
        return getattr(self._d, key, default)


class _NodeView:
    def __init__(self, pyg_data):
        self.data = _NodeDataView(pyg_data)


class PatchGraph:
    def __init__(self, pyg_data):
        self.data = pyg_data
        self.nodes = {"patch": _NodeView(self.data)}

    @property
    def ntypes(self):
        return ["patch"]

    @property
    def device(self):
        for k in ["mm", "pos", "spe", "phi_mean", "edge_index"]:
            if hasattr(self.data, k):
                t = getattr(self.data, k)
                if torch.is_tensor(t):
                    return t.device
        return torch.device("cpu")

    def to(self, device):
        self.data = self.data.to(device)
        self.nodes = {"patch": _NodeView(self.data)}
        return self

    def num_nodes(self, ntype= "patch"):
        if ntype != "patch":
            return 0
        return int(self.data.num_nodes)

    def batch_num_nodes(self, ntype="patch"):
        ptr = self.data.ptr
        return (ptr[1:] - ptr[:-1]).to(torch.long)


@dataclass
class GraphItem:
    g: PatchGraph
    pid: str


class SurfaceProteinDirDatasetPatch(Dataset):
    def __init__(self, data_root, geo_k=16, spe_k_patch=16, patch_k=32, patch_knn_k=8, p_min=16, p_max=192,
                 k_min=16, k_max=48, fps_seed=0, r2p_knn_k=8, all_assign_path=None):
        self.esp_root = data_root
        self.data_root = data_root
        self.geo_k = geo_k
        self.spe_k_patch = spe_k_patch
        self.patch_k = patch_k
        self.patch_knn_k = patch_knn_k
        self.p_min = p_min
        self.p_max = p_max
        self.k_min = k_min
        self.k_max = k_max
        self.fps_seed = fps_seed
        self.r2p_knn_k = int(r2p_knn_k)
        self.all_assign_path = all_assign_path
        self.all_assign = load_all_assign_features(all_assign_path)

        self.pids = sorted([os.path.basename(p) for p in glob.glob(os.path.join(data_root, '*')) if os.path.isdir(p)])

    def __len__(self):
        return len(self.pids)

    def _adaptive_PK(self, N):
        K_base = int(self.patch_k)
        P = int(round(N / max(1, K_base)))
        P = int(np.clip(P, self.p_min, self.p_max))
        K = int(round(N / max(1, P)))
        K = int(np.clip(K, self.k_min, self.k_max))
        P = int(min(P, max(1, N // max(1, self.k_min))))
        P = max(1, P)
        return P, K

    @staticmethod
    def _knn_patch_to_res_edges(patch_xyz: np.ndarray, res_xyz: np.ndarray, k: int):
        P = int(patch_xyz.shape[0])

        diff = patch_xyz[:, None, :] - res_xyz[None, :, :]
        dist2 = (diff * diff).sum(axis=-1)
        nn_idx = np.argsort(dist2, axis=1)[:, :k]

        src = nn_idx.reshape(-1)
        dst = np.repeat(np.arange(P, dtype=np.int64), k)
        return src.astype(np.int64), dst.astype(np.int64)

    def _find_pdb(self, prot_dir: str):
        cands = sorted(glob.glob(os.path.join(prot_dir, "*.pdb")))
        return cands[0] if len(cands) > 0 else None

    def _load_backbone_residue_centroids_and_feat(self, pid: str):
        prot_dir = os.path.join(self.esp_root, pid)
        pdb_path = self._find_pdb(prot_dir)
        if pdb_path is None:
            raise FileNotFoundError(f"No .pdb file found for pid={pid} under {prot_dir}")

        ca_coords = {}
        res_name = {}
        with open(pdb_path, "r") as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                atom = line[12:16].strip()
                if atom != "CA":
                    continue
                rname = line[17:20].strip()
                chain = line[21].strip()
                resseq = line[22:26].strip()
                icode = line[26].strip()
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                key = (chain, resseq, icode)
                ca_coords[key] = (x, y, z)
                res_name[key] = rname

        keys = list(ca_coords.keys())
        if len(keys) == 0:
            raise ValueError(f"No CA atoms found in pdb for pid={pid}: {pdb_path}")

        res_xyz = np.array([ca_coords[k] for k in keys], dtype=np.float32)
        feat = np.zeros((len(keys), 7), dtype=np.float32)
        for i, k in enumerate(keys):
            aa3 = res_name.get(k, "UNK")
            if aa3 not in AA3_TO_ASSIGN_IDX:
                raise ValueError(
                    f"Unsupported residue '{aa3}' in pid={pid}, pdb={pdb_path}, "
                    f"residue_key={k}. all_assign.txt only supports: {AA3_ASSIGN_ORDER}"
                )
            feat[i] = self.all_assign[AA3_TO_ASSIGN_IDX[aa3]]
        return res_xyz, feat

    def _load_patch_from_pointtxt(self, prot_dir: str, idx: int):
        pt_path = os.path.join(prot_dir, "point.txt")

        surf = read_point_txt(pt_path)
        pts = surf.astype(np.float32)

        phi = surf[:, 0].copy()
        xyz = surf[:, 1:4].copy()
        N = xyz.shape[0]

        idxs16, dist16 = knn_indices(xyz, k=16)
        idxs24, dist24 = knn_indices(xyz, k=24)

        gradn, logp = discrete_gradient_features_from_knn(phi, xyz, idxs16, dist16)
        normals, kH, kG = estimate_normals_pca_from_knn(xyz, idxs16)
        s_pers = knn_density_at_scale(dist24, k=24)
        lscale = local_scale_feature_from_knn(dist16, k=16)

        gradn, logp, kH, kG, lscale = stabilize_aux_features(
            gradn, logp, kH, kG, lscale, clip_q=0.995
        )
        x_surf = np.stack([phi, gradn, logp, lscale, kH, kG, s_pers], axis=1).astype(np.float32)

        P, K = self._adaptive_PK(N)
        centers_idx = fps_indices(xyz, P=P, seed=(self.fps_seed + idx) % 1000000)

        center_xyz = xyz[centers_idx]
        assign = assign_points_to_centers(xyz, center_xyz)

        patch_idx, point_mask = build_patches_no_overlap(
            xyz=xyz, centers_idx=centers_idx, assign=assign, K=K, seed=(self.fps_seed + idx + 17) % 1000000
        )

        mm_patch = compute_patch_meanmax(
            x_surf=x_surf, patch_idx=patch_idx, point_mask=point_mask, invalid_fill=-1e9
        ).astype(np.float32)

        phi_mean = mm_patch[:, 0:1].astype(np.float32)

        pm = point_mask.astype(np.float32)
        denom = np.maximum(pm.sum(axis=1, keepdims=True), 1e-6)
        pts_xyz = xyz[patch_idx]
        patch_xyz_raw = (pts_xyz * pm[..., None]).sum(axis=1) / denom  # raw coords (P,3)

        xyz_norm, _, _ = _normalize_xyz_per_protein(patch_xyz_raw.astype(np.float32))

        patch_xyz = patch_xyz_raw.astype(np.float32)
        xyz_norm = xyz_norm.astype(np.float32)

        kk = int(min(self.patch_knn_k, max(1, P - 1)))
        p_edges = patch_knn_edges(patch_xyz_raw.astype(np.float32), k=kk)

        if len(p_edges) == 0:
            edge_index_patch = np.zeros((2, 0), dtype=np.int64)
        else:
            edge_index_patch = np.array(p_edges, dtype=np.int64).T

        A = build_adjacency_from_edges(P, p_edges, sym=True)
        spe_patch = laplacian_pe(A, k=self.spe_k_patch).astype(np.float32)

        return pts, mm_patch, patch_xyz, xyz_norm, phi_mean, edge_index_patch, spe_patch

    def __getitem__(self, idx):

        pid = self.pids[idx]
        prot_dir = os.path.join(self.esp_root, pid)

        pts, mm_patch, patch_xyz, xyz_norm, phi_mean, edge_index_patch, spe_patch = self._load_patch_from_pointtxt(prot_dir, idx)

        P = int(mm_patch.shape[0])

        res_xyz_raw, res_feat = self._load_backbone_residue_centroids_and_feat(pid)
        R = int(res_xyz_raw.shape[0])

        src_r, dst_p = self._knn_patch_to_res_edges(
            patch_xyz.astype(np.float32),
            res_xyz_raw.astype(np.float32),
            int(self.r2p_knn_k),
        )

        N = R + P
        patch_offset = R

        pos_all = np.zeros((N, 3), dtype=np.float32)
        if R > 0:
            pos_all[:R] = res_xyz_raw.astype(np.float32)
        pos_all[patch_offset:patch_offset + P] = patch_xyz.astype(np.float32)

        phi_all = np.zeros((N, 1), dtype=np.float32)
        phi_mean_arr = phi_mean.astype(np.float32)
        if phi_mean_arr.ndim == 1:
            phi_all[patch_offset:patch_offset + P, 0] = phi_mean_arr.reshape(P)
        else:
            phi_all[patch_offset:patch_offset + P, 0] = phi_mean_arr.reshape(P)

        spe_patch = spe_patch.astype(np.float32)
        spe_k = int(spe_patch.shape[1]) if spe_patch.ndim == 2 else int(self.spe_k_patch)
        spe_all = np.zeros((N, spe_k), dtype=np.float32)
        spe_all[patch_offset:patch_offset + P] = spe_patch

        mm_patch = mm_patch.astype(np.float32)
        mm_dim = int(mm_patch.shape[1])
        mm_all = np.zeros((N, mm_dim), dtype=np.float32)
        mm_all[patch_offset:patch_offset + P] = mm_patch

        xyz_all = np.zeros((N, 3), dtype=np.float32)
        xyz_all[patch_offset:patch_offset + P] = xyz_norm.astype(np.float32)

        is_patch = np.zeros((N,), dtype=np.int64)
        is_patch[patch_offset:patch_offset + P] = 1

        # edges: combine r2p + p2p, plus edge_type
        edge_index_patch = edge_index_patch.astype(np.int64)

        edge_index_patch_shift = edge_index_patch.copy()
        if edge_index_patch_shift.shape[1] > 0:
            edge_index_patch_shift[0, :] += patch_offset
            edge_index_patch_shift[1, :] += patch_offset

        edge_index_r2p = np.stack([src_r, dst_p + patch_offset], axis=0).astype(np.int64)

        E_rp = int(edge_index_r2p.shape[1])
        E_pp = int(edge_index_patch_shift.shape[1])
        edge_index_all = np.concatenate([edge_index_r2p, edge_index_patch_shift], axis=1)
        edge_type = np.concatenate([
            np.zeros((E_rp,), dtype=np.int64),
            np.ones((E_pp,), dtype=np.int64),
        ], axis=0)

        C = int(res_feat.shape[1])
        x_res_all = np.zeros((N, C), dtype=np.float32)
        if R > 0 and C > 0:
            x_res_all[:R] = res_feat.astype(np.float32)

        data = Data(
            edge_index=torch.from_numpy(edge_index_all).long(),
            num_nodes=int(N),
            pos=torch.from_numpy(pos_all).float(),
            phi_mean=torch.from_numpy(phi_all).float(),
            spe=torch.from_numpy(spe_all).float(),
            mm=torch.from_numpy(mm_all).float(),
            xyz=torch.from_numpy(xyz_all).float(),
            is_patch=torch.from_numpy(is_patch).long(),
            x_res=torch.from_numpy(x_res_all).float(),
            edge_type=torch.from_numpy(edge_type).long(),
        )
        data.num_res = int(R)
        data.num_patch = int(P)
        data.patch_offset = int(patch_offset)

        g = PatchGraph(data)
        return GraphItem(g=g, pid=pid)


class SurfaceGraphFileDataset(Dataset):
    def __init__(self, bin_root):
        self.bin_root = bin_root
        self.files = sorted(glob.glob(os.path.join(bin_root, '*.pt')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        obj = torch.load(f, map_location="cpu")
        g = PatchGraph(obj)
        pid = os.path.splitext(os.path.basename(f))[0]
        return GraphItem(g=g, pid=pid)


def collate_graphitems(items):
    gs = []
    pids = []
    for it in items:
        if ("patch" in it.g.ntypes) and (it.g.num_nodes("patch") > 0):
            gs.append(it.g.data)
            pids.append(it.pid)

    batched = Batch.from_data_list(gs)
    return {"g": PatchGraph(batched), "pids": pids}


def prepare_surface_patch_graphs(data_root, save_root, skip_existing=True, geo_k=16, spe_k_patch=16, patch_k=32,
                                 patch_knn_k=8, p_min=16, p_max=192, k_min=16, k_max=48, fps_seed=0,
                                 all_assign_path=None):
    os.makedirs(save_root, exist_ok=True)
    ds = SurfaceProteinDirDatasetPatch(
        data_root=data_root,
        geo_k=geo_k,
        spe_k_patch=spe_k_patch,
        patch_k=patch_k,
        patch_knn_k=patch_knn_k,
        p_min=p_min,
        p_max=p_max,
        k_min=k_min,
        k_max=k_max,
        fps_seed=fps_seed,
        all_assign_path=all_assign_path,
    )
    total = len(ds)
    built = 0
    skipped = 0

    pbar = tqdm(total=total, desc="patch_graphs_pyg", dynamic_ncols=True, leave=True)

    for i in range(total):
        pid = ds.pids[i]
        outp = os.path.join(save_root, f"{pid}.pt")

        if skip_existing and os.path.exists(outp):
            skipped += 1
            pbar.set_postfix_str(f"skip={skipped} build={built}")
            pbar.update(1)
            continue

        item = ds[i]
        torch.save(item.g.data, outp)
        built += 1
        pbar.set_postfix_str(f"skip={skipped} build={built} last=build:{pid}")
        pbar.update(1)

    pbar.close()
    print(f"[done] built {built}, skipped {skipped}, total {total}, save_root={save_root}")

def _normalize_xyz_per_protein(xyz: np.ndarray):
    center = xyz.mean(axis=0).astype(np.float32)
    x0 = (xyz - center[None, :]).astype(np.float32)
    scale = float(np.sqrt((x0 ** 2).sum(axis=1).mean()) + 1e-6)
    x1 = (x0 / scale).astype(np.float32)
    x1 = np.clip(x1, -5.0, 5.0).astype(np.float32)
    return x1, center, scale
