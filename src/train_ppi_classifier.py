import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import torch.utils.data as tud
from torch.utils.data import DataLoader

from dataloader_surface import SurfaceGraphFileDataset, collate_graphitems
from models_surface import HESQ_Net, pretrain_losses_patch
from models_ppi import LinkPredGCN
from utils import (
    set_seed,
    ensure_dir,
    save_json,
    evaluat_metrics,
    default_path,
    require_prebuilt_root,
    PatchReadout,
    CLS_MAP,
    load_embeddings,
    read_seq_dict_ordered,
    build_ppi_and_labels,
    build_mp_edge_index,
    split_edges_with_mode,
    run_epoch,
    eval_epoch_full_micro,
)


def run_pretrain(args, ckpt_path):
    device = args.device

    prebuilt_root = require_prebuilt_root(args)
    ds_tr = SurfaceGraphFileDataset(prebuilt_root)

    dl_tr = tud.DataLoader(
        ds_tr,
        batch_size=args.pre_batch,
        shuffle=True,
        collate_fn=collate_graphitems,
    )

    model = HESQ_Net(
        hid=args.hid,
        depth=args.depth,
        heads=getattr(args, "heads", 4),
        rbf_k=getattr(args, "rbf_k", 8),
        mask_ratio_patch=getattr(args, "mask_patch_init", 0.6),
        dropout=getattr(args, "dropout_pre", 0.1),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.pre_lr,
        weight_decay=args.pre_wd,
    )

    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    os.makedirs(ckpt_dir, exist_ok=True)
    pretrain_log_path = os.path.join(ckpt_dir, "pretrain_log.txt")

    with open(pretrain_log_path, "w", encoding="utf-8") as lg:
        lg.write("=== [A] PRETRAIN Patch-VQMAE ===\n")
        lg.write("Columns: ep | train_loss\n")
        lg.flush()

        for ep in range(1, args.pre_epochs + 1):
            if hasattr(args, "mask_patch_init"):
                cur_mask_patch = args.mask_patch_init + getattr(args, "mask_patch_step", 0.0) * (ep - 1)
                cur_mask_patch = min(cur_mask_patch, getattr(args, "mask_patch_max", cur_mask_patch))
                cur_mask_patch = max(cur_mask_patch, 0.0)
            else:
                cur_mask_patch = model.mask_ratio_patch
            model.mask_ratio_patch = float(cur_mask_patch)

            model.train()
            sum_loss = 0.0
            n_step = 0

            for batch in dl_tr:
                g = batch["g"].to(device)

                out = model(g)
                preds, vq, hdict = out

                loss = pretrain_losses_patch(
                    g, preds, vq, hdict,
                    w_recon=args.w_tgt
                )

                opt.zero_grad()
                loss.backward()
                opt.step()

                sum_loss += float(loss.item())
                n_step += 1

            tr = sum_loss / max(1, n_step)
            msg = f"[A:EP {ep:03d}] train_loss {tr:.4f}"
            print(msg)
            lg.write(msg + "\n")
            lg.flush()

        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": vars(args),
            },
            ckpt_path,
        )

        done_msg = f"[A] Done. Final epoch model saved to {ckpt_path}"
        print(done_msg)
        lg.write(done_msg + "\n")
        lg.flush()

    print("[A] Pretrain log written to:", pretrain_log_path)


@torch.no_grad()
def run_export(args, ckpt_path, out_embed_pt):
    device = args.device

    prebuilt_root = require_prebuilt_root(args)
    ds = SurfaceGraphFileDataset(prebuilt_root)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_graphitems)

    ckpt = torch.load(ckpt_path, map_location=device)

    model = HESQ_Net(
        patch_spe_dim=getattr(args, "spe_k_patch", 16),
        hid=args.hid,
        depth=args.depth,
        heads=getattr(args, "heads", 4),
        rbf_k=getattr(args, "rbf_k", 8),
        mask_ratio_patch=0.0,
        dropout=getattr(args, "dropout_pre", 0.1),
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    readout = PatchReadout().to(device)
    readout.eval()

    embed_dict = {}

    for batch in dl:
        pids = batch.get("pids", [])
        g = batch["g"].to(device)

        preds, vq, hdict = model(g)
        zq = vq

        emb = readout(g, zq)
        embed_dict[pids[0]] = emb.detach().float().cpu()

    torch.save(embed_dict, out_embed_pt)
    print("[B] saved embeddings only:", out_embed_pt)


def run_ppi(args, out_dir, embed_pt, results_txt_path):
    set_seed(args.seed)
    ensure_dir(out_dir, overwrite=False)

    embed_map = load_embeddings(embed_pt)
    if not isinstance(embed_map, dict) or len(embed_map) == 0:
        raise ValueError(f"Invalid embedding file: {embed_pt}")
    hid = next(iter(embed_map.values())).numel()

    protein_name_to_id = read_seq_dict_ordered(args.seq_dict)

    missing_pids = [pid for pid in protein_name_to_id.keys() if pid not in embed_map]
    if missing_pids:
        raise KeyError(
            f"Embedding file is missing {len(missing_pids)} PIDs from sequence dictionary. "
            f"Examples: {missing_pids[:10]}"
        )

    nodes, edges, labels = build_ppi_and_labels(
        args.ppi_actions,
        protein_name_to_id,
    )
    print(
        f"[C] {args.dataset}-{args.split_mode} | #nodes={len(nodes)}  #edges={len(edges)} | "
        f"node_order=sequence_dictionary"
    )

    X = torch.zeros((len(nodes), hid), dtype=torch.float32)
    for i, pid in enumerate(nodes):
        if pid is None:
            raise ValueError(f"nodes[{i}] is None; sequence dictionary mapping is broken.")
        X[i] = embed_map[pid].float()

    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    X = (X - mu) / sd

    proc_dir = getattr(args, "proc_dir", None)

    if proc_dir is None:
        ppi_actions_path = getattr(args, "ppi_actions", None)
        if ppi_actions_path:
            proc_dir = os.path.dirname(os.path.abspath(ppi_actions_path))

    if proc_dir is None:
        data_root = getattr(args, "data_root", None)
        if data_root:
            proc_dir = os.path.join(os.path.abspath(data_root), "processed_data", args.dataset)

    if proc_dir is None:
        proc_dir = default_path(os.path.join("data", "processed_data", args.dataset))

    os.makedirs(proc_dir, exist_ok=True)
    predefined_split = os.path.join(proc_dir, f"{args.dataset}_{args.split_mode}_split.json")
    print(f"[C] split lookup path = {predefined_split}")

    if os.path.isfile(predefined_split):
        print(f"[C] Using predefined split: {predefined_split}")
        with open(predefined_split, "r", encoding="utf-8") as f:
            split_obj = json.load(f)
        tr_idx = split_obj["train_index"]
        va_idx = split_obj["val_index"]
        te_idx = split_obj["test_index"]
    else:
        print(f"[C] No predefined split found, generate new split with mode={args.split_mode}")
        tr_idx, va_idx, te_idx = split_edges_with_mode(edges, args.split_mode, seed=args.seed)

    save_json(
        {
            "dataset": args.dataset,
            "split_mode": args.split_mode,
            "node_order": "sequence_dictionary",
            "train_index": tr_idx,
            "val_index": va_idx,
            "test_index": te_idx,
        },
        os.path.join(out_dir, f"{args.dataset}_{args.split_mode}_split_seed{args.seed}.json")
    )

    save_json(
        {
            "dataset": args.dataset,
            "split_mode": args.split_mode,
            "node_order": "sequence_dictionary",
            "train_edges": [edges[i] for i in tr_idx],
            "val_edges": [edges[i] for i in va_idx],
            "test_edges": [edges[i] for i in te_idx],
        },
        os.path.join(out_dir, f"{args.dataset}_{args.split_mode}_split_edges_seed{args.seed}.json")
    )

    graph_only_train = bool(getattr(args, "graph_only_train", True))
    test_use_all_graph = bool(getattr(args, "test_use_all_graph", True))

    mp_edges_train = [edges[i] for i in tr_idx]
    edge_index_train = build_mp_edge_index(mp_edges_train, num_nodes=len(nodes))
    edge_index_all = build_mp_edge_index(edges, num_nodes=len(nodes))

    if graph_only_train:
        print(f"[C] graph_only_train=True | train_mp_edges={len(mp_edges_train)} | all_mp_edges={len(edges)}")
    else:
        print(f"[C] graph_only_train=False | all_mp_edges={len(edges)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    labels = labels.to(device)
    edge_index_train = edge_index_train.to(device)
    edge_index_all = edge_index_all.to(device)

    g_tr = Data(
        edge_index=edge_index_train if graph_only_train else edge_index_all,
        num_nodes=len(nodes)
    )
    g_test = Data(
        edge_index=edge_index_all if (graph_only_train and test_use_all_graph) else g_tr.edge_index,
        num_nodes=len(nodes)
    )

    if graph_only_train:
        print(
            f"[C] test_use_all_graph={test_use_all_graph} | "
            f"test_mp_graph={'all_graph' if (graph_only_train and test_use_all_graph) else 'train_graph'}"
        )

    edges_t = torch.as_tensor(np.asarray(edges), dtype=torch.long, device=device)

    in_dim = int(X.shape[1])
    model = LinkPredGCN(
        n_layers=getattr(args, "n_layers", 2),
        in_dim=in_dim,
        hidden=getattr(args, "hidden", 512),
        out_dim=getattr(args, "out_dim", 64),
        class_num=len(CLS_MAP),
        dropout=getattr(args, "dropout", 0.5)
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    sched_patience = int(getattr(args, "sched_patience", 10))
    sched_factor = float(getattr(args, "sched_factor", 0.5))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=sched_factor, patience=sched_patience, verbose=True
    )

    loss_fn = nn.BCEWithLogitsLoss()

    best = {"val": 0.0, "test_at_val": 0.0, "test_best": 0.0, "epoch": 0}
    es = 0
    es_patience = int(getattr(args, "patience", 50))
    es_min_delta = float(getattr(args, "es_min_delta", 0.0))

    log_path = os.path.join(out_dir, f"train_log_{args.dataset}_{args.split_mode}.txt")
    best_ckpt_path = os.path.join(
        out_dir,
        f"best_classifier_{args.dataset}_{args.split_mode}.pt"
    )

    with open(log_path, "w", encoding="utf-8") as lg:
        lg.write(f"===== {args.dataset}-{args.split_mode} | patch_only=True =====\n")
        lg.write(f"surface_embed={embed_pt}\n")
        lg.write(f"in_dim={in_dim}\n")
        lg.write(f"best_ckpt={best_ckpt_path}\n")
        lg.write("ep | tr(loss,f1) | va(loss,f1) | te(loss,f1) | lr\n")
        lg.flush()

        for ep in range(1, int(args.epochs) + 1):
            tr_loss, tr_f1 = run_epoch(
                model=model,
                graph=g_tr,
                X=X,
                edges_t=edges_t,
                labels=labels,
                indices=tr_idx,
                batch_size=int(args.batch_size),
                device=device,
                loss_fn=loss_fn,
                opt=opt,
                train=True
            )
            va_loss, va_f1 = eval_epoch_full_micro(
                model=model,
                graph=g_test,
                X=X,
                edges_t=edges_t,
                labels=labels,
                indices=va_idx,
                batch_size=int(args.batch_size),
                device=device,
                loss_fn=loss_fn
            )
            te_loss, te_f1 = eval_epoch_full_micro(
                model=model,
                graph=g_test,
                X=X,
                edges_t=edges_t,
                labels=labels,
                indices=te_idx,
                batch_size=int(args.batch_size),
                device=device,
                loss_fn=loss_fn
            )

            scheduler.step(va_loss)
            cur_lr = opt.param_groups[0]["lr"]

            msg = (
                f"[EP {ep:03d}] "
                f"tr {tr_loss:.4f}/{tr_f1:.4f}  "
                f"va {va_loss:.4f}/{va_f1:.4f}  "
                f"te {te_loss:.4f}/{te_f1:.4f}  "
                f"lr {cur_lr:.2e}"
            )
            print(msg)
            lg.write(msg + "\n")
            lg.flush()

            if va_f1 > best["val"] + es_min_delta:
                best["val"] = va_f1
                best["test_at_val"] = te_f1
                best["epoch"] = ep
                es = 0

                torch.save(
                    {
                        "epoch": ep,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_f1": va_f1,
                        "test_f1_at_best_val": te_f1,
                        "val_loss": va_loss,
                        "test_loss": te_loss,
                        "args": vars(args),
                        "in_dim": in_dim,
                        "surface_embed": embed_pt,
                        "graph_only_train": graph_only_train,
                        "test_use_all_graph": test_use_all_graph,
                    },
                    best_ckpt_path
                )

                save_msg = (
                    f"[BEST] ep={ep} | val_f1={va_f1:.4f} | "
                    f"test_f1_at_best_val={te_f1:.4f} | saved={best_ckpt_path}"
                )
                print(save_msg)
                lg.write(save_msg + "\n")
                lg.flush()
            else:
                es += 1

            best["test_best"] = max(best["test_best"], te_f1)

            if es >= es_patience:
                lg.write(f"[STOP] early stop at ep={ep}, best_val={best['val']:.4f} @ep={best['epoch']}\n")
                lg.flush()
                break

    with open(results_txt_path, "w", encoding="utf-8") as rf:
        rf.write(f"{args.dataset}-{args.split_mode}\n")
        rf.write(f"best_val_f1={best['val']:.6f} (ep={best['epoch']})\n")
        rf.write(f"test_f1_at_best_val={best['test_at_val']:.6f}\n")
        rf.write(f"test_best_f1={best['test_best']:.6f}\n")
        rf.write(f"surface_embed={embed_pt}\n")
        rf.write(f"in_dim={in_dim}\n")
        rf.write(f"graph_only_train={graph_only_train}\n")
        rf.write(f"test_use_all_graph={test_use_all_graph}\n")
        rf.write(f"best_ckpt={best_ckpt_path}\n")

    print("[C] results saved to:", results_txt_path)
    print("[C] train log saved to:", log_path)
    print("[C] best classifier saved to:", best_ckpt_path)
