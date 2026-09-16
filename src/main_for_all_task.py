import os, sys
from datetime import datetime
import copy
import json

from utils import ensure_dir, set_seed, save_json

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from train_ppi_classifier import (
    run_pretrain, run_export, run_ppi,
)

from dataloader_surface import (
    prepare_surface_patch_graphs,
)


def _paths_for(dataset, split, run_idx):
    root = os.path.abspath(os.path.join(CUR_DIR, '..'))
    esp_root = os.path.join(root, 'data', 'esp_data', f'target_data_stable_{dataset}')
    ppi_actions = os.path.join(root, 'data', 'processed_data', dataset, f'protein.actions.{dataset}.txt')
    seq_dict    = os.path.join(root, 'data', 'processed_data', dataset, f'protein.{dataset}.sequences.dictionary.csv')
    all_assign  = os.path.join(root, 'data', 'processed_data', dataset, 'all_assign.txt')
    workdir     = os.path.join(root, 'runs_src', dataset, split, str(run_idx))
    return esp_root, ppi_actions, seq_dict, all_assign, workdir


def _print_task_header(dataset, split, run_idx, seed):
    line = '=' * 70
    print(f'\n{line}\n>>> TASK: {dataset}-{split}-run{run_idx} (seed={seed}) <<<\n{line}\n', flush=True)


def _patch_cache_dir(esp_root, base):
    tag = (
        f"patch_pyg"
        f"_bb2patch"
        f"_assign7"
    )
    return esp_root + f"_{tag}_graphs"


def _bb_cache_dir(esp_root, base):
    tag = (
        f"bb_pyg"
        f"_knn{base.bb_knn_k}"
        f"_spe{base.spe_k_bb}"
        f"_seq{1 if base.bb_use_seq_edges else 0}"
        f"_xyznorm{1 if base.bb_xyz_norm else 0}"
        f"_cc"
    )
    return esp_root + f"_{tag}_graphs"


def _bb_ckpt_name(base):
    tag = (
        f"bbGT"
        f"_hid{base.bb_hid}_d{base.bb_depth}"
        f"_m{base.bb_mask_ratio:.2f}"
    )
    return f"best_{tag}.pt"


def _bb_embed_name(base):
    tag = (
        f"bbEmb"
        f"_hid{base.bb_hid}_d{base.bb_depth}"
    )
    return f"backbone_{tag}.pt"


def _ckpt_name(base):
    tag = (
        f"patchVQMAE"
        f"_hid{base.hid}_d{base.depth}"
        f"_cb{base.codebook_p}"
        f"_m{base.mask_patch_init:.2f}-{base.mask_patch_max:.2f}"
    )
    return f"best_{tag}.pt"


def _embed_name(base):
    tag = (
        f"patchEmb"
        f"_hid{base.hid}_d{base.depth}"
        f"_cb{base.codebook_p}"
    )
    return f"surface_{tag}.pt"


def _load_task_config(cfg_path):
    if not os.path.exists(cfg_path):
        print(f"[WARN] config.json not found at: {cfg_path}. Use defaults in main_for_all_task.py")
        return {}

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            print(f"[WARN] config.json root must be dict, got {type(cfg)}. Ignore config.")
            return {}
        return cfg
    except Exception as e:
        print(f"[WARN] Failed to read config.json ({cfg_path}): {e}. Ignore config.")
        return {}


def _get_task_overrides(task_cfg, ds, sp):
    d = task_cfg.get(ds, {})
    if not isinstance(d, dict):
        return {}
    s = d.get(sp, {})
    if not isinstance(s, dict):
        return {}
    return s


def _apply_overrides(args, over):
    for k, v in over.items():
        setattr(args, k, v)


def main():
    datasets = ['SHS27k', 'SHS148k', 'STRING']
    splits   = ['random', 'dfs', 'bfs']

    root = os.path.abspath(os.path.join(CUR_DIR, '..'))
    runs_root = os.path.join(root, 'runs_src')
    ensure_dir(runs_root)

    overall = os.path.join(runs_root, 'ALL_RESULTS.txt')
    with open(overall, 'w', encoding='utf-8') as agg:
        agg.write(f'# ALL RESULTS generated at {datetime.now().isoformat()}\n\n')

    cfg_path = os.path.join(CUR_DIR, "config.json")
    task_cfg = _load_task_config(cfg_path)

    class Args:
        pass

    base = Args()
    base.device = "cuda:0"
    print("Using device:", base.device)
    base.seed = 0

    base.geo_k = 16

    # Patch 图超参数
    base.spe_k_patch = 16
    base.patch_k = 32              # patch 内点数基准
    base.patch_knn_k = 16          # patch graph KNN
    base.p_min = 16
    base.p_max = 192
    base.k_min = 16
    base.k_max = 48
    base.fps_seed = 0
    base.disjoint_patches = True

    # 预训练相关（Patch-VQMAE）
    base.pre_epochs = 50
    base.pre_batch = 32
    base.pre_lr = 1e-3
    base.pre_wd = 5e-4

    # 掩码率模拟退火
    base.mask_patch_init = 0.50
    base.mask_patch_step = 0.02
    base.mask_patch_max  = 1.00

    # Patch-VQMAE 模型超参数
    base.hid = 256
    base.depth = 6
    base.heads = 4
    base.rbf_k = 8
    base.codebook_p = 1024
    base.commitment = 0.25
    base.dropout_pre = 0.1

    # Patch loss 超参数
    base.w_tgt = 1.0
    base.unmask_coef = 0.2
    base.use_lscale_weight = True
    base.beta_kl = 10

    # downstream
    base.batch_size = 2048
    base.lr = 1e-3
    base.wd = 5e-4
    base.epochs = 500
    base.log_num = 10
    base.patience = 50
    base.es_min_delta = 0.0
    base.sched_patience = 20
    base.sched_factor = 0.5
    base.hidden = 512
    base.dropout = 0.5
    base.graph_only_train = True
    base.embed_norm = "zscore"

    base.force_pretrain = False
    base.force_export = False
    base.force_rebuild_graphs = False

    base.seed_inc = True

    base.repeats_SHS27k_random   = 0
    base.repeats_SHS27k_dfs      = 0
    base.repeats_SHS27k_bfs      = 0

    base.repeats_SHS148k_random  = 100
    base.repeats_SHS148k_dfs     = 100
    base.repeats_SHS148k_bfs     = 100

    base.repeats_STRING_random   = 0
    base.repeats_STRING_dfs      = 0
    base.repeats_STRING_bfs      = 0

    done_pretrain = set()

    GRAPH_RELATED_KEYS = {
        "geo_k", "spe_k_patch", "patch_k", "patch_knn_k",
        "p_min", "p_max", "k_min", "k_max", "fps_seed", "disjoint_patches"
    }

    for ds in datasets:
        total_repeats = 0
        for sp in splits:
            rep_attr = f'repeats_{ds}_{sp}'
            total_repeats += int(getattr(base, rep_attr, 1))
        if total_repeats <= 0:
            print(f"[{ds}] No tasks configured. Skip dataset.")
            continue

        esp_root, ppi_actions, seq_dict, all_assign, _ = _paths_for(ds, splits[0], 1)

        prebuilt_root = _patch_cache_dir(esp_root, base)
        print(f'=== [BIN] PREPARE patch surface graphs (dataset={ds}, fps_seed={base.fps_seed}) ===')
        prepare_surface_patch_graphs(
            data_root=esp_root,
            save_root=prebuilt_root,
            skip_existing=(not base.force_rebuild_graphs),
            geo_k=base.geo_k,
            spe_k_patch=base.spe_k_patch,
            patch_k=base.patch_k,
            patch_knn_k=base.patch_knn_k,
            p_min=base.p_min,
            p_max=base.p_max,
            k_min=base.k_min,
            k_max=base.k_max,
            fps_seed=base.fps_seed,
            all_assign_path=all_assign,
        )

        for sp in splits:
            rep_attr = f'repeats_{ds}_{sp}'
            repeats = int(getattr(base, rep_attr, 1))

            for run_idx in range(1, repeats + 1):
                over = _get_task_overrides(task_cfg, ds, sp)

                if "seed" in over:
                    seed = int(over["seed"])
                else:
                    if getattr(base, "seed_inc", True):
                        seed = int(base.seed + (run_idx - 1))
                    else:
                        seed = int(base.seed)

                _print_task_header(ds, sp, run_idx, seed)

                esp_root, ppi_actions, seq_dict, all_assign, workdir = _paths_for(ds, sp, run_idx)

                args = copy.deepcopy(base)
                args.dataset = ds
                args.split_mode = sp
                args.esp_root = esp_root
                args.ppi_actions = ppi_actions
                args.seq_dict = seq_dict
                args.all_assign = all_assign
                args.workdir = workdir

                args.prebuilt_root = prebuilt_root

                _apply_overrides(args, over)

                args.seed = int(seed)
                set_seed(args.seed)

                for k in list(over.keys()):
                    if k in GRAPH_RELATED_KEYS:
                        bv = getattr(base, k, None)
                        av = getattr(args, k, None)
                        if av != bv:
                            print(f"[WARN] override '{k}'={av} differs from base.{k}={bv}, "
                                  f"but prebuilt PATCH graphs were already prepared using base.* (no rebuild).")

                args.fps_seed = int(base.fps_seed)

                pretrain_workdir = os.path.join(root, 'runs_src', ds, 'pretrain')
                ckpt_dir = os.path.join(pretrain_workdir, 'ckpts')
                embed_dir = os.path.join(pretrain_workdir, 'embeds')
                ensure_dir(pretrain_workdir)
                ensure_dir(ckpt_dir)
                ensure_dir(embed_dir)

                out_dir = os.path.join(args.workdir, 'ppi_results')
                ensure_dir(out_dir)

                cfg_path_pre = os.path.join(pretrain_workdir, 'config_patch_vqmae.json')
                if (not os.path.exists(cfg_path_pre)) or args.force_pretrain or args.force_export:
                    save_json(vars(args), cfg_path_pre)

                cfg_path_run = os.path.join(args.workdir, 'config_run.json')
                save_json(vars(args), cfg_path_run)

                ckpt_path = os.path.join(ckpt_dir, _ckpt_name(args))
                out_embed = os.path.join(embed_dir, _embed_name(args))

                results_txt = os.path.join(out_dir, f'results_{ds}_{sp}.txt')

                key = (ds, int(args.seed))

                if key not in done_pretrain:
                    done_pretrain.add(key)

                    if (not os.path.exists(ckpt_path)) or args.force_pretrain:
                        print(f'=== [A] PRETRAIN Patch-VQMAE (dataset={ds}, seed={args.seed}) ===')
                        run_pretrain(args, ckpt_path)
                    else:
                        print(f'=== [A] PRETRAIN skipped (found ckpt): {ckpt_path} ===')

                    if (not os.path.exists(out_embed)) or args.force_export:
                        print(f'=== [B] EXPORT protein embeddings (dataset={ds}, seed={args.seed}) ===')
                        run_export(args, ckpt_path, out_embed)
                    else:
                        print(f'=== [B] EXPORT skipped (found embeds): {out_embed} ===')
                else:
                    print(f'=== [A/B] upstream already prepared for (dataset={ds}, seed={args.seed}) ===')

                # ===== [C] downstream: PATCH ONLY (NO BB) =====
                print(f'=== [C] TRAIN PPI classifier ({ds}-{sp}, run {run_idx}, seed={args.seed}) ===')
                run_ppi(args, out_dir, out_embed, results_txt)

                with open(results_txt, 'r', encoding='utf-8') as rf, \
                     open(overall, 'a', encoding='utf-8') as agg:
                    agg.write(f'\n----- {ds}-{sp}-run{run_idx} (seed={args.seed}) -----\n')
                    agg.write(rf.read().strip() + '\n')

    print('\nAll tasks (with repeats) finished.')
    print('Aggregated results ->', overall)


if __name__ == '__main__':
    main()
