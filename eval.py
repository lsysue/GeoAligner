# eval.py

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# 引入项目模块
from utils.config import load_config, Config, config_to_dict, arg_parser
from datasets.img2geo_dataset import Img2GeoDataset
from encoders.image_encoder import ImageEncoder
from encoders.location_encoder import GPSEncoder
from utils.metrics import Evaluator, METRIC_TOP_KS, METRIC_THRESHOLDS_KM
from utils.reporter import dataset_name_from_csv
from reasoners.retriever import Retriever

RESULTS_DIR = "./results"
RESULTS_CSV = "eval.csv"

def _strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def _select_encoder_state_dict(checkpoint, key, use_ema=True):
    if use_ema and isinstance(checkpoint, dict):
        ema = checkpoint.get("ema", None)
        if isinstance(ema, dict):
            ema_key = ema.get(key, None)
            if isinstance(ema_key, dict):
                shadow = ema_key.get("shadow", None)
                if isinstance(shadow, dict):
                    print(f"Using EMA {key} weights from checkpoint.")
                    return _strip_module_prefix(shadow)

    state_dict = checkpoint.get(key, checkpoint)
    print(f"Using non-EMA {key} weights from checkpoint.")
    return _strip_module_prefix(state_dict)


def _collect_metric_fields(metrics_dict):
    out = {}
    for k in METRIC_TOP_KS:
        for t in METRIC_THRESHOLDS_KM:
            key = f"r@{k}_{t}km"
            out[key] = float(metrics_dict.get(key, float('nan')))
    out["median_error_km"] = float(metrics_dict.get("median_error_km", float('nan')))
    return out


def _csv_columns():
    metric_cols = []
    for k in METRIC_TOP_KS:
        for t in METRIC_THRESHOLDS_KM:
            metric_cols.append(f"r@{k}_{t}km")

    return [
        "timestamp",
        "test_set",
        "run_dir",
        "checkpoint",
        "top_k",
        "nprobe",
        "use_ivf",
        "nlist",
        "use_rerank",
        "rerank_topk",
        "rerank_fusion_mode",
        "rerank_s_weight",
        "rerank_dynamic_temperature",
        "rerank_dynamic_consistency_topk",
        "rerank_dynamic_weight_gap",
        "rerank_dynamic_weight_entropy",
        "rerank_dynamic_weight_consistency",
        "rerank_dynamic_alpha_min",
        "rerank_dynamic_alpha_max",
        "ot_eps",
        "ot_iters",
        "kde_weight",
        "kde_sigma_km",
        *metric_cols,
        "median_error_km",
    ]


def _compute_metrics_from_all_query_dists(all_query_dists):
    metrics_out = {}
    if all_query_dists.size == 0:
        metrics_out["median_error_km"] = float('nan')
        return metrics_out

    topk_retrieval = all_query_dists.shape[1]
    for k in METRIC_TOP_KS:
        if k > topk_retrieval:
            continue
        min_dists_topk = np.min(all_query_dists[:, :k], axis=1)
        for t in METRIC_THRESHOLDS_KM:
            metrics_out[f"r@{k}_{t}km"] = float(np.mean(min_dists_topk <= t))
        if k == 1:
            metrics_out["median_error_km"] = float(np.median(min_dists_topk))
    return metrics_out


def _print_metrics(metrics_out, topk_retrieval):
    print("\n" + "=" * 50)
    print("Geodesic Distance Recall Evaluation")
    print("=" * 50)

    for k in METRIC_TOP_KS:
        if k > topk_retrieval:
            continue
        print(f"\n--- Metrics @ Top-{k} ---")
        for t in METRIC_THRESHOLDS_KM:
            key = f"r@{k}_{t}km"
            if key in metrics_out:
                print(f"Recall @ {t:4d} km: {metrics_out[key] * 100:.6f}%")
        if k == 1 and "median_error_km" in metrics_out:
            print(f"Median Error: {metrics_out['median_error_km']:.6f} km")

    print("=" * 50)


def _append_eval_record(run_dir, checkpoint, cfg, run_meta, metrics_dict):
    save_dir = os.path.join(RESULTS_DIR, run_dir)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, RESULTS_CSV)
    retrieval_cfg = cfg.retrieval

    record = {
        "timestamp": run_meta["timestamp"],
        "test_set": run_meta["test_set"],
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "top_k": int(retrieval_cfg.top_k),
        "nprobe": int(retrieval_cfg.nprobe),
        "use_ivf": bool(getattr(retrieval_cfg, "use_ivf", True)),
        "nlist": int(getattr(retrieval_cfg, "nlist", 4096)),
        "use_rerank": bool(retrieval_cfg.use_rerank),
        "rerank_topk": int(getattr(retrieval_cfg, "rerank_topk", retrieval_cfg.top_k)),
        "rerank_fusion_mode": str(retrieval_cfg.rerank_fusion_mode),
        "rerank_s_weight": float(retrieval_cfg.rerank_s_weight),
        "rerank_dynamic_temperature": float(retrieval_cfg.rerank_dynamic_temperature),
        "rerank_dynamic_consistency_topk": int(retrieval_cfg.rerank_dynamic_consistency_topk),
        "rerank_dynamic_weight_gap": float(retrieval_cfg.rerank_dynamic_weight_gap),
        "rerank_dynamic_weight_entropy": float(retrieval_cfg.rerank_dynamic_weight_entropy),
        "rerank_dynamic_weight_consistency": float(retrieval_cfg.rerank_dynamic_weight_consistency),
        "rerank_dynamic_alpha_min": float(retrieval_cfg.rerank_dynamic_alpha_min),
        "rerank_dynamic_alpha_max": float(retrieval_cfg.rerank_dynamic_alpha_max),
        "ot_eps": float(retrieval_cfg.ot_eps),
        "ot_iters": int(retrieval_cfg.ot_iters),
        "kde_weight": float(retrieval_cfg.kde_weight),
        "kde_sigma_km": float(retrieval_cfg.kde_sigma_km),
    }

    record.update(_collect_metric_fields(metrics_dict))

    # Ensure stable schema and column order.
    new_row_df = pd.DataFrame([record])
    # 保持列顺序一致
    cols = _csv_columns()
    new_row_df = new_row_df.reindex(columns=cols)

    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        merged_df = pd.concat([old_df, new_row_df], ignore_index=True, sort=False)
        merged_df.to_csv(csv_path, index=False)
    else:
        new_row_df.to_csv(csv_path, index=False)
    print(f"Saved eval record to: {csv_path}")

def _load_eval_models(cfg, checkpoint_path, device):
    print(f"Loading models from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    use_ema = bool(getattr(cfg.train, "use_ema", False))

    img_cfg = cfg.model.image if hasattr(cfg.model, "image") else None
    img_cfg.img_size = cfg.data.img_size
    gps_cfg = cfg.model.gps if hasattr(cfg.model, "gps") else None

    image_encoder = ImageEncoder(img_cfg).to(device)
    image_encoder.load_state_dict(_select_encoder_state_dict(checkpoint, "image_encoder", use_ema=use_ema))
    image_encoder.eval()

    gps_encoder = None
    retrieval_cfg = cfg.retrieval
    if bool(retrieval_cfg.use_rerank):
        print("Re-ranking is ENABLED. Loading GPSEncoder...")
        gps_encoder = GPSEncoder(gps_cfg).to(device)
        gps_encoder.load_state_dict(_select_encoder_state_dict(checkpoint, "gps_encoder", use_ema=use_ema))
        gps_encoder.eval()

    return image_encoder, gps_encoder


def _build_eval_dataloader(cfg):
    test_transform = transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = Img2GeoDataset(
        csv_file=cfg.data.test.csv_file,
        img_dir=cfg.data.test.img_dir,
        transform=test_transform,
        s2_levels=getattr(cfg.model.gps, "s2_levels", [3, 6, 9, 11, 13]),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    return dataset, dataloader


def run_eval_batches(
    cfg,
    dataloader,
    image_encoder,
    gps_encoder,
    retriever,
    device,
):
    top_k = int(cfg.retrieval.top_k)

    alpha_monitor = []
    dist_chunks = []
    diag_records = {}

    if hasattr(retriever, "coords_lookup_tensor"):
        coords_lookup = retriever.coords_lookup_tensor
    else:
        coords_lookup = torch.as_tensor(
            retriever.coords_lookup,
            device=device,
            dtype=torch.float32,
        )

    base_dataset = dataloader.dataset
    if hasattr(base_dataset, "dataset"):
        metadata = base_dataset.dataset.geo_metadata
        indices = base_dataset.indices
    else:
        metadata = base_dataset.geo_metadata
        indices = None
    
    amp_enabled = bool(getattr(cfg.eval, "eval_amp", False)) and device.type == "cuda"

    with torch.inference_mode():
        for batch_idx, (_, images, gps_coords, _) in enumerate(tqdm(dataloader)):
            images = images.to(device, non_blocking=True)
            gps_coords = gps_coords.to(device, dtype=torch.float32, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                img_embeds = image_encoder(images)

            retrieve_out = retriever(
                img_embeds=img_embeds,
                top_k=top_k,
                gps_encoder=gps_encoder,
            )

            final_idx_np = retrieve_out["final_idx"]
            final_idx = torch.as_tensor(final_idx_np, device=device, dtype=torch.long)
            final_coords = coords_lookup[final_idx]  # (B, K, 2)

            alpha_batch = retrieve_out["alpha_batch_cpu"]
            if alpha_batch is not None:
                alpha_monitor.append(alpha_batch)

            gps_coords_ext = gps_coords.unsqueeze(1)  # (B,1,2)
            dists = Evaluator.haversine_km(gps_coords_ext, final_coords)  # (B,K)

            dists_cpu = torch.empty_like(dists, device="cpu", pin_memory=True)
            dists_cpu.copy_(dists, non_blocking=True)

            dist_chunks.append(dists_cpu)

            if "alpha_batch_cpu" in retrieve_out and retrieve_out["alpha_batch_cpu"] is not None:
                alphas = retrieve_out["alpha_batch_cpu"].numpy()
                batch_size = images.shape[0]
                
                for b in range(batch_size):
                    # 计算当前样本在整个数据集中的全局索引
                    global_ptr = batch_idx * dataloader.batch_size + b
                    
                    # 映射回原始 metadata 的行索引
                    actual_idx = indices[global_ptr] if indices is not None else global_ptr
                    img_name = metadata.iloc[actual_idx]['IMG_ID']
                    
                    diag_records[img_name] = {
                        "alpha": float(alphas[b]),
                        "top1_dist_km": float(dists[b, 0].item()),
                    }


    all_query_dists = torch.cat(dist_chunks, dim=0).numpy().astype(np.float32, copy=False)

    return {
        "top_k_retrieval": top_k,
        "all_query_dists": all_query_dists,
        "alpha_monitor": alpha_monitor,
    }


def evaluate(args):
    cfg = load_config(args.config, overrides=args.overrides)
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.run_dir and args.checkpoint:
        checkpoint_path = os.path.join(cfg.dirs.ckpt_dir, args.run_dir, args.checkpoint)
    elif args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        args.run_dir = os.path.basename(os.path.dirname(checkpoint_path))
    run_tag = args.run_dir

    if args.index_dir:
        index_dir = args.index_dir
    else:
        index_dir = os.path.join(cfg.dirs.index_dir, args.run_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_meta = {"timestamp": timestamp, "test_set": dataset_name_from_csv(cfg.data.test.csv_file)}
    print(f"Running evaluation on {device}...")

    retriever = Retriever(cfg=cfg, run_tag=run_tag, device=device)

    image_encoder, gps_encoder = _load_eval_models(cfg, checkpoint_path, device)
    dataset, dataloader = _build_eval_dataloader(cfg)

    print(f"Starting evaluation on {len(dataset)} query images...")
    outputs = run_eval_batches(
        cfg=cfg,
        dataloader=dataloader,
        image_encoder=image_encoder,
        gps_encoder=gps_encoder,
        retriever=retriever,
        device=device,
    )

    metrics_out = _compute_metrics_from_all_query_dists(outputs["all_query_dists"])
    topk_for_report = outputs["top_k_retrieval"]

    if len(outputs["alpha_monitor"]) > 0:
        alpha_all = torch.cat(outputs["alpha_monitor"], dim=0)
        print(
            "Dynamic alpha summary: "
            f"mean={alpha_all.mean().item():.4f}, "
            f"std={alpha_all.std().item():.4f}, "
            f"min={alpha_all.min().item():.4f}, "
            f"max={alpha_all.max().item():.4f}"
        )

    run_dir_for_log = args.run_dir if args.run_dir else os.path.basename(os.path.normpath(index_dir))
    ckpt_for_log = os.path.basename(checkpoint_path)
    _append_eval_record(run_dir_for_log, ckpt_for_log, cfg, run_meta, metrics_out)
    _print_metrics(metrics_out, topk_for_report)

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    evaluate(args)