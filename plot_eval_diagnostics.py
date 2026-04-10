#!/usr/bin/env python3

import os
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from geopy.distance import geodesic

from utils.config import load_config
from datasets.img2geo_dataset import Img2GeoDataset
# from encoders.image_encoder_20260325 import ImageEncoder, ImageEncoderConfig
from encoders.image_encoder import ImageEncoder, ImageEncoderConfig
from encoders.location_encoder import GPSEncoder, GPSEncoderConfig


BASE_CHECKPOINT_ROOT = "/data/lsy/repos/3_GeoAligner/checkpoints"
BASE_ANALYSIS_GALLERY_DIR = "/data/lsy/repos/3_GeoAligner/analysis/galleries"
BASE_ANALYSIS_CACHE_DIR = "/data/lsy/repos/3_GeoAligner/analysis/cache"
os.makedirs(BASE_ANALYSIS_CACHE_DIR, exist_ok=True)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run representation diagnostics from checkpoint")

    parser.add_argument("--config", default="configs/config.yaml", help="Config path")
    parser.add_argument("--run_dir", default=None, help="Checkpoint run directory under fixed checkpoint root")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth", help="Checkpoint filename or absolute path")
    parser.add_argument("--gallery_nearest_csv", default=None, help="Optional precomputed nearest-gallery CSV under analysis/galleries")
    parser.add_argument("--gallery_topk_npz", default=None, help="Optional precomputed query-to-train topk geodesic npz under analysis/galleries")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k similarity-ranked gallery items to analyze per query")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights if available in checkpoint")

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save plots; defaults to ./analysis/plots/<run_tag>.",
    )
    parser.add_argument(
        "--max_scatter_points",
        type=int,
        default=30000,
        help="Subsample large scatter plots to this many points for readability.",
    )
    return parser.parse_args()


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


def _resolve_output_dir(output_dir, run_tag):
    if output_dir:
        resolved = output_dir
    else:
        resolved = os.path.join("./analysis/plots", run_tag)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def _gallery_cache_paths(run_tag):
    prefix = os.path.join(BASE_ANALYSIS_CACHE_DIR, f"gallery_repr_{run_tag}")
    return {
        "s": f"{prefix}_s.npy",
        "g": f"{prefix}_g.npy",
        "meta": f"{prefix}_meta.npz",
    }


def _infer_dataset_name_from_path(path_like):
    path_text = str(path_like).lower()
    if "im2gps3k" in path_text or "im2gps_3k" in path_text:
        return "im2gps3k"
    if "yfcc4k" in path_text:
        return "yfcc4k"
    if "yfcc26k" in path_text or "yfcc25600" in path_text:
        return "yfcc26k"
    if "mp16" in path_text:
        if "filtered" in path_text:
            return "mp16pro"
        else:
            return "mp16profull"
    return "unknown"

def load_nearest_gallery_map(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    required = {
        "query_img_id",
        "gt_lat",
        "gt_lon",
        "nearest_gallery_img_id",
        "nearest_gallery_lat",
        "nearest_gallery_lon",
        "nearest_haversine_km",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Nearest-gallery CSV missing columns: {sorted(missing)}")
    return df


def load_gallery_topk_npz(npz_path):
    if not npz_path or not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    required = {"query_img_id", "topk_gallery_idx", "topk_haversine_km"}
    missing = required.difference(set(data.files))
    if missing:
        raise ValueError(f"Topk npz missing arrays: {sorted(missing)}")

    qids = data["query_img_id"].astype(str)
    topk_idx = data["topk_gallery_idx"].astype(np.int64)
    topk_dist = data["topk_haversine_km"].astype(np.float32)
    row_map = {qid: i for i, qid in enumerate(qids.tolist())}
    return {
        "row_map": row_map,
        "topk_idx": topk_idx,
        "topk_dist": topk_dist,
        "topk_k": int(topk_idx.shape[1]) if topk_idx.ndim == 2 else 0,
    }

def _maybe_sample(df, max_points, seed=42):
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed).sort_index()


def _plot_percentage_hist(ax, values, label, color, bins=60):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    counts, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pct = counts.astype(np.float32) * (100.0 / float(values.size))
    ax.plot(centers, pct, label=label, color=color, linewidth=2)


def _plot_rank_percentage(ax, values, label, color):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    ranks = values.astype(np.int64)
    ranks = ranks[ranks >= 1]
    if ranks.size == 0:
        return

    max_rank = int(np.max(ranks))
    counts = np.bincount(ranks, minlength=max_rank + 1)[1:]
    x = np.arange(1, max_rank + 1, dtype=np.int64)
    pct = counts.astype(np.float32) * (100.0 / float(ranks.size))
    ax.plot(x, pct, label=label, color=color, linewidth=2, marker="o", markersize=3)

def _style_axes(ax):
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.tick_params(labelsize=10)


def _compute_rank_distance_stats(df_gallery, top_k):
    rank_cols = [f"sim_rank_{rank_pos}_dist_to_gt_km" for rank_pos in range(1, top_k + 1)]
    rank_matrix = df_gallery[rank_cols].to_numpy(dtype=np.float32)

    medians = np.full(top_k, np.nan, dtype=np.float32)
    q25 = np.full(top_k, np.nan, dtype=np.float32)
    q75 = np.full(top_k, np.nan, dtype=np.float32)

    valid_cols = np.isfinite(rank_matrix).any(axis=0)
    if np.any(valid_cols):
        valid_matrix = rank_matrix[:, valid_cols]
        medians[valid_cols] = np.nanmedian(valid_matrix, axis=0)
        q25[valid_cols] = np.nanquantile(valid_matrix, 0.25, axis=0)
        q75[valid_cols] = np.nanquantile(valid_matrix, 0.75, axis=0)

    return medians, q25, q75


def _compute_g_token_similarity(query_g_tokens, gallery_g_tokens):
    # query_g_tokens: [B, T, D], gallery_g_tokens: [N, T, D]
    # returns [B, N] using mean token-wise cosine similarity.
    token_sim = torch.einsum("btd,ntd->bnt", query_g_tokens, gallery_g_tokens)
    return token_sim.mean(dim=-1)


def _chunked_multi_mode_topk_and_rank(
    query_s_vecs,
    query_g_tokens,
    gallery_s_vecs_cpu,
    gallery_g_tokens_cpu,
    top_k,
    device,
    chunk_size=1024,
    oracle_indices=None,
):
    mode_names = ("s", "g", "sg")
    num_gallery = int(gallery_g_tokens_cpu.shape[0])
    valid_oracle_mask = None
    oracle_scores = None
    oracle_rank = None

    if oracle_indices is not None:
        oracle_indices = oracle_indices.to(device)
        valid_oracle_mask = oracle_indices >= 0
        oracle_scores = {mode_name: torch.full((query_s_vecs.shape[0],), float("nan"), device=device, dtype=torch.float32) for mode_name in mode_names}
        oracle_rank = {mode_name: torch.zeros((query_s_vecs.shape[0],), device=device, dtype=torch.int64) for mode_name in mode_names}

        if torch.any(valid_oracle_mask):
            row_idx = torch.nonzero(valid_oracle_mask, as_tuple=False).squeeze(1)
            oracle_idx_np = oracle_indices[row_idx].detach().cpu().numpy().astype(np.int64)

            oracle_s_np = np.array(gallery_s_vecs_cpu[oracle_idx_np], dtype=np.float32, copy=True)
            oracle_g_np = np.array(gallery_g_tokens_cpu[oracle_idx_np], dtype=np.float32, copy=True)

            oracle_s = torch.from_numpy(oracle_s_np).to(device)
            oracle_g = torch.from_numpy(oracle_g_np).to(device)

            query_s_sel = query_s_vecs[row_idx]
            query_g_sel = query_g_tokens[row_idx]

            s_score = (query_s_sel * oracle_s).sum(dim=1)
            g_score = (query_g_sel * oracle_g).sum(dim=-1).mean(dim=1)
            sg_score = 0.5 * (s_score + g_score)

            oracle_scores["s"][row_idx] = s_score
            oracle_scores["g"][row_idx] = g_score
            oracle_scores["sg"][row_idx] = sg_score

        for mode_name in mode_names:
            if torch.any(valid_oracle_mask & torch.isnan(oracle_scores[mode_name])):
                raise RuntimeError(f"Failed to compute oracle scores for mode {mode_name}")

    best_scores = {mode_name: None for mode_name in mode_names}
    best_indices = {mode_name: None for mode_name in mode_names}

    def _compute_chunk_scores(gallery_s_chunk, gallery_g_chunk):
        score_s = query_s_vecs @ gallery_s_chunk.T
        score_g = _compute_g_token_similarity(query_g_tokens, gallery_g_chunk)
        return {
            "s": score_s,
            "g": score_g,
            "sg": 0.5 * (score_s + score_g),
        }

    for start in range(0, num_gallery, chunk_size):
        end = min(start + chunk_size, num_gallery)
        chunk_indices = torch.arange(start, end, device=device).unsqueeze(0).expand(query_s_vecs.shape[0], -1)
        gallery_s_np = np.array(gallery_s_vecs_cpu[start:end], dtype=np.float32, copy=True)
        gallery_g_np = np.array(gallery_g_tokens_cpu[start:end], dtype=np.float32, copy=True)
        gallery_s_chunk = torch.from_numpy(gallery_s_np).to(device)
        gallery_g_chunk = torch.from_numpy(gallery_g_np).to(device)
        chunk_scores_by_mode = _compute_chunk_scores(gallery_s_chunk, gallery_g_chunk)

        for mode_name in mode_names:
            chunk_scores = chunk_scores_by_mode[mode_name]
            if best_scores[mode_name] is None:
                current_k = min(top_k, chunk_scores.shape[1])
                mode_best_scores, best_pos = torch.topk(chunk_scores, k=current_k, dim=1, largest=True)
                mode_best_indices = chunk_indices.gather(1, best_pos)
                best_scores[mode_name] = mode_best_scores
                best_indices[mode_name] = mode_best_indices
            else:
                merged_scores = torch.cat([best_scores[mode_name], chunk_scores], dim=1)
                merged_indices = torch.cat([best_indices[mode_name], chunk_indices], dim=1)
                mode_best_scores, best_pos = torch.topk(merged_scores, k=top_k, dim=1, largest=True)
                mode_best_indices = merged_indices.gather(1, best_pos)
                best_scores[mode_name] = mode_best_scores
                best_indices[mode_name] = mode_best_indices

        if oracle_scores is not None and torch.any(valid_oracle_mask):
            row_idx = torch.nonzero(valid_oracle_mask, as_tuple=False).squeeze(1)
            for mode_name in mode_names:
                rank_incr = (chunk_scores_by_mode[mode_name][row_idx] >= oracle_scores[mode_name][row_idx].unsqueeze(1)).sum(dim=1).to(torch.int64)
                oracle_rank[mode_name][row_idx] += rank_incr

    for mode_name in mode_names:
        if best_scores[mode_name] is None or best_indices[mode_name] is None:
            raise RuntimeError(f"Failed to compute top-k scores for mode {mode_name}")

    if oracle_rank is not None:
        for mode_name in mode_names:
            oracle_rank[mode_name][~valid_oracle_mask] = 0
    
    results = {
        mode_name: {
            "topk_scores": best_scores[mode_name],
            "topk_indices": best_indices[mode_name],
            "oracle_scores": None if oracle_scores is None else oracle_scores[mode_name],
            "oracle_ranks": None if oracle_rank is None else oracle_rank[mode_name],
        }
        for mode_name in mode_names
    }
    return results


def _compute_pos_neg_margin_rank(sim_matrix):
    diag = torch.diag(sim_matrix)
    sim_no_diag = sim_matrix.clone()
    sim_no_diag.fill_diagonal_(-1e9)
    hard_neg = sim_no_diag.max(dim=1).values
    rank = (sim_matrix >= diag.unsqueeze(1)).sum(dim=1).to(torch.int64)
    margin = diag - hard_neg
    return diag, hard_neg, margin, rank

def _geodesic_km(lat1, lon1, lat2, lon2):
    return float(geodesic((lat1, lon1), (lat2, lon2)).km)


def _save_repr_similarity_hist(df_repr, out_dir, query_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df_repr["s_pos_sim"], bins=50, alpha=0.75, label="S positive", color="#1b9e77")
    ax.hist(df_repr["s_neg_sim"], bins=50, alpha=0.55, label="S hard-negative", color="#d95f0e")
    ax.set_title(f"S-space similarity", fontsize=13)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Sample count")
    ax.legend(frameon=False)
    _style_axes(ax)
    out_path = os.path.join(out_dir, f"{query_name}_s_similarity_hist.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def _save_mode_gallery_distance_hist(df_gallery, out_dir, query_name, mode_name, color):
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_percentage_hist(ax, df_gallery["nearest_gallery_dist_to_gt_km"], "Nearest gallery to GT", "#2b8cbe")
    _plot_percentage_hist(ax, df_gallery[f"top1_{mode_name}_dist_to_gt_km"], f"Top-1 {mode_name.upper()} candidate", color)
    ax.set_title(f"Top1 distance distribution ({mode_name.upper()})", fontsize=13)
    ax.set_xlabel("GT-to-GPS geodesic (km)")
    ax.set_ylabel("Query percentage (%)")
    ax.legend(frameon=False)
    _style_axes(ax)
    out_path = os.path.join(out_dir, f"{query_name}_{mode_name}_top1_vs_oracle_distance_distribution.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_repr_margin_hist(df_repr, out_dir, query_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df_repr["s_margin"], bins=50, alpha=0.85, color="#2b8cbe")
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"S-space margin (pos - hard-neg)", fontsize=13)
    ax.set_xlabel("Margin")
    ax.set_ylabel("Sample count")
    _style_axes(ax)
    out_path = os.path.join(out_dir, f"{query_name}_s_margin_hist.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def _save_mode_gallery_rank_hist(df_gallery, out_dir, query_name, mode_name, color):
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_rank_percentage(ax, df_gallery[f"oracle_{mode_name}_gallery_rank"], f"Nearest gallery rank ({mode_name.upper()})", color)
    ax.set_title(f"Oracle nearest-gallery rank distribution ({mode_name.upper()})", fontsize=13)
    ax.set_xlabel("Similarity rank (1 is best)")
    ax.set_ylabel("Query percentage (%)")
    ax.legend(frameon=False)
    _style_axes(ax)
    out_path = os.path.join(out_dir, f"{query_name}_{mode_name}_oracle_rank_distribution.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_repr_rank_distribution(df_repr, out_dir, query_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_rank_percentage(ax, df_repr["s_pos_rank_in_batch"], "S pos rank in batch", "#7570b3")
    ax.set_title(f"Positive rank distribution", fontsize=13)
    ax.set_xlabel("Rank (1 is best)")
    ax.set_ylabel("Query percentage (%)")
    ax.legend(frameon=False)
    _style_axes(ax)
    out_path = os.path.join(out_dir, f"{query_name}_rank_distribution.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def _save_mode_gallery_rank_curve(df_gallery, out_dir, query_name, mode_name, top_k, color):
    rank_grid = np.arange(1, top_k + 1, dtype=np.int64)
    medians, q25, q75 = _compute_rank_distance_stats(df_gallery, top_k)
    valid_rank_mask = np.isfinite(medians) & np.isfinite(q25) & np.isfinite(q75)

    fig, ax = plt.subplots(figsize=(8, 6))
    if np.any(valid_rank_mask):
        ax.plot(rank_grid[valid_rank_mask], medians[valid_rank_mask], color=color, linewidth=2, label="Median distance")
        ax.fill_between(rank_grid[valid_rank_mask], q25[valid_rank_mask], q75[valid_rank_mask], color=color, alpha=0.2, label="IQR")
    else:
        ax.text(0.5, 0.5, "No valid rank-distance data", ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_title(f"Rank distance curve ({mode_name.upper()})", fontsize=13)
    ax.set_xlabel("Similarity rank")
    ax.set_ylabel("GT-to-ranked-GPS geodesic (km)")
    _style_axes(ax)
    ax.legend(frameon=False)
    out_path = os.path.join(out_dir, f"{query_name}_{mode_name}_rank_distance_curve.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_repr_sg_scatter(df_repr, out_dir, max_points, query_name):
    sampled = _maybe_sample(df_repr, max_points=max_points)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        sampled["s_pos_sim"],
        sampled["g_pos_sim"],
        s=10,
        alpha=0.35,
        color="#e7298a",
        edgecolors="none",
    )
    ax.set_title(f"S vs G positive similarity", fontsize=13)
    ax.set_xlabel("S positive cosine")
    ax.set_ylabel("G positive cosine")
    _style_axes(ax)
    out_path = os.path.join(out_dir, f"{query_name}_s_vs_g_scatter.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def _save_mode_gallery_rank_scatter(df_gallery, out_dir, max_points, query_name, mode_name, color):
    sample_df = _maybe_sample(df_gallery, max_points=max_points)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        sample_df[f"oracle_{mode_name}_gallery_rank"],
        sample_df[f"nearest_gallery_dist_to_gt_km"],
        s=10,
        alpha=0.35,
        color=color,
        edgecolors="none",
    )
    ax.set_title(f"Oracle rank vs oracle distance ({mode_name.upper()})", fontsize=13)
    ax.set_xlabel("Similarity rank of nearest gallery")
    ax.set_ylabel("GT-to-nearest-gallery geodesic (km)")
    ax.set_yscale("log")
    _style_axes(ax)
    out_path = os.path.join(out_dir, f"{query_name}_{mode_name}_oracle_rank_vs_distance.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_diagnostics(args):
    t0 = time.time()
    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    if os.path.isabs(args.checkpoint):
        checkpoint_path = args.checkpoint
        run_tag = os.path.basename(os.path.dirname(checkpoint_path.rstrip("/")))
    else:
        if not args.run_dir:
            raise ValueError("--run_dir required when --checkpoint is not an absolute path")
        args.run_dir = os.path.basename(args.run_dir.rstrip("/"))
        checkpoint_path = os.path.join(BASE_CHECKPOINT_ROOT, args.run_dir, args.checkpoint)
        run_tag = args.run_dir

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not hasattr(cfg, "data") or not hasattr(cfg.data, "test") or not hasattr(cfg.data, "train"):
        raise ValueError("Config must contain both data.test and data.train")

    gallery_cfg = cfg.data.train
    query_cfg = cfg.data.test
    gallery_name = _infer_dataset_name_from_path(gallery_cfg.csv_file)
    query_name = _infer_dataset_name_from_path(query_cfg.csv_file)
    allowed_query_names = {"im2gps3k", "yfcc4k", "yfcc26k"}
    if query_name not in allowed_query_names:
        raise ValueError(f"Unsupported query_name '{query_name}' inferred from data.test.csv_file; expected one of {sorted(allowed_query_names)}")

    transform = transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = Img2GeoDataset(csv_file=query_cfg.csv_file, img_dir=query_cfg.img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    use_ema = bool(args.use_ema or getattr(cfg.train, "use_ema", False))

    img_cfg = ImageEncoderConfig(
        vit_name=cfg.model.image.vit_name,
        img_size=cfg.data.img_size,
        s_dim=cfg.model.image.s_dim,
        g_dim=cfg.model.image.g_dim,
        n_g_tokens=cfg.model.image.n_g_tokens,
        use_landmark=False,
    )
    image_encoder = ImageEncoder(img_cfg).to(device)
    image_encoder.load_state_dict(_select_encoder_state_dict(checkpoint, "image_encoder", use_ema=use_ema))
    image_encoder.eval()

    gps_cfg = GPSEncoderConfig(
        s2_levels=cfg.model.gps.s2_levels,
        s2_embed_dim=cfg.model.gps.s2_embed_dim,
        s2_num_buckets=getattr(cfg.model.gps, "s2_num_buckets", 2**17),
        s2_embed_dropout=getattr(cfg.model.gps, "s2_embed_dropout", 0.1),
        s2_feature_dropout=getattr(cfg.model.gps, "s2_feature_dropout", 0.1),
        transformer_nhead=getattr(cfg.model.gps, "transformer_nhead", 4),
        transformer_nlayers=getattr(cfg.model.gps, "transformer_nlayers", 2),
        transformer_dropout=getattr(cfg.model.gps, "transformer_dropout", 0.1),
        fourier_n_freqs=cfg.model.gps.fourier_n_freqs,
        continuous_geo_mode=getattr(cfg.model.gps, "continuous_geo_mode", "unit_sphere"),
        n_g_tokens=cfg.model.gps.n_g_tokens,
        base_scale_multiplier=cfg.model.gps.base_scale_multiplier,
        min_scale_deg=getattr(cfg.model.gps, "min_scale_deg", 1e-4),
        max_scale_deg=getattr(cfg.model.gps, "max_scale_deg", 1.0),
        lon_scale_cos_epsilon=getattr(cfg.model.gps, "lon_scale_cos_epsilon", 0.15),
        sampling_mode_train=getattr(cfg.model.gps, "sampling_mode_train", "random"),
        sampling_mode_eval=getattr(cfg.model.gps, "sampling_mode_eval", "same_as_train"),
        sampling_seed=getattr(cfg.model.gps, "sampling_seed", 42),
        s_dim=cfg.model.gps.s_dim,
        g_dim=cfg.model.gps.g_dim,
    )
    gps_encoder = GPSEncoder(gps_cfg).to(device)
    gps_encoder.load_state_dict(_select_encoder_state_dict(checkpoint, "gps_encoder", use_ema=use_ema))
    gps_encoder.eval()

    if not args.gallery_nearest_csv and query_name != "unknown":
        args.gallery_nearest_csv = os.path.join(BASE_ANALYSIS_GALLERY_DIR, f"{query_name}_to_{gallery_name}_nearest.csv")
    nearest_gallery_df = load_nearest_gallery_map(args.gallery_nearest_csv)
    nearest_gallery_map = None
    if nearest_gallery_df is not None:
        nearest_gallery_map = nearest_gallery_df.set_index("query_img_id").to_dict(orient="index")
        print(f"Loaded nearest-gallery reference CSV: {args.gallery_nearest_csv}")

    if not args.gallery_topk_npz and query_name != "unknown":
        default_topk_npz = os.path.join(BASE_ANALYSIS_GALLERY_DIR, f"{query_name}_to_{gallery_name}_top{args.top_k}.npz")
        args.gallery_topk_npz = default_topk_npz if os.path.exists(default_topk_npz) else None
    gallery_topk_data = load_gallery_topk_npz(args.gallery_topk_npz)
    if gallery_topk_data is not None:
        print(f"Loaded top-k geodesic cache: {args.gallery_topk_npz}")

    
    gallery_dataset = Img2GeoDataset(csv_file=gallery_cfg.csv_file, img_dir=gallery_cfg.img_dir, transform=transform)
    gallery_loader = DataLoader(gallery_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    gallery_cache_paths = _gallery_cache_paths(run_tag)

    gallery_s_vecs = None
    gallery_g_tokens = None
    gallery_coords_np = None
    gallery_img_ids = None

    if all(os.path.exists(path) for path in gallery_cache_paths.values()):
        t_cache_load = time.time()
        gallery_s_vecs = np.load(gallery_cache_paths["s"], mmap_mode="r")
        gallery_g_tokens = np.load(gallery_cache_paths["g"], mmap_mode="r")
        meta_cache = np.load(gallery_cache_paths["meta"], allow_pickle=True)
        gallery_coords_np = meta_cache["gallery_coords"].astype(np.float32)
        gallery_img_ids = meta_cache["gallery_img_ids"].astype(str)
        print(f"Loaded gallery embedding cache: {gallery_cache_paths['s']} ({time.time() - t_cache_load:.2f}s)")

    if gallery_s_vecs is None or gallery_g_tokens is None:
        t_gallery = time.time()
        gallery_total = len(gallery_loader.dataset)
        gallery_s_vecs_np = None
        gallery_g_tokens_np = None
        write_offset = 0
        with torch.no_grad():
            for images, gps_coords, s2_tokens in tqdm(gallery_loader, desc="Encoding gallery GPS representations"):
                gps_coords = gps_coords.to(device)
                s2_tokens = s2_tokens.to(device)
                gallery_embeds = gps_encoder(gps_coords, s2_tokens)
                gallery_s = torch.nn.functional.normalize(gallery_embeds["s_vector"], p=2, dim=-1).detach().cpu().numpy().astype(np.float32)
                gallery_g = torch.nn.functional.normalize(gallery_embeds["g_tokens"], p=2, dim=-1).detach().cpu().numpy().astype(np.float32)

                if gallery_s_vecs_np is None:
                    gallery_s_vecs_np = np.lib.format.open_memmap(
                        gallery_cache_paths["s"],
                        mode="w+",
                        dtype=np.float32,
                        shape=(gallery_total, gallery_s.shape[1]),
                    )
                    gallery_g_tokens_np = np.lib.format.open_memmap(
                        gallery_cache_paths["g"],
                        mode="w+",
                        dtype=np.float32,
                        shape=(gallery_total,) + gallery_g.shape[1:],
                    )

                batch_size = gallery_s.shape[0]
                next_offset = write_offset + batch_size
                gallery_s_vecs_np[write_offset:next_offset] = gallery_s
                gallery_g_tokens_np[write_offset:next_offset] = gallery_g
                write_offset = next_offset

        gallery_s_vecs_np.flush()
        gallery_g_tokens_np.flush()
        del gallery_s_vecs_np
        del gallery_g_tokens_np

        gallery_s_vecs = np.load(gallery_cache_paths["s"], mmap_mode="r")
        gallery_g_tokens = np.load(gallery_cache_paths["g"], mmap_mode="r")
        gallery_coords_np = gallery_dataset.geo_metadata[["LAT", "LON"]].to_numpy(dtype=np.float32)
        gallery_img_ids = gallery_dataset.geo_metadata["IMG_ID"].astype(str).to_numpy()
        np.savez_compressed(
            gallery_cache_paths["meta"],
            gallery_coords=gallery_coords_np.astype(np.float32),
            gallery_img_ids=gallery_img_ids,
        )
        print(f"Encoded gallery representations in {time.time() - t_gallery:.2f}s, cached at: {gallery_cache_paths['s']}")

    gallery_id_to_index = {img_id: idx for idx, img_id in enumerate(gallery_img_ids.tolist())}

    stats = {
        "s_pos_sim": [],
        "s_neg_sim": [],
        "s_margin": [],
        "s_pos_rank_in_batch": [],
        "g_pos_sim": [],
        "g_neg_sim": [],
        "g_margin": [],
        "g_pos_rank_in_batch": [],
        "sg_pos_sim": [],
        "sg_neg_sim": [],
        "sg_margin": [],
        "sg_pos_rank_in_batch": [],
    }

    processed_batches = 0
    with torch.no_grad():
        for images, gps_coords, s2_tokens in tqdm(dataloader, desc="Batch representation diagnostics"):
            images = images.to(device)
            gps_coords = gps_coords.to(device)
            s2_tokens = s2_tokens.to(device)
            bsz = images.shape[0]
            if bsz < 2:
                continue

            img = image_encoder(images)
            gps = gps_encoder(gps_coords, s2_tokens)

            s_img = torch.nn.functional.normalize(img["s_vector"], p=2, dim=-1)
            s_gps = torch.nn.functional.normalize(gps["s_vector"], p=2, dim=-1)
            sim_s = torch.matmul(s_img, s_gps.T)

            g_img = torch.nn.functional.normalize(img["g_tokens"], p=2, dim=-1)
            g_gps = torch.nn.functional.normalize(gps["g_tokens"], p=2, dim=-1)
            sim_g = _compute_g_token_similarity(g_img, g_gps)
            sim_sg = 0.5 * (sim_s + sim_g)

            s_diag, s_hard_neg, s_margin, s_rank = _compute_pos_neg_margin_rank(sim_s)
            g_diag, g_hard_neg, g_margin, g_rank = _compute_pos_neg_margin_rank(sim_g)
            sg_diag, sg_hard_neg, sg_margin, sg_rank = _compute_pos_neg_margin_rank(sim_sg)

            stats["s_pos_sim"].extend(s_diag.detach().cpu().numpy().tolist())
            stats["s_neg_sim"].extend(s_hard_neg.detach().cpu().numpy().tolist())
            stats["s_margin"].extend(s_margin.detach().cpu().numpy().tolist())
            stats["s_pos_rank_in_batch"].extend(s_rank.detach().cpu().numpy().tolist())

            stats["g_pos_sim"].extend(g_diag.detach().cpu().numpy().tolist())
            stats["g_neg_sim"].extend(g_hard_neg.detach().cpu().numpy().tolist())
            stats["g_margin"].extend(g_margin.detach().cpu().numpy().tolist())
            stats["g_pos_rank_in_batch"].extend(g_rank.detach().cpu().numpy().tolist())

            stats["sg_pos_sim"].extend(sg_diag.detach().cpu().numpy().tolist())
            stats["sg_neg_sim"].extend(sg_hard_neg.detach().cpu().numpy().tolist())
            stats["sg_margin"].extend(sg_margin.detach().cpu().numpy().tolist())
            stats["sg_pos_rank_in_batch"].extend(sg_rank.detach().cpu().numpy().tolist())

            processed_batches += 1

    df_repr = pd.DataFrame(stats)
    if len(df_repr) == 0:
        raise RuntimeError("No valid samples were collected for representation diagnostics")

    query_gallery_rows = []
    query_img_ids = dataset.geo_metadata["IMG_ID"].astype(str).to_numpy()
    query_offset = 0
    top_k = min(max(1, int(args.top_k)), int(gallery_s_vecs.shape[0]))
    gallery_chunk_size = int(getattr(cfg.data, "gallery_chunk_size", 256))

    with torch.no_grad():
        for images, gps_coords, s2_tokens in tqdm(dataloader, desc="Query-vs-gallery diagnostics"):
            gps_coords_cpu = gps_coords
            images = images.to(device)
            gps_coords = gps_coords.to(device)
            s2_tokens = s2_tokens.to(device)

            query_embeds = image_encoder(images)
            query_s_vecs = torch.nn.functional.normalize(query_embeds["s_vector"], p=2, dim=-1)
            query_g_tokens = torch.nn.functional.normalize(query_embeds["g_tokens"], p=2, dim=-1)

            gps_coords_np = gps_coords_cpu.numpy()

            nearest_gallery_idx_list = []
            nearest_gallery_img_id_list = []
            nearest_gallery_lat_list = []
            nearest_gallery_lon_list = []
            nearest_gallery_dist_list = []
            oracle_gallery_idx_list = []

            for i in range(images.shape[0]):
                query_img_id = str(query_img_ids[query_offset + i])
                nearest_ref = nearest_gallery_map.get(query_img_id) if nearest_gallery_map is not None else None
                if nearest_ref is not None:
                    nearest_gallery_lat = float(nearest_ref["nearest_gallery_lat"])
                    nearest_gallery_lon = float(nearest_ref["nearest_gallery_lon"])
                    nearest_gallery_dist = float(nearest_ref["nearest_haversine_km"])
                    nearest_gallery_img_id = str(nearest_ref["nearest_gallery_img_id"])
                    nearest_gallery_idx = gallery_id_to_index.get(nearest_gallery_img_id, None)
                else:
                    nearest_gallery_lat = np.nan
                    nearest_gallery_lon = np.nan
                    nearest_gallery_dist = np.nan
                    nearest_gallery_img_id = None
                    nearest_gallery_idx = None

                nearest_gallery_idx_list.append(-1 if nearest_gallery_idx is None else int(nearest_gallery_idx))
                nearest_gallery_img_id_list.append(nearest_gallery_img_id)
                nearest_gallery_lat_list.append(nearest_gallery_lat)
                nearest_gallery_lon_list.append(nearest_gallery_lon)
                nearest_gallery_dist_list.append(nearest_gallery_dist)
                oracle_gallery_idx_list.append(nearest_gallery_idx)

            oracle_indices_tensor = torch.tensor(nearest_gallery_idx_list, device=device, dtype=torch.int64)
            mode_results = _chunked_multi_mode_topk_and_rank(
                query_s_vecs=query_s_vecs,
                query_g_tokens=query_g_tokens,
                gallery_s_vecs_cpu=gallery_s_vecs,
                gallery_g_tokens_cpu=gallery_g_tokens,
                top_k=top_k,
                device=device,
                chunk_size=gallery_chunk_size,
                oracle_indices=oracle_indices_tensor,
            )

            topk_by_mode = {
                mode_name: {
                    "scores": mode_results[mode_name]["topk_scores"].detach().cpu().numpy(),
                    "indices": mode_results[mode_name]["topk_indices"].detach().cpu().numpy(),
                }
                for mode_name in ("s", "g", "sg")
            }

            for i in range(images.shape[0]):
                query_img_id = str(query_img_ids[query_offset + i])
                query_lat = float(gps_coords_np[i, 0])
                query_lon = float(gps_coords_np[i, 1])
                pre_idx_to_dist = None
                if gallery_topk_data is not None:
                    row_idx = gallery_topk_data["row_map"].get(query_img_id, None)
                    if row_idx is not None:
                        idx_row = gallery_topk_data["topk_idx"][row_idx]
                        dist_row = gallery_topk_data["topk_dist"][row_idx]
                        pre_idx_to_dist = {int(k): float(v) for k, v in zip(idx_row.tolist(), dist_row.tolist())}

                candidate_indices = set()
                for mode_name in ("s", "g", "sg"):
                    candidate_indices.update(int(idx) for idx in topk_by_mode[mode_name]["indices"][i].tolist())

                distance_cache = pre_idx_to_dist if pre_idx_to_dist is not None else {}
                for cand_idx in candidate_indices:
                    if cand_idx in distance_cache:
                        continue
                    cand_lat = float(gallery_coords_np[cand_idx, 0])
                    cand_lon = float(gallery_coords_np[cand_idx, 1])
                    distance_cache[cand_idx] = _geodesic_km(query_lat, query_lon, cand_lat, cand_lon)

                nearest_gallery_img_id = nearest_gallery_img_id_list[i]
                nearest_gallery_lat = nearest_gallery_lat_list[i]
                nearest_gallery_lon = nearest_gallery_lon_list[i]
                nearest_gallery_dist = nearest_gallery_dist_list[i]
                nearest_gallery_idx = oracle_gallery_idx_list[i]

                row = {
                    "query_img_id": query_img_id,
                    "gt_lat": query_lat,
                    "gt_lon": query_lon,
                    "nearest_gallery_img_id": nearest_gallery_img_id,
                    "nearest_gallery_lat": nearest_gallery_lat,
                    "nearest_gallery_lon": nearest_gallery_lon,
                    "nearest_gallery_dist_to_gt_km": nearest_gallery_dist,
                }

                for mode_name in ("s", "g", "sg"):
                    mode_topk = topk_by_mode[mode_name]
                    mode_topk_indices = mode_topk["indices"]
                    mode_topk_scores = mode_topk["scores"]
                    mode_result = mode_results[mode_name]

                    top1_idx = int(mode_topk_indices[i, 0])
                    top1_lat = float(gallery_coords_np[top1_idx, 0])
                    top1_lon = float(gallery_coords_np[top1_idx, 1])
                    top1_dist = distance_cache[top1_idx]

                    if mode_name == "s":
                        if nearest_gallery_idx is not None:
                            oracle_score = float(mode_result["oracle_scores"][i].item())
                            oracle_rank = int(mode_result["oracle_ranks"][i].item())
                            oracle_lat = float(nearest_gallery_lat_list[i])
                            oracle_lon = float(nearest_gallery_lon_list[i])
                        else:
                            oracle_score = np.nan
                            oracle_rank = np.nan
                            oracle_lat = np.nan
                            oracle_lon = np.nan
                    elif mode_name == "g":
                        if nearest_gallery_idx is not None:
                            oracle_score = float(mode_result["oracle_scores"][i].item())
                            oracle_rank = int(mode_result["oracle_ranks"][i].item())
                            oracle_lat = float(nearest_gallery_lat_list[i])
                            oracle_lon = float(nearest_gallery_lon_list[i])
                        else:
                            oracle_score = np.nan
                            oracle_rank = np.nan
                            oracle_lat = np.nan
                            oracle_lon = np.nan
                    else:
                        if nearest_gallery_idx is not None:
                            oracle_score = float(mode_result["oracle_scores"][i].item())
                            oracle_rank = int(mode_result["oracle_ranks"][i].item())
                            oracle_lat = float(nearest_gallery_lat_list[i])
                            oracle_lon = float(nearest_gallery_lon_list[i])
                        else:
                            oracle_score = np.nan
                            oracle_rank = np.nan
                            oracle_lat = np.nan
                            oracle_lon = np.nan

                    row[f"top1_{mode_name}_lat"] = top1_lat
                    row[f"top1_{mode_name}_lon"] = top1_lon
                    row[f"top1_{mode_name}_dist_to_gt_km"] = top1_dist
                    row[f"top1_{mode_name}_score"] = float(mode_topk_scores[i, 0])

                    row[f"oracle_{mode_name}_gallery_lat"] = oracle_lat
                    row[f"oracle_{mode_name}_gallery_lon"] = oracle_lon
                    row[f"oracle_{mode_name}_gallery_rank"] = oracle_rank
                    row[f"oracle_{mode_name}_gallery_score"] = oracle_score

                    for rank_pos in range(top_k):
                        cand_idx = int(mode_topk_indices[i, rank_pos])
                        row[f"{mode_name}_sim_rank_{rank_pos + 1}_dist_to_gt_km"] = distance_cache[cand_idx]

                # Backward-compatible aliases for existing plotting code (s-only).
                row["top1_sim_lat"] = row["top1_s_lat"]
                row["top1_sim_lon"] = row["top1_s_lon"]
                row["top1_sim_dist_to_gt_km"] = row["top1_s_dist_to_gt_km"]
                row["top1_sim_score"] = row["top1_s_score"]
                row["oracle_gallery_lat"] = row["oracle_s_gallery_lat"]
                row["oracle_gallery_lon"] = row["oracle_s_gallery_lon"]
                row["oracle_gallery_rank"] = row["oracle_s_gallery_rank"]
                row["oracle_gallery_score"] = row["oracle_s_gallery_score"]
                for rank_pos in range(top_k):
                    row[f"sim_rank_{rank_pos + 1}_dist_to_gt_km"] = row[f"s_sim_rank_{rank_pos + 1}_dist_to_gt_km"]

                query_gallery_rows.append(row)

            query_offset += images.shape[0]

    df_gallery = pd.DataFrame(query_gallery_rows)
    if len(df_gallery) == 0:
        raise RuntimeError("No valid query-gallery samples were collected for representation diagnostics")

    out_dir = _resolve_output_dir(args.output_dir, run_tag)
    csv_path = os.path.join(out_dir, f"{query_name}_repr_diagnostics.csv")
    df_repr.to_csv(csv_path, index=False)

    gallery_csv_path = os.path.join(out_dir, f"{query_name}_query_gallery_diagnostics.csv")
    df_gallery.to_csv(gallery_csv_path, index=False)

    generated = [
        _save_repr_similarity_hist(df_repr, out_dir, query_name),
        _save_repr_margin_hist(df_repr, out_dir, query_name),
        _save_repr_rank_distribution(df_repr, out_dir, query_name),
        _save_repr_sg_scatter(df_repr, out_dir, args.max_scatter_points, query_name),
    ]

    # Query-vs-gallery plots requested by the user.
    # Query-vs-gallery plots: generate mode-specific outputs for s, g, and s+g.
    gallery_generated = []
    mode_styles = {
        "s": "#d95f0e",
        "g": "#1b9e77",
        "sg": "#7570b3",
    }
    for mode_name, color in mode_styles.items():
        gallery_generated.append(_save_mode_gallery_distance_hist(df_gallery, out_dir, query_name, mode_name, color))
        gallery_generated.append(_save_mode_gallery_rank_hist(df_gallery, out_dir, query_name, mode_name, color))
        gallery_generated.append(_save_mode_gallery_rank_curve(df_gallery, out_dir, query_name, mode_name, top_k, color))
        gallery_generated.append(_save_mode_gallery_rank_scatter(df_gallery, out_dir, args.max_scatter_points, query_name, mode_name, color))

    generated.extend(gallery_generated)

    summary = {
        "samples": int(len(df_repr)),
        "batches": int(processed_batches),
        "s_pos_mean": float(df_repr["s_pos_sim"].mean()),
        "s_neg_mean": float(df_repr["s_neg_sim"].mean()),
        "s_margin_mean": float(df_repr["s_margin"].mean()),
        "rank1_rate": float((df_repr["s_pos_rank_in_batch"] == 1).mean()),
        "g_pos_mean": float(df_repr["g_pos_sim"].mean()),
        "g_neg_mean": float(df_repr["g_neg_sim"].mean()),
        "g_margin_mean": float(df_repr["g_margin"].mean()),
        "g_rank1_rate": float((df_repr["g_pos_rank_in_batch"] == 1).mean()),
        "sg_pos_mean": float(df_repr["sg_pos_sim"].mean()),
        "sg_neg_mean": float(df_repr["sg_neg_sim"].mean()),
        "sg_margin_mean": float(df_repr["sg_margin"].mean()),
        "sg_rank1_rate": float((df_repr["sg_pos_rank_in_batch"] == 1).mean()),
        "oracle_rank1_rate": float((df_gallery["oracle_gallery_rank"] == 1).mean()),
        "oracle_rank_median": float(df_gallery["oracle_gallery_rank"].median()),
        "top1_sim_dist_mean": float(df_gallery["top1_sim_dist_to_gt_km"].mean()),
        "oracle_dist_mean": float(df_gallery["nearest_gallery_dist_to_gt_km"].mean()),
        "top1_distance_coverage": float(np.mean(np.isfinite(df_gallery["top1_sim_dist_to_gt_km"].to_numpy(dtype=np.float32)))),
    }

    for mode_name in ("s", "g", "sg"):
        rank_col = f"oracle_{mode_name}_gallery_rank"
        top1_dist_col = f"top1_{mode_name}_dist_to_gt_km"
        summary[f"{mode_name}_oracle_rank1_rate"] = float((df_gallery[rank_col] == 1).mean())
        summary[f"{mode_name}_oracle_rank_median"] = float(df_gallery[rank_col].median())
        summary[f"{mode_name}_top1_dist_mean"] = float(df_gallery[top1_dist_col].mean())
        summary[f"{mode_name}_top1_distance_coverage"] = float(
            np.mean(np.isfinite(df_gallery[top1_dist_col].to_numpy(dtype=np.float32)))
        )
    summary_path = os.path.join(out_dir, f"{query_name}_repr_summary.json")
    pd.Series(summary).to_json(summary_path, indent=2)

    print(f"Saved representation diagnostics CSV to: {csv_path}")
    print(f"Saved query-gallery diagnostics CSV to: {gallery_csv_path}")
    print(f"Saved representation summary to: {summary_path}")
    print(f"Saved {len(generated)} representation figures to: {out_dir}")
    print(f"Total diagnostics runtime: {time.time() - t0:.2f}s")
    for path in generated:
        print(f"- {path}")


def main():
    args = _parse_args()
    run_diagnostics(args)


if __name__ == "__main__":
    main()
