import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from geoclip import GeoCLIP
from utils.config import load_config, build_from_defaults
from utils.metrics import Evaluator
from utils.reporter import dataset_name_from_csv
from datasets.img2geo_dataset import Img2GeoDataset
from datasets.geoclip_dataset import GeoCLIPDataset
from encoders.image_encoder import ImageEncoder, ImageEncoderConfig
from encoders.location_encoder import GPSEncoder, GPSEncoderConfig
from analysis.analysis import (
    geographic_pair_similarity, semantic_pair_similarity, 
    update_topk, update_rank,
    compute_repr_pos_neg)
from analysis.plot import (
    save_repr_pos_neg_hist, save_repr_margin_hist, save_repr_rank_distribution, save_repr_sg_scatter,
    save_retrv_distance_hist, save_retrv_rank_hist, save_retrv_rank_curve, save_oracle_rank_scatter,
    save_repr_vs_retrv_scatter,)

BASE_CHECKPOINT_ROOT = "/data/lsy/repos/3_GeoAligner/checkpoints"
BASE_GEOCLIP_ROOT = "/data/lsy/repos/Baseline_GeoAligner_GeoCLIP"
BASE_ANALYSIS_GALLERY_DIR = "/data/lsy/repos/3_GeoAligner/analysis/galleries"
BASE_ANALYSIS_CACHE_DIR = "/data/lsy/repos/3_GeoAligner/analysis/cache"
os.makedirs(BASE_ANALYSIS_CACHE_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Run representation diagnostics from checkpoint")
    parser.add_argument(
        "--backend",
        default="geoaligner",
        choices=["geoaligner", "geoclip"],
        help="Diagnostics backend: GeoAligner checkpoint or GeoCLIP pretrained model.",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Config path")
    parser.add_argument("--run_dir", default=None, help="Checkpoint run directory under fixed checkpoint root")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth", help="Checkpoint filename or absolute path")
    parser.add_argument("--gallery_nearest_csv", default=None, help="Optional precomputed nearest-gallery CSV under analysis/galleries")
    parser.add_argument("--gallery_topk_npz", default=None, help="Optional precomputed query-to-train topk geodesic npz under analysis/galleries")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k similarity-ranked gallery items to analyze per query")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights if available in checkpoint")
    parser.add_argument("--output_dir", default=None, help="Directory to save plots; defaults to ./analysis/plots/<run_tag>.")
    parser.add_argument("--max_scatter_points", type=int, default=30000, help="Subsample large scatter plots to this many points for readability.")
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

def _build_image_encoder(cfg, device, checkpoint=None, use_ema=False):
    img_cfg = getattr(cfg.model, "image", None)
    if not isinstance(img_cfg, ImageEncoderConfig):
        img_cfg = build_from_defaults(ImageEncoderConfig, img_cfg)
    img_cfg.use_landmark = False
    image_encoder = ImageEncoder(img_cfg).to(device)
    if checkpoint is not None:
        image_encoder.load_state_dict(_select_encoder_state_dict(checkpoint, "image_encoder", use_ema=use_ema))
    image_encoder.eval()
    return image_encoder

def _build_gps_encoder(cfg, device, checkpoint=None, use_ema=False):
    gps_cfg = getattr(cfg.model, "gps", None)
    if not isinstance(gps_cfg, GPSEncoderConfig):
        gps_cfg = build_from_defaults(GPSEncoderConfig, gps_cfg)
    gps_encoder = GPSEncoder(gps_cfg).to(device)
    if checkpoint is not None:
        gps_encoder.load_state_dict(_select_encoder_state_dict(checkpoint, "gps_encoder", use_ema=use_ema))
    gps_encoder.eval()
    return gps_encoder

def build_backend(args, cfg, device):
    gallery_name = dataset_name_from_csv(cfg.data.train.csv_file)

    if args.backend == "geoclip":
        
        model = GeoCLIP(from_pretrained=True).to(device).eval()

        dataset_cls = GeoCLIPDataset
        ds_kwargs = {"preprocess": model.image_encoder.preprocess_image}
        run_tag = f"{gallery_name}_geoclip"
        modes = ["s"]

        def encode_image(imgs):
            feats = F.normalize(model.image_encoder(imgs), dim=-1)
            return {"s_vector": feats, "g_tokens": None}

        def encode_gps(gps, extra):
            feats = F.normalize(model.location_encoder(gps), dim=-1)
            return {"s_vector": feats, "g_tokens": None}

    else:
        transform = transforms.Compose([
            transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        if os.path.isabs(args.checkpoint):
            checkpoint_path = args.checkpoint
            run_tag = os.path.basename(os.path.dirname(checkpoint_path.rstrip("/")))
        else:
            run_dir = os.path.basename(args.run_dir.rstrip("/"))
            checkpoint_path = os.path.join(BASE_CHECKPOINT_ROOT, run_dir, args.checkpoint)
            run_tag = run_dir
        checkpoint = torch.load(checkpoint_path, map_location=device)
        use_ema = bool(args.use_ema or getattr(cfg.train, "use_ema", False))

        img_enc = _build_image_encoder(cfg, device, checkpoint, use_ema)
        gps_enc = _build_gps_encoder(cfg, device, checkpoint, use_ema)

        dataset_cls = Img2GeoDataset
        ds_kwargs = {"transform": transform}
        modes = ["s", "g", "sg"]

        def encode_image(imgs):
            out = img_enc(imgs)
            return {
                "s_vector": F.normalize(out["s_vector"], dim=-1),
                "g_tokens": F.normalize(out["g_tokens"], dim=-1),
            }

        def encode_gps(gps, extra):
            out = gps_enc(gps, extra)
            return {
                "s_vector": F.normalize(out["s_vector"], dim=-1),
                "g_tokens": F.normalize(out["g_tokens"], dim=-1),
            }

    loaders = {}
    for split, csv, img_dir in [
        ("query", cfg.data.test.csv_file, cfg.data.test.img_dir),
        ("gallery", cfg.data.train.csv_file, cfg.data.train.img_dir),
    ]:
        ds = dataset_cls(csv_file=csv, img_dir=img_dir, **ds_kwargs)
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return {
        "backend": args.backend,
        "modes": modes,
        "query_loader": loaders["query"],
        "gallery_loader": loaders["gallery"],
        "image_encode_fn": encode_image,
        "gps_encode_fn": encode_gps,
        "run_tag": run_tag,
    }

def load_gallery_topk_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    qids = data["query_img_id"].astype(str)
    topk_idx = data["topk_gallery_idx"].astype(np.int64)
    topk_dist = data["topk_haversine_km"].astype(np.float32)

    # 构建 query_id -> row index 映射
    row_map = {qid: i for i, qid in enumerate(qids.tolist())}

    return {
        "row_map": row_map,
        "oracle_topk_indices": topk_idx,
        "oracle_topk_dists": topk_dist,
    }

def compute_query_embeddings(backend, device):
    loader = backend["query_loader"]
    image_encode_fn = backend["image_encode_fn"]
    gps_encode_fn = backend["gps_encode_fn"]
    modes = backend["modes"]

    all_image_s = []
    all_image_g = [] if "g" in modes else None
    all_gps_s = []
    all_gps_g = [] if "g" in modes else None
    coords_list = []
    img_ids_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encoding queries ({backend['run_tag']})"):
            
            img_ids, images, gps_coords, *extras = batch
            s2_tokens = extras[0] if extras else None

            gps_coords = gps_coords.to(device)
            if s2_tokens is not None:
                s2_tokens = s2_tokens.to(device)

            images = images.to(device)
            image_out = image_encode_fn(images)
            gps_out = gps_encode_fn(gps_coords, s2_tokens)

            all_image_s.append(image_out["s_vector"].cpu())
            if all_image_g is not None and image_out.get("g_tokens") is not None:
                all_image_g.append(image_out["g_tokens"].cpu())
            all_gps_s.append(gps_out["s_vector"].cpu())
            if all_gps_g is not None and gps_out.get("g_tokens") is not None:
                all_gps_g.append(gps_out["g_tokens"].cpu())
            coords_list.append(gps_coords.cpu())
            img_ids_list.extend([str(i) for i in img_ids])

    return {
        "image_s": torch.cat(all_image_s, dim=0),
        "image_g": torch.cat(all_image_g, dim=0) if all_image_g is not None else None,
        "gps_s": torch.cat(all_gps_s, dim=0),
        "gps_g": torch.cat(all_gps_g, dim=0) if all_gps_g is not None else None,
        "coords": torch.cat(coords_list, dim=0).float(),
        "img_ids": np.asarray(img_ids_list, dtype=str),
    }

def compute_gallery_embeddings(backend, device):
    run_tag = backend["run_tag"]
    loader = backend["gallery_loader"]
    gps_encode_fn = backend["gps_encode_fn"]
    modes = backend["modes"]
    use_g = "g" in modes

    s_path = os.path.join(BASE_ANALYSIS_CACHE_DIR, f"{run_tag}_s.npy")
    g_path = os.path.join(BASE_ANALYSIS_CACHE_DIR, f"{run_tag}_g.npy")
    meta_path = os.path.join(BASE_ANALYSIS_CACHE_DIR, f"{run_tag}_meta.npz")
    print(s_path, g_path, meta_path)

    if os.path.exists(s_path) and os.path.exists(meta_path):
        print(f"Loading gallery cache: {run_tag}")

        s_vecs = np.load(s_path, mmap_mode="r")
        g_tokens = np.load(g_path, mmap_mode="r") if (use_g and os.path.exists(g_path)) else None

        meta = np.load(meta_path, allow_pickle=True)
        return {
            "s": s_vecs,
            "g": g_tokens,
            "coords": meta["coords"].astype(np.float32),
            "img_ids": meta["img_ids"].astype(str),
        }

    print(f"Encoding gallery embeddings: {run_tag}")

    total = len(loader.dataset)

    s_memmap, g_memmap = None, None
    coords_arr = np.zeros((total, 2), dtype=np.float32)
    img_ids_arr = np.empty(total, dtype=object)
    offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encoding {run_tag}"):

            img_ids, _, gps_coords, *extras = batch
            s2_tokens = extras[0] if extras else None

            gps_coords = gps_coords.to(device)
            if s2_tokens is not None:
                s2_tokens = s2_tokens.to(device)

            out = gps_encode_fn(gps_coords, s2_tokens)

            s_vec = out["s_vector"].detach().cpu().numpy().astype(np.float32)
            bs = s_vec.shape[0]

            if s_memmap is None:
                s_memmap = np.lib.format.open_memmap(
                    s_path, mode="w+", dtype=np.float32, shape=(total, s_vec.shape[1])
                )

                if use_g and out.get("g_tokens") is not None:
                    g_shape = out["g_tokens"].shape[1:]
                    g_memmap = np.lib.format.open_memmap(
                        g_path, mode="w+", dtype=np.float32, shape=(total, *g_shape)
                    )

            s_memmap[offset:offset+bs] = s_vec
            if g_memmap is not None:
                g_memmap[offset:offset+bs] = out["g_tokens"].detach().cpu().numpy().astype(np.float32)

            coords_arr[offset:offset+bs] = gps_coords.cpu().numpy().astype(np.float32)
            img_ids_arr[offset:offset+bs] = img_ids

            offset += bs

    if s_memmap is not None:
        s_memmap.flush()
        del s_memmap
    if g_memmap is not None:
        g_memmap.flush()
        del g_memmap

    coords = coords_arr
    img_ids = img_ids_arr.astype(str)

    np.savez_compressed(meta_path, coords=coords, img_ids=img_ids)
    print(f"Saved gallery cache metadata: {meta_path}")

    s_vecs = np.load(s_path, mmap_mode="r")
    g_tokens = np.load(g_path, mmap_mode="r") if use_g else None

    return {
        "s": s_vecs,
        "g": g_tokens,
        "coords": coords,
        "img_ids": img_ids,
    }

def compute_chunked_retrv_stats(
    query_emb,  # dict{str: torch.Tensor or None}
    gallery_emb,    # dict{str: np.array or None}
    modes,
    top_k,
    device,
    oracle_indices,
    chunk_size=4096,
):
    """
    Chunked multi-mode retrieval: safely compute top-k and streaming rank.
    """
    state = {
        mode: {
            "pred_topk_indices": None, "pred_topk_scores": None, "pred_topk_dists": None,
            "oracle_rank": None, "oracle_score": None, 
            "gt_rank": None, "gt_score": None} 
        for mode in modes
    }

    num_gallery = gallery_emb["s"].shape[0]
    batch_size = query_emb["image_s"].shape[0]

    query_coords = query_emb["coords"].to(device, non_blocking=True)
    gallery_coords = torch.from_numpy(gallery_emb["coords"]).float().to(device, non_blocking=True)
    query_emb_s = query_emb["image_s"].to(device, non_blocking=True)

    query_emb_g = None
    if query_emb.get("image_g") is not None:
        query_emb_g = query_emb["image_g"].to(device, non_blocking=True)

    query_emb = {"image_s": query_emb_s, "image_g": query_emb_g}

    oracle_emb_s = torch.from_numpy(gallery_emb["s"][oracle_indices].copy()).float().to(device, non_blocking=True)
    oracle_emb_g = None
    if gallery_emb.get("g") is not None:
        oracle_emb_g = torch.from_numpy(gallery_emb["g"][oracle_indices].copy()).float().to(device, non_blocking=True)
    
    oracle_emb = {"s": oracle_emb_s, "g": oracle_emb_g}

    def compute_oracle_score(query_emb, oracle_emb, modes):
        pos_scores = {"s": None, "g": None, "sg": None}
        query_s = query_emb["image_s"]        # Tensor [B, D]
        gallery_s = oracle_emb["s"]   # Tensor [B, D]
        pos_s = (query_s * gallery_s).sum(dim=-1)
        pos_scores["s"] = pos_s
        if "g" in modes or "sg" in modes:
            query_g = query_emb.get("image_g")        # Tensor [B, N, D]
            gallery_g = oracle_emb.get("g")   # Tensor [B, M, D]
            # MaxSim diagonal (efficient)
            sim = torch.einsum("bnd,bmd->bnm", query_g, gallery_g)
            pos_g = sim.max(dim=2)[0].mean(dim=1)

            if "g" in modes:
                pos_scores["g"] = pos_g
            if "sg" in modes:
                pos_scores["sg"] = 0.5 * (pos_s + pos_g)
        return pos_scores

    oracle_score = compute_oracle_score(query_emb, oracle_emb, modes)

    def compute_chunk_similarity(query_emb, gallery_chunk_emb, modes, q_chunk_size=128):
        query_s = query_emb["image_s"]
        gallery_s = gallery_chunk_emb["s"]
        query_batch_size = query_s.shape[0]

        sims = {"s": None, "g": None, "sg": None}
        sim_s = semantic_pair_similarity(query_s, gallery_s)
        sims["s"] = sim_s
        if "g" in modes or "sg" in modes:
            sim_g_list = []
            query_g = query_emb["image_g"]
            gallery_g = gallery_chunk_emb["g"]
            for i in range(0, query_batch_size, q_chunk_size):
                end_i = min(i + q_chunk_size, query_batch_size)
                query_chunk_g = query_g[i:end_i]
                
                sim_g_chunk = geographic_pair_similarity(query_chunk_g, gallery_g)
                sim_g_list.append(sim_g_chunk)
            sim_g = torch.cat(sim_g_list, dim=0)
            
            if "g" in modes:
                sims["g"] = sim_g
            if "sg" in modes:
                sims["sg"] = 0.5 * (sim_s + sim_g)
        return sims

    with torch.no_grad():
        for start in tqdm(range(0, num_gallery, chunk_size), desc="Retrieving"):
            end = min(start + chunk_size, num_gallery)

            chunk_s = torch.from_numpy(gallery_emb["s"][start:end].copy()).float().to(device, non_blocking=True)
            chunk_g = None
            if gallery_emb.get("g") is not None:
                chunk_g = torch.from_numpy(gallery_emb["g"][start:end].copy()).float().to(device, non_blocking=True)
                
            chunked_gallery_emb = {"s": chunk_s, "g": chunk_g}

            chunk_scores = compute_chunk_similarity(query_emb, chunked_gallery_emb, modes)

            # 生成全局绝对索引
            chunk_indices = torch.arange(start, end, device=device).unsqueeze(0).expand(batch_size, -1)

            # 更新各 Mode 状态
            for mode in modes:
                state[mode]["pred_topk_scores"], state[mode]["pred_topk_indices"] = update_topk(
                    state[mode]["pred_topk_scores"],
                    state[mode]["pred_topk_indices"],
                    chunk_scores[mode],
                    chunk_indices,
                    top_k,
                )

                state[mode]["oracle_rank"] = update_rank(
                    state[mode]["oracle_rank"],
                    chunk_scores[mode],
                    oracle_score[mode],
                )

    for mode in modes:
        pred_topk_indices = state[mode]["pred_topk_indices"]
        pred_coords = gallery_coords[pred_topk_indices]
        pred_topk_dists = Evaluator.haversine_km(query_coords[:, None, :], pred_coords)
        
        state[mode]["oracle_rank"] = (state[mode]["oracle_rank"] + 1).cpu()
        state[mode]["oracle_score"] = oracle_score[mode].cpu()
        state[mode]["pred_topk_indices"] = state[mode]["pred_topk_indices"].cpu()
        state[mode]["pred_topk_scores"] = state[mode]["pred_topk_scores"].cpu()
        state[mode]["pred_topk_dists"] = pred_topk_dists.cpu()
        
    return state

def compute_repr_stats(query_emb, modes, semi_hard_q=0.9):
    """
    计算 representation 层面的 pos / neg / margin / rank
    基于 query 内部 image ↔ gps 的 pairwise similarity
    """

    stats = {}

    image_s = query_emb["image_s"]
    gps_s = query_emb["gps_s"]

    image_g = query_emb.get("image_g")
    gps_g = query_emb.get("gps_g")

    with torch.no_grad():

        sim_s = semantic_pair_similarity(image_s, gps_s)  # (B, B)
        pos, neg, margin, rank = compute_repr_pos_neg(sim_s, semi_hard_q)

        stats["s"] = {
            "repr_pos": pos.cpu().numpy(),
            "repr_neg": neg.cpu().numpy(),
            "repr_margin": margin.cpu().numpy(),
            "repr_rank": rank.cpu().numpy(),
        }

        if "g" in modes and image_g is not None and gps_g is not None:
            sim_g = geographic_pair_similarity(image_g, gps_g)  # (B, B)
            pos, neg, margin, rank = compute_repr_pos_neg(sim_g, semi_hard_q)

            stats["g"] = {
                "repr_pos": pos.cpu().numpy(),
                "repr_neg": neg.cpu().numpy(),
                "repr_margin": margin.cpu().numpy(),
                "repr_rank": rank.cpu().numpy(),
            }

        if "sg" in modes and "s" in stats and "g" in stats:
            sim_sg = 0.5 * (sim_s + sim_g)

            pos, neg, margin, rank = compute_repr_pos_neg(sim_sg, semi_hard_q)

            stats["sg"] = {
                "repr_pos": pos.cpu().numpy(),
                "repr_neg": neg.cpu().numpy(),
                "repr_margin": margin.cpu().numpy(),
                "repr_rank": rank.cpu().numpy(),
            }

    return stats

def run_retrieval_eval(
    backend,
    device,
    topk_npz_path,
    top_k=10,
    chunk_size=4096,
):
    query_emb = compute_query_embeddings(backend, device)
    gallery_emb = compute_gallery_embeddings(backend, device)
    query_ids = query_emb["img_ids"]

    oracle_data = load_gallery_topk_npz(topk_npz_path)
    row_map = oracle_data["row_map"]
    oracle_topk_indices = oracle_data["oracle_topk_indices"]
    oracle_top1_indices = np.array([
        oracle_topk_indices[row_map[qid], 0]
        for qid in query_ids
    ], dtype=np.int64)

    retrv_state = compute_chunked_retrv_stats(
        query_emb=query_emb,
        gallery_emb=gallery_emb,
        modes=backend["modes"],
        top_k=top_k,
        device=device,
        oracle_indices=oracle_top1_indices,
        chunk_size=chunk_size,
    )

    repr_state = compute_repr_stats(
        query_emb=query_emb, 
        modes=backend["modes"])

    retrv_rows, repr_rows = [], []
    query_coords = query_emb["coords"].cpu().numpy()
    gallery_coords = gallery_emb["coords"]
    oracle_topk_dists = oracle_data["oracle_topk_dists"]

    for mode in backend["modes"]:
        retrv_out = retrv_state[mode]
        pred_topk_indices = retrv_out["pred_topk_indices"].numpy()
        pred_topk_scores = retrv_out["pred_topk_scores"].numpy()
        oracle_rank = retrv_out["oracle_rank"].numpy()
        oracle_score = retrv_out["oracle_score"].numpy()
        retrv_pos = oracle_score
        retrv_neg = np.quantile(pred_topk_scores, q=0.9, axis=1)
        retrv_margin = retrv_pos - retrv_neg

        for i, qid in enumerate(query_ids):
            gt_lat, gt_lon = query_coords[i]
            oracle_top1_dist = oracle_topk_dists[row_map[qid], 0]

            pred_top1_idx = int(pred_topk_indices[i, 0])
            pred_top1_lat, pred_top1_lon = gallery_coords[pred_top1_idx]
            pred_top1_dist = float(retrv_out["pred_topk_dists"][i, 0])

            retrv_row = {
                "query_img_id": qid,
                "mode": mode,
                "gt_lat": float(gt_lat),
                "gt_lon": float(gt_lon),
                "oracle_rank": int(oracle_rank[i]),
                "oracle_score": float(oracle_score[i]),
                "oracle_top1_dist": float(oracle_top1_dist),
                "pred_top1_idx": pred_top1_idx,
                "pred_top1_score": float(pred_topk_scores[i, 0]),
                "pred_top1_lat": float(pred_top1_lat),
                "pred_top1_lon": float(pred_top1_lon),
                "pred_top1_dist": float(pred_top1_dist),
                "retrv_pos": float(retrv_pos[i]),
                "retrv_neg": float(retrv_neg[i]),
                "retrv_margin": float(retrv_margin[i]),
            }

            for k in range(top_k):
                retrv_row[f"pred_rank_{k+1}_idx"] = int(pred_topk_indices[i, k])
                retrv_row[f"pred_rank_{k+1}_score"] = float(pred_topk_scores[i, k])
                retrv_row[f"pred_rank_{k+1}_dist_km"] = float(retrv_out["pred_topk_dists"][i, k].item())

            retrv_rows.append(retrv_row)

        repr_out = repr_state[mode]
        repr_pos = repr_out["repr_pos"]
        repr_neg = repr_out["repr_neg"]
        repr_margin = repr_out["repr_margin"]
        repr_rank = repr_out["repr_rank"]

        for i, qid in enumerate(query_ids):
            repr_row = {
                "query_img_id": qid,
                "mode": mode,

                "repr_pos": float(repr_pos[i]),
                "repr_neg": float(repr_neg[i]),
                "repr_margin": float(repr_margin[i]),
                "repr_rank": int(repr_rank[i]),
            }

            repr_rows.append(repr_row)
    
    retrv_results = pd.DataFrame(retrv_rows)
    repr_results = pd.DataFrame(repr_rows)

    return {
        "query_ids": query_ids,
        "query_emb": query_emb,       # diagnostics 可能会用
        "gallery_emb": gallery_emb,   # diagnostics 可能会用
        "oracle_data": oracle_data,           # 包含 topk_dist
        "retrv_results": retrv_results,
        "repr_results": repr_results
    }

def run_diagnostics(eval_results, out_dir, query_name, backend, max_points=8000):
    """
    运行全套诊断并生成图表报告。
    """    
    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)
    
    df_repr = eval_results["repr_results"]
    df_retrv = eval_results["retrv_results"]
    modes = backend.get("modes", [])
    
    mode_colors = {
        "s": "#1b9e77",      # 视觉特征：绿色系
        "g": "#d95f0e",      # 语义/地理特征：橙色系
        "sg": "#7570b3", # 融合特征：紫色系
    }
    # 默认颜色 fallback
    default_colors = ["#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
    
    # 1. 遍历模式，生成各自的图表
    for i, mode in enumerate(modes):
        color = mode_colors.get(mode, default_colors[i % len(default_colors)])
        print(f"Diagnosing {mode.upper()} mode ...")
        
        # --- Representation (表示层) 分析 ---
        save_repr_pos_neg_hist(df_repr, out_dir, query_name, mode, color)
        save_repr_margin_hist(df_repr, out_dir, query_name, mode, color)
        save_repr_rank_distribution(df_repr, out_dir, query_name, mode, color)
        
        # --- Retrieval (检索层) 分析 ---
        save_retrv_distance_hist(df_retrv, out_dir, query_name, mode, color)
        save_retrv_rank_hist(df_retrv, out_dir, query_name, mode, color)
        save_retrv_rank_curve(df_retrv, out_dir, query_name, mode, color)
        save_oracle_rank_scatter(df_retrv, out_dir, max_points, query_name, mode, color)
        
        # --- 交叉诊断 (Bridge) ---
        save_repr_vs_retrv_scatter(df_retrv, df_repr, out_dir, query_name, mode, max_points, color)

    # 2. 跨模式图表 (如果同时存在 s 和 g)
    if "s" in modes and "g" in modes:
        print("s vs g cross-mode diagnostics ...")
        save_repr_sg_scatter(df_repr, out_dir, max_points, query_name)

    print(f"Finished! Plots saved to: {out_dir}")

def main():
    args = parse_args()
    print(f"Loading config from {args.config}...")
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    backend = build_backend(args, cfg, device)
    run_tag = backend["run_tag"]
    query_name = dataset_name_from_csv(cfg.data.test.csv_file)
    gallery_name = dataset_name_from_csv(cfg.data.train.csv_file)

    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join("./analysis/plots", f"{run_tag}_{query_name}")
    os.makedirs(out_dir, exist_ok=True)

    default_topk_npz = os.path.join(BASE_ANALYSIS_GALLERY_DIR, f"{query_name}_to_{gallery_name}_top{args.top_k}.npz")
    if not args.gallery_topk_npz:
        args.gallery_topk_npz = default_topk_npz

    # 调用你之前写好的评测核心函数
    eval_results = run_retrieval_eval(
        backend=backend,
        device=device,
        topk_npz_path=args.gallery_topk_npz,
        top_k=args.top_k,
    )
    print(f"Eval completed! ")
    print(f"  - Repr Results: {len(eval_results['repr_results'])} rows")
    print(f"  - Retrv Results: {len(eval_results['retrv_results'])} rows")
    print("=" * 50)
    
    run_diagnostics(
        eval_results=eval_results,
        out_dir=out_dir,
        query_name=query_name,
        backend=backend,
        max_points=args.max_scatter_points
    )

    save_path = os.path.join(out_dir, f"{query_name}_eval_results.pkl")
    save_dict = {
        "retrv_results": eval_results["retrv_results"],
        "repr_results": eval_results["repr_results"],
        "run_tag": run_tag,
        "query_name": query_name
    }
    pd.to_pickle(save_dict, save_path)

if __name__ == "__main__":
    main()
