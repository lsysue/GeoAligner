import os
import math
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from utils.config import Config
from aligners.alignmenthub import dynamic_fuse_scores


@dataclass
class RetrieverConfig:
    """Configuration for retrieval and reranking behavior."""
    top_k: int = 100
    nprobe: int = 64
    use_ivf: bool = True
    nlist: int = 4096
    use_rerank: bool = False
    rerank_topk: int = 100
    rerank_fusion_mode: str = "weighted"
    rerank_s_weight: float = 0.8
    rerank_dynamic_temperature: float = 0.07
    rerank_dynamic_consistency_topk: int = 10
    rerank_dynamic_weight_gap: float = 0.45
    rerank_dynamic_weight_entropy: float = 0.35
    rerank_dynamic_weight_consistency: float = 0.20
    rerank_dynamic_alpha_min: float = 0.05
    rerank_dynamic_alpha_max: float = 0.95
    ot_eps: float = 0.05
    ot_iters: int = 3
    kde_weight: float = 0.0
    kde_sigma_km: float = 50.0

class Retriever(nn.Module):
    """
    Two-stage retrieval:
    1. S-space (macro recall, FAISS)
    2. G-space (micro rerank, OT / token alignment)
    """

    def __init__(self, cfg: Config, run_tag, device):
        super().__init__()
        self.cfg = cfg
        self.retrieval_cfg = cfg.retrieval
        self.device = device
        self.index_dir = os.path.join(cfg.dirs.index_dir, run_tag)
        self.cache_path = os.path.join(cfg.dirs.cache_dir, run_tag)
        s_path = "_".join([self.cache_path, "s.npy"])
        g_path = "_".join([self.cache_path, "g.npy"])

        (
            self.index,
            self.coords_lookup,
            self.s2_lookup,
        ) = self.load_index_bundle(self.index_dir)

        self.gps_s_lookup = None
        self.gps_g_lookup = None

        if os.path.exists(s_path) and os.path.exists(g_path):
            print(f"Loading GPS cache from {s_path} and {g_path}...")
            self.gps_s_lookup = torch.from_numpy(np.load(s_path, mmap_mode="r")).float()
            self.gps_g_lookup = torch.from_numpy(np.load(g_path, mmap_mode="r")).float()

            if self.gps_s_lookup is not None:
                print("Loaded cached GPS s_vector")
            if self.gps_g_lookup is not None:
                print("Loaded cached GPS g_tokens")

    def load_index_bundle(self, index_dir):
        index_path = os.path.join(index_dir, "geo_index.faiss")
        meta_path = os.path.join(index_dir, "geo_metadata.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(index_path)

        index = faiss.read_index(index_path)
        index.nprobe = self.retrieval_cfg.nprobe

        metadata_df = pd.read_pickle(meta_path)

        coords = metadata_df[["LAT", "LON"]].values.astype(np.float32)

        s2_lookup = None
        if "S2_TOKENS" in metadata_df.columns:
            s2_lookup = np.stack(metadata_df["S2_TOKENS"].values)

        return index, coords, s2_lookup

    def coarse_retrieval(self, query_s, top_k):
        query_np = query_s.detach().cpu().numpy().astype(np.float32, copy=False)
        faiss.normalize_L2(query_np)
        _, indices = self.index.search(query_np, int(top_k))
        return indices

    def gather_candidates(self, indices):
        indices = torch.as_tensor(indices, device=self.device)

        if not hasattr(self, "coords_lookup_tensor"):
            self.coords_lookup_tensor = torch.tensor(
                self.coords_lookup, dtype=torch.float32, device=self.device
            )

        coords = self.coords_lookup_tensor[indices]

        out = {"coords": coords, "idx": indices}

        if self.s2_lookup is not None:
            if not hasattr(self, "s2_lookup_tensor"):
                self.s2_lookup_tensor = torch.tensor(
                    self.s2_lookup, dtype=torch.long, device=self.device
                )
            out["s2"] = self.s2_lookup_tensor[indices]

        return out

    def encode_candidates(self, cand, gps_encoder, B, K):
        gps_cfg = gps_encoder.cfg
        coords = cand["coords"]
        flat_coords = coords.view(-1, 2)
        flat_s2 = cand["s2"].view(-1, cand["s2"].shape[-1])

        gps_embeds = gps_encoder(flat_coords, flat_s2)

        s_vec = gps_embeds["s_vector"].view(B, K, -1)
        g_tok = gps_embeds["g_tokens"].view(B, K, gps_cfg.n_g_tokens, -1)

        return s_vec, g_tok

    def _compute_s_scores(self, query_s, cand_s):
        return torch.einsum("bd,bkd->bk", query_s, cand_s)

    @torch.no_grad()
    def _compute_g_scores(self, query_g, cand_g):
        return self._sinkhorn_ot(
            query_g,
            cand_g,
            eps=self.retrieval_cfg.ot_eps,
            iters=self.retrieval_cfg.ot_iters,
        )

    @staticmethod
    def _sinkhorn_ot(img_g, gps_g, eps=0.05, iters=3):
        sim = torch.einsum("bnd,bkmd->bknm", img_g, gps_g)

        u = torch.zeros_like(sim[..., 0])  # (B, K, N)
        v = torch.zeros_like(sim[..., 0, :]) # (B, K, M)

        for _ in range(iters):
            # 交替在 M (dim=-1) 和 N (dim=-2) 维度上更新
            u = eps * torch.logsumexp((sim - v.unsqueeze(-2)) / eps, dim=-1)
            v = eps * torch.logsumexp((sim - u.unsqueeze(-1)) / eps, dim=-2)

        # 计算最优传输计划 (Transport Plan) PI
        PI = torch.exp((sim - u.unsqueeze(-1) - v.unsqueeze(-2)) / eps)
        
        # 最终得分：相似度的加权和
        score = torch.sum(PI * sim, dim=(-2, -1))
        return score

    def fuse_scores(self, s_scores, g_scores):
        mode = self.retrieval_cfg.rerank_fusion_mode

        if mode == "dynamic":
            hybrid, alpha, _ = dynamic_fuse_scores(
                s_scores=s_scores,
                g_scores=g_scores,
                temperature=self.retrieval_cfg.rerank_dynamic_temperature,
                consistency_topk=self.retrieval_cfg.rerank_dynamic_consistency_topk,
                weight_gap=self.retrieval_cfg.rerank_dynamic_weight_gap,
                weight_entropy=self.retrieval_cfg.rerank_dynamic_weight_entropy,
                weight_consistency=self.retrieval_cfg.rerank_dynamic_weight_consistency,
                alpha_min=self.retrieval_cfg.rerank_dynamic_alpha_min,
                alpha_max=self.retrieval_cfg.rerank_dynamic_alpha_max,
                normalize_before_fuse=True,
            )
            return hybrid, alpha

        # fallback
        s = self._minmax(s_scores)
        g = self._minmax(g_scores)
        return self.retrieval_cfg.rerank_s_weight * s + (1 - self.retrieval_cfg.rerank_s_weight) * g, None

    @staticmethod
    def _minmax(x):
        return (x - x.min(dim=1, keepdim=True)[0]) / (
            x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0] + 1e-6
        )

    @torch.no_grad()
    def forward(self, img_embeds, top_k, gps_encoder=None):

        # ================= S-space =================
        query_s = F.normalize(img_embeds["s_vector"], dim=-1)
        coarse_idx = self.coarse_retrieval(query_s, top_k)

        if not self.retrieval_cfg.use_rerank:
            return {
                "final_idx": coarse_idx,
                "coarse_idx": coarse_idx,
                "alpha_batch_cpu": None,
            }

        # ================= Candidate =================
        cand = self.gather_candidates(coarse_idx)
        B, K = cand["idx"].shape

        # ================= Encode =================
        use_cache = self.gps_s_lookup is not None and self.gps_g_lookup is not None
        if use_cache:
            idx = cand["idx"].cpu().numpy()
            cand_s = self.gps_s_lookup[idx].to(self.device)
            cand_g = self.gps_g_lookup[idx].to(self.device)
        else:
            cand_s, cand_g = self.encode_candidates(
                cand, gps_encoder, B, K
            )

        cand_s = F.normalize(cand_s, dim=-1)
        query_g = F.normalize(img_embeds["g_tokens"], dim=-1)
        cand_g = F.normalize(cand_g, dim=-1)

        # ================= S score =================
        s_scores = self._compute_s_scores(query_s, cand_s)

        # ================= Rerank pruning =================
        rerank_k = min(K, getattr(self.retrieval_cfg, "rerank_topk", K))
        print(rerank_k)
        topk_s_scores, topk_idx = torch.topk(s_scores, rerank_k, dim=1)

        if use_cache:
            topk_global_idx = torch.gather(cand["idx"], 1, topk_idx) 
            cand_g_small = self.gps_g_lookup[topk_global_idx.cpu()].to(self.device)
        else:
            cand_g_small = torch.gather(
                cand_g,
                1,
                topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, cand_g.size(2), cand_g.size(3)),
            )

        # ================= G score =================
        g_small = self._compute_g_scores(query_g, cand_g_small)
        s_small = torch.gather(s_scores, 1, topk_idx)

        hybrid_small, alpha = self.fuse_scores(s_small, g_small)
        final_scores = s_scores.clone()

        offset_to_ensure_top_tier = 1000.0
        final_scores.scatter_(1, topk_idx, hybrid_small + offset_to_ensure_top_tier)

        # ================= Rerank =================
        order = torch.argsort(final_scores, dim=1, descending=True)
        final_idx = torch.gather(cand["idx"], 1, order).cpu().numpy()

        return {
            "final_idx": final_idx,
            "coarse_idx": coarse_idx,
            "alpha_batch_cpu": alpha.cpu() if alpha is not None else None,
        }