# alignment_hub.py
from typing import Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .semantic_aligner import SemanticAligner
from .geographic_aligner import GeoAligner


class AlignmentHubConfig:
    """AlignmentHub configuration with defaults, similar to encoder config classes."""

    def __init__(
        self,
        s_dim: int = 768,
        g_dim: int = 256,
        loss_weight_s: float = 1.0,
        loss_weight_g: float = 1.0,
        temperature: float = 0.07,
        semantic_queue_size: int = 4096,
        geo_scorer: str = "late",
        hybrid_lambda: float = 0.2,
        sinkhorn_eps: float = 0.05,
        sinkhorn_iters: int = 3,
        ot_mass_mode: str = "uniform",
        ot_unbalanced_tau: float = 1.0,
        ot_partial_mass: float = 1.0,
        sg_fusion_mode: str = "mean",
        sg_fusion_weight: float = 0.5,
        sg_dynamic_temperature: float = 0.07,
        sg_dynamic_consistency_topk: int = 10,
        sg_dynamic_weight_gap: float = 0.45,
        sg_dynamic_weight_entropy: float = 0.35,
        sg_dynamic_weight_consistency: float = 0.20,
        sg_dynamic_alpha_min: float = 0.05,
        sg_dynamic_alpha_max: float = 0.95,
    ):
        self.s_dim = s_dim
        self.g_dim = g_dim
        self.loss_weight_s = loss_weight_s
        self.loss_weight_g = loss_weight_g
        self.temperature = temperature
        self.semantic_queue_size = semantic_queue_size
        self.geo_scorer = geo_scorer
        self.hybrid_lambda = hybrid_lambda
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.ot_mass_mode = ot_mass_mode
        self.ot_unbalanced_tau = ot_unbalanced_tau
        self.ot_partial_mass = ot_partial_mass
        self.sg_fusion_mode = sg_fusion_mode
        self.sg_fusion_weight = sg_fusion_weight
        self.sg_dynamic_temperature = sg_dynamic_temperature
        self.sg_dynamic_consistency_topk = sg_dynamic_consistency_topk
        self.sg_dynamic_weight_gap = sg_dynamic_weight_gap
        self.sg_dynamic_weight_entropy = sg_dynamic_weight_entropy
        self.sg_dynamic_weight_consistency = sg_dynamic_weight_consistency
        self.sg_dynamic_alpha_min = sg_dynamic_alpha_min
        self.sg_dynamic_alpha_max = sg_dynamic_alpha_max

def min_max_norm(scores: torch.Tensor) -> torch.Tensor:
    if scores.dim() != 2:
        raise ValueError("scores must be a 2D tensor [batch, num_candidates]")
    mins = scores.min(dim=1, keepdim=True).values
    maxs = scores.max(dim=1, keepdim=True).values
    return (scores - mins) / (maxs - mins + 1e-6)


def _normalized_entropy(prob: torch.Tensor) -> torch.Tensor:
    num_candidates = max(int(prob.shape[1]), 2)
    entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum(dim=1)
    return entropy / math.log(num_candidates)


def _top1_top2_gap(prob: torch.Tensor) -> torch.Tensor:
    sorted_prob = torch.sort(prob, dim=1, descending=True).values
    if sorted_prob.shape[1] < 2:
        return torch.zeros(sorted_prob.shape[0], device=prob.device, dtype=prob.dtype)
    return sorted_prob[:, 0] - sorted_prob[:, 1]


def _topk_overlap_ratio(
    s_scores: torch.Tensor,
    g_scores: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    k = max(1, min(int(topk), int(s_scores.shape[1]), int(g_scores.shape[1])))
    s_idx = torch.topk(s_scores, k=k, dim=1, largest=True, sorted=False).indices
    g_idx = torch.topk(g_scores, k=k, dim=1, largest=True, sorted=False).indices
    overlap = (s_idx.unsqueeze(2) == g_idx.unsqueeze(1)).any(dim=2).float().sum(dim=1)
    return overlap / float(k)


@torch.no_grad()
def dynamic_fuse_scores(
    s_scores: torch.Tensor,
    g_scores: torch.Tensor,
    temperature: float = 0.07,
    consistency_topk: int = 10,
    weight_gap: float = 0.45,
    weight_entropy: float = 0.35,
    weight_consistency: float = 0.20,
    alpha_min: float = 0.05,
    alpha_max: float = 0.95,
    normalize_before_fuse: bool = True,
):
    if s_scores.shape != g_scores.shape:
        raise ValueError("s_scores and g_scores must share the same shape")
    if s_scores.dim() != 2:
        raise ValueError("s_scores and g_scores must be 2D tensors [batch, num_candidates]")

    s_fuse = min_max_norm(s_scores) if normalize_before_fuse else s_scores
    g_fuse = min_max_norm(g_scores) if normalize_before_fuse else g_scores

    temp = max(float(temperature), 1e-4)
    s_prob = F.softmax(s_fuse / temp, dim=1)
    g_prob = F.softmax(g_fuse / temp, dim=1)

    s_gap = _top1_top2_gap(s_prob)
    g_gap = _top1_top2_gap(g_prob)
    s_entropy = _normalized_entropy(s_prob)
    g_entropy = _normalized_entropy(g_prob)
    overlap = _topk_overlap_ratio(s_fuse, g_fuse, topk=consistency_topk)

    conf_s = (
        float(weight_gap) * s_gap
        + float(weight_entropy) * (1.0 - s_entropy)
        + float(weight_consistency) * overlap
    )
    conf_g = (
        float(weight_gap) * g_gap
        + float(weight_entropy) * (1.0 - g_entropy)
        + float(weight_consistency) * overlap
    )

    alpha = conf_s / (conf_s + conf_g + 1e-6)
    alpha = alpha.clamp(min=float(alpha_min), max=float(alpha_max))

    fused_scores = alpha.unsqueeze(1) * s_fuse + (1.0 - alpha.unsqueeze(1)) * g_fuse
    stats = {
        "alpha": alpha,
        "s_gap": s_gap,
        "g_gap": g_gap,
        "s_entropy": s_entropy,
        "g_entropy": g_entropy,
        "topk_overlap": overlap,
        "conf_s": conf_s,
        "conf_g": conf_g,
    }
    return fused_scores, alpha, stats

class AlignmentHub(nn.Module):
    def __init__(self, cfg: Optional[AlignmentHubConfig] = None, **kwargs):
        super().__init__()
        # Backward-compatible path: allow AlignmentHub(...kwargs) as before.
        if cfg is None:
            cfg = AlignmentHubConfig(**kwargs)

        self.loss_weight_s = cfg.loss_weight_s
        self.loss_weight_g = cfg.loss_weight_g
        self.sg_fusion_mode = cfg.sg_fusion_mode
        self.sg_fusion_weight = cfg.sg_fusion_weight
        self.sg_dynamic_temperature = cfg.sg_dynamic_temperature
        self.sg_dynamic_consistency_topk = cfg.sg_dynamic_consistency_topk
        self.sg_dynamic_weight_gap = cfg.sg_dynamic_weight_gap
        self.sg_dynamic_weight_entropy = cfg.sg_dynamic_weight_entropy
        self.sg_dynamic_weight_consistency = cfg.sg_dynamic_weight_consistency
        self.sg_dynamic_alpha_min = cfg.sg_dynamic_alpha_min
        self.sg_dynamic_alpha_max = cfg.sg_dynamic_alpha_max

        # S-Space: 全局语义对齐 (InfoNCE)
        self.semantic_aligner = SemanticAligner(
            s_dim=cfg.s_dim,
            temperature=cfg.temperature,
            queue_size=cfg.semantic_queue_size,
        )
        # G-Space: 地理结构对齐 (Symmetric Attention + InfoNCE)
        self.geo_aligner = GeoAligner(
            g_dim=cfg.g_dim,
            temperature=cfg.temperature,
            geo_scorer=cfg.geo_scorer,
            hybrid_lambda=cfg.hybrid_lambda,
            sinkhorn_eps=cfg.sinkhorn_eps,
            sinkhorn_iters=cfg.sinkhorn_iters,
            ot_mass_mode=cfg.ot_mass_mode,
            ot_unbalanced_tau=cfg.ot_unbalanced_tau,
            ot_partial_mass=cfg.ot_partial_mass,
        )

    def fuse_pair_similarity(
        self,
        s_sim: torch.Tensor,
        g_sim: torch.Tensor,
        fusion_mode: Optional[str] = None,
        fusion_weight: Optional[float] = None,
    ) -> torch.Tensor:
        """Fuse semantic and geographic similarity matrices for SG retrieval."""
        mode = self.sg_fusion_mode if fusion_mode is None else fusion_mode
        weight = self.sg_fusion_weight if fusion_weight is None else fusion_weight
        if mode == "s_only":
            return s_sim
        if mode == "g_only":
            return g_sim
        if mode == "max":
            return torch.maximum(s_sim, g_sim)
        if mode == "weighted":
            return weight * s_sim + (1.0 - weight) * g_sim
        if mode == "dynamic":
            fused, _, _ = dynamic_fuse_scores(
                s_scores=s_sim,
                g_scores=g_sim,
                temperature=self.sg_dynamic_temperature,
                consistency_topk=self.sg_dynamic_consistency_topk,
                weight_gap=self.sg_dynamic_weight_gap,
                weight_entropy=self.sg_dynamic_weight_entropy,
                weight_consistency=self.sg_dynamic_weight_consistency,
                alpha_min=self.sg_dynamic_alpha_min,
                alpha_max=self.sg_dynamic_alpha_max,
                normalize_before_fuse=True,
            )
            return fused
        return 0.5 * (s_sim + g_sim)

    def compute_pair_similarity(
        self,
        image_s_vec: torch.Tensor,
        gps_s_vec: torch.Tensor,
        image_g_tokens: torch.Tensor,
        gps_g_tokens: torch.Tensor,
        fusion_mode: Optional[str] = None,
        fusion_weight: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute fused SG similarity matrix from semantic and geographic branches."""
        s_sim = self.semantic_aligner.compute_pair_similarity(image_s_vec, gps_s_vec)
        g_sim = self.geo_aligner.compute_pair_similarity(image_g_tokens, gps_g_tokens)
        return self.fuse_pair_similarity(s_sim, g_sim, fusion_mode=fusion_mode, fusion_weight=fusion_weight)

    def forward(
        self, 
        image_embeds: Dict[str, torch.Tensor], 
        gps_embeds: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        输入:
            image_embeds: {
                "s_vector": (B, s_dim),
                "g_tokens": (B, n_g_tokens, g_dim)
            }
            gps_embeds: {
                "s_vector": (B, s_dim),
                "g_tokens": (B, n_g_tokens, g_dim)
            }
        输出:
            Dict 包含各部分 loss 和总 loss
        """
        # 1. 计算 S-Space Loss (粗粒度)
        # Image Global Vector <-> GPS (S2 Context + Center Fourier)
        s_loss = self.semantic_aligner(
            image_embeds["s_vector"], 
            gps_embeds["s_vector"]
        )

        # 2. 计算 G-Space Loss (细粒度)
        # Image Patch Tokens <-> GPS Neighborhood Point Cloud
        # 这一步内部会进行 Cross-Attention，尝试找到最佳匹配结构
        g_loss = self.geo_aligner(
            image_embeds["g_tokens"], 
            gps_embeds["g_tokens"]
        )

        # 3. 加权求和
        total_loss = (self.loss_weight_s * s_loss) + (self.loss_weight_g * g_loss)

        return {
            "loss": total_loss,       # 用于 backward
            "semantic_loss": s_loss,  # 用于日志记录
            "geo_loss": g_loss        # 用于日志记录
        }