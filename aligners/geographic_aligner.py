# geo_aligner_symmetric.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class GeoAligner(nn.Module):
    """
    对称的地理结构化对齐模块 (Symmetric Geo Aligner)

    该模块通过对称的交叉注意力机制，对齐来自两个不同模态的地理令牌序列
    (例如，Image g_tokens 和 GPS g_tokens)。
    """
    def __init__(
        self,
        g_dim: int,
        temperature: float = 0.07,
        geo_scorer: str = "late",
        hybrid_lambda: float = 0.2,
        sinkhorn_eps: float = 0.05,
        sinkhorn_iters: int = 3,
        ot_mass_mode: str = "uniform",
        ot_unbalanced_tau: float = 1.0,
        ot_partial_mass: float = 1.0,
    ):
        super().__init__()
        geo_scorer = str(geo_scorer).lower()
        if geo_scorer not in {"late", "ot", "hybrid"}:
            raise ValueError(
                f"Unsupported geo_scorer={geo_scorer}. Expected one of ['late', 'ot', 'hybrid']"
            )
        ot_mass_mode = str(ot_mass_mode).lower()
        if ot_mass_mode not in {"uniform", "confidence"}:
            raise ValueError(
                f"Unsupported ot_mass_mode={ot_mass_mode}. Expected one of ['uniform', 'confidence']"
            )

        self.g_dim = g_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        self.geo_scorer = geo_scorer
        self.hybrid_lambda = float(min(max(hybrid_lambda, 0.0), 1.0))
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.ot_mass_mode = ot_mass_mode
        self.ot_unbalanced_tau = float(min(max(ot_unbalanced_tau, 0.0), 1.0))
        self.ot_partial_mass = float(min(max(ot_partial_mass, 0.05), 1.0))

    @torch.no_grad()
    def gather_features(self, features):
        if not dist.is_initialized():
            return features
        gathered_features = [torch.zeros_like(features) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_features, features)
        return torch.cat(gathered_features, dim=0)

    @torch.no_grad()
    def _build_global_labels(self, local_batch_size: int, device: torch.device) -> torch.Tensor:
        if not dist.is_initialized():
            return torch.arange(local_batch_size, device=device)

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_bs_tensor = torch.tensor([local_batch_size], device=device, dtype=torch.long)
        gathered_bs = [torch.zeros_like(local_bs_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_bs, local_bs_tensor)
        batch_sizes = torch.cat(gathered_bs, dim=0)
        global_offset = int(batch_sizes[:rank].sum().item())
        return torch.arange(local_batch_size, device=device) + global_offset

    def _sinkhorn_ot_score(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sim_matrix: (B, Global_B, N, M)
        Returns:
            (B, Global_B)
        """
        bsz, global_bsz, n_tokens, m_tokens = sim_matrix.shape

        if self.ot_mass_mode == "confidence":
            # Confidence mass: stronger tokens receive larger transport budget.
            mu = F.softmax(sim_matrix.max(dim=-1).values, dim=-1)
            nu = F.softmax(sim_matrix.max(dim=-2).values, dim=-1)
        else:
            mu = torch.full(
                (bsz, global_bsz, n_tokens),
                1.0 / max(n_tokens, 1),
                device=sim_matrix.device,
                dtype=sim_matrix.dtype,
            )
            nu = torch.full(
                (bsz, global_bsz, m_tokens),
                1.0 / max(m_tokens, 1),
                device=sim_matrix.device,
                dtype=sim_matrix.dtype,
            )
        mu = mu * self.ot_partial_mass
        nu = nu * self.ot_partial_mass
        log_mu = torch.log(mu)
        log_nu = torch.log(nu)

        # log(K) where K = exp(sim / eps)
        log_k = sim_matrix / self.sinkhorn_eps
        log_u = torch.zeros_like(mu)
        log_v = torch.zeros_like(nu)

        tau = self.ot_unbalanced_tau
        for _ in range(self.sinkhorn_iters):
            upd_u = log_mu - torch.logsumexp(log_k + log_v.unsqueeze(-2), dim=-1)
            upd_v = log_nu - torch.logsumexp(log_k + log_u.unsqueeze(-1), dim=-2)
            if tau < 1.0:
                log_u = tau * upd_u + (1.0 - tau) * log_u
                log_v = tau * upd_v + (1.0 - tau) * log_v
            else:
                log_u = upd_u
                log_v = upd_v

        log_p = log_k + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
        p = torch.exp(log_p)
        return torch.sum(p * sim_matrix, dim=(-2, -1))

    def cross_attention_aggregation(
        self, 
        query_source_tokens: torch.Tensor, 
        key_value_source_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        一个辅助函数，执行交叉注意力的聚合过程。
        
        Args:
            query_source_tokens: 用于生成 Query 的源 token 序列。
            key_value_source_tokens: 用于生成 Key 和 Value 的源 token 序列。
        
        Returns:
            聚合后的向量。
        """
        # 1. 生成 Query: 对 query_source 进行平均池化
        query = query_source_tokens.mean(dim=1, keepdim=True) # (B, 1, g_dim)
        
        # 2. Key 和 Value 来自另一个模态
        keys = key_value_source_tokens
        values = key_value_source_tokens

        # 3. 计算注意力并聚合
        attn_scores = torch.matmul(query, keys.transpose(1, 2)) / (self.g_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        aggregated_vector = torch.matmul(attn_weights, values).squeeze(1) # (B, g_dim)
        
        return aggregated_vector

    def compute_pair_similarity(
        self,
        image_g_tokens: torch.Tensor,
        gps_g_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise geographic similarity matrix between image and GPS tokens."""
        image_g_tokens = F.normalize(image_g_tokens, p=2, dim=-1)
        gps_g_tokens = F.normalize(gps_g_tokens, p=2, dim=-1)

        sims_img2gps = torch.einsum('bnd,gmd->bgnm', image_g_tokens, gps_g_tokens)
        late_img2gps = sims_img2gps.max(dim=3)[0].mean(dim=2)

        if self.geo_scorer == "late":
            return late_img2gps

        ot_img2gps = self._sinkhorn_ot_score(sims_img2gps)
        if self.geo_scorer == "ot":
            return ot_img2gps

        alpha = self.hybrid_lambda
        return (1.0 - alpha) * late_img2gps + alpha * ot_img2gps

    def forward(self, image_g_tokens: torch.Tensor, gps_g_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_g_tokens (torch.Tensor): 图像的地理令牌序列, 形状 (B, n_img_tokens, g_dim)
            gps_g_tokens (torch.Tensor): GPS 的地理令牌序列, 形状 (B, n_gps_tokens, g_dim)
        """
        # 归一化输入
        image_g_tokens = F.normalize(image_g_tokens, p=2, dim=-1)
        gps_g_tokens = F.normalize(gps_g_tokens, p=2, dim=-1)

        global_image_g_tokens = self.gather_features(image_g_tokens) # (Global_B, N, D)
        global_gps_g_tokens = self.gather_features(gps_g_tokens)     # (Global_B, M, D)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)

        sims_img2gps = torch.einsum('bnd,gmd->bgnm', image_g_tokens, global_gps_g_tokens)
        sims_gps2img = torch.einsum('bmd,gnd->bgmn', gps_g_tokens, global_image_g_tokens)

        late_img2gps = sims_img2gps.max(dim=3)[0].mean(dim=2)
        late_gps2img = sims_gps2img.max(dim=3)[0].mean(dim=2)

        if self.geo_scorer == "late":
            score_img2gps = late_img2gps
            score_gps2img = late_gps2img
        else:
            ot_img2gps = self._sinkhorn_ot_score(sims_img2gps)
            ot_gps2img = self._sinkhorn_ot_score(sims_gps2img)
            if self.geo_scorer == "ot":
                score_img2gps = ot_img2gps
                score_gps2img = ot_gps2img
            else:
                alpha = self.hybrid_lambda
                score_img2gps = (1.0 - alpha) * late_img2gps + alpha * ot_img2gps
                score_gps2img = (1.0 - alpha) * late_gps2img + alpha * ot_gps2img

        logits_img2gps = score_img2gps * logit_scale
        logits_gps2img = score_gps2img * logit_scale
        
        local_B = logits_img2gps.shape[0]
        labels = self._build_global_labels(local_B, image_g_tokens.device)

        loss_img2gps = F.cross_entropy(logits_img2gps, labels)
        loss_gps2img = F.cross_entropy(logits_gps2img, labels)
        
        return (loss_img2gps + loss_gps2img) / 2