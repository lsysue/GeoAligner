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
    def __init__(self, g_dim: int, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.g_dim = g_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))

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

        logits_img2gps = sims_img2gps.max(dim=3)[0].mean(dim=2) * logit_scale
        logits_gps2img = sims_gps2img.max(dim=3)[0].mean(dim=2) * logit_scale
        
        local_B = logits_img2gps.shape[0]
        labels = self._build_global_labels(local_B, image_g_tokens.device)

        loss_img2gps = F.cross_entropy(logits_img2gps, labels)
        loss_gps2img = F.cross_entropy(logits_gps2img, labels)
        
        return (loss_img2gps + loss_gps2img) / 2