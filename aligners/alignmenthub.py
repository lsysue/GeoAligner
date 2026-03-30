# alignment_hub.py
"""
观察semantic_loss和geo_loss的变化曲线。
Gemini猜测：semantic_loss应该较快下降，geo_loss可能下降较慢但更稳定；依情况调高geo_loss权重。
"""
from typing import Dict
import torch
import torch.nn as nn
from .semantic_aligner import SemanticAligner
from .geographic_aligner import GeoAligner

class AlignmentHub(nn.Module):
    def __init__(
        self,
        s_dim: int,
        g_dim: int,
        loss_weight_s: float = 1.0,
        loss_weight_g: float = 1.0,
        temperature: float = 0.07,
        semantic_queue_size: int = 4096,
    ):
        super().__init__()
        self.loss_weight_s = loss_weight_s
        self.loss_weight_g = loss_weight_g

        # S-Space: 全局语义对齐 (InfoNCE)
        self.semantic_aligner = SemanticAligner(
            s_dim=s_dim,
            temperature=temperature,
            queue_size=semantic_queue_size,
        )
        # G-Space: 地理结构对齐 (Symmetric Attention + InfoNCE)
        self.geo_aligner = GeoAligner(g_dim=g_dim, temperature=temperature)

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