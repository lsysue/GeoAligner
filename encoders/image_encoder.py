# encoders/image_encoder.py

from typing import Optional, Dict, Any, Tuple
import math
import torch
import torch.nn as nn
from torchvision import transforms as T

from PIL import Image
import numpy as np

import timm
from transformers import CLIPVisionModel, AutoImageProcessor

class ImageEncoderConfig:
    """
    配置项
    - vit_name: 默认为 Swin Transformer Base
    - img_size: 输入图像尺寸
    - s_dim: S-space 维度 (全局语义)
    - g_dim: G-space 维度 (地理 Token)
    - n_g_tokens: 输出的地理 Token 数量 (通过 Attention 聚合得到)
    """
    def __init__(
        self,
        # 改为 Swin 的默认模型名称
        vit_name: str = "swin_base_patch4_window7_224", 
        img_size: int = 224,
        s_dim: int = 768,
        g_dim: int = 256,
        n_g_tokens: int = 64,
        g_agg_nhead: int = 8,
        g_dropout: float = 0.1,
        g_ffn_mult: int = 4,
        use_landmark: bool = True,
        add_2d_positional_encoding: bool = True,
        normalize_input: bool = False,
        freeze_backbone: bool = False,
    ):
        self.vit_name = vit_name
        self.img_size = img_size
        self.s_dim = s_dim
        self.g_dim = g_dim
        self.n_g_tokens = n_g_tokens
        self.g_agg_nhead = g_agg_nhead
        self.g_dropout = g_dropout
        self.g_ffn_mult = g_ffn_mult
        self.use_landmark = use_landmark
        self.add_2d_positional_encoding = add_2d_positional_encoding
        self.normalize_input = normalize_input
        self.freeze_backbone = freeze_backbone

class SimpleLandmarkHead(nn.Module):
    """
    轻量 landmark head：在 feature tokens 上做检测。
    对于 SwinT，输入是 Flatten 后的 Feature Map Tokens。
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.score_head = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, 1)
        )
        # 回归框 (cx, cy, w, h) 相对坐标
        self.reg_head = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, 4),
            nn.Tanh() 
        )

    def forward(self, patch_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        # patch_tokens: (B, N, C)
        scores = self.score_head(patch_tokens).squeeze(-1)  # (B, N)
        bbox = self.reg_head(patch_tokens)                  # (B, N, 4)
        return {"scores": scores, "bbox": bbox}
    
    def decode(self, outputs, img_size):
        """
        将相对坐标映射回像素坐标
        """
        scores = outputs["scores"]
        bbox = outputs["bbox"]

        if isinstance(img_size, (tuple, list)):
            img_h, img_w = img_size
        else:
            img_h, img_w = img_size, img_size
        
        # 映射 [-1, 1] -> [0, image size]
        cx = (bbox[..., 0] + 1) / 2 * img_w
        cy = (bbox[..., 1] + 1) / 2 * img_h
        w = (bbox[..., 2] + 1) / 2 * img_w / 2
        h = (bbox[..., 3] + 1) / 2 * img_h / 2

        x1 = (cx - w).clamp(0, img_w)
        y1 = (cy - h).clamp(0, img_h)
        x2 = (cx + w).clamp(0, img_w)
        y2 = (cy + h).clamp(0, img_h)

        return {"scores": scores, "bboxes": torch.stack([x1, y1, x2, y2], dim=-1)}

class ImageEncoder(nn.Module):
    """
    ImageEncoder (Unified Version)
    - Backbones:
        1) Swin Transformer (via timm)
        2) CLIP Vision (via transformers)
    - Outputs:
        s_vector: (B, s_dim) - 全局语义
        g_tokens: (B, n_g_tokens, g_dim) - 地理空间 Token
    """
    def __init__(self, cfg: ImageEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.is_clip_backbone = "clip" in cfg.vit_name.lower()

        if self.is_clip_backbone:
            self.backbone = CLIPVisionModel.from_pretrained(cfg.vit_name)
            self.image_processor = AutoImageProcessor.from_pretrained(cfg.vit_name)

            image_mean = getattr(self.image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
            image_std = getattr(self.image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
            self.register_buffer("img_mean", torch.tensor(image_mean).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("img_std", torch.tensor(image_std).view(1, 3, 1, 1), persistent=False)

            self.backbone_feature_dim = int(self.backbone.config.hidden_size)
        else:
            # print(f"Loading backbone: {cfg.vit_name} ...")
            self.backbone = timm.create_model(cfg.vit_name, pretrained=True, features_only=False)

            pretrained_cfg = getattr(self.backbone, "pretrained_cfg", {}) or {}
            img_mean = pretrained_cfg.get("mean", (0.485, 0.456, 0.406))
            img_std = pretrained_cfg.get("std", (0.229, 0.224, 0.225))
            self.register_buffer("img_mean", torch.tensor(img_mean).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("img_std", torch.tensor(img_std).view(1, 3, 1, 1), persistent=False)

            # 获取 Swin 输出的特征维度 (通常 Swin Base 是 1024, Tiny 是 768)
            self.backbone_feature_dim = self.backbone.num_features
            # print(f"Backbone feature dim: {self.backbone_feature_dim}")

        # 2. 映射层 (Mapping Layers)
        # Global Projection: Backbone Features -> S-space
        self.global_proj = nn.Linear(self.backbone_feature_dim, cfg.s_dim)

        # 可选冻结预训练 backbone，仅训练新增 head 与对齐模块
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 3. Landmark Head (Optional)
        if cfg.use_landmark:
            self.landmark_head = SimpleLandmarkHead(in_dim=self.backbone_feature_dim)
        else:
            self.landmark_head = None

        # 4. G-Tokens Aggregation (Projected Multi-Head Cross-Attention)
        self.n_g_tokens = cfg.n_g_tokens
        if cfg.g_dim % cfg.g_agg_nhead != 0:
            raise ValueError(
                f"g_dim ({cfg.g_dim}) must be divisible by g_agg_nhead ({cfg.g_agg_nhead})"
            )
        self.g_query = nn.Parameter(torch.randn(1, self.n_g_tokens, cfg.g_dim))
        self.patch_kv_proj = nn.Linear(self.backbone_feature_dim, cfg.g_dim)
        self.g_cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.g_dim,
            num_heads=cfg.g_agg_nhead,
            batch_first=True,
            dropout=cfg.g_dropout,
        )
        self.g_attn_norm = nn.LayerNorm(cfg.g_dim)
        self.g_ffn_norm = nn.LayerNorm(cfg.g_dim)
        self.g_ffn = nn.Sequential(
            nn.Linear(cfg.g_dim, cfg.g_dim * cfg.g_ffn_mult),
            nn.GELU(),
            nn.Dropout(cfg.g_dropout),
            nn.Linear(cfg.g_dim * cfg.g_ffn_mult, cfg.g_dim),
            nn.Dropout(cfg.g_dropout),
        )

        # LayerNorms
        self.ln_s = nn.LayerNorm(cfg.s_dim)
        self.ln_g = nn.LayerNorm(cfg.g_dim)

    def preprocess_image(self, image):
        if not self.is_clip_backbone:
            raise RuntimeError("preprocess_image is only available when using a CLIP backbone")
        return self.image_processor(images=image, return_tensors="pt")["pixel_values"]

    def _build_2d_sincos_pos_embed(
        self,
        h: int,
        w: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dim % 4 != 0:
            raise ValueError(f"g_dim must be divisible by 4 for 2D sin-cos positional encoding, got {dim}")

        y, x = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        y = y.reshape(-1)
        x = x.reshape(-1)

        half_dim = dim // 2
        quarter_dim = dim // 4
        inv_freq = 1.0 / (10000 ** (torch.arange(quarter_dim, device=device, dtype=dtype) / max(quarter_dim, 1)))

        pos_x = x.unsqueeze(1) * inv_freq.unsqueeze(0)
        pos_y = y.unsqueeze(1) * inv_freq.unsqueeze(0)

        emb_x = torch.cat([torch.sin(pos_x), torch.cos(pos_x)], dim=1)
        emb_y = torch.cat([torch.sin(pos_y), torch.cos(pos_y)], dim=1)
        pos = torch.cat([emb_y, emb_x], dim=1)

        if pos.shape[1] != dim or half_dim != emb_x.shape[1]:
            raise RuntimeError("Unexpected positional embedding shape construction error")
        return pos.unsqueeze(0)

    def forward(self, images: torch.Tensor) -> Dict[str, Any]:
        """
        images: (B, 3, H, W)
        """
        B = images.shape[0]

        if self.cfg.normalize_input:
            images = (images - self.img_mean) / self.img_std

        spatial_hw = None

        # --- 1. Backbone Forward ---
        if self.is_clip_backbone:
            # CLIP last_hidden_state: (B, 1 + N_patch, C)
            vision_out = self.backbone(pixel_values=images, return_dict=True)
            tokens = vision_out.last_hidden_state

            if vision_out.pooler_output is not None:
                global_vec = vision_out.pooler_output
            else:
                global_vec = tokens[:, 0, :]

            patch_tokens = tokens[:, 1:, :]
            n_patch = patch_tokens.shape[1]
            side = int(math.sqrt(n_patch))
            if side * side == n_patch:
                spatial_hw = (side, side)
        else:
            # timm 的 SwinT forward_features 通常返回 (B, H/32, W/32, C)
            features = self.backbone.forward_features(images)

            # 如果是 (B, h, w, C) 且 C 是特征维度，则转换为 (B, C, h, w)
            if features.dim() == 4 and features.shape[-1] == self.backbone_feature_dim:
                features = features.permute(0, 3, 1, 2)  # (B, h, w, C) -> (B, C, h, w)

            # 处理 Swin 的输出形状，确保统一为 (B, N, C) 和 (B, C)
            if features.dim() == 4:
                # 1. Global Vector: Global Average Pooling
                global_vec = features.mean(dim=[-2, -1])  # (B, C)

                # 2. Patch Tokens: Flatten spatial dims
                patch_tokens = features.flatten(2).transpose(1, 2)
                spatial_hw = (features.shape[-2], features.shape[-1])

            elif features.dim() == 3:
                # Case: (B, L, C)
                global_vec = features.mean(dim=1)  # (B, C)
                patch_tokens = features
            else:
                raise ValueError(f"Unexpected output shape from backbone: {features.shape}")

        # --- 2. S-Space Mapping ---
        s_vec = self.global_proj(global_vec)
        s_vec = self.ln_s(s_vec) # (B, s_dim)

        # --- 3. G-Space Mapping & Aggregation ---
        # Q: (B, n_g_tokens, g_dim)
        q = self.g_query.expand(B, -1, -1)
        # K/V: (B, N_raw, g_dim)
        kv = self.patch_kv_proj(patch_tokens)

        if self.cfg.add_2d_positional_encoding and spatial_hw is not None:
            h, w = spatial_hw
            pos_emb = self._build_2d_sincos_pos_embed(
                h=h,
                w=w,
                dim=self.cfg.g_dim,
                device=kv.device,
                dtype=kv.dtype,
            )
            kv = kv + pos_emb

        # Cross-attention + FFN (Transformer-style residual blocks)
        q_norm = self.g_attn_norm(q)
        attn_out, _ = self.g_cross_attn(
            query=q_norm,
            key=kv,
            value=kv,
            need_weights=False,
        )
        g_tokens = q + attn_out
        g_tokens = g_tokens + self.g_ffn(self.g_ffn_norm(g_tokens))
        g_tokens = self.ln_g(g_tokens) # (B, n_g_tokens, g_dim)

        outputs = {
            "s_vector": s_vec, 
            "g_tokens": g_tokens
        }

        # --- 4. Landmark Detection (Optional) ---
        if self.landmark_head is not None:
            lm_raw = self.landmark_head(patch_tokens)
            lm_decoded = self.landmark_head.decode(lm_raw, images.shape[-2:])
            outputs["landmarks"] = lm_decoded

        return outputs

if __name__ == "__main__":
    # 测试代码
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 使用 Swin Base
    cfg = ImageEncoderConfig(
        vit_name="swin_base_patch4_window7_224",
        img_size=224,
        s_dim=768,
        g_dim=256,
        n_g_tokens=64,
        use_landmark=True,
        normalize_input=False,
    )
    
    model = ImageEncoder(cfg).to(device)
    model.eval()
    print("Swin Transformer Encoder initialized successfully.")
    
    img = Image.open("/data/lsy/datas/im2gps3k/images/31700873_d7c4159106_22_25159586@N00.jpg").convert("RGB")
    img = img.resize((cfg.img_size, cfg.img_size))
    transform_ops = [T.ToTensor()]
    if cfg.normalize_input:
        transform_ops.append(T.Normalize(mean=tuple(model.img_mean.view(-1).tolist()), std=tuple(model.img_std.view(-1).tolist())))
    transform = T.Compose(transform_ops)
    image_tensor = transform(img).unsqueeze(0).to(device)  # (B, 3, H, W)

    with torch.no_grad():
        out = model(image_tensor)
        
    print("s_vector shape:", out["s_vector"].shape)     # Expected: (2, 768)
    print("g_tokens shape:", out["g_tokens"].shape)     # Expected: (2, 64, 256)
    if "landmarks" in out:
        print("landmarks bbox shape:", out["landmarks"]["bboxes"].shape) # Expected: (2, 49, 4) (7x7 window output)
        