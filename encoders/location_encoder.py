# location_encoder.py
# Requires: torch, s2sphere
# pip install torch s2sphere

from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import s2sphere
import math

# --- 辅助模块：傅里叶特征编码器 ---
class FourierFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, n_freqs: int):
        super().__init__()
        self.n_freqs = n_freqs
        self.input_dim = input_dim
        self.freq_bands = nn.Parameter(
            2.0 ** torch.arange(n_freqs) * math.pi, requires_grad=False
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = x.unsqueeze(-1) * self.freq_bands
        features = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        return features.flatten(-2)

class GPSEncoderConfig:
    """
    混合 GPS 编码器配置
    """
    def __init__(
        self,
        # S2 相关配置
        s2_levels: List[int] = [3, 6, 9, 11, 13] ,
        s2_embed_dim: int = 128,
        s2_num_buckets: int = 2**17,
        s2_embed_dropout: float = 0.1,
        s2_feature_dropout: float = 0.1,
        transformer_nhead: int = 4,
        transformer_nlayers: int = 2,
        transformer_dropout: float = 0.1,
        # 傅里叶特征相关配置
        fourier_n_freqs: int = 10,
        # continuous_geo_mode supports: latlon_linear | unit_sphere
        continuous_geo_mode: str = "unit_sphere",
        n_g_tokens: int = 64,
        # g_token_neighborhood_scale: float = 0.01,
        base_scale_multiplier: float = 0.5, # 基础尺度乘数
        min_scale_deg: float = 1e-4,
        max_scale_deg: float = 1.0,
        lon_scale_cos_epsilon: float = 0.15,
        # sampling_mode_* supports: random | random_seeded | sunflower | grid
        sampling_mode_train: str = "random",
        # sampling_mode_eval also supports: same_as_train
        sampling_mode_eval: Optional[str] = "same_as_train",
        sampling_seed: int = 42,
        # 最终输出维度
        g_dim: int = 256,
        s_dim: int = 768,
    ):
        self.s2_levels = sorted(s2_levels)
        self.s2_embed_dim = s2_embed_dim
        self.s2_num_buckets = s2_num_buckets
        self.s2_embed_dropout = s2_embed_dropout
        self.s2_feature_dropout = s2_feature_dropout
        self.transformer_nhead = transformer_nhead
        self.transformer_nlayers = transformer_nlayers
        self.transformer_dropout = transformer_dropout

        self.fourier_n_freqs = fourier_n_freqs
        self.continuous_geo_mode = continuous_geo_mode
        self.n_g_tokens = n_g_tokens
        # self.g_token_neighborhood_scale = g_token_neighborhood_scale
        self.base_scale_multiplier = base_scale_multiplier
        self.min_scale_deg = min_scale_deg
        self.max_scale_deg = max_scale_deg
        self.lon_scale_cos_epsilon = lon_scale_cos_epsilon
        self.sampling_mode_train = sampling_mode_train
        self.sampling_mode_eval = sampling_mode_eval
        self.sampling_seed = sampling_seed
        
        self.g_dim = g_dim
        self.s_dim = s_dim

    def refresh_derived_fields(self):
        self.s2_levels = sorted(self.s2_levels)
        self.num_s2_levels = len(self.s2_levels)
        if self.continuous_geo_mode not in {"latlon_linear", "unit_sphere"}:
            raise ValueError(
                f"Unsupported continuous_geo_mode: {self.continuous_geo_mode}. "
                f"Expected one of ['latlon_linear', 'unit_sphere']"
            )
        self.continuous_input_dim = 3 if self.continuous_geo_mode == "unit_sphere" else 2
        self.fourier_dim = self.continuous_input_dim * 2 * self.fourier_n_freqs
        
class GPSEncoder(nn.Module):
    """
    混合 GPS 编码器，结合了 S2 结构化信息和傅里叶特征的连续精度
    """
    def __init__(self, cfg: GPSEncoderConfig):
        super().__init__()
        self.cfg = cfg

        if self.cfg.s2_num_buckets <= 0:
            raise ValueError(f"s2_num_buckets must be > 0, got {self.cfg.s2_num_buckets}")
        if self.cfg.min_scale_deg <= 0 or self.cfg.max_scale_deg <= 0:
            raise ValueError("min_scale_deg and max_scale_deg must be positive")
        if self.cfg.min_scale_deg > self.cfg.max_scale_deg:
            raise ValueError(
                f"min_scale_deg ({self.cfg.min_scale_deg}) must be <= max_scale_deg ({self.cfg.max_scale_deg})"
            )
        if not (0.0 < self.cfg.lon_scale_cos_epsilon <= 1.0):
            raise ValueError(
                f"lon_scale_cos_epsilon must be in (0, 1], got {self.cfg.lon_scale_cos_epsilon}"
            )

        # Recompute derived fields once here to guard against externally mutated cfg values.
        self.cfg.refresh_derived_fields()
        if self.cfg.num_s2_levels <= 0:
            raise ValueError("s2_levels must contain at least one level")

        # --- Part 1: S2 离散编码（宏观上下文） ---
        self.s2_shared_embedding = nn.Embedding(
            num_embeddings=self.cfg.s2_num_buckets,
            embedding_dim=self.cfg.s2_embed_dim,
        )
        self.s2_level_embedding = nn.Embedding(
            num_embeddings=self.cfg.num_s2_levels,
            embedding_dim=self.cfg.s2_embed_dim,
        )
        self.s2_embed_dropout = nn.Dropout(self.cfg.s2_embed_dropout)
        self.s2_feature_dropout = nn.Dropout(self.cfg.s2_feature_dropout)

        s2_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.s2_embed_dim,
            nhead=self.cfg.transformer_nhead,
            dropout=self.cfg.transformer_dropout,
            batch_first=True,
        )
        self.s2_s_vector_transformer = nn.TransformerEncoder(
            s2_encoder_layer, num_layers=self.cfg.transformer_nlayers
        )
        
        # --- Part 2: 傅里叶特征连续编码（微观精度） ---
        self.fourier_encoder = FourierFeatureEncoder(
            input_dim=self.cfg.continuous_input_dim,
            n_freqs=self.cfg.fourier_n_freqs,
        )

        # --- Part 3: 特征融合 (MLPs) ---
        # S-vector 融合
        s2_s_vec_dim = self.cfg.s2_embed_dim
        fourier_s_vec_dim = self.cfg.fourier_dim
        self.s_vector_fusion_mlp = nn.Sequential(
            nn.Linear(s2_s_vec_dim + fourier_s_vec_dim, self.cfg.s_dim),
            nn.ReLU(),
            nn.LayerNorm(self.cfg.s_dim)
        )
        
        # G-tokens 融合
        s2_g_token_dim = self.cfg.s2_embed_dim
        fourier_g_token_dim = self.cfg.fourier_dim
        self.g_tokens_fusion_mlp = nn.Sequential(
            nn.Linear(s2_g_token_dim + fourier_g_token_dim, self.cfg.g_dim),
            nn.ReLU(),
            nn.LayerNorm(self.cfg.g_dim)
        )

    def _encode_continuous_coords(self, gps_coordinates: torch.Tensor) -> torch.Tensor:
        if self.cfg.continuous_geo_mode == "latlon_linear":
            lat = gps_coordinates[..., 0] / 90.0
            lon = gps_coordinates[..., 1] / 180.0
            return torch.stack([lat, lon], dim=-1)

        lat_rad = torch.deg2rad(gps_coordinates[..., 0])
        lon_rad = torch.deg2rad(gps_coordinates[..., 1])
        cos_lat = torch.cos(lat_rad)
        x = cos_lat * torch.cos(lon_rad)
        y = cos_lat * torch.sin(lon_rad)
        z = torch.sin(lat_rad)
        return torch.stack([x, y, z], dim=-1)
    
    def _get_adaptive_scale(self) -> float:
        """
        根据最大 S2 level 动态计算邻域采样的经纬度范围 (degree)。
        Level 10 ~ 0.1度, Level 20 ~ 0.0001度。
        近似公式: 180 / (2 ^ level)
        """
        max_level = max(self.cfg.s2_levels) if self.cfg.s2_levels else 0
        
        # S2 Cell 的几何特性：Level 每加 1，边长大约减半。
        # Level 0 覆盖 1/6 地球，Level 30 是厘米级。
        # 180.0 / (2 ** max_level) 是一个启发式公式，估算该 Level 对应的经纬度跨度(degree)。
        approx_cell_size_deg = 180.0 / (2 ** max_level)

        # 乘以基数倍率。如果 base=0.5，说明我们在 Cell 的一半大小范围内采样。        
        adaptive_scale = approx_cell_size_deg * self.cfg.base_scale_multiplier
        adaptive_scale = max(self.cfg.min_scale_deg, adaptive_scale)
        adaptive_scale = min(self.cfg.max_scale_deg, adaptive_scale)
        return adaptive_scale

    def _resolve_sampling_mode(self) -> str:
        mode = self.cfg.sampling_mode_train if self.training else self.cfg.sampling_mode_eval
        if mode in {None, "same_as_train"}:
            mode = self.cfg.sampling_mode_train
        supported_modes = {"random", "random_seeded", "sunflower", "grid"}
        if mode not in supported_modes:
            raise ValueError(
                f"Unsupported sampling mode: {mode}. "
                f"Expected one of {sorted(supported_modes)}"
            )
        return mode

    def _sample_sunflower_offsets(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        idx = torch.arange(n, device=device, dtype=dtype)
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        radius = torch.sqrt((idx + 0.5) / max(n, 1))
        theta = idx * golden_angle
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        return torch.stack([x, y], dim=-1)

    def _sample_grid_offsets(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid_side = max(1, math.ceil(math.sqrt(n)))
        coords = torch.linspace(-1.0, 1.0, steps=grid_side, device=device, dtype=dtype)
        # Reorder axes from center to boundary to avoid top-biased truncation when n is not a perfect square.
        center = (grid_side - 1) / 2.0
        axis_order = torch.argsort(torch.abs(torch.arange(grid_side, device=device, dtype=dtype) - center))
        ordered = coords[axis_order]
        mesh_y, mesh_x = torch.meshgrid(ordered, ordered, indexing="ij")
        grid = torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=-1)
        return grid[:n]

    def _sample_neighborhood_offsets(
        self,
        center_lat: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, float]:
        adaptive_scale = self._get_adaptive_scale()
        n = self.cfg.n_g_tokens

        mode = self._resolve_sampling_mode()
        if mode == "random":
            base_offsets = torch.randn(batch_size, n, 2, device=device, dtype=dtype)
        elif mode == "random_seeded":
            cpu_gen = torch.Generator(device="cpu")
            cpu_gen.manual_seed(self.cfg.sampling_seed)
            seeded = torch.randn(batch_size, n, 2, generator=cpu_gen, dtype=dtype)
            base_offsets = seeded.to(device=device)
        elif mode == "sunflower":
            sunflower = self._sample_sunflower_offsets(n, device=device, dtype=dtype)
            base_offsets = sunflower.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            grid = self._sample_grid_offsets(n, device=device, dtype=dtype)
            base_offsets = grid.unsqueeze(0).expand(batch_size, -1, -1)

        lat_offsets = base_offsets[..., 0] * adaptive_scale
        lon_offsets_base = base_offsets[..., 1] * adaptive_scale

        lat_rad = torch.deg2rad(center_lat).unsqueeze(1)
        cos_lat = torch.cos(lat_rad).clamp_min(self.cfg.lon_scale_cos_epsilon)
        lon_offsets = lon_offsets_base / cos_lat

        offsets = torch.stack([lat_offsets, lon_offsets], dim=-1)
        return offsets, adaptive_scale

    def forward(self, gps_coordinates: torch.Tensor, s2_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        gps_coordinates: (B, 2) [Lat, Lon]
        s2_tokens: (B, Num_Levels) 直接从 DataLoader 传入
        """
        if gps_coordinates.dim() != 2 or gps_coordinates.shape[1] != 2:
            raise ValueError(
                f"gps_coordinates must have shape (B, 2), got {tuple(gps_coordinates.shape)}"
            )

        device = gps_coordinates.device
        B = gps_coordinates.shape[0]
        expected_levels = len(self.cfg.s2_levels)

        if s2_tokens.dim() != 2:
            raise ValueError(
                f"s2_tokens must have shape (B, Num_Levels), got {tuple(s2_tokens.shape)}"
            )
        if s2_tokens.shape[0] != B:
            raise ValueError(
                f"Batch size mismatch: gps has {B}, s2_tokens has {s2_tokens.shape[0]}"
            )
        if s2_tokens.shape[1] != expected_levels:
            raise ValueError(
                f"s2_tokens second dim must equal len(s2_levels)={expected_levels}, got {s2_tokens.shape[1]}"
            )
        if s2_tokens.device != device:
            s2_tokens = s2_tokens.to(device)
        s2_tokens = s2_tokens.long()

        # --- S2 特征提取 ---
        
        # 2. Embedding 查表
        level_embeddings = []
        for i in range(expected_levels):
            # 取出所有 batch 在当前 level 的 token
            tokens_at_level = s2_tokens[:, i]  # (B,)
            # Hash Trick: 因为 S2 ID 很大 (64位整数)，Embedding 表存不下。
            # 我们取模 (%) 映射到 Embedding 大小。这是一种常见的压缩手段。
            hashed_tensor = tokens_at_level % self.cfg.s2_num_buckets
            token_emb = self.s2_shared_embedding(hashed_tensor)

            level_index = torch.full((B,), i, device=device, dtype=torch.long)
            level_emb = self.s2_level_embedding(level_index)

            level_embeddings.append(token_emb + level_emb)
        
        # 堆叠: (B, Num_Levels, Embed_Dim)
        s2_stacked_embeddings = torch.stack(level_embeddings, dim=1)
        s2_stacked_embeddings = self.s2_embed_dropout(s2_stacked_embeddings)

        # 3. S2 Transformer 交互
        # S2 路径的 s_vector (来自 Transformer)
        s2_transformer_out = self.s2_s_vector_transformer(s2_stacked_embeddings)
        s2_s_vector = s2_transformer_out.mean(dim=1)
        # S2 路径的 g_token (来自求和)
        s2_g_token_base = torch.sum(s2_stacked_embeddings, dim=1)
        s2_s_vector = self.s2_feature_dropout(s2_s_vector)
        s2_g_token_base = self.s2_feature_dropout(s2_g_token_base)

        # --- 傅里叶特征提取 ---
        # 1. Center (S-vector component)
        encoded_center_gps = self._encode_continuous_coords(gps_coordinates)
        fourier_s_vector = self.fourier_encoder(encoded_center_gps)
        
        # 2. Neighborhood (G-tokens component) - [Adaptive Neighborhood Sampling]
        center_points = gps_coordinates.unsqueeze(1)    # (B, 1, 2)
        offsets, _ = self._sample_neighborhood_offsets(
            center_lat=gps_coordinates[:, 0],
            batch_size=B,
            device=device,
            dtype=gps_coordinates.dtype,
        )
        neighborhood_points = center_points + offsets

        n_lat = neighborhood_points[..., 0]
        n_lon = neighborhood_points[..., 1]
        n_lat_clamped = torch.clamp(n_lat, -90, 90)
        n_lon_wrapped = (n_lon + 180.0) % 360.0 - 180.0  # 经度环绕处理

        neighborhood_points = torch.stack([n_lat_clamped, n_lon_wrapped], dim=-1)

        # 归一化邻域点
        encoded_neighborhood = self._encode_continuous_coords(neighborhood_points)
        fourier_g_tokens = self.fourier_encoder(encoded_neighborhood)

        # --- 特征融合 ---
        # 1. S-vector Fusion
        combined_s_features = torch.cat([s2_s_vector, fourier_s_vector], dim=-1)    # (B, S2_Dim + Fourier_Dim)
        s_vector = self.s_vector_fusion_mlp(combined_s_features)                    # (B, s_dim)   

        # 2. G-tokens Fusion
        # 将 S2 的 info 扩展到每个 G-token (因为所有 G-token 都在同一个 S2 cell 内)
        s2_g_token_expanded = s2_g_token_base.unsqueeze(1).expand(-1, self.cfg.n_g_tokens, -1)  # (B, n_g_tokens, S2_Dim)
        combined_g_features = torch.cat([s2_g_token_expanded, fourier_g_tokens], dim=-1)    # (B, n_g_tokens, S2_Dim + Fourier_Dim)
        g_tokens = self.g_tokens_fusion_mlp(combined_g_features)           # (B, n_g_tokens, g_dim)
        
        return {
            "s_vector": s_vector,
            "g_tokens": g_tokens,
        }

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPSEncoderConfig(
        s2_levels=[3, 9, 13],
        base_scale_multiplier=1.0,
        g_dim=256,
        s_dim=768,
        n_g_tokens=64,
        fourier_n_freqs=10,
        # g_token_neighborhood_scale=0.01,
    )

    gps_encoder = GPSEncoder(config).to(device)
    gps_encoder.eval()

    dummy_gps_data = torch.tensor([
        [40.7580, -73.9855], # New York
        [51.5055, -0.0754],  # London
        [39.9163, 116.3972]  # Beijing
    ], device=device, dtype=torch.float32)

    # Build S2 tokens for each configured level.
    s2_rows = []
    for lat, lon in dummy_gps_data.cpu().tolist():
        latlng = s2sphere.LatLng.from_degrees(lat, lon)
        cell_id = s2sphere.CellId.from_lat_lng(latlng)
        # Keep values in signed int64 range for torch.long.
        s2_rows.append([
            cell_id.parent(level).id() & ((1 << 63) - 1)
            for level in config.s2_levels
        ])
    dummy_s2_tokens = torch.tensor(s2_rows, device=device, dtype=torch.long)

    with torch.no_grad():
        output = gps_encoder(dummy_gps_data, dummy_s2_tokens)

    print("--- GPSEncoder with Fourier Features Test ---")
    
    s_vec = output["s_vector"]
    g_tokens = output["g_tokens"]
    
    print(f"Output s_vector shape: {s_vec.shape}")
    print(f"Output g_tokens shape: {g_tokens.shape}")

    expected_s_shape = (dummy_gps_data.shape[0], config.s_dim)
    expected_g_shape = (dummy_gps_data.shape[0], config.n_g_tokens, config.g_dim)

    assert s_vec.shape == expected_s_shape, \
        f"S-vector shape mismatch! Expected {expected_s_shape}, got {s_vec.shape}"
    assert g_tokens.shape == expected_g_shape, \
        f"G-tokens shape mismatch! Expected {expected_g_shape}, got {g_tokens.shape}"
    
    print("\nSuccessfully generated s_vector and g_tokens with correct shapes.")