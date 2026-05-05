import s2sphere
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class GeoDataset(Dataset):
    """
    Lightweight dataset for indexing or GPS-only tasks.
    It bypasses image loading to maximize I/O performance.
    """
    def __init__(self, csv_file: str, s2_levels=[3, 6, 9, 11, 13]):
        """
        Args:
            csv_file: Path to the metadata CSV file.
            lat_col: Name of the latitude column in CSV.
            lon_col: Name of the longitude column in CSV.
        """
        print(f"Loading GeoDataset from {csv_file}...")
        required_cols = ['IMG_ID', 'LAT', 'LON']
        self.geo_metadata = pd.read_csv(csv_file, usecols=required_cols)
        self.s2_levels = s2_levels

        # 预先提取经纬度并转为 numpy 数组，加速访问
        # 确保列名匹配你的 CSV 格式（你的 YAML 中显示是 LAT/LON 或 lat/lon）
        self.coords = self.geo_metadata[['LAT', 'LON']].values.astype(np.float32)
        
        print(f"GeoDataset loaded with {len(self.coords)} samples.")

    def __len__(self):
        return len(self.coords)

    def _latlon_to_s2_tokens(self, lat, lon):
        try:
            cell_id = s2sphere.CellId.from_lat_lng(s2sphere.LatLng.from_degrees(lat, lon))
            return [cell_id.parent(level).id() for level in self.s2_levels]
        except Exception:
            return [0] * len(self.s2_levels)

    def __getitem__(self, idx):
        lat = self.coords[idx, 0]
        lon = self.coords[idx, 1]
        
        img_id = self.geo_metadata.at[idx, 'IMG_ID']
        dummy_img = torch.zeros(1) # 占位符
        gps_tensor = torch.tensor([lat, lon], dtype=torch.float32)
        
        # 这里的 s2_levels 应该和 Img2GeoDataset 保持一致
        s2_tokens = self._latlon_to_s2_tokens(lat, lon) 
        s2_np_uint64 = np.array(s2_tokens, dtype=np.uint64)
        s2_np_int64 = s2_np_uint64.astype(np.int64)
        s2_tensor = torch.from_numpy(s2_np_int64)

        return img_id, dummy_img, gps_tensor, s2_tensor