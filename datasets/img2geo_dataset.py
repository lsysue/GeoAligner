# datasets/img2geo_dataset.py

import os
import torch
import s2sphere
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Img2GeoDataset(Dataset):
    """
    针对 MP16-Pro 数据集优化的 (图像, GPS) 数据加载器。
    """
    def __init__(self, csv_file: str, img_dir: str, transform: transforms.Compose = None, s2_levels=[3, 6, 9, 11, 13]):
        """
        Args:
            csv_file (str): MP16-Pro 的 CSV 文件路径 (e.g., MP16_Pro_filtered.csv)
            img_dir (str): 图像根目录 (e.g., /home/xxx/data/MP16-Pro/images)
            transform (callable, optional): 图像预处理 transforms
        """
        super().__init__()
        
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f"CSV文件未找到: {csv_file}")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"图像目录未找到: {img_dir}")
            
        self.img_dir = img_dir
        self.transform = transform
        
        # --- 内存优化 ---
        # MP16-Pro CSV 包含很多列 ('region', 'country' 等)，训练只需要 ID 和 坐标。
        # 使用 usecols 只读取必要的列，显著减少内存占用。
        required_cols = ['IMG_ID', 'LAT', 'LON']
        
        # print(f"正在加载数据集元数据: {csv_file} ...")
        try:
            self.geo_metadata = pd.read_csv(csv_file, usecols=required_cols)
        except ValueError as e:
            # 如果列名不匹配（以防万一），尝试读取前几行并报错
            df_sample = pd.read_csv(csv_file, nrows=1)
            raise ValueError(f"CSV列名不匹配。需要: {required_cols}, 实际: {df_sample.columns.tolist()}") from e

        # 过滤掉无效坐标 (简单的范围检查)
        # 有效范围: Lat [-90, 90], Lon [-180, 180]
        initial_len = len(self.geo_metadata)
        self.geo_metadata = self.geo_metadata[
            (self.geo_metadata['LAT'] >= -90) & (self.geo_metadata['LAT'] <= 90) &
            (self.geo_metadata['LON'] >= -180) & (self.geo_metadata['LON'] <= 180)
        ].reset_index(drop=True)
        
        if len(self.geo_metadata) < initial_len:
            print(f"警告: 已过滤掉 {initial_len - len(self.geo_metadata)} 条无效坐标的数据。")

        self.s2_levels = s2_levels

        # print(f"数据集加载完成，共 {len(self.geo_metadata)} 个样本。")

    def __len__(self) -> int:
        return len(self.geo_metadata)

    def _latlon_to_s2_tokens(self, lat, lon):
        try:
            cell_id = s2sphere.CellId.from_lat_lng(s2sphere.LatLng.from_degrees(lat, lon))
            return [cell_id.parent(level).id() for level in self.s2_levels]
        except Exception:
            return [0] * len(self.s2_levels)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            image (Tensor): (3, H, W)
            gps_tensor (Tensor): (2,) -> [Latitude, Longitude]
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 1. 获取图像路径
        # MP16 CSV 中的 IMG_ID 格式如 "92_17_5276763594.jpg"，已包含后缀
        img_filename = self.geo_metadata.at[idx, 'IMG_ID']
        img_path = os.path.join(self.img_dir, img_filename)
        
        # 2. 获取 GPS 坐标
        # 顺序严格为 [Latitude, Longitude]
        lat = self.geo_metadata.at[idx, 'LAT']
        lon = self.geo_metadata.at[idx, 'LON']
        
        # 3. 加载图像
        try:
            # 使用 convert('RGB') 确保处理 1通道(灰度)或 4通道(RGBA) 图片不报错
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError, IOError):
            # 鲁棒性处理：如果图片损坏或丢失，尝试获取下一个
            # 注意：这在训练中是可以接受的，但在验证集严格评估时可能需要记录
            # print(f"警告: 无法读取图片 {img_path}，尝试下一个...")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)
        
        # 4. 转换 GPS 为 Tensor
        # 确保是 float32 类型
        gps_tensor = torch.tensor([lat, lon], dtype=torch.float32)

        s2_tokens = self._latlon_to_s2_tokens(lat, lon)
        s2_np_uint64 = np.array(s2_tokens, dtype=np.uint64)
        s2_np_int64 = s2_np_uint64.astype(np.int64)

        s2_tensor = torch.from_numpy(s2_np_int64)

        return image, gps_tensor, s2_tensor

if __name__ == '__main__':
    # 简单的测试块
    # 请替换为你的真实路径进行测试
    TEST_CSV = "/home/lsy/data/MP16-Pro/metadata/MP16_Pro_filtered.csv"
    TEST_IMG_DIR = "/home/lsy/data/MP16-Pro/images"
    
    if os.path.exists(TEST_CSV):
        ds = Img2GeoDataset(TEST_CSV, TEST_IMG_DIR)
        img, gps = ds[0]
        print(f"Test Sample 0:")
        print(f"Image Shape: {img.size if hasattr(img, 'size') else img.shape}")
        print(f"GPS: {gps} (Type: {gps.dtype})")
    else:
        print("测试路径不存在，跳过测试。")