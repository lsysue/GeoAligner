import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class GeoCLIPDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, preprocess):
        super().__init__()
        self.img_dir = img_dir
        self.preprocess = preprocess
        required_cols = ["IMG_ID", "LAT", "LON"]
        self.geo_metadata = pd.read_csv(csv_file, usecols=required_cols).reset_index(drop=True)

        initial_len = len(self.geo_metadata)
        self.geo_metadata = self.geo_metadata[
            self.geo_metadata["LAT"].between(-90.0, 90.0, inclusive="both")
            & self.geo_metadata["LON"].between(-180.0, 180.0, inclusive="both")
        ].reset_index(drop=True)

        if len(self.geo_metadata) < initial_len:
            print(f"Filtered out {initial_len - len(self.geo_metadata)} invalid coordinate rows.")

    def __len__(self) -> int:
        return len(self.geo_metadata)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = str(self.geo_metadata.at[idx, "IMG_ID"])
        img_path = os.path.join(self.img_dir, img_id)
        lat = float(self.geo_metadata.at[idx, "LAT"])
        lon = float(self.geo_metadata.at[idx, "LON"])

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError, IOError):
            return self.__getitem__((idx + 1) % len(self))

        img_tensor = self.preprocess(image).squeeze(0)
        gps_tensor = torch.tensor([lat, lon], dtype=torch.float32)
        return img_id, img_tensor, gps_tensor