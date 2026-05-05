from importlib.resources import path
import os
import json
import faiss
import s2sphere
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from utils.config import load_config, config_to_dict, arg_parser, Config
from datasets.geo_dataset import GeoDataset
from encoders.location_encoder import GPSEncoder

@dataclass
class IndexerConfig:
    use_ivf: bool = True
    nlist: int = 100

class IndexBuilder:

    def __init__(self, cfg: Config, checkpoint_path: str, device):
        self.cfg = cfg
        self.index_cfg = getattr(cfg.retrieval, 'indexer', None)
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.run_tag = os.path.basename(os.path.dirname(checkpoint_path))
        self.output_dir = os.path.join(cfg.dirs.index_dir, self.run_tag)
        os.makedirs(self.output_dir, exist_ok=True)

        self.batch_size = cfg.data.batch_size
        self.num_workers = cfg.data.num_workers

        self.use_ivf = bool(self.index_cfg.use_ivf)
        self.nlist = int(self.index_cfg.nlist)
        self.cache_dir = os.path.join(cfg.dirs.root_dir, cfg.dirs.cache_dir)

        self.cache_path = os.path.join(self.cache_dir, self.run_tag)
        

    def build_dataloader(self):
        dataset = GeoDataset(
            csv_file=self.cfg.data.train.csv_file,
            s2_levels=self.cfg.model.gps.s2_levels
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return dataset, loader

    def build_gps_encoder(self):
        gps_cfg = self.cfg.model.gps
        model = GPSEncoder(gps_cfg).to(self.device)

        print(f"Loading GPS encoder from {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)

        state_dict = ckpt.get("gps_encoder", ckpt)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        model.eval()
        return model

    def load_from_cache(self, s_path, meta_path):
        print(f"Loading cache from {self.cache_path}")

        s_vectors = np.load(s_path, mmap_mode="r")

        meta = np.load(meta_path, allow_pickle=True)
        img_ids = meta["img_ids"]
        coords = meta["coords"]

        print(f"Loaded {len(s_vectors)} vectors from cache")
        return s_vectors, coords
    
    @torch.no_grad()
    def encode_gallery(self, loader, gps_encoder):

        total = len(loader.dataset)

        s_memmap = np.lib.format.open_memmap(
            "_".join([self.cache_path, "s.npy"]),
            mode="w+",
            dtype=np.float32,
            shape=(total, self.cfg.model.gps.s_dim)
        )

        coords = np.zeros((total, 2), dtype=np.float32)
        s2_all = []

        offset = 0

        for img_id, _, gps, s2 in tqdm(loader, desc="Encoding GPS"):

            gps = gps.to(self.device)
            s2 = s2.to(self.device)

            out = gps_encoder(gps, s2)

            s_vec = torch.nn.functional.normalize(out["s_vector"], dim=-1)
            s_np = s_vec.cpu().numpy().astype(np.float32)

            bs = s_np.shape[0]

            s_memmap[offset:offset+bs] = s_np
            coords[offset:offset+bs] = gps.cpu().numpy().astype(np.float32)
            s2_all.append(s2.cpu().numpy())

            offset += bs

        s_memmap.flush()
        s2_tokens = np.concatenate(s2_all, axis=0)

        return s_memmap, coords, s2_tokens

    def build_faiss(self, vectors):
        dim = vectors.shape[1]
        use_ivf = self.use_ivf

        if not use_ivf:
            print("Using Flat index")
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)
            return index

        # IVF
        nlist = self.nlist
        print(f"Building IVF index: nlist={nlist}")

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        # training
        n_train = min(len(vectors), 100000)

        indices = np.random.choice(len(vectors), n_train, replace=False)
        train_samples = np.ascontiguousarray(vectors[indices], dtype=np.float32)

        print("Training FAISS index...")
        index.train(train_samples)

        print("Adding vectors...")
        for i in tqdm(range(0, len(vectors), 10000)):
            index.add(vectors[i:i+10000])

        return index

    def save_metadata(self, coords, s2_tokens):
        metadata_df = pd.DataFrame({
            "LAT": coords[:, 0],
            "LON": coords[:, 1],
            "S2_TOKENS": list(s2_tokens)})
        
        meta_path = os.path.join(self.output_dir, "geo_metadata.pkl")
        metadata_df.to_pickle(meta_path)
        print(f"Saved Meta → {meta_path}")

    def run(self):

        s_path = "_".join([self.cache_path, "s.npy"])
        meta_path = "_".join([self.cache_path, "meta.npz"])
        print(s_path, meta_path)
        cache_exists = os.path.exists(s_path) and os.path.exists(meta_path)
        if cache_exists:
            s_vectors, coords = self.load_from_cache(s_path, meta_path)
            print("Calculating S2 tokens from cached coordinates...")
            lat = coords[:, 0]
            lon = coords[:, 1]
            s2_levels = self.cfg.model.gps.s2_levels
            
            # 使用现有的 latlon_to_s2 函数[cite: 4]
            s2_tokens = np.array(
                [latlon_to_s2(la, lo, s2_levels) for la, lo in zip(lat, lon)],
                dtype=np.int64,
            )
            print("Skipping encoding since cache is used.")
        else:
            dataset, loader = self.build_dataloader()
            gps_encoder = self.build_gps_encoder()

            print(f"Total samples: {len(dataset)}")

            s_vectors, coords, s2_tokens = self.encode_gallery(loader, gps_encoder)

        print("Building FAISS index...")
        vectors = np.array(s_vectors, dtype=np.float32, copy=True)
        faiss.normalize_L2(vectors)
        index = self.build_faiss(vectors)

        index_path = os.path.join(self.output_dir, "geo_index.faiss")
        faiss.write_index(index, index_path)

        print(f"Saved FAISS → {index_path}")

        self.save_metadata(coords, s2_tokens)

        # 保存配置（方便复现）
        cfg_path = os.path.join(self.output_dir, "build_config.json")
        with open(cfg_path, "w") as f:
            json.dump(config_to_dict(self.cfg), f, indent=2)

        print("Index building complete.")


def latlon_to_s2(lat, lon, levels):
    try:
        cell = s2sphere.CellId.from_lat_lng(
            s2sphere.LatLng.from_degrees(float(lat), float(lon))
        )
        return [(cell.parent(l).id() & ((1 << 63) - 1)) for l in levels]
    except Exception:
        return [0] * len(levels)

def _resolve_checkpoint_path(args):
    if args.checkpoint_path:
        return args.checkpoint_path

    if not args.run_dir:
        raise ValueError("Either --checkpoint_path or --run_dir must be provided.")

    return os.path.join(args.ckpt_dir, args.run_dir, args.checkpoint)


def main():
    parser = arg_parser()
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.checkpoint_path = _resolve_checkpoint_path(args)

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    builder = IndexBuilder(cfg, args.checkpoint_path, device)
    builder.run()


if __name__ == "__main__":
    main()