import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from geopy.distance import geodesic

try:
    from pyproj import Geod
except Exception:
    Geod = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

DEFAULT_DATASET_CSVS: Dict[str, str] = {
    "mp16profull": "/data/lsy/datas/MP16-Pro/metadata/MP16_Pro_places365.csv",
    "mp16pro": "/data/lsy/datas/MP16-Pro/metadata/MP16_Pro_filtered.csv",
    "yfcc26k": "/data/lsy/datas/yfcc26k/metadata/yfcc25600_places365.csv",
    "yfcc4k": "/data/lsy/datas/yfcc4k/metadata/yfcc4k_places365.csv",
    "im2gps3k": "/data/lsy/datas/im2gps3k/metadata/im2gps3k_places365.csv",
}

def _read_lat_lon_table(csv_path: str, source_name: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols_upper = {c.upper(): c for c in df.columns}

    img_col = cols_upper.get("IMG_ID", None)
    lat_col = cols_upper.get("LAT", None)
    lon_col = cols_upper.get("LON", None)

    if img_col is None:
        if "id" in df.columns:
            img_col = "id"
        elif "IMG_ID" in df.columns:
            img_col = "IMG_ID"
        else:
            raise ValueError(f"IMG_ID/id column not found in {source_name}: {csv_path}")

    if lat_col is None:
        lat_col = "lat" if "lat" in df.columns else None
    if lon_col is None:
        lon_col = "lon" if "lon" in df.columns else None

    if lat_col is None or lon_col is None:
        raise ValueError(f"LAT/LON columns not found in {source_name}: {csv_path}")

    out = pd.DataFrame(
        {
            "IMG_ID": df[img_col].astype(str),
            "LAT": pd.to_numeric(df[lat_col], errors="coerce"),
            "LON": pd.to_numeric(df[lon_col], errors="coerce"),
        }
    )

    valid = (
        out["LAT"].between(-90.0, 90.0, inclusive="both")
        & out["LON"].between(-180.0, 180.0, inclusive="both")
    )
    print(f"Read {len(out)} rows from {source_name}: {csv_path}, valid lat/lon: {valid.sum()}")
    out = out[valid].reset_index(drop=True)
    return out


def _geodesic_topk(
    q_lat: np.ndarray,
    q_lon: np.ndarray,
    g_lat: np.ndarray,
    g_lon: np.ndarray,
    chunk_size: int,
    topk: int,
    candidate_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_q = q_lat.shape[0]
    n_g = g_lat.shape[0]
    if n_g == 0:
        raise ValueError("Gallery is empty after filtering valid LAT/LON rows.")

    k = max(1, min(int(topk), n_g))
    c = max(0, int(candidate_k))
    topk_idx = np.full((n_q, k), -1, dtype=np.int64)
    topk_dist = np.full((n_q, k), np.inf, dtype=np.float64)

    gallery_points = list(zip(g_lat.astype(np.float64), g_lon.astype(np.float64)))
    g_lat_arr = g_lat.astype(np.float64)
    g_lon_arr = g_lon.astype(np.float64)
    geod = Geod(ellps="WGS84") if Geod is not None else None

    g_lat_rad = np.deg2rad(g_lat.astype(np.float64))[None, :]
    g_lon_rad = np.deg2rad(g_lon.astype(np.float64))[None, :]
    desc = "Geodesic top-k (exact)" if c <= 0 else "Geodesic top-k (approx)"
    pbar = tqdm(total=n_q, desc=desc, unit="query") if tqdm is not None else None

    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)

        chunk_q_lat = q_lat[start:end].astype(np.float64)
        chunk_q_lon = q_lon[start:end].astype(np.float64)
        ql = np.deg2rad(chunk_q_lat)[:, None]
        qo = np.deg2rad(chunk_q_lon)[:, None]

        cand_idx_chunk = None
        if c > 0:
            c_eff = max(k, min(c, n_g))
            # Fast candidate pre-selection with vectorized haversine.
            dlat = g_lat_rad - ql
            dlon = g_lon_rad - qo
            a = np.sin(dlat / 2.0) ** 2 + np.cos(ql) * np.cos(g_lat_rad) * np.sin(dlon / 2.0) ** 2
            approx = 2.0 * np.arctan2(np.sqrt(np.maximum(a, 0.0)), np.sqrt(np.maximum(1.0 - a, 0.0)))
            cand_idx_chunk = np.argpartition(approx, c_eff - 1, axis=1)[:, :c_eff]

        for local_i, (qlat, qlon) in enumerate(zip(chunk_q_lat, chunk_q_lon)):
            query_point = (float(qlat), float(qlon))
            if c <= 0:
                if geod is not None:
                    lon1 = np.full(n_g, float(qlon), dtype=np.float64)
                    lat1 = np.full(n_g, float(qlat), dtype=np.float64)
                    _, _, dist_m = geod.inv(lon1, lat1, g_lon_arr, g_lat_arr)
                    distances = np.asarray(dist_m, dtype=np.float64) / 1000.0
                    candidate_idx = np.arange(n_g, dtype=np.int64)
                else:
                    distances = np.fromiter(
                        (geodesic(query_point, gpt).km for gpt in gallery_points),
                        dtype=np.float64,
                        count=len(gallery_points),
                    )
                    candidate_idx = np.arange(len(gallery_points), dtype=np.int64)
            else:
                candidate_idx = cand_idx_chunk[local_i]
                candidate_points = [gallery_points[int(i)] for i in candidate_idx]
                distances = np.fromiter(
                    (geodesic(query_point, gpt).km for gpt in candidate_points),
                    dtype=np.float64,
                    count=len(candidate_points),
                )

            # Use partial sort then sort selected top-k for stable nearest-to-farthest order.
            local_topk = np.argpartition(distances, k - 1)[:k]
            local_order = np.argsort(distances[local_topk])
            selected_local = local_topk[local_order]
            selected_global = candidate_idx[selected_local].astype(np.int64)

            topk_idx[start + local_i, :] = selected_global
            topk_dist[start + local_i, :] = distances[selected_local]
            if pbar is not None:
                pbar.update(1)

        if pbar is None:
            print(f"Processed queries: {end}/{n_q}")

    if pbar is not None:
        pbar.close()

    return topk_idx, topk_dist


def _save_topk_results(
    query_name: str,
    gallery_name: str,
    output_dir: str,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    topk: int,
    topk_idx: np.ndarray,
    topk_dist: np.ndarray,
) -> str:
    npz_path = os.path.join(output_dir, f"{query_name}_to_{gallery_name}_top{topk}.npz")
    np.savez_compressed(
        npz_path,
        query_img_id=query_ids.astype(str),
        gallery_img_id=gallery_ids.astype(str),
        topk_gallery_idx=topk_idx.astype(np.int64),
        topk_haversine_km=topk_dist.astype(np.float32),
    )
    return npz_path


def precompute_dataset(
    query_name: str,
    gallery_name: str,
    query_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    output_dir: str,
    chunk_size: int,
    topk: int,
    candidate_k: int,
) -> Tuple[str, str]:
    print(f"\n=== Dataset: {query_name} ===")
    print(f"Valid query rows: {len(query_df)}")

    q_lat = query_df["LAT"].to_numpy(dtype=np.float64)
    q_lon = query_df["LON"].to_numpy(dtype=np.float64)
    g_lat = gallery_df["LAT"].to_numpy(dtype=np.float64)
    g_lon = gallery_df["LON"].to_numpy(dtype=np.float64)

    topk_idx, topk_dist_km = _geodesic_topk(
        q_lat,
        q_lon,
        g_lat,
        g_lon,
        chunk_size=chunk_size,
        topk=topk,
        candidate_k=candidate_k,
    )

    nn_idx = topk_idx[:, 0]
    nn_dist_km = topk_dist_km[:, 0]
    nn_rows = gallery_df.iloc[nn_idx].reset_index(drop=True)

    out_df = pd.DataFrame(
        {
            "query_img_id": query_df["IMG_ID"],
            "gt_lat": query_df["LAT"],
            "gt_lon": query_df["LON"],
            "nearest_gallery_img_id": nn_rows["IMG_ID"].astype(str),
            "nearest_gallery_lat": nn_rows["LAT"].to_numpy(dtype=np.float64),
            "nearest_gallery_lon": nn_rows["LON"].to_numpy(dtype=np.float64),
            "nearest_haversine_km": nn_dist_km,
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{query_name}_to_{gallery_name}_nearest.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    topk_path = _save_topk_results(
        query_name=query_name,
        gallery_name=gallery_name,
        output_dir=output_dir,
        query_ids=query_df["IMG_ID"].to_numpy(dtype=str),
        gallery_ids=gallery_df["IMG_ID"].to_numpy(dtype=str),
        topk=topk,
        topk_idx=topk_idx,
        topk_dist=topk_dist_km,
    )
    print(f"Saved: {topk_path}")

    return out_path, topk_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gallery_name",
        type=str,
        default="mp16pro",
        choices=list(DEFAULT_DATASET_CSVS.keys()),
        help="Gallery dataset name. Must be one key in DEFAULT_DATASET_CSVS.",
    )
    parser.add_argument(
        "--query_name",
        type=str,
        required=True,
        choices=list(DEFAULT_DATASET_CSVS.keys()),
        help="Query dataset name. Must be one key in DEFAULT_DATASET_CSVS.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./galleries/",
        help="Directory to save output CSV files.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Query chunk size for batched nearest search.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Save top-k nearest gallery distances per query.",
    )
    parser.add_argument(
        "--candidate_k",
        type=int,
        default=0,
        help="Approximate mode only: number of fast haversine candidates per query before geodesic reranking. Set <=0 for exact full-gallery geodesic.",
    )
    args = parser.parse_args()

    gallery_source = DEFAULT_DATASET_CSVS[args.gallery_name]
    query_source = DEFAULT_DATASET_CSVS[args.query_name]

    gallery_df = _read_lat_lon_table(gallery_source, source_name=f"gallery/{args.gallery_name}")
    query_df = _read_lat_lon_table(query_source, source_name=f"query/{args.query_name}")
    print(f"Gallery source ({args.gallery_name}): {gallery_source}")
    print(f"Gallery points: {len(gallery_df)}")
    print(f"Query source ({args.query_name}): {query_source}")

    precompute_dataset(
        query_name=args.query_name,
        gallery_name=args.gallery_name,
        query_df=query_df,
        gallery_df=gallery_df,
        output_dir=args.output_dir,
        chunk_size=max(int(args.chunk_size), 1),
        topk=max(int(args.topk), 1),
        candidate_k=int(args.candidate_k),
    )


if __name__ == "__main__":
    main()