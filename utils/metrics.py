import torch

from utils.ddp import Accumulator


class Evaluator:
    """Distance and hit-rate evaluation utilities shared by retrieval and generation."""

    def __init__(self, thresholds_km, distance_backend: str = "haversine"):
        self.thresholds_km = tuple(float(t) for t in thresholds_km)
        self.distance_backend = str(distance_backend).lower()
        if self.distance_backend not in {"haversine", "geodesic"}:
            raise ValueError("distance_backend must be 'haversine' or 'geodesic'")

    @staticmethod
    def haversine_km(query_coords: torch.Tensor, cand_coords: torch.Tensor) -> torch.Tensor:
        lat1 = torch.deg2rad(query_coords[..., 0])
        lon1 = torch.deg2rad(query_coords[..., 1])
        lat2 = torch.deg2rad(cand_coords[..., 0])
        lon2 = torch.deg2rad(cand_coords[..., 1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a.clamp_min(1e-12)), torch.sqrt((1 - a).clamp_min(1e-12)))
        return 6371.0 * c

    @staticmethod
    def geodesic_km(query_coords: torch.Tensor, cand_coords: torch.Tensor) -> torch.Tensor:
        from geopy.distance import geodesic

        query_np = query_coords.detach().cpu().numpy().reshape(-1, 2)
        cand_np = cand_coords.detach().cpu().numpy().reshape(-1, 2)
        dists = [geodesic((float(q[0]), float(q[1])), (float(c[0]), float(c[1]))).km for q, c in zip(query_np, cand_np)]
        out = torch.tensor(dists, device=query_coords.device, dtype=query_coords.dtype)
        return out.view(query_coords.shape[:-1])

    def pairwise_distance_km(self, query_coords: torch.Tensor, cand_coords: torch.Tensor) -> torch.Tensor:
        if self.distance_backend == "haversine":
            return self.haversine_km(query_coords, cand_coords)
        return self.geodesic_km(query_coords, cand_coords)

    def hit_counts_from_min_dist(self, min_dists_km: torch.Tensor, thresholds_tensor: torch.Tensor) -> torch.Tensor:
        return (min_dists_km.unsqueeze(1) <= thresholds_tensor.unsqueeze(0)).sum(dim=0).to(torch.long)


class Retriever:
    """Top-k retrieval helper for similarity matrices."""
    @staticmethod
    def init_recall_state(ks, thresholds, distance_backend: str = "haversine"):
        return {
            "ks": tuple(int(k) for k in ks),
            "thresholds": tuple(float(t) for t in thresholds),
            "accumulator": Accumulator(),
            "threshold_tensor": None,
            "evaluator": Evaluator(thresholds, distance_backend=distance_backend),
        }

    @staticmethod
    @torch.no_grad()
    def update_recall_state(state: dict, sim: torch.Tensor, query_coords: torch.Tensor, candidate_coords: torch.Tensor = None):
        if sim.numel() == 0:
            return

        ks = state["ks"]
        thresholds = state["thresholds"]
        accumulator = state["accumulator"]
        evaluator = state["evaluator"]

        accumulator.add_count("n", int(sim.shape[0]))
        query_coords = query_coords.detach()
        if candidate_coords is None:
            candidate_coords = query_coords
        else:
            candidate_coords = candidate_coords.detach()

        n_k = len(ks)
        n_t = len(thresholds)
        batch_hits = torch.zeros((n_k, n_t), device=sim.device, dtype=torch.long)

        threshold_tensor = state.get("threshold_tensor", None)
        if threshold_tensor is None or threshold_tensor.shape[0] != n_t or threshold_tensor.device != sim.device:
            threshold_tensor = torch.tensor(thresholds, device=sim.device, dtype=sim.dtype)
            state["threshold_tensor"] = threshold_tensor

        if sim.dim() != 2:
            raise ValueError("similarity matrix must be 2D: [num_queries, num_candidates]")
        max_k = min(max(ks), sim.shape[1])
        topk_idx = torch.topk(sim, k=max_k, dim=1, largest=True, sorted=True).indices

        cand_coords = candidate_coords[topk_idx]
        query_coords_expanded = query_coords.unsqueeze(1).expand(-1, max_k, -1)
        dists_km = evaluator.pairwise_distance_km(query_coords_expanded, cand_coords)

        for row_idx, k in enumerate(ks):
            effective_k = min(k, max_k)
            min_dists = dists_km[:, :effective_k].min(dim=1).values
            hit_counts = evaluator.hit_counts_from_min_dist(min_dists, threshold_tensor)
            batch_hits[row_idx] += hit_counts
            if k == 1:
                accumulator.append_vector("top1", min_dists)

        accumulator.add_tensor_sum("hits", batch_hits)

    @staticmethod
    @torch.no_grad()
    def finalize_recall_state(state: dict, device: torch.device) -> dict:
        ks = state["ks"]
        thresholds = state["thresholds"]
        accumulator = state["accumulator"]

        total_n = max(accumulator.reduced_count("n", device=device), 1)

        hit_tensor = accumulator.reduced_tensor_sum(
            "hits",
            shape=(len(ks), len(thresholds)),
            device=device,
            dtype=torch.long,
        ).reshape(-1)

        hit_vals = hit_tensor.detach().cpu().tolist()
        out = {}
        idx = 0
        for k in ks:
            for t in thresholds:
                out[f"r@{k}_{int(t)}km"] = hit_vals[idx] / total_n
                idx += 1

        top1_all = accumulator.gathered_vector("top1", device=device, dtype=torch.float32)
        out["median_error_km"] = torch.median(top1_all).item() if top1_all.numel() > 0 else float("nan")
        return out


class Generator:
    """Top-k candidate extraction helper for generation outputs."""

    @staticmethod
    def topk_from_scores(candidate_scores: torch.Tensor, top_k: int):
        if candidate_scores.dim() != 2:
            raise ValueError("candidate_scores must be 2D: [batch, num_candidates]")
        k = min(int(top_k), int(candidate_scores.shape[1]))
        topk = torch.topk(candidate_scores, k=k, dim=1, largest=True, sorted=True)
        return topk.values, topk.indices

    @staticmethod
    def gather_topk_coords(candidate_coords: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        if candidate_coords.dim() != 3:
            raise ValueError("candidate_coords must be 3D: [batch, num_candidates, 2]")
        idx = topk_idx.unsqueeze(-1).expand(-1, -1, candidate_coords.shape[-1])
        return torch.gather(candidate_coords, dim=1, index=idx)


