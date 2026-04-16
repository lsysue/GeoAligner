import torch


def semantic_pair_similarity(image_s_vectors: torch.Tensor, gps_s_vectors: torch.Tensor) -> torch.Tensor:
    """Compute pairwise semantic cosine similarity matrix.

    Args:
        image_s_vectors: (B, D)
        gps_s_vectors: (G, D)

    Returns:
        Pairwise similarity matrix of shape (B, G).
    """
    image_s_vectors = torch.nn.functional.normalize(image_s_vectors, p=2, dim=-1)
    gps_s_vectors = torch.nn.functional.normalize(gps_s_vectors, p=2, dim=-1)
    return image_s_vectors @ gps_s_vectors.T

def geographic_pair_similarity(image_g_tokens: torch.Tensor, gps_g_tokens: torch.Tensor) -> torch.Tensor:
    """Compute pairwise geographic similarity using late interaction.

    Args:
        image_g_tokens: (B, N, D)
        gps_g_tokens: (G, M, D)

    Returns:
        Pairwise similarity matrix of shape (B, G).
    """
    image_g_tokens = torch.nn.functional.normalize(image_g_tokens, p=2, dim=-1)
    gps_g_tokens = torch.nn.functional.normalize(gps_g_tokens, p=2, dim=-1)

    token_sim = torch.einsum("bnd,gmd->bgnm", image_g_tokens, gps_g_tokens)
    return token_sim.max(dim=3)[0].mean(dim=2)


def pos_neg_stats(sim: torch.Tensor, semi_hard_q: float = 0.9):
    """Compute positive/negative/gap from a square similarity matrix.

    Negative is defined as semi-hard: row-wise quantile over off-diagonal entries,
    then averaged over rows.
    """
    if sim.dim() != 2:
        raise ValueError("sim_matrix must be 2D: [B, G]")
    if sim.shape[0] != sim.shape[1]:
        raise ValueError("sim_matrix must be square [B, B] to compute pos/neg stats")
    if not (0.0 <= semi_hard_q <= 1.0):
        raise ValueError("semi_hard_q must be in [0, 1]")

    pos = torch.diagonal(sim).mean()
    if sim.shape[0] > 1:
        mask = ~torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
        neg_matrix = sim[mask].view(sim.shape[0], sim.shape[1] - 1)
        row_neg = torch.quantile(neg_matrix, q=semi_hard_q, dim=1)
        neg = row_neg.mean()
    else:
        neg = torch.tensor(0.0, device=sim.device)

    return pos.item(), neg.item(), (pos - neg).item()


def pos_neg_stats_per_sample(sim: torch.Tensor, semi_hard_q: float = 0.9):
    """Compute per-sample pos/neg/margin/rank from a square similarity matrix.

    Returns:
        pos_diag: (B,) diagonal positive similarities.
        neg: (B,) row-wise semi-hard negative similarities.
        margin: (B,) pos_diag - neg.
        rank: (B,) rank of diagonal element in each row (1 is best).
    """
    if sim.dim() != 2:
        raise ValueError("sim_matrix must be 2D: [B, G]")
    if sim.shape[0] != sim.shape[1]:
        raise ValueError("sim_matrix must be square [B, B] to compute per-sample stats")
    if not (0.0 <= semi_hard_q <= 1.0):
        raise ValueError("semi_hard_q must be in [0, 1]")

    pos_diag = torch.diagonal(sim)
    if sim.shape[0] > 1:
        mask = ~torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
        neg_matrix = sim[mask].view(sim.shape[0], sim.shape[1] - 1)
        neg = torch.quantile(neg_matrix, q=semi_hard_q, dim=1)
    else:
        neg = torch.zeros_like(pos_diag)

    margin = pos_diag - neg
    rank = (sim >= pos_diag.unsqueeze(1)).sum(dim=1).to(torch.int64)
    return pos_diag, neg, margin, rank


