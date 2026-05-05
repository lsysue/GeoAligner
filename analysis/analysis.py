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

def update_topk(prev_scores, prev_indices, new_scores, new_indices, top_k):
    """
    Merge previous top-k results with new chunk scores safely.
    """
    # 修复：确保请求的 k 不会超过当前张量的元素数量
    curr_k = min(top_k, new_scores.shape[1])
    
    if prev_scores is None:
        best_scores, pos = torch.topk(new_scores, k=curr_k, dim=1)
        best_indices = new_indices.gather(1, pos)
        return best_scores, best_indices

    # 拼接历史 Top-K 与当前 Chunk
    merged_scores = torch.cat([prev_scores, new_scores], dim=1)
    merged_indices = torch.cat([prev_indices, new_indices], dim=1)

    # 再次安全提取 Top-K
    final_k = min(top_k, merged_scores.shape[1])
    best_scores, pos = torch.topk(merged_scores, k=final_k, dim=1)
    best_indices = merged_indices.gather(1, pos)

    return best_scores, best_indices

def update_rank(prev_rank, sim_chunk, pos_scores):
    """
    Accumulate rank across chunks. (pos_scores: Query 与 Oracle 真值的相似度)
    """
    # 计算当前 Chunk 中有多少个分数大于等于 Oracle 分数
    count = (sim_chunk >= pos_scores.unsqueeze(1)).sum(dim=1)

    if prev_rank is None:
        return count
    return prev_rank + count

def compute_repr_pos_neg(sim: torch.Tensor, semi_hard_q: float = 0.9):
    """Compute per-sample pos/neg/margin/rank from a square similarity matrix.

    Returns:
        pos_diag: (B,) diagonal positive similarities.
        neg: (B,) row-wise semi-hard negative similarities.
        margin: (B,) pos_diag - neg.
        rank: (B,) rank of diagonal element in each row (1 is best).
    """
    pos = torch.diagonal(sim)
    if sim.shape[0] > 1:
        mask = ~torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
        neg_matrix = sim[mask].view(sim.shape[0], sim.shape[1] - 1)
        neg = torch.quantile(neg_matrix, q=semi_hard_q, dim=1)
    else:
        neg = torch.zeros_like(pos)

    margin = pos - neg
    rank = (sim >= pos.unsqueeze(1)).sum(dim=1).to(torch.int64)
    return pos, neg, margin, rank


