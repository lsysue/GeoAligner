import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _style_axes(ax):
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.tick_params(labelsize=10)

def _plot_percentage_hist(ax, values, label, color, bins=60):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    weights = np.ones_like(values) * (100.0 / values.size)

    ax.hist(
        values,
        bins=bins,
        weights=weights,
        label=label,
        color=color,
        alpha=0.6,
        edgecolor='black'
    )

def _plot_rank_percentage(ax, values, label, color):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    ranks = values.astype(np.int64)
    ranks = ranks[ranks >= 1]
    if ranks.size == 0:
        return

    max_rank = int(np.max(ranks))
    counts = np.bincount(ranks, minlength=max_rank + 1)[1:]
    x = np.arange(1, max_rank + 1, dtype=np.int64)
    pct = counts.astype(np.float32) * (100.0 / float(ranks.size))
    ax.plot(x, pct, label=label, color=color, linewidth=2, marker="o", markersize=3)

def save_repr_pos_neg_hist(df_repr, out_dir, query_name, mode_name, color):
    # 筛选当前 mode 的数据
    df_mode = df_repr[df_repr["mode"] == mode_name]

    pos = np.asarray(df_mode["repr_pos"], dtype=np.float32)
    neg = np.asarray(df_mode["repr_neg"], dtype=np.float32)

    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    
    fig, ax = plt.subplots(figsize=(8, 6))

    all_vals = np.concatenate([pos, neg])
    bins = np.linspace(all_vals.min(), all_vals.max(), 60)
    
    # 绘制正样本和负样本的分布
    _plot_percentage_hist(ax, pos, f"Positive ({mode_name.upper()})", color, bins=bins)
    _plot_percentage_hist(ax, neg, f"Semi-hard negative ({mode_name.upper()})", "#7f8c8d", bins=bins) # 负样本用灰色对比
    
    ax.set_title(f"Representation similarity: {mode_name.upper()}", fontsize=13)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Query percentage (%)")
    ax.legend(frameon=False)
    _style_axes(ax)
    
    out_path = os.path.join(out_dir, f"{query_name}_{mode_name}_repr_similarity.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def save_repr_margin_hist(df_repr, out_dir, query_name, mode_name, color):
    # 筛选当前 mode 的数据
    df_mode = df_repr[df_repr["mode"] == mode_name]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 使用百分比分布图
    _plot_percentage_hist(ax, df_mode["repr_margin"], f"{mode_name.upper()} Margin", color)
    
    # 画出 0.0 阈值线：左边是分类错误的（负样本比正样本得分高），右边是正确的
    ax.axvline(0.0, color="#c0392b", linewidth=1.5, linestyle="--", alpha=0.8)
    
    ax.set_title(f"Representation margin: {mode_name.upper()}", fontsize=13)
    ax.set_xlabel("Margin (Pos - Neg)")
    ax.set_ylabel("Query percentage (%)")
    _style_axes(ax)
    
    out_path = os.path.join(out_dir, f"{query_name}_{mode_name}_repr_margin.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def save_repr_rank_distribution(df_repr, out_dir, query_name, mode_name, color):
    # 筛选对应 mode
    df_mode = df_repr[df_repr["mode"] == mode_name]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # 这里的 repr_rank 是 run_retrieval_eval 中存入的 Batch 内排名
    _plot_rank_percentage(ax, df_mode["repr_rank"], f"Pos rank in batch ({mode_name.upper()})", color)

    ax.set_title(f"Representation Rank Distribution: {mode_name.upper()}", fontsize=13)
    ax.set_xlabel("Rank (1 is best)")
    ax.set_ylabel("Query percentage (%)")
    ax.set_xscale("log")
    ax.set_xlim(1, df_mode["repr_rank"].max())
    ax.legend(frameon=False)
    _style_axes(ax)

    out_path = os.path.join(out_dir, f"{query_name}_{mode_name}_repr_rank_dist.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def save_repr_sg_scatter(df_repr, out_dir, max_points, query_name):
    # 因为 df_repr 包含所有 mode，我们只画一次，所以需要透视一下或者只取特定行
    # 假设我们想看同一个 query 在不同 mode 下的 pos_sim
    df_s = df_repr[df_repr["mode"] == "s"][["query_img_id", "repr_pos"]].rename(columns={"repr_pos": "s_pos"})
    df_g = df_repr[df_repr["mode"] == "g"][["query_img_id", "repr_pos"]].rename(columns={"repr_pos": "g_pos"})
    merged = pd.merge(df_s, df_g, on="query_img_id")
    
    sampled = merged.sample(n=min(len(merged), max_points), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sampled["s_pos"], sampled["g_pos"], s=10, alpha=0.35, color="#e7298a")
    
    # 画一条 y=x 线参考
    lims = [0, 1]
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
    
    ax.set_title(f"S vs G positive similarity correlation", fontsize=13)
    ax.set_xlabel("S-space positive score")
    ax.set_ylabel("G-space positive score")
    _style_axes(ax)
    out_path = os.path.join(out_dir, f"{query_name}_s_vs_g_correlation.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def save_retrv_distance_hist(df_retrv, out_dir, query_name, mode_name, color):
    # 只筛选当前 mode 的数据
    df_mode = df_retrv[df_retrv["mode"] == mode_name]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # Oracle 是地表最强参考（数据库里最近的点）
    _plot_percentage_hist(ax, df_mode["oracle_top1_dist"], "Nearest in Gallery (Oracle)", "#2b8cbe")
    # Pred 是模型认为最近的点
    _plot_percentage_hist(ax, df_mode["pred_top1_dist"], f"Predicted Top-1 ({mode_name.upper()})", color)

    ax.set_xscale("log")
    
    ax.set_title(f"Distance distribution ({mode_name.upper()})", fontsize=13)
    ax.set_xlabel("Error distance (km)")
    ax.set_ylabel("Query percentage (%)")
    ax.legend(frameon=False)
    _style_axes(ax)
    out_path = os.path.join(out_dir, f"{query_name}_{mode_name}_distance_compare.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def save_retrv_rank_hist(df_retrv, out_dir, query_name, mode_name, color):
    df_mode = df_retrv[df_retrv["mode"] == mode_name]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # 使用你定义的 _plot_rank_percentage
    _plot_rank_percentage(ax, df_mode["oracle_rank"], f"Oracle rank in {mode_name.upper()}", color)
    
    ax.set_title(f"Oracle nearest-gallery rank ({mode_name.upper()})", fontsize=13)
    ax.set_xlabel("Rank (1 is best)")
    ax.set_ylabel("Query percentage (%)")
    ax.set_xscale("log")
    ax.set_xlim(1, df_mode["oracle_rank"].max())
    _style_axes(ax)
    out_path = os.path.join(out_dir, f"{query_name}_{mode_name}_oracle_rank.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def save_retrv_rank_curve(df_retrv, out_dir, query_name, mode_name, color):
    # 1. 提取当前 mode 的数据
    df_mode = df_retrv[df_retrv["mode"] == mode_name]
    
    # 2. 从列名中识别所有 pred_rank{k}_dist_km
    dist_cols = [c for c in df_mode.columns if "pred_rank" in c and "_dist_km" in c]
    dist_cols.sort(key=lambda x: int(x.split('_')[2])) # 确保按 1, 2, 3... 排序
    
    if not dist_cols:
        print(f"Warning: No distance columns found for {mode_name}")
        return None

    topk_dists = df_mode[dist_cols].to_numpy()
    rank_grid = np.arange(1, len(dist_cols) + 1)
    
    medians = np.nanmedian(topk_dists, axis=0)
    q25 = np.nanquantile(topk_dists, 0.25, axis=0)
    q75 = np.nanquantile(topk_dists, 0.75, axis=0)
    valid_mask = np.isfinite(medians)

    fig, ax = plt.subplots(figsize=(8, 6))
    if np.any(valid_mask):
        ax.plot(rank_grid[valid_mask], medians[valid_mask], color=color, linewidth=2, label="Median distance", marker='o', markersize=4)
        ax.fill_between(rank_grid[valid_mask], q25[valid_mask], q75[valid_mask], color=color, alpha=0.2, label="IQR (25th-75th)")
    else:
        ax.text(0.5, 0.5, "No valid rank-distance data", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(f"Rank-Distance Calibration ({mode_name.upper()})", fontsize=13)
    ax.set_xlabel("Similarity Rank")
    ax.set_ylabel("GT-to-Ranked-GPS distance (km)")
    ax.legend(frameon=False)
    _style_axes(ax)

    out_path = os.path.join(out_dir, f"{query_name}_{mode_name}_rank_dist_curve.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_oracle_rank_scatter(df_retrv, out_dir, max_points, query_name, mode_name, color):
    # 1. 筛选对应 mode
    df_mode = df_retrv[df_retrv["mode"] == mode_name]
    
    # 2. 采样，防止点太多
    sample_size = min(len(df_mode), max_points)
    sample_df = df_mode.sample(n=sample_size, random_state=42)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        sample_df["oracle_rank"],
        sample_df["oracle_top1_dist"],
        s=12,
        alpha=0.4,
        color=color,
        edgecolors="none",
    )
    
    ax.set_title(f"Oracle Rank vs Distance ({mode_name.upper()})", fontsize=13)
    ax.set_xlabel("Similarity Rank of nearest gallery point")
    ax.set_ylabel("True Distance to nearest point (km)")
    ax.set_yscale("log") # 距离通常跨度很大，对数轴更直观
    
    # 细节微调：如果 rank 集中在前面，可以限制一下 x 轴
    # ax.set_xlim(0, 100) 
    
    _style_axes(ax)
    out_path = os.path.join(out_dir, f"{query_name}_{mode_name}_oracle_scatter.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def save_repr_vs_retrv_scatter(
    df_retrv,
    df_repr,
    out_dir,
    query_name,
    mode_name,
    max_points=5000,
    color="#e74c3c",
):
    df_r = df_retrv[df_retrv["mode"] == mode_name]
    df_p = df_repr[df_repr["mode"] == mode_name]

    merged = pd.merge(
        df_r,
        df_p,
        on=["query_img_id", "mode"],
        how="inner"
    )

    if len(merged) == 0:
        print(f"[WARN] No data for mode={mode_name}")
        return None
    sample_df = merged.sample(
        n=min(len(merged), max_points),
        random_state=42
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    x = sample_df["repr_margin"]
    y = np.maximum(sample_df["pred_top1_dist"], 1e-3)

    ax.scatter(x, y, s=12, alpha=0.35, color=color, edgecolors="none")

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)  # margin=0

    ax.set_yscale("log")  # 非常重要（距离是长尾）
    ax.set_title(f"Repr vs Retrieval ({mode_name.upper()})", fontsize=13)
    ax.set_xlabel("Representation margin (pos - neg)")
    ax.set_ylabel("Top-1 geodesic error (km, log scale)")

    _style_axes(ax)

    out_path = os.path.join(
        out_dir,
        f"{query_name}_{mode_name}_repr_vs_retrv.png"
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path