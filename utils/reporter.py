import csv
import logging
import os
import sys

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils.config import save_config


def rank_metric_fieldnames(split_prefix: str, modality_prefix: str, ks, thresholds):
    fields = [f"{split_prefix}_{modality_prefix}_median_error_km"]
    for k in ks:
        for t in thresholds:
            fields.append(f"{split_prefix}_{modality_prefix}_r{k}_{t}km")
    return fields


def add_rank_metrics_to_row(row: dict, metrics: dict, split_prefix: str, modality_prefix: str, ks, thresholds):
    row[f"{split_prefix}_{modality_prefix}_median_error_km"] = metrics.get("median_error_km", float("nan"))
    for k in ks:
        for t in thresholds:
            row[f"{split_prefix}_{modality_prefix}_r{k}_{t}km"] = metrics.get(
                f"r@{k}_{t}km", float("nan")
            )


def write_rank_metrics_to_tensorboard(writer, epoch: int, metrics: dict, split: str, modality: str, eval_ks, thresholds):
    split_upper = split.upper()
    modality_upper = modality.upper()
    for k in eval_ks:
        for t in thresholds:
            writer.add_scalar(
                f"Metric/{split_upper}_{modality_upper}_R{k}_{t}km",
                metrics.get(f"r@{k}_{t}km", float("nan")) * 100.0,
                epoch,
            )
    writer.add_scalar(f"Metric/{split_upper}_{modality_upper}_MedianError_km", metrics.get("median_error_km", float("nan")), epoch)


def format_retrieval_line(metrics: dict, modality: str, split_label: str, eval_ks, thresholds):
    threshold_labels = "/".join(str(t) for t in thresholds)
    topk_parts = []
    for k in eval_ks:
        scores = "/".join(
            f"{metrics.get(f'r@{k}_{t}km', 0.0) * 100:.2f}" for t in thresholds
        )
        topk_parts.append(f"R@{k}({threshold_labels}km)={scores}%")
    return (
        f"{split_label}({modality.upper()}): MedErr@1={metrics.get('median_error_km', float('nan')):.2f}km, "
        + ", ".join(topk_parts)
    )


def dataset_name_from_csv(csv_file: str) -> str:
    csv_lower = str(csv_file).lower()
    if "mp16_pro" in csv_lower or "mp16-pro" in csv_lower:
        return "mp16pro"
    if "yfcc26k" in csv_lower or "yfcc25600" in csv_lower:
        return "yfcc26k"
    if "yfcc4k" in csv_lower:
        return "yfcc4k"
    if "im2gps3k" in csv_lower or "im2gps_3k" in csv_lower:
        return "im2gps3k"
    return "unknown"


def setup_reporter(*, is_master: bool, base_output_dir: str, cfg, overrides, world_size: int):
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers = []

    if is_master:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trainset_name = dataset_name_from_csv(cfg.data.train.csv_file)
        out_dir = os.path.join(base_output_dir, f"{trainset_name}_{timestamp}")
        os.makedirs(out_dir, exist_ok=True)
        # Save the fully resolved runtime config (defaults + yaml + CLI overrides).
        save_config(cfg, os.path.join(out_dir, "config.yaml"))

        log_file = os.path.join(out_dir, "train_ddp.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - Rank 0 - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="w"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logger = logging.getLogger(__name__)
        writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb_logs"))
        logger.info(f"DDP Training Started. World Size: {world_size}")
        logger.info(f"Resolved config saved to: {os.path.join(out_dir, 'config.yaml')}")
        if overrides:
            logger.info(f"Config overrides: {overrides}")
    else:
        out_dir = None
        logging.basicConfig(level=logging.ERROR)
        logger = logging.getLogger(__name__)
        writer = None

    return logger, writer, out_dir


def log_train_step_diagnostics(
    *,
    writer,
    global_step: int,
    s_pos,
    s_neg,
    s_margin,
    g_pos,
    g_neg,
    g_margin,
    sem_scale,
    geo_scale,
):
    writer.add_scalar("Diag/S_Pos_Sim", s_pos, global_step)
    writer.add_scalar("Diag/S_Neg_Sim", s_neg, global_step)
    writer.add_scalar("Diag/S_Margin", s_margin, global_step)
    writer.add_scalar("Diag/G_Pos_Score", g_pos, global_step)
    writer.add_scalar("Diag/G_Neg_Score", g_neg, global_step)
    writer.add_scalar("Diag/G_Margin", g_margin, global_step)
    writer.add_scalar("LogitScale/Semantic", sem_scale, global_step)
    writer.add_scalar("LogitScale/Geo", geo_scale, global_step)


def report_epoch_records(
    *,
    logger,
    writer,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    avg_loss: float,
    avg_s_loss: float,
    avg_g_loss: float,
    avg_val_loss: float,
    avg_val_s_loss: float,
    avg_val_g_loss: float,
    train_diag: dict,
    train_s_metrics: dict,
    train_g_metrics: dict,
    val_s_metrics: dict,
    val_g_metrics: dict,
    use_ema: bool,
    eval_thresholds_km,
    eval_ks,
    metrics_csv_path: str,
    metrics_fieldnames,
    history: dict,
):
    lr_groups = [f"g{i}:{group['lr']:.2e}" for i, group in enumerate(optimizer.param_groups)]
    logger.info(
        f"Epoch {epoch+1} | Train Avg Loss: {avg_loss:.4f} | S: {avg_s_loss:.4f} | G: {avg_g_loss:.4f} "
        f"| Val Avg Loss: {avg_val_loss:.4f} | S: {avg_val_s_loss:.4f} | G: {avg_val_g_loss:.4f} "
        f"| Margins(S/G): {train_diag['s_margin']:.4f}/{train_diag['g_margin']:.4f} "
        f"| Scales(S/G): {train_diag['sem_scale']:.3f}/{train_diag['geo_scale']:.3f} "
        f"| LR[{', '.join(lr_groups)}] "
    )
    logger.info(
        f"Epoch {epoch+1} Retrieval(Train) | "
        f"{format_retrieval_line(train_s_metrics, 's', 'Train', eval_ks, eval_thresholds_km)} | "
        f"{format_retrieval_line(train_g_metrics, 'g', 'Train', eval_ks, eval_thresholds_km)}"
    )
    logger.info(
        f"Epoch {epoch+1} Retrieval(Val) | "
        f"{format_retrieval_line(val_s_metrics, 's', 'Val', eval_ks, eval_thresholds_km)} | "
        f"{format_retrieval_line(val_g_metrics, 'g', 'Val', eval_ks, eval_thresholds_km)}"
    )
    if use_ema:
        logger.info("Validation used EMA weights.")

    history['total_loss'].append(avg_loss)
    history['s_loss'].append(avg_s_loss)
    history['g_loss'].append(avg_g_loss)

    writer.add_scalar('Loss/Train_total_Loss', avg_loss, epoch)
    writer.add_scalar('Loss/Train_S_Loss', avg_s_loss, epoch)
    writer.add_scalar('Loss/Train_G_Loss', avg_g_loss, epoch)

    writer.add_scalar('Loss/Val_total_Loss', avg_val_loss, epoch)
    writer.add_scalar('Loss/Val_S_Loss', avg_val_s_loss, epoch)
    writer.add_scalar('Loss/Val_G_Loss', avg_val_g_loss, epoch)

    write_rank_metrics_to_tensorboard(
        writer,
        epoch,
        train_s_metrics,
        split='Train',
        modality='s',
        eval_ks=eval_ks,
        thresholds=eval_thresholds_km,
    )
    write_rank_metrics_to_tensorboard(
        writer,
        epoch,
        train_g_metrics,
        split='Train',
        modality='g',
        eval_ks=eval_ks,
        thresholds=eval_thresholds_km,
    )
    write_rank_metrics_to_tensorboard(
        writer,
        epoch,
        val_s_metrics,
        split='Val',
        modality='s',
        eval_ks=eval_ks,
        thresholds=eval_thresholds_km,
    )
    write_rank_metrics_to_tensorboard(
        writer,
        epoch,
        val_g_metrics,
        split='Val',
        modality='g',
        eval_ks=eval_ks,
        thresholds=eval_thresholds_km,
    )

    writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)

    with open(metrics_csv_path, 'a', newline='') as f:
        row = {
            'epoch': epoch + 1,
            'train_total_loss': avg_loss,
            'train_s_loss': avg_s_loss,
            'train_g_loss': avg_g_loss,
            'val_total_loss': avg_val_loss,
            'val_s_loss': avg_val_s_loss,
            'val_g_loss': avg_val_g_loss,
        }
        add_rank_metrics_to_row(row, train_s_metrics, split_prefix='train', modality_prefix='s', ks=eval_ks, thresholds=eval_thresholds_km)
        add_rank_metrics_to_row(row, train_g_metrics, split_prefix='train', modality_prefix='g', ks=eval_ks, thresholds=eval_thresholds_km)
        add_rank_metrics_to_row(row, val_s_metrics, split_prefix='val', modality_prefix='s', ks=eval_ks, thresholds=eval_thresholds_km)
        add_rank_metrics_to_row(row, val_g_metrics, split_prefix='val', modality_prefix='g', ks=eval_ks, thresholds=eval_thresholds_km)
        csv.DictWriter(f, fieldnames=metrics_fieldnames).writerow(row)


def finalize_reporter(*, logger, writer, history: dict, out_dir: str):
    logger.info("Generating final plots...")
    figure = None
    try:
        figure = plt.figure(figsize=(10, 6))
        plt.plot(history["total_loss"], label="Total Loss")
        plt.plot(history["s_loss"], label="S Loss", linestyle="--")
        plt.plot(history["g_loss"], label="G Loss", linestyle="--")
        plt.legend()
        plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    except Exception as exc:
        logger.warning(f"Plotting failed (likely headless env): {exc}")
    finally:
        if figure is not None:
            plt.close(figure)
        if writer is not None:
            writer.close()
