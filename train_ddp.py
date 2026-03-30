import os
import sys
import time
import yaml
import math
import csv
import argparse
import logging
import shutil
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# --- 引入项目模块 ---
from utils.config import load_config
from datasets.img2geo_dataset import Img2GeoDataset
from encoders.image_encoder import ImageEncoder, ImageEncoderConfig
from encoders.location_encoder import GPSEncoder, GPSEncoderConfig
from aligners.alignmenthub import AlignmentHub

torch.multiprocessing.set_sharing_strategy('file_system')

def setup_ddp():
    """初始化 DDP 环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Fallback for debugging
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    return rank, local_rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

def save_checkpoint(state, filename):
    torch.save(state, filename)

class ModelEMA:
    def __init__(self, module: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        with torch.no_grad():
            for name, tensor in module.state_dict().items():
                self.shadow[name] = tensor.detach().clone()

    @torch.no_grad()
    def update(self, module: torch.nn.Module):
        state = module.state_dict()
        for name, tensor in state.items():
            src = tensor.detach()
            if name not in self.shadow:
                self.shadow[name] = src.clone()
                continue

            # EMA for floating tensors; direct copy for integer/bool buffers (e.g. queue pointers).
            if src.is_floating_point() or src.is_complex():
                self.shadow[name].mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(src)

    @torch.no_grad()
    def store(self, module: torch.nn.Module):
        self.backup = {}
        for name, tensor in module.state_dict().items():
            self.backup[name] = tensor.detach().clone()

    @torch.no_grad()
    def copy_to(self, module: torch.nn.Module):
        module.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, module: torch.nn.Module):
        if self.backup:
            module.load_state_dict(self.backup, strict=True)
            self.backup = {}

    def state_dict(self):
        return {
            'decay': self.decay,
            'shadow': self.shadow,
        }

    def load_state_dict(self, state):
        self.decay = float(state.get('decay', self.decay))
        shadow = state.get('shadow', None)
        if shadow is not None:
            self.shadow = {k: v.detach().clone() for k, v in shadow.items()}

def _module_grad_norm(module: torch.nn.Module) -> float:
    grads = [p.grad.detach() for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([torch.norm(g) for g in grads])).item()

@torch.no_grad()
def _pairwise_pos_neg_stats(a: torch.Tensor, b: torch.Tensor):
    a = torch.nn.functional.normalize(a, p=2, dim=-1)
    b = torch.nn.functional.normalize(b, p=2, dim=-1)
    sim = a @ b.T
    pos = torch.diagonal(sim).mean()
    if sim.shape[0] > 1:
        mask = ~torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
        neg = sim[mask].mean()
    else:
        neg = torch.tensor(0.0, device=sim.device)
    return pos.item(), neg.item(), (pos - neg).item()

@torch.no_grad()
def _geo_pos_neg_stats(image_g_tokens: torch.Tensor, gps_g_tokens: torch.Tensor):
    image_g_tokens = torch.nn.functional.normalize(image_g_tokens, p=2, dim=-1)
    gps_g_tokens = torch.nn.functional.normalize(gps_g_tokens, p=2, dim=-1)

    pos_sim = torch.einsum('bnd,bmd->bnm', image_g_tokens, gps_g_tokens)
    pos_score = pos_sim.max(dim=2)[0].mean(dim=1).mean()

    if image_g_tokens.shape[0] > 1:
        neg_gps = torch.roll(gps_g_tokens, shifts=1, dims=0)
        neg_sim = torch.einsum('bnd,bmd->bnm', image_g_tokens, neg_gps)
        neg_score = neg_sim.max(dim=2)[0].mean(dim=1).mean()
    else:
        neg_score = torch.tensor(0.0, device=image_g_tokens.device)

    return pos_score.item(), neg_score.item(), (pos_score - neg_score).item()

@torch.no_grad()
def _haversine_km(query_coords: torch.Tensor, cand_coords: torch.Tensor) -> torch.Tensor:
    # query_coords/cand_coords: (..., 2) with [lat, lon] in degrees
    lat1 = torch.deg2rad(query_coords[..., 0])
    lon1 = torch.deg2rad(query_coords[..., 1])
    lat2 = torch.deg2rad(cand_coords[..., 0])
    lon2 = torch.deg2rad(cand_coords[..., 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a.clamp_min(1e-12)), torch.sqrt((1 - a).clamp_min(1e-12)))
    return 6371.0 * c

def _init_recall_tracker(ks, thresholds):
    return {
        "n": 0,
        "hits": {k: {t: 0 for t in thresholds} for k in ks},
        "top1_dists": [],
    }

@torch.no_grad()
def _update_recall_tracker(tracker, image_s_vec, gps_s_vec, gps_coords, ks, thresholds):
    if image_s_vec.numel() == 0:
        return

    image_s_vec = torch.nn.functional.normalize(image_s_vec, p=2, dim=-1)
    gps_s_vec = torch.nn.functional.normalize(gps_s_vec, p=2, dim=-1)
    sim = image_s_vec @ gps_s_vec.T

    batch_size = sim.shape[0]
    tracker["n"] += int(batch_size)
    gps_coords = gps_coords.detach()

    for k in ks:
        effective_k = min(k, sim.shape[1])
        topk_idx = torch.topk(sim, k=effective_k, dim=1, largest=True).indices
        cand_coords = gps_coords[topk_idx]
        query_coords = gps_coords.unsqueeze(1).expand(-1, effective_k, -1)
        dists_km = _haversine_km(query_coords, cand_coords)
        min_dists = dists_km.min(dim=1).values

        for t in thresholds:
            tracker["hits"][k][t] += int((min_dists <= t).sum().item())

        if k == 1:
            tracker["top1_dists"].append(min_dists.detach().cpu())

@torch.no_grad()
def _finalize_recall_tracker(tracker, ks, thresholds, device):
    n_tensor = torch.tensor([tracker["n"]], device=device, dtype=torch.long)
    if dist.is_initialized():
        dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
    total_n = max(int(n_tensor.item()), 1)

    out = {}
    for k in ks:
        for t in thresholds:
            hit_tensor = torch.tensor([tracker["hits"][k][t]], device=device, dtype=torch.long)
            if dist.is_initialized():
                dist.all_reduce(hit_tensor, op=dist.ReduceOp.SUM)
            out[f"r@{k}_{t}km"] = hit_tensor.item() / total_n

    if tracker["top1_dists"]:
        local_top1 = torch.cat(tracker["top1_dists"], dim=0).to(device=device)
    else:
        local_top1 = torch.empty(0, device=device)

    if dist.is_initialized():
        world_size = dist.get_world_size()
        local_len = torch.tensor([local_top1.numel()], device=device, dtype=torch.long)
        len_list = [torch.zeros_like(local_len) for _ in range(world_size)]
        dist.all_gather(len_list, local_len)
        lens = [int(x.item()) for x in len_list]
        max_len = max(lens) if lens else 0

        if max_len > 0:
            padded = torch.zeros(max_len, device=device, dtype=local_top1.dtype)
            if local_top1.numel() > 0:
                padded[:local_top1.numel()] = local_top1
            gathered = [torch.zeros_like(padded) for _ in range(world_size)]
            dist.all_gather(gathered, padded)
            merged = []
            for tensor_i, len_i in zip(gathered, lens):
                if len_i > 0:
                    merged.append(tensor_i[:len_i])
            top1_all = torch.cat(merged, dim=0) if merged else torch.empty(0, device=device)
        else:
            top1_all = torch.empty(0, device=device)
    else:
        top1_all = local_top1

    out["median_error_km"] = torch.median(top1_all).item() if top1_all.numel() > 0 else float("nan")
    return out

def train_per_epoch(image_encoder, gps_encoder, alignment_hub, dataloader, optimizer, scheduler, scaler, device, writer, global_step, epoch, is_master, world_size, cfg, ema_dict=None):
    sampler = dataloader.sampler
    sampler.set_epoch(epoch)
    
    image_encoder.train()
    gps_encoder.train()
    alignment_hub.train()
    
    total_loss_epoch = torch.zeros(1, device=device)
    s_loss_epoch = torch.zeros(1, device=device)
    g_loss_epoch = torch.zeros(1, device=device)

    diag_sums = {
        "s_margin": 0.0,
        "g_margin": 0.0,
        "sem_scale": 0.0,
        "geo_scale": 0.0,
        "img_grad": 0.0,
        "gps_grad": 0.0,
        "align_grad": 0.0,
    }
    diag_count = 0

    eval_ks = [1, 5, 10]
    eval_thresholds = [1, 25, 200, 750, 2500]
    train_recall_tracker = _init_recall_tracker(eval_ks, eval_thresholds)

    align_module = alignment_hub.module if hasattr(alignment_hub, "module") else alignment_hub
    
    # 只有 Rank 0 打印进度条，避免刷屏
    if is_master:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}")
    else:
        pbar = dataloader

    for batch_idx, (images, gps_coords, s2_tokens) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        gps_coords = gps_coords.to(device, non_blocking=True)
        s2_tokens = s2_tokens.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            img_embeds = image_encoder(images)
            gps_embeds = gps_encoder(gps_coords, s2_tokens)
            loss_dict = alignment_hub(img_embeds, gps_embeds)
            loss = loss_dict["loss"]

        # 记录嵌入分布（每100步记录一次）
        if is_master and batch_idx % 100 == 0:
            if isinstance(img_embeds, dict):
                if "s_vector" in img_embeds:
                    writer.add_histogram('Embeddings/Image/s_vector', img_embeds["s_vector"].detach(), global_step)
                if "g_tokens" in img_embeds:
                    writer.add_histogram('Embeddings/Image/g_tokens', img_embeds["g_tokens"].detach(), global_step)
            else:
                writer.add_histogram('Embeddings/Image', img_embeds.detach(), global_step)

            if isinstance(gps_embeds, dict):
                if "s_vector" in gps_embeds:
                    writer.add_histogram('Embeddings/GPS/s_vector', gps_embeds["s_vector"].detach(), global_step)
                if "g_tokens" in gps_embeds:
                    writer.add_histogram('Embeddings/GPS/g_tokens', gps_embeds["g_tokens"].detach(), global_step)
            else:
                writer.add_histogram('Embeddings/GPS', gps_embeds.detach(), global_step)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        image_grad_norm = _module_grad_norm(image_encoder)
        gps_grad_norm = _module_grad_norm(gps_encoder)
        align_grad_norm = _module_grad_norm(alignment_hub)

        params = list(image_encoder.parameters()) + \
         list(gps_encoder.parameters()) + \
         list(alignment_hub.parameters())
        grads = [p.grad.detach() for p in params if p.grad is not None]
        if grads:
            grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads]))
        else:
            grad_norm = torch.tensor(0.0, device=device)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if ema_dict is not None:
            ema_dict['image_encoder'].update(image_encoder.module)
            ema_dict['gps_encoder'].update(gps_encoder.module)
            ema_dict['alignment_hub'].update(alignment_hub.module)

        if is_master:
            writer.add_scalar('Gradient_Norm', grad_norm.item(), global_step)
            writer.add_scalar('Gradient_Norm/ImageEncoder', image_grad_norm, global_step)
            writer.add_scalar('Gradient_Norm/GPSEncoder', gps_grad_norm, global_step)
            writer.add_scalar('Gradient_Norm/AlignmentHub', align_grad_norm, global_step)

        # Detach to save memory
        total_loss_epoch += loss.detach()
        s_loss_epoch += loss_dict["semantic_loss"].detach()
        g_loss_epoch += loss_dict["geo_loss"].detach()

        if isinstance(img_embeds, dict) and isinstance(gps_embeds, dict):
            if "s_vector" in img_embeds and "s_vector" in gps_embeds:
                _update_recall_tracker(
                    train_recall_tracker,
                    img_embeds["s_vector"].detach(),
                    gps_embeds["s_vector"].detach(),
                    gps_coords,
                    eval_ks,
                    eval_thresholds,
                )

        if is_master:
            current_loss = loss.item()
            if batch_idx % cfg.train.log_interval == 0:
                s_margin = None
                g_margin = None
                if isinstance(img_embeds, dict) and isinstance(gps_embeds, dict):
                    if "s_vector" in img_embeds and "s_vector" in gps_embeds:
                        s_pos, s_neg, s_margin = _pairwise_pos_neg_stats(
                            img_embeds["s_vector"].detach(),
                            gps_embeds["s_vector"].detach(),
                        )
                        writer.add_scalar('Diag/S_Pos_Sim', s_pos, global_step)
                        writer.add_scalar('Diag/S_Neg_Sim', s_neg, global_step)
                        writer.add_scalar('Diag/S_Margin', s_margin, global_step)

                    if "g_tokens" in img_embeds and "g_tokens" in gps_embeds:
                        g_pos, g_neg, g_margin = _geo_pos_neg_stats(
                            img_embeds["g_tokens"].detach(),
                            gps_embeds["g_tokens"].detach(),
                        )
                        writer.add_scalar('Diag/G_Pos_Score', g_pos, global_step)
                        writer.add_scalar('Diag/G_Neg_Score', g_neg, global_step)
                        writer.add_scalar('Diag/G_Margin', g_margin, global_step)

                sem_scale = align_module.semantic_aligner.logit_scale.exp().clamp(max=100.0).item()
                geo_scale = align_module.geo_aligner.logit_scale.exp().clamp(max=100.0).item()
                writer.add_scalar('LogitScale/Semantic', sem_scale, global_step)
                writer.add_scalar('LogitScale/Geo', geo_scale, global_step)

                queue_ptr = align_module.semantic_aligner.queue_ptr.item()

                if s_margin is not None:
                    diag_sums["s_margin"] += s_margin
                if g_margin is not None:
                    diag_sums["g_margin"] += g_margin
                diag_sums["sem_scale"] += sem_scale
                diag_sums["geo_scale"] += geo_scale
                diag_sums["img_grad"] += image_grad_norm
                diag_sums["gps_grad"] += gps_grad_norm
                diag_sums["align_grad"] += align_grad_norm
                diag_count += 1

            pbar.set_postfix({"Loss": f"{current_loss:.4f}"})

        global_step += 1

    # Reduce losses from all devices
    dist.all_reduce(total_loss_epoch, op=dist.ReduceOp.SUM)
    dist.all_reduce(s_loss_epoch, op=dist.ReduceOp.SUM)
    dist.all_reduce(g_loss_epoch, op=dist.ReduceOp.SUM)
    
    avg_loss = total_loss_epoch.item() / (len(dataloader) * world_size)
    avg_s_loss = s_loss_epoch.item() / (len(dataloader) * world_size)
    avg_g_loss = g_loss_epoch.item() / (len(dataloader) * world_size)

    if diag_count > 0:
        diag_avg = {k: v / diag_count for k, v in diag_sums.items()}
    else:
        diag_avg = {k: float('nan') for k in diag_sums.keys()}

    train_metric_results = _finalize_recall_tracker(train_recall_tracker, eval_ks, eval_thresholds, device)

    return avg_loss, avg_s_loss, avg_g_loss, global_step, diag_avg, train_metric_results

def _apply_ema_for_eval(ema_dict, modules_dict):
    if ema_dict is None:
        return
    for key, ema in ema_dict.items():
        ema.store(modules_dict[key])
        ema.copy_to(modules_dict[key])

def _restore_from_ema(ema_dict, modules_dict):
    if ema_dict is None:
        return
    for key, ema in ema_dict.items():
        ema.restore(modules_dict[key])

def val_per_epoch(image_encoder, gps_encoder, alignment_hub, val_dataloader, device, writer, epoch, is_master, world_size):
    if val_dataloader is None:
        return float('nan'), float('nan'), float('nan'), {}
    
    image_encoder.eval()
    gps_encoder.eval()
    alignment_hub.eval()
    
    val_total_loss = torch.zeros(1, device=device)
    val_s_loss = torch.zeros(1, device=device)
    val_g_loss = torch.zeros(1, device=device)

    eval_ks = [1, 5, 10]
    eval_thresholds = [1, 25, 200, 750, 2500]
    val_recall_tracker = _init_recall_tracker(eval_ks, eval_thresholds)
    
    with torch.no_grad():
        for val_images, val_gps_coords, val_s2_tokens in val_dataloader:
            val_images = val_images.to(device, non_blocking=True)
            val_gps_coords = val_gps_coords.to(device, non_blocking=True)
            val_s2_tokens = val_s2_tokens.to(device, non_blocking=True)
            val_img_embeds = image_encoder(val_images)
            val_gps_embeds = gps_encoder(val_gps_coords, val_s2_tokens)
            val_loss_dict = alignment_hub(val_img_embeds, val_gps_embeds)
            val_total_loss += val_loss_dict["loss"].detach()
            val_s_loss += val_loss_dict["semantic_loss"].detach()
            val_g_loss += val_loss_dict["geo_loss"].detach()

            if isinstance(val_img_embeds, dict) and isinstance(val_gps_embeds, dict):
                if "s_vector" in val_img_embeds and "s_vector" in val_gps_embeds:
                    _update_recall_tracker(
                        val_recall_tracker,
                        val_img_embeds["s_vector"].detach(),
                        val_gps_embeds["s_vector"].detach(),
                        val_gps_coords,
                        eval_ks,
                        eval_thresholds,
                    )
    
    dist.all_reduce(val_total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_s_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_g_loss, op=dist.ReduceOp.SUM)
    
    avg_val_loss = val_total_loss.item() / (len(val_dataloader) * world_size)
    avg_val_s_loss = val_s_loss.item() / (len(val_dataloader) * world_size)
    avg_val_g_loss = val_g_loss.item() / (len(val_dataloader) * world_size)

    val_metric_results = _finalize_recall_tracker(val_recall_tracker, eval_ks, eval_thresholds, device)

    return avg_val_loss, avg_val_s_loss, avg_val_g_loss, val_metric_results

def main(args):
    # 1. DDP 初始化
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_master = (rank == 0)
    writer = None
    # 2. 加载配置
    cfg = load_config(args.config)
    
    # --- [FIX 1] 强力日志控制 ---
    # 只有 Rank 0 输出 INFO，其他 Rank 只输出 ERROR
    log_level = logging.INFO if is_master else logging.ERROR
    
    # 清除之前的 handlers (防止 tqdm 或其他库干扰)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers = []
        
    if is_master:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_out_dir = cfg.output_dir
        out_dir = os.path.join(base_out_dir, f"mp16_pro_{timestamp}")
        os.makedirs(out_dir, exist_ok=True)

        shutil.copy(args.config, os.path.join(out_dir, 'config.yaml'))  # 保存配置文件副本
        log_file = os.path.join(out_dir, 'train_ddp.log')
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - Rank 0 - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"DDP Training Started. World Size: {world_size}")
        
        # TensorBoard
        tb_dir = os.path.join(out_dir, "tb_logs")
        writer = SummaryWriter(log_dir=tb_dir)
    else:
        # 非主进程仅仅配置一个 NullHandler 或者简单的 Error Handler
        logging.basicConfig(level=logging.ERROR)
        logger = logging.getLogger(__name__)

    # 3. 数据准备
    if is_master: logger.info("Initializing Datasets...")
    
    train_transform = transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validation should be deterministic for stable and comparable metrics.
    val_transform = transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = Img2GeoDataset(
        csv_file=cfg.data.train.csv_file,
        img_dir=cfg.data.train.img_dir,
        transform=train_transform,
        s2_levels=cfg.model.gps.s2_levels
    )

    if 'debug' in args.config or args.debug:
        debug_size = cfg.data.n_data
        if is_master: logger.info(f"[DEBUG] Truncating dataset to {debug_size}")
        indices = list(range(min(len(dataset), debug_size)))
        dataset = Subset(dataset, indices)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.data.num_workers > 0 else False
    )

    # 验证数据集
    if hasattr(cfg, 'data') and hasattr(cfg.data, 'val') and cfg.data.val is not None:
        val_dataset = Img2GeoDataset(
            csv_file=cfg.data.val.csv_file,
            img_dir=cfg.data.val.img_dir,
            transform=val_transform,
            s2_levels=cfg.model.gps.s2_levels
        )
        if args.debug:
            val_indices = list(range(min(len(val_dataset), debug_size)))
            val_dataset = Subset(val_dataset, val_indices)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            sampler=val_sampler,
            drop_last=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True if cfg.data.num_workers > 0 else False
        )
    else:
        val_dataloader = None

    # 4. 模型初始化
    if is_master: logger.info("Initializing Models...")

    # Encoder Configs
    img_cfg = ImageEncoderConfig(
        vit_name=cfg.model.image.vit_name,
        img_size=cfg.data.img_size,
        s_dim=cfg.model.image.s_dim,
        g_dim=cfg.model.image.g_dim,
        n_g_tokens=cfg.model.image.n_g_tokens,
        use_landmark=False
    )
    
    gps_cfg = GPSEncoderConfig(
        s2_levels=cfg.model.gps.s2_levels,
        s2_embed_dim=cfg.model.gps.s2_embed_dim,
        s2_num_buckets=getattr(cfg.model.gps, 's2_num_buckets', 2**17),
        s2_embed_dropout=getattr(cfg.model.gps, 's2_embed_dropout', 0.1),
        s2_feature_dropout=getattr(cfg.model.gps, 's2_feature_dropout', 0.1),
        transformer_nhead=getattr(cfg.model.gps, 'transformer_nhead', 4),
        transformer_nlayers=getattr(cfg.model.gps, 'transformer_nlayers', 2),
        transformer_dropout=getattr(cfg.model.gps, 'transformer_dropout', 0.1),
        fourier_n_freqs=cfg.model.gps.fourier_n_freqs,
        continuous_geo_mode=getattr(cfg.model.gps, 'continuous_geo_mode', 'unit_sphere'),
        n_g_tokens=cfg.model.gps.n_g_tokens,
        base_scale_multiplier=cfg.model.gps.base_scale_multiplier,
        min_scale_deg=getattr(cfg.model.gps, 'min_scale_deg', 1e-4),
        max_scale_deg=getattr(cfg.model.gps, 'max_scale_deg', 1.0),
        lon_scale_cos_epsilon=getattr(cfg.model.gps, 'lon_scale_cos_epsilon', 0.15),
        sampling_mode_train=getattr(cfg.model.gps, 'sampling_mode_train', 'random'),
        sampling_mode_eval=getattr(cfg.model.gps, 'sampling_mode_eval', 'same_as_train'),
        sampling_seed=getattr(cfg.model.gps, 'sampling_seed', 42),
        s_dim=cfg.model.gps.s_dim,
        g_dim=cfg.model.gps.g_dim
    )

    # Initialize Modules on GPU
    image_encoder = ImageEncoder(img_cfg).to(device)
    gps_encoder = GPSEncoder(gps_cfg).to(device)
    alignment_hub = AlignmentHub(
        s_dim=cfg.model.image.s_dim,
        g_dim=cfg.model.image.g_dim,
        loss_weight_s=cfg.model.alignment.loss_weight_s,
        loss_weight_g=cfg.model.alignment.loss_weight_g,
        temperature=getattr(cfg.model.alignment, 'temperature', 0.07),
        semantic_queue_size=getattr(cfg.model.alignment, 'semantic_queue_size', 4096),
    ).to(device)

    image_encoder = DDP(image_encoder, device_ids=[local_rank], find_unused_parameters=True)
    gps_encoder = DDP(gps_encoder, device_ids=[local_rank], find_unused_parameters=True)
    alignment_hub = DDP(alignment_hub, device_ids=[local_rank], find_unused_parameters=True)

    # 5. 优化器
    base_lr = float(cfg.train.learning_rate)
    image_lr_mult = float(getattr(cfg.train, 'lr_mult_image', 0.1))
    gps_lr_mult = float(getattr(cfg.train, 'lr_mult_gps', 0.5))
    align_lr_mult = float(getattr(cfg.train, 'lr_mult_align', 0.5))
    logit_scale_lr_mult = float(getattr(cfg.train, 'lr_mult_logit_scale', 0.01))

    align_named_params = list(alignment_hub.module.named_parameters())
    logit_scale_params = [p for n, p in align_named_params if n.endswith('logit_scale')]
    align_core_params = [p for n, p in align_named_params if not n.endswith('logit_scale')]

    optimizer_grouped_parameters = [
        {"params": image_encoder.module.parameters(), "lr": base_lr * image_lr_mult},
        {"params": gps_encoder.module.parameters(), "lr": base_lr * gps_lr_mult},
        {"params": align_core_params, "lr": base_lr * align_lr_mult},
    ]
    if logit_scale_params:
        optimizer_grouped_parameters.append({
            "params": logit_scale_params,
            "lr": base_lr * logit_scale_lr_mult,
        })
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        weight_decay=float(cfg.train.weight_decay)
    )
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup_epochs = int(getattr(cfg.train, 'warmup_epochs', 0))
    warmup_steps = warmup_epochs * steps_per_epoch
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()

    # 5.1 EMA
    ema_decay = float(getattr(cfg.train, 'ema_decay', 0.999))
    use_ema = bool(getattr(cfg.train, 'use_ema', True))
    ema_dict = None
    if use_ema:
        ema_dict = {
            'image_encoder': ModelEMA(image_encoder.module, decay=ema_decay),
            'gps_encoder': ModelEMA(gps_encoder.module, decay=ema_decay),
            'alignment_hub': ModelEMA(alignment_hub.module, decay=ema_decay),
        }
        if is_master:
            logger.info(f"EMA enabled. decay={ema_decay}")

    # 6. Resume
    start_epoch = 0
    if cfg.train.resume_path:
        if is_master: logger.info(f"Loading checkpoint {cfg.train.resume_path}")
        # Map location is crucial for DDP resume
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(cfg.train.resume_path, map_location=map_location)

        image_encoder.module.load_state_dict(checkpoint['image_encoder'])
        gps_encoder.module.load_state_dict(checkpoint['gps_encoder'])
        alignment_hub.module.load_state_dict(checkpoint['alignment_hub'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            if is_master: logger.warning(f"Scheduler state missing! Fast-forwarding scheduler to step {start_epoch * steps_per_epoch}...")
            # 手动执行 step，让学习率衰减到 resume 对应的 step 数值
            completed_steps = start_epoch * steps_per_epoch
            for _ in range(completed_steps):
                scheduler.step()

        # 尝试加载 scaler，如果不存在则忽略（它会自适应）
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        else:
            if is_master: logger.warning("Scaler state missing. It will self-calibrate.")

        if use_ema and ('ema' in checkpoint):
            ckpt_ema = checkpoint['ema']
            if 'image_encoder' in ckpt_ema:
                ema_dict['image_encoder'].load_state_dict(ckpt_ema['image_encoder'])
            if 'gps_encoder' in ckpt_ema:
                ema_dict['gps_encoder'].load_state_dict(ckpt_ema['gps_encoder'])
            if 'alignment_hub' in ckpt_ema:
                ema_dict['alignment_hub'].load_state_dict(ckpt_ema['alignment_hub'])
            if is_master:
                logger.info("EMA state restored from checkpoint.")
        elif use_ema and is_master:
            logger.warning("EMA state missing in checkpoint. Re-initialized EMA from current weights.")
        
        if is_master: logger.info(f"Resumed from epoch {start_epoch}")

    history = {'total_loss': [], 's_loss': [], 'g_loss': []}
    global_step = start_epoch * steps_per_epoch

    # Selection and stopping controls
    best_val_r1_1km = float('-inf')
    best_epoch = -1
    no_improve_epochs = 0
    early_stopping_patience = 10

    metrics_csv_path = None
    metrics_fieldnames = [
        'epoch',
        'train_total_loss',
        'train_s_loss',
        'train_g_loss',
        'train_median_error_km',
        'train_r1_1km',
        'train_r1_25km',
        'train_r1_200km',
        'train_r1_750km',
        'train_r1_2500km',
        'train_r5_1km',
        'train_r5_25km',
        'train_r5_200km',
        'train_r5_750km',
        'train_r5_2500km',
        'train_r10_1km',
        'train_r10_25km',
        'train_r10_200km',
        'train_r10_750km',
        'train_r10_2500km',
        'val_total_loss',
        'val_s_loss',
        'val_g_loss',
        'val_median_error_km',
        'val_r1_1km',
        'val_r1_25km',
        'val_r1_200km',
        'val_r1_750km',
        'val_r1_2500km',
        'val_r5_1km',
        'val_r5_25km',
        'val_r5_200km',
        'val_r5_750km',
        'val_r5_2500km',
        'val_r10_1km',
        'val_r10_25km',
        'val_r10_200km',
        'val_r10_750km',
        'val_r10_2500km',
        'best_val_r1_1km_so_far',
    ]
    if is_master:
        metrics_csv_path = os.path.join(out_dir, 'metrics.csv')
        with open(metrics_csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=metrics_fieldnames).writeheader()

    if is_master: logger.info("Start Training Loop...")

    for epoch in range(start_epoch, cfg.train.epochs):
        # 训练epoch
        avg_loss, avg_s_loss, avg_g_loss, global_step, train_diag, train_metrics = train_per_epoch(
            image_encoder, gps_encoder, alignment_hub, dataloader, optimizer, scheduler, scaler, device, writer, global_step, epoch, is_master, world_size, cfg, ema_dict=ema_dict
        )

        if use_ema:
            _apply_ema_for_eval(ema_dict, {
                'image_encoder': image_encoder.module,
                'gps_encoder': gps_encoder.module,
                'alignment_hub': alignment_hub.module,
            })

        # 验证epoch
        val_results = val_per_epoch(image_encoder, gps_encoder, alignment_hub, val_dataloader, device, writer, epoch, is_master, world_size)
        avg_val_loss, avg_val_s_loss, avg_val_g_loss, val_metrics = val_results

        if use_ema:
            _restore_from_ema(ema_dict, {
                'image_encoder': image_encoder.module,
                'gps_encoder': gps_encoder.module,
                'alignment_hub': alignment_hub.module,
            })
        current_val_r1_1km = float(val_metrics.get('r@1_1km', float('nan')))
        improved = False
        if not math.isnan(current_val_r1_1km) and current_val_r1_1km > best_val_r1_1km:
            best_val_r1_1km = current_val_r1_1km
            best_epoch = epoch
            no_improve_epochs = 0
            improved = True
        else:
            no_improve_epochs += 1

        if is_master:
            lr_groups = [f"g{i}:{group['lr']:.2e}" for i, group in enumerate(optimizer.param_groups)]
            logger.info(
                f"Epoch {epoch+1} | Train Avg Loss: {avg_loss:.4f} | S: {avg_s_loss:.4f} | G: {avg_g_loss:.4f} "
                f"| Val Avg Loss: {avg_val_loss:.4f} | S: {avg_val_s_loss:.4f} | G: {avg_val_g_loss:.4f} "
                f"| Margins(S/G): {train_diag['s_margin']:.4f}/{train_diag['g_margin']:.4f} "
                f"| Scales(S/G): {train_diag['sem_scale']:.3f}/{train_diag['geo_scale']:.3f} "
                f"| GradNorm(I/G/A): {train_diag['img_grad']:.3f}/{train_diag['gps_grad']:.3f}/{train_diag['align_grad']:.3f} "
                f"| LR[{', '.join(lr_groups)}] "
            )
            logger.info(
                f"Epoch {epoch+1} Retrieval | "
                f"Train: MedErr@1={train_metrics.get('median_error_km', float('nan')):.2f}km, "
                f"R@1(1/25/200km)={train_metrics.get('r@1_1km', 0.0)*100:.2f}/"
                f"{train_metrics.get('r@1_25km', 0.0)*100:.2f}/"
                f"{train_metrics.get('r@1_200km', 0.0)*100:.2f}%, "
                f"R@5@200km={train_metrics.get('r@5_200km', 0.0)*100:.2f}%, "
                f"R@10@200km={train_metrics.get('r@10_200km', 0.0)*100:.2f}% | "
                f"Val: MedErr@1={val_metrics.get('median_error_km', float('nan')):.2f}km, "
                f"R@1(1/25/200km)={val_metrics.get('r@1_1km', 0.0)*100:.2f}/"
                f"{val_metrics.get('r@1_25km', 0.0)*100:.2f}/"
                f"{val_metrics.get('r@1_200km', 0.0)*100:.2f}%, "
                f"R@5@200km={val_metrics.get('r@5_200km', 0.0)*100:.2f}%, "
                f"R@10@200km={val_metrics.get('r@10_200km', 0.0)*100:.2f}%"
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

            for t in [1, 25, 200, 750, 2500]:
                writer.add_scalar(f'Metric/Train_R1_{t}km', train_metrics.get(f'r@1_{t}km', float('nan')) * 100.0, epoch)
                writer.add_scalar(f'Metric/Val_R1_{t}km', val_metrics.get(f'r@1_{t}km', float('nan')) * 100.0, epoch)
            writer.add_scalar('Metric/Train_R5_200km', train_metrics.get('r@5_200km', float('nan')) * 100.0, epoch)
            writer.add_scalar('Metric/Train_R10_200km', train_metrics.get('r@10_200km', float('nan')) * 100.0, epoch)
            writer.add_scalar('Metric/Val_R5_200km', val_metrics.get('r@5_200km', float('nan')) * 100.0, epoch)
            writer.add_scalar('Metric/Val_R10_200km', val_metrics.get('r@10_200km', float('nan')) * 100.0, epoch)
            writer.add_scalar('Metric/Train_MedianError_km', train_metrics.get('median_error_km', float('nan')), epoch)
            writer.add_scalar('Metric/Val_MedianError_km', val_metrics.get('median_error_km', float('nan')), epoch)

            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)

            # Append one row per epoch to metrics CSV in checkpoint directory.
            with open(metrics_csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=metrics_fieldnames).writerow({
                    'epoch': epoch + 1,
                    'train_total_loss': avg_loss,
                    'train_s_loss': avg_s_loss,
                    'train_g_loss': avg_g_loss,
                    'train_median_error_km': train_metrics.get('median_error_km', float('nan')),
                    'train_r1_1km': train_metrics.get('r@1_1km', float('nan')),
                    'train_r1_25km': train_metrics.get('r@1_25km', float('nan')),
                    'train_r1_200km': train_metrics.get('r@1_200km', float('nan')),
                    'train_r1_750km': train_metrics.get('r@1_750km', float('nan')),
                    'train_r1_2500km': train_metrics.get('r@1_2500km', float('nan')),
                    'train_r5_1km': train_metrics.get('r@5_1km', float('nan')),
                    'train_r5_25km': train_metrics.get('r@5_25km', float('nan')),
                    'train_r5_200km': train_metrics.get('r@5_200km', float('nan')),
                    'train_r5_750km': train_metrics.get('r@5_750km', float('nan')),
                    'train_r5_2500km': train_metrics.get('r@5_2500km', float('nan')),
                    'train_r10_1km': train_metrics.get('r@10_1km', float('nan')),
                    'train_r10_25km': train_metrics.get('r@10_25km', float('nan')),
                    'train_r10_200km': train_metrics.get('r@10_200km', float('nan')),
                    'train_r10_750km': train_metrics.get('r@10_750km', float('nan')),
                    'train_r10_2500km': train_metrics.get('r@10_2500km', float('nan')),
                    'val_total_loss': avg_val_loss,
                    'val_s_loss': avg_val_s_loss,
                    'val_g_loss': avg_val_g_loss,
                    'val_median_error_km': val_metrics.get('median_error_km', float('nan')),
                    'val_r1_1km': val_metrics.get('r@1_1km', float('nan')),
                    'val_r1_25km': val_metrics.get('r@1_25km', float('nan')),
                    'val_r1_200km': val_metrics.get('r@1_200km', float('nan')),
                    'val_r1_750km': val_metrics.get('r@1_750km', float('nan')),
                    'val_r1_2500km': val_metrics.get('r@1_2500km', float('nan')),
                    'val_r5_1km': val_metrics.get('r@5_1km', float('nan')),
                    'val_r5_25km': val_metrics.get('r@5_25km', float('nan')),
                    'val_r5_200km': val_metrics.get('r@5_200km', float('nan')),
                    'val_r5_750km': val_metrics.get('r@5_750km', float('nan')),
                    'val_r5_2500km': val_metrics.get('r@5_2500km', float('nan')),
                    'val_r10_1km': val_metrics.get('r@10_1km', float('nan')),
                    'val_r10_25km': val_metrics.get('r@10_25km', float('nan')),
                    'val_r10_200km': val_metrics.get('r@10_200km', float('nan')),
                    'val_r10_750km': val_metrics.get('r@10_750km', float('nan')),
                    'val_r10_2500km': val_metrics.get('r@10_2500km', float('nan')),
                    'best_val_r1_1km_so_far': best_val_r1_1km,
                })

            if improved:
                best_path = os.path.join(out_dir, 'checkpoint_best.pth')
                save_checkpoint({
                    'epoch': epoch,
                    'image_encoder': image_encoder.module.state_dict(),
                    'gps_encoder': gps_encoder.module.state_dict(),
                    'alignment_hub': alignment_hub.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_val_r1_1km': best_val_r1_1km,
                    'best_epoch': best_epoch,
                    'ema': {
                        'image_encoder': ema_dict['image_encoder'].state_dict(),
                        'gps_encoder': ema_dict['gps_encoder'].state_dict(),
                        'alignment_hub': ema_dict['alignment_hub'].state_dict(),
                    } if use_ema else None,
                }, best_path)
                logger.info(
                    f"Best checkpoint updated @ epoch {epoch+1}: "
                    f"Val R@1(1km)={best_val_r1_1km*100:.2f}% | {best_path}"
                )
            
            if (epoch + 1) % cfg.train.save_interval == 0:
                save_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch+1}.pth")
                save_checkpoint({
                    'epoch': epoch,
                    'image_encoder': image_encoder.module.state_dict(),
                    'gps_encoder': gps_encoder.module.state_dict(),
                    'alignment_hub': alignment_hub.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'ema': {
                        'image_encoder': ema_dict['image_encoder'].state_dict(),
                        'gps_encoder': ema_dict['gps_encoder'].state_dict(),
                        'alignment_hub': ema_dict['alignment_hub'].state_dict(),
                    } if use_ema else None,
                }, save_path)
                logger.info(f"Checkpoint saved: {save_path}")

            if no_improve_epochs >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch+1}. "
                    f"No Val R@1(1km) improvement for {early_stopping_patience} epochs. "
                    f"Best epoch: {best_epoch+1}, Best Val R@1(1km): {best_val_r1_1km*100:.2f}%"
                )

        if is_master and (epoch + 1) % 5 == 0:  # 每10个epoch记录一次
            for name, param in image_encoder.named_parameters():
                writer.add_histogram(f'ImageEncoder/{name}', param.detach(), epoch)
            for name, param in gps_encoder.named_parameters():
                writer.add_histogram(f'GPSEncoder/{name}', param.detach(), epoch)
            for name, param in alignment_hub.named_parameters():
                writer.add_histogram(f'AlignmentHub/{name}', param.detach(), epoch)

        if no_improve_epochs >= early_stopping_patience:
            break

    if is_master:
        logger.info("Generating final plots...")
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(history['total_loss'], label='Total Loss')
            plt.plot(history['s_loss'], label='S Loss', linestyle='--')
            plt.plot(history['g_loss'], label='G Loss', linestyle='--')
            plt.legend()
            plt.savefig(os.path.join(out_dir, 'loss_curve.png'))
            writer.close()
        except Exception as e:
            logger.warning(f"Plotting failed (likely headless env): {e}")
            
    cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)