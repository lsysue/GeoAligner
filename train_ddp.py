import os
import math
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from tqdm import tqdm

from utils.config import Config, arg_parser, build_from_defaults, load_config
from utils.ddp import DDP
from utils.metrics import Retriever
from utils.reporter import (
    setup_reporter,
    log_train_step_diagnostics,
    rank_metric_fieldnames,
    report_epoch_records,
    finalize_reporter,
)
from utils.ema import ModelEMA
from analysis.analysis import pos_neg_stats
from datasets.img2geo_dataset import Img2GeoDataset
from encoders.image_encoder import ImageEncoder, ImageEncoderConfig
from encoders.location_encoder import GPSEncoder, GPSEncoderConfig
from aligners.alignmenthub import AlignmentHub, AlignmentHubConfig

torch.multiprocessing.set_sharing_strategy('file_system')

EVAL_KS = (1, 5, 10)
EVAL_THRESHOLDS_KM = (1, 25, 200, 750, 2500)

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
    }
    diag_count = 0

    train_s_recall_state = Retriever.init_recall_state(EVAL_KS, EVAL_THRESHOLDS_KM)
    train_g_recall_state = Retriever.init_recall_state(EVAL_KS, EVAL_THRESHOLDS_KM)

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

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            img_embeds = image_encoder(images)
            gps_embeds = gps_encoder(gps_coords, s2_tokens)
            loss_dict = alignment_hub(img_embeds, gps_embeds)
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if ema_dict is not None:
            ema_dict['image_encoder'].update(image_encoder.module)
            ema_dict['gps_encoder'].update(gps_encoder.module)
            ema_dict['alignment_hub'].update(alignment_hub.module)

        # Detach to save memory
        total_loss_epoch += loss.detach()
        s_loss_epoch += loss_dict["semantic_loss"].detach()
        g_loss_epoch += loss_dict["geo_loss"].detach()

        image_s_vectors = img_embeds["s_vector"]
        image_g_tokens = img_embeds["g_tokens"]
        gps_s_vectors = gps_embeds["s_vector"]
        gps_g_tokens = gps_embeds["g_tokens"]
        s_sim = align_module.semantic_aligner.compute_pair_similarity(
            image_s_vectors.detach(),
            gps_s_vectors.detach(),
        )
        Retriever.update_recall_state(train_s_recall_state, s_sim, gps_coords)
        g_sim = align_module.geo_aligner.compute_pair_similarity(
            image_g_tokens.detach(),
            gps_g_tokens.detach(),
        )
        Retriever.update_recall_state(train_g_recall_state, g_sim, gps_coords)

        if is_master:
            if batch_idx % cfg.train.log_interval == 0:
                current_loss = loss.detach().item()
                s_pos, s_neg, s_margin = pos_neg_stats(s_sim)
                g_pos, g_neg, g_margin = pos_neg_stats(g_sim)

                sem_scale = align_module.semantic_aligner.logit_scale.exp().clamp(max=100.0).item()
                geo_scale = align_module.geo_aligner.logit_scale.exp().clamp(max=100.0).item()
                log_train_step_diagnostics(
                    writer=writer,
                    global_step=global_step,
                    s_pos=s_pos,
                    s_neg=s_neg,
                    s_margin=s_margin,
                    g_pos=g_pos,
                    g_neg=g_neg,
                    g_margin=g_margin,
                    sem_scale=sem_scale,
                    geo_scale=geo_scale,
                )

                diag_sums["s_margin"] += s_margin
                diag_sums["g_margin"] += g_margin
                diag_sums["sem_scale"] += sem_scale
                diag_sums["geo_scale"] += geo_scale
                diag_count += 1

                pbar.set_postfix({"Loss": f"{current_loss:.4f}"})

        global_step += 1

    # Reduce losses from all devices
    DDP.all_reduce_sum_(total_loss_epoch)
    DDP.all_reduce_sum_(s_loss_epoch)
    DDP.all_reduce_sum_(g_loss_epoch)
    
    avg_loss = total_loss_epoch.item() / (len(dataloader) * world_size)
    avg_s_loss = s_loss_epoch.item() / (len(dataloader) * world_size)
    avg_g_loss = g_loss_epoch.item() / (len(dataloader) * world_size)

    if diag_count > 0:
        diag_avg = {k: v / diag_count for k, v in diag_sums.items()}
    else:
        diag_avg = {k: float('nan') for k in diag_sums.keys()}

    train_s_metrics = Retriever.finalize_recall_state(train_s_recall_state, device)
    train_g_metrics = Retriever.finalize_recall_state(train_g_recall_state, device)

    return avg_loss, avg_s_loss, avg_g_loss, global_step, diag_avg, train_s_metrics, train_g_metrics

def val_per_epoch(image_encoder, gps_encoder, alignment_hub, val_dataloader, device, world_size):
    if val_dataloader is None:
        return float('nan'), float('nan'), float('nan'), {}, {}
    
    image_encoder.eval()
    gps_encoder.eval()
    alignment_hub.eval()
    
    val_total_loss = torch.zeros(1, device=device)
    val_s_loss = torch.zeros(1, device=device)
    val_g_loss = torch.zeros(1, device=device)

    val_s_recall_state = Retriever.init_recall_state(EVAL_KS, EVAL_THRESHOLDS_KM)
    val_g_recall_state = Retriever.init_recall_state(EVAL_KS, EVAL_THRESHOLDS_KM)
    align_module = alignment_hub.module if hasattr(alignment_hub, "module") else alignment_hub
    
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

            val_image_s_vectors = val_img_embeds["s_vector"]
            val_image_g_tokens = val_img_embeds["g_tokens"]
            val_gps_s_vectors = val_gps_embeds["s_vector"]
            val_gps_g_tokens = val_gps_embeds["g_tokens"]
            s_sim = align_module.semantic_aligner.compute_pair_similarity(
                val_image_s_vectors.detach(),
                val_gps_s_vectors.detach(),
            )
            Retriever.update_recall_state(val_s_recall_state, s_sim, val_gps_coords)
            g_sim = align_module.geo_aligner.compute_pair_similarity(
                val_image_g_tokens.detach(),
                val_gps_g_tokens.detach(),
            )
            Retriever.update_recall_state(val_g_recall_state, g_sim, val_gps_coords)
    
    DDP.all_reduce_sum_(val_total_loss)
    DDP.all_reduce_sum_(val_s_loss)
    DDP.all_reduce_sum_(val_g_loss)
    
    avg_val_loss = val_total_loss.item() / (len(val_dataloader) * world_size)
    avg_val_s_loss = val_s_loss.item() / (len(val_dataloader) * world_size)
    avg_val_g_loss = val_g_loss.item() / (len(val_dataloader) * world_size)

    val_s_metrics = Retriever.finalize_recall_state(val_s_recall_state, device)
    val_g_metrics = Retriever.finalize_recall_state(val_g_recall_state, device)

    return avg_val_loss, avg_val_s_loss, avg_val_g_loss, val_s_metrics, val_g_metrics

def main(args):
    # 1. DDP 初始化
    rank, local_rank, world_size = DDP.setup()
    device = torch.device(f"cuda:{local_rank}")
    is_master = (rank == 0)
    writer = None
    # 2. 加载配置
    cfg = load_config(args.config, overrides=args.overrides)
    # Build module configs as: class defaults -> YAML -> CLI overrides (already merged in cfg).
    img_cfg = build_from_defaults(ImageEncoderConfig, getattr(cfg.model, 'image', None))
    img_cfg.img_size = cfg.data.img_size
    gps_cfg = build_from_defaults(GPSEncoderConfig, getattr(cfg.model, 'gps', None))
    align_cfg = build_from_defaults(AlignmentHubConfig, getattr(cfg.model, 'alignment', None))
    align_cfg.s_dim = img_cfg.s_dim
    align_cfg.g_dim = img_cfg.g_dim

    # Persist the effective runtime config used by training:
    # top-level cfg (already YAML + --set merged) + module defaults resolved via build_from_defaults.
    resolved_cfg = Config.from_dict(cfg.to_dict())
    if not hasattr(resolved_cfg, 'model') or not isinstance(resolved_cfg.model, Config):
        resolved_cfg.model = Config()
    resolved_cfg.model.image = Config.from_dict(img_cfg.to_dict())
    resolved_cfg.model.gps = Config.from_dict(gps_cfg.to_dict())
    resolved_cfg.model.alignment = Config.from_dict(align_cfg.to_dict())

    logger, writer, out_dir = setup_reporter(
        is_master=is_master,
        base_output_dir=cfg.output_dir,
        cfg=resolved_cfg,
        overrides=args.overrides,
        world_size=world_size,
    )

    # 3. 数据准备
    if is_master:
        logger.info("Initializing Datasets...")

    use_debug = 'debug' in args.config or args.debug
    num_workers = cfg.data.num_workers
    persistent_workers = num_workers > 0
    
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
        s2_levels=gps_cfg.s2_levels
    )

    if use_debug:
        debug_size = cfg.data.n_data
        if is_master:
            logger.info(f"[DEBUG] Truncating dataset to {debug_size}")
        indices = list(range(min(len(dataset), debug_size)))
        dataset = Subset(dataset, indices)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )

    # 验证数据集
    val_dataset = Img2GeoDataset(
        csv_file=cfg.data.val.csv_file,
        img_dir=cfg.data.val.img_dir,
        transform=val_transform,
        s2_levels=gps_cfg.s2_levels
    )
    if use_debug:
        val_indices = list(range(min(len(val_dataset), debug_size)))
        val_dataset = Subset(val_dataset, val_indices)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )

    # 4. 模型初始化
    if is_master:
        logger.info("Initializing Models...")

    # Initialize Modules on GPU
    image_encoder = ImageEncoder(img_cfg).to(device)
    gps_encoder = GPSEncoder(gps_cfg).to(device)
    alignment_hub = AlignmentHub(align_cfg).to(device)

    image_encoder = DDP.wrap(image_encoder, local_rank=local_rank, find_unused_parameters=True)
    gps_encoder = DDP.wrap(gps_encoder, local_rank=local_rank, find_unused_parameters=True)
    alignment_hub = DDP.wrap(alignment_hub, local_rank=local_rank, find_unused_parameters=True)

    # 5. 优化器
    base_lr = float(cfg.train.learning_rate)
    image_lr_mult = float(cfg.train.lr_mult_image)
    gps_lr_mult = float(cfg.train.lr_mult_gps)
    align_lr_mult = float(cfg.train.lr_mult_align)
    logit_scale_lr_mult = float(cfg.train.lr_mult_logit_scale)

    align_named_params = list(alignment_hub.module.named_parameters())
    logit_scale_params = [p for n, p in align_named_params if n.endswith('logit_scale')]
    align_core_params = [p for n, p in align_named_params if not n.endswith('logit_scale')]
    image_trainable_params = [p for p in image_encoder.module.parameters() if p.requires_grad]
    gps_trainable_params = [p for p in gps_encoder.module.parameters() if p.requires_grad]

    optimizer_grouped_parameters = [
        {"params": image_trainable_params, "lr": base_lr * image_lr_mult},
        {"params": gps_trainable_params, "lr": base_lr * gps_lr_mult},
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
    warmup_epochs = int(cfg.train.warmup_epochs)
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
    ema_decay = float(cfg.train.ema_decay)
    use_ema = bool(cfg.train.use_ema)
    ema_dict = None
    if use_ema:
        ema_dict = {
            'image_encoder': ModelEMA(image_encoder.module, decay=ema_decay),
            'gps_encoder': ModelEMA(gps_encoder.module, decay=ema_decay),
            'alignment_hub': ModelEMA(alignment_hub.module, decay=ema_decay),
        }
        if is_master:
            logger.info(f"EMA enabled. decay={ema_decay}")
    ema_modules = {
        'image_encoder': image_encoder.module,
        'gps_encoder': gps_encoder.module,
        'alignment_hub': alignment_hub.module,
    } if use_ema else None

    # 6. Resume
    start_epoch = 0
    if cfg.train.resume_path:
        if is_master:
            logger.info(f"Loading checkpoint {cfg.train.resume_path}")
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
            if is_master:
                logger.warning(f"Scheduler state missing! Fast-forwarding scheduler to step {start_epoch * steps_per_epoch}...")
            completed_steps = start_epoch * steps_per_epoch
            for _ in range(completed_steps):
                scheduler.step()

        # 尝试加载 scaler，如果不存在则忽略（它会自适应）
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        else:
            if is_master:
                logger.warning("Scaler state missing. It will self-calibrate.")

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
        
        if is_master:
            logger.info(f"Resumed from epoch {start_epoch}")

    history = {'total_loss': [], 's_loss': [], 'g_loss': []}
    global_step = start_epoch * steps_per_epoch

    # Selection and stopping controls
    best_val_s_r1_1km = float('-inf')
    best_epoch = -1
    no_improve_epochs = 0
    early_stopping_patience = 10

    metrics_csv_path = None
    metrics_fieldnames = [
        'epoch',
        'train_total_loss',
        'train_s_loss',
        'train_g_loss',
    ]
    metrics_fieldnames.extend(rank_metric_fieldnames('train', 's', EVAL_KS, EVAL_THRESHOLDS_KM))
    metrics_fieldnames.extend(rank_metric_fieldnames('train', 'g', EVAL_KS, EVAL_THRESHOLDS_KM))
    metrics_fieldnames.extend(['val_total_loss', 'val_s_loss', 'val_g_loss'])
    metrics_fieldnames.extend(rank_metric_fieldnames('val', 's', EVAL_KS, EVAL_THRESHOLDS_KM))
    metrics_fieldnames.extend(rank_metric_fieldnames('val', 'g', EVAL_KS, EVAL_THRESHOLDS_KM))
    metrics_fieldnames.append('best_val_s_r1_1km_so_far')
    if is_master:
        metrics_csv_path = os.path.join(out_dir, 'metrics.csv')
        with open(metrics_csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=metrics_fieldnames).writeheader()

    if is_master:
        logger.info("Start Training Loop...")

    for epoch in range(start_epoch, cfg.train.epochs):
        # 训练epoch
        avg_loss, avg_s_loss, avg_g_loss, global_step, train_diag, train_s_metrics, train_g_metrics = train_per_epoch(
            image_encoder, gps_encoder, alignment_hub, dataloader, optimizer, scheduler, scaler, device, writer, global_step, epoch, is_master, world_size, cfg, ema_dict=ema_dict
        )

        if use_ema:
            for key, ema in ema_dict.items():
                ema.store(ema_modules[key])
                ema.copy_to(ema_modules[key])

        # 验证epoch
        val_results = val_per_epoch(image_encoder, gps_encoder, alignment_hub, val_dataloader, device, world_size)
        avg_val_loss, avg_val_s_loss, avg_val_g_loss, val_s_metrics, val_g_metrics = val_results

        if use_ema:
            for key, ema in ema_dict.items():
                ema.restore(ema_modules[key])
        current_val_s_r1_1km = float(val_s_metrics.get('r@1_1km', float('nan')))
        improved = False
        if not math.isnan(current_val_s_r1_1km) and current_val_s_r1_1km > best_val_s_r1_1km:
            best_val_s_r1_1km = current_val_s_r1_1km
            best_epoch = epoch
            no_improve_epochs = 0
            improved = True
        else:
            no_improve_epochs += 1

        if is_master:
            report_epoch_records(
                logger=logger,
                writer=writer,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                avg_loss=avg_loss,
                avg_s_loss=avg_s_loss,
                avg_g_loss=avg_g_loss,
                avg_val_loss=avg_val_loss,
                avg_val_s_loss=avg_val_s_loss,
                avg_val_g_loss=avg_val_g_loss,
                train_diag=train_diag,
                train_s_metrics=train_s_metrics,
                train_g_metrics=train_g_metrics,
                val_s_metrics=val_s_metrics,
                val_g_metrics=val_g_metrics,
                use_ema=use_ema,
                eval_thresholds_km=EVAL_THRESHOLDS_KM,
                eval_ks=EVAL_KS,
                metrics_csv_path=metrics_csv_path,
                metrics_fieldnames=metrics_fieldnames,
                history=history,
            )

            checkpoint_state = {
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
            }

            if improved:
                best_path = os.path.join(out_dir, 'checkpoint_best.pth')
                torch.save({**checkpoint_state, 'best_val_s_r1_1km': best_val_s_r1_1km, 'best_epoch': best_epoch}, best_path)
                logger.info(
                    f"Best checkpoint updated @ epoch {epoch+1}: "
                    f"Val S-R@1(1km)={best_val_s_r1_1km*100:.2f}% | {best_path}"
                )
            
            if (epoch + 1) % cfg.train.save_interval == 0:
                save_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save(checkpoint_state, save_path)
                logger.info(f"Checkpoint saved: {save_path}")

            should_stop = no_improve_epochs >= early_stopping_patience
            if should_stop:
                logger.info(
                    f"Early stopping triggered at epoch {epoch+1}. "
                    f"No Val S-R@1(1km) improvement for {early_stopping_patience} epochs. "
                    f"Best epoch: {best_epoch+1}, Best Val S-R@1(1km): {best_val_s_r1_1km*100:.2f}%"
                )

        if no_improve_epochs >= early_stopping_patience:
            break

    if is_master:
        finalize_reporter(logger=logger, writer=writer, history=history, out_dir=out_dir)
            
    DDP.cleanup()

if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    main(args)