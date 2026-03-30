import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class SemanticAligner(nn.Module):
    def __init__(self, s_dim, temperature=0.07, queue_size=4096):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if queue_size <= 0:
            raise ValueError(f"queue_size must be > 0, got {queue_size}")
        # 温度参数，控制着loss的锐度
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))

        self.queue_size = queue_size
        # 注册为 buffer，会随 checkpoint 保存，但不参与梯度更新
        self.register_buffer("image_queue", torch.randn(s_dim, queue_size))
        self.register_buffer("gps_queue", torch.randn(s_dim, queue_size))
        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.gps_queue = F.normalize(self.gps_queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def gather_features(self, features):
        """收集所有 GPU 上的特征以扩大负样本池"""
        if not dist.is_initialized():
            return features
        
        gathered_features = [torch.zeros_like(features) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_features, features)
        # 将所有卡的特征拼接起来
        return torch.cat(gathered_features, dim=0)

    @torch.no_grad()
    def _build_global_labels(self, local_batch_size: int, device: torch.device) -> torch.Tensor:
        if not dist.is_initialized():
            return torch.arange(local_batch_size, device=device)

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_bs_tensor = torch.tensor([local_batch_size], device=device, dtype=torch.long)
        gathered_bs = [torch.zeros_like(local_bs_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_bs, local_bs_tensor)
        batch_sizes = torch.cat(gathered_bs, dim=0)
        global_offset = int(batch_sizes[:rank].sum().item())
        return torch.arange(local_batch_size, device=device) + global_offset

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feats, gps_feats):
        """更新历史队列 (FIFO)"""
        batch_size = image_feats.shape[0]
        if batch_size == 0:
            return

        if batch_size > self.queue_size:
            image_feats = image_feats[-self.queue_size:]
            gps_feats = gps_feats[-self.queue_size:]
            batch_size = self.queue_size

        ptr = int(self.queue_ptr)

        first_chunk = min(self.queue_size - ptr, batch_size)
        second_chunk = batch_size - first_chunk

        self.image_queue[:, ptr:ptr + first_chunk] = image_feats[:first_chunk].T
        self.gps_queue[:, ptr:ptr + first_chunk] = gps_feats[:first_chunk].T

        if second_chunk > 0:
            self.image_queue[:, :second_chunk] = image_feats[first_chunk:].T
            self.gps_queue[:, :second_chunk] = gps_feats[first_chunk:].T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def forward(self, image_s_vec, gps_s_vec):
        """
        计算图像、GPS之间的多模态对比损失
        Args:
            image_s_vec (B, D): 图像 s_vector
            gps_s_vec (B, D): GPS s_vector
        """
        # 归一化特征
        image_s_vec = F.normalize(image_s_vec, p=2, dim=-1)
        gps_s_vec = F.normalize(gps_s_vec, p=2, dim=-1)

        global_image_s_vec = self.gather_features(image_s_vec)
        global_gps_s_vec = self.gather_features(gps_s_vec)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)

        # 计算全局相似度矩阵: (Local_B, Global_B)
        logits_img2gps = (image_s_vec @ global_gps_s_vec.T)
        logits_gps2img = (gps_s_vec @ global_image_s_vec.T)
        
        logits_img2queue = (image_s_vec @ self.gps_queue.clone().detach())
        logits_gps2queue = (gps_s_vec @ self.image_queue.clone().detach())

        logits_img = torch.cat([logits_img2gps, logits_img2queue], dim=1) * logit_scale
        logits_gps = torch.cat([logits_gps2img, logits_gps2queue], dim=1) * logit_scale

        # 【修复3】对齐正确的 Label 索引
        # 如果是单卡，labels 就是 0, 1, 2...
        # 如果是多卡，当前卡 (Rank) 的样本在全局张量中的位置需要加上 rank * batch_size
        local_B = image_s_vec.shape[0]
        labels = self._build_global_labels(local_B, image_s_vec.device)

        loss_img2gps = F.cross_entropy(logits_img, labels)
        loss_gps2img = F.cross_entropy(logits_gps, labels)

        if self.training:
            self._dequeue_and_enqueue(global_image_s_vec, global_gps_s_vec)
        
        return (loss_img2gps + loss_gps2img) / 2