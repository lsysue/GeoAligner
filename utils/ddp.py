import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as TorchDDP


class DDP:
    """Centralized DDP utility API (setup/cleanup/wrap/collectives)."""

    @staticmethod
    def is_distributed() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def get_world_size() -> int:
        return dist.get_world_size() if DDP.is_distributed() else 1

    @staticmethod
    def get_rank() -> int:
        return dist.get_rank() if DDP.is_distributed() else 0

    @staticmethod
    def setup():
        """Initialize DDP process group and return rank/local_rank/world_size."""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            rank = 0
            world_size = 1
            local_rank = 0
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")

        torch.cuda.set_device(local_rank)
        if not DDP.is_distributed():
            dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank, world_size

    @staticmethod
    def cleanup():
        if DDP.is_distributed():
            dist.destroy_process_group()

    @staticmethod
    def wrap(module: torch.nn.Module, local_rank: int, find_unused_parameters: bool = True):
        return TorchDDP(module, device_ids=[local_rank], find_unused_parameters=find_unused_parameters)

    @staticmethod
    def all_reduce_sum_(tensor: torch.Tensor):
        if DDP.is_distributed():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    @staticmethod
    def all_gather_variable_length_1d(local_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """All-gather variable-length 1D tensors and concatenate them in rank order."""
        if local_tensor.dim() != 1:
            raise ValueError("all_gather_variable_length_1d expects a 1D tensor")

        if not DDP.is_distributed():
            return local_tensor

        world_size = dist.get_world_size()
        local_len = torch.tensor([local_tensor.numel()], device=device, dtype=torch.long)
        len_list = [torch.zeros_like(local_len) for _ in range(world_size)]
        dist.all_gather(len_list, local_len)
        lengths = [int(x.item()) for x in len_list]
        max_len = max(lengths) if lengths else 0

        if max_len == 0:
            return torch.empty(0, device=device, dtype=local_tensor.dtype)

        padded = torch.zeros(max_len, device=device, dtype=local_tensor.dtype)
        if local_tensor.numel() > 0:
            padded[: local_tensor.numel()] = local_tensor

        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)

        merged = []
        for tensor_i, len_i in zip(gathered, lengths):
            if len_i > 0:
                merged.append(tensor_i[:len_i])
        return torch.cat(merged, dim=0) if merged else torch.empty(0, device=device, dtype=local_tensor.dtype)


class Accumulator:
    """Generic accumulator with DDP-aware reduction and variable-length gather helpers."""

    def __init__(self):
        self._counts = {}
        self._tensor_sums = {}
        self._vectors = {}

    def add_count(self, key: str, value: int):
        self._counts[key] = int(self._counts.get(key, 0)) + int(value)

    def add_tensor_sum(self, key: str, tensor: torch.Tensor):
        local = tensor.detach()
        if key not in self._tensor_sums:
            self._tensor_sums[key] = local.clone()
            return

        cur = self._tensor_sums[key]
        if cur.device != local.device:
            cur = cur.to(local.device)
        self._tensor_sums[key] = cur + local

    def append_vector(self, key: str, vector_1d: torch.Tensor):
        local = vector_1d.detach().reshape(-1)
        self._vectors.setdefault(key, []).append(local)

    def reduced_count(self, key: str, device: torch.device) -> int:
        local = torch.tensor([int(self._counts.get(key, 0))], device=device, dtype=torch.long)
        DDP.all_reduce_sum_(local)
        return int(local.item())

    def reduced_tensor_sum(self, key: str, shape, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if key in self._tensor_sums:
            out = self._tensor_sums[key]
            if out.device != device:
                out = out.to(device)
        else:
            out = torch.zeros(shape, device=device, dtype=dtype)

        DDP.all_reduce_sum_(out)
        return out

    def gathered_vector(self, key: str, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        vectors = self._vectors.get(key, [])
        if vectors:
            local = torch.cat(
                [v.to(device=device) if v.device != device else v for v in vectors],
                dim=0,
            )
        else:
            local = torch.empty(0, device=device, dtype=dtype)

        if local.dtype != dtype:
            local = local.to(dtype=dtype)

        if DDP.is_distributed():
            return DDP.all_gather_variable_length_1d(local, device=device)
        return local
