import torch


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
        shadow = state.get('shadow')
        if shadow is not None:
            self.shadow = {k: v.detach().clone() for k, v in shadow.items()}