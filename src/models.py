"""Model factory. Every architecture choice is a config key (`model.*`) so experiments are yaml diffs."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def _expand_first_conv(model: nn.Module, in_channels: int) -> None:
    """Replace the encoder's first conv (built for 1 input channel = the ImageNet RGB weights summed, smp's rule)
    by one with `in_channels` inputs whose extra input channels start at zero. The network therefore starts exactly
    as the 1-channel baseline and learns to use the extra channels from zero (model.zero_init_extra_channels)."""
    for name, module in model.encoder.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 1:
            new = nn.Conv2d(in_channels, module.out_channels, module.kernel_size, module.stride, module.padding,
                            module.dilation, module.groups, bias=module.bias is not None)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, :1] = module.weight
                if module.bias is not None:
                    new.bias.copy_(module.bias)
            parent = model.encoder
            *path, leaf = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, leaf, new)
            return
    raise RuntimeError("no 1-channel first conv found in the encoder")


class HighResStem(nn.Module):
    """Full-resolution skip for the last decoder stage (model.highres_stem; SenNet 1st / Contrails 2nd place).
    smp Unet's last decoder block upsamples to full resolution without a stride-1 skip, so boundaries come from
    stride-2 features; a filament is ~11 px wide, so 1 px of boundary error costs ~15 % IoU. Two stride-1 3x3 convs on
    the input are concatenated with the decoder output and added back as a residual whose last conv starts at zero:
    the network starts exactly as the base model and learns to use the full-resolution evidence."""

    def __init__(self, base: nn.Module, in_channels: int, width: int = 16):
        super().__init__()
        self.base = base
        dec = base.segmentation_head[0].in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(dec + width, dec, 3, padding=1, bias=False), nn.BatchNorm2d(dec), nn.ReLU(inplace=True),
            nn.Conv2d(dec, dec, 3, padding=1))
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.base.decoder(self.base.encoder(x))
        d = d + self.fuse(torch.cat([d, self.stem(x)], dim=1))
        return self.base.segmentation_head(d)


def build_model(cfg: dict, in_channels: int, classes: int = 1) -> nn.Module:
    """classes = 1 (mask) + 1 when the spine auxiliary head is on (target.spine_width > 0); channel 0 is the mask.
    model.zero_init_extra_channels: build for 1 channel and expand the first conv with zero weights for channels 1.."""
    zero_init = bool(cfg.get("zero_init_extra_channels", False)) and in_channels > 1
    arch = getattr(smp, cfg["arch"])
    model = arch(
        encoder_name=cfg["encoder"],
        encoder_weights=cfg.get("encoder_weights"),
        in_channels=1 if zero_init else in_channels,
        classes=classes,
    )
    if zero_init:
        _expand_first_conv(model, in_channels)
    if cfg.get("grad_checkpointing", False):
        # timm ('tu-*') encoders: recompute activations in the backward pass (memory for compute; model.grad_checkpointing)
        inner = getattr(model.encoder, "model", None)
        if inner is None or not hasattr(inner, "set_grad_checkpointing"):
            raise ValueError(f"encoder {cfg['encoder']} does not support gradient checkpointing")
        inner.set_grad_checkpointing(True)
    if cfg.get("highres_stem", False):
        model = HighResStem(model, in_channels, int(cfg.get("highres_width", 16)))
    return model
