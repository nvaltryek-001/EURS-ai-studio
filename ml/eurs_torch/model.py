"""EURS-Net: shared encoder, dual decoders with restoration-to-segmentation feature fusion.

    degraded x ──► Encoder (ResNet-18/34 or MobileNetV3-L, ImageNet init)
                    │  multi-scale features F1..F5
          ┌─────────┴─────────┐
   Restoration decoder   Segmentation decoder
   (U-Net, residual out)  (U-Net; at every scale it also receives the
          │                restoration decoder's features: "R2S fusion")
     restored = clip(x + tanh(r))      logits (C classes)

The fusion lets the segmenter use colour-corrected evidence without a second forward pass,
and the restoration branch receives gradients from the segmentation loss through the fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv_bn(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class DSConv(nn.Module):
    """Depthwise-separable conv block (MobileNet-style): dw3x3 -> BN -> ReLU -> pw1x1 -> BN -> ReLU."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class TinyEncoder(nn.Sequential):
    """~0.1M-parameter encoder for the embedded track (stride 2, 4, 8, 16, 32)."""

    def __init__(self, ch=(16, 24, 40, 64, 96)):
        stages = [nn.Sequential(conv_bn(3, ch[0]), DSConv(ch[0], ch[0], 2))]
        for a, b in zip(ch[:-1], ch[1:]):
            stages.append(nn.Sequential(DSConv(a, b, 2), DSConv(b, b)))
        super().__init__(*stages)


class Encoder(nn.Module):
    """Returns 5 feature maps at strides 2, 4, 8, 16, 32."""

    def __init__(self, name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.name = name
        if name == "tiny":
            self.stages = nn.ModuleList(list(TinyEncoder()))  # trained from scratch
        elif name.startswith("resnet"):
            w = "DEFAULT" if pretrained else None
            net = getattr(torchvision.models, name)(weights=w)
            self.stages = nn.ModuleList([
                nn.Sequential(net.conv1, net.bn1, net.relu),          # /2
                nn.Sequential(net.maxpool, net.layer1),               # /4
                net.layer2, net.layer3, net.layer4,                   # /8 /16 /32
            ])
        elif name == "mobilenet_v3_large":
            net = torchvision.models.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
            f = net.features
            # Stage boundaries at strides 2, 4, 8, 16, 32 for torchvision's MobileNetV3-Large.
            self.stages = nn.ModuleList([f[:2], f[2:4], f[4:7], f[7:13], f[13:]])
        else:
            raise ValueError(name)
        # ImageNet normalisation is folded in so the exported model takes plain [0,1] RGB.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.eval()  # probe channel counts without touching pretrained BatchNorm statistics
        with torch.no_grad():
            self.channels = [t.shape[1] for t in self.forward(torch.zeros(1, 3, 64, 64))]
        self.train()

    def forward(self, x):
        x = (x - self.mean) / self.std
        feats = []
        for s in self.stages:
            x = s(x)
            feats.append(x)
        return feats


class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout, cextra=0, light=False):
        super().__init__()
        c = cin + cskip + cextra
        self.conv = (nn.Sequential(DSConv(c, cout), DSConv(cout, cout)) if light
                     else nn.Sequential(conv_bn(c, cout), conv_bn(cout, cout)))

    def forward(self, x, skip, extra=None):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        parts = [x, skip] + ([extra] if extra is not None else [])
        return self.conv(torch.cat(parts, 1))


class Decoder(nn.Module):
    def __init__(self, enc_ch, widths=(256, 128, 64, 32), fuse_ch=None, light=False):
        super().__init__()
        fuse_ch = fuse_ch or [0] * len(widths)
        c5, c4, c3, c2, c1 = enc_ch[::-1]
        self.center = conv_bn(c5, widths[0], k=1 if light else 3)
        self.blocks = nn.ModuleList([
            UpBlock(widths[0], c4, widths[0], fuse_ch[0], light),
            UpBlock(widths[0], c3, widths[1], fuse_ch[1], light),
            UpBlock(widths[1], c2, widths[2], fuse_ch[2], light),
            UpBlock(widths[2], c1, widths[3], fuse_ch[3], light),
        ])
        self.out_ch = list(widths)

    def forward(self, feats, extras=None):
        f1, f2, f3, f4, f5 = feats
        x = self.center(f5)
        outs = []
        for blk, skip, i in zip(self.blocks, (f4, f3, f2, f1), range(4)):
            x = blk(x, skip, None if extras is None else extras[i])
            outs.append(x)
        return outs  # strides 16, 8, 4, 2


class EURSNet(nn.Module):
    def __init__(self, num_classes: int = 5, encoder: str = "resnet18", pretrained: bool = True,
                 fusion: bool = True, widths=None):
        super().__init__()
        light = encoder == "tiny"
        widths = widths or ((64, 48, 32, 16) if light else (256, 128, 64, 32))
        head = 16 if light else 32
        self.encoder = Encoder(encoder, pretrained)
        self.fusion = fusion
        self.res_dec = Decoder(self.encoder.channels, widths, light=light)
        self.seg_dec = Decoder(self.encoder.channels, widths, fuse_ch=list(widths) if fusion else None, light=light)
        self.res_head = nn.Sequential(conv_bn(widths[-1] + 3, head), nn.Conv2d(head, 3, 3, padding=1))
        self.seg_head = nn.Sequential(conv_bn(widths[-1], head), nn.Conv2d(head, num_classes, 1))

    def forward(self, x):
        feats = self.encoder(x)
        r = self.res_dec(feats)
        s = self.seg_dec(feats, r if self.fusion else None)
        H, W = x.shape[-2:]
        rf = F.interpolate(r[-1], size=(H, W), mode="bilinear", align_corners=False)
        restored = torch.clamp(x + torch.tanh(self.res_head(torch.cat([rf, x], 1))), 0, 1)
        logits = F.interpolate(self.seg_head(s[-1]), size=(H, W), mode="bilinear", align_corners=False)
        return restored, logits


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
