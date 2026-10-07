"""Model zoo for Project #9 medmodel-discovery.

- PLANNet: novel pyramidal latent-attention architecture (Candidate A).
- GCViTLitePerceiverLite: faithful small PyTorch analog of Habib's
  GCViT + Perceiver IO hybrid (flat latent bottleneck baseline).
- ResNet18Small, ViTTiny: standard baselines.
All sized for 28x28 MedMNIST inputs, CPU-trainable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class ConvStem(nn.Module):
    """3-stage conv stem: 28x28 -> 14x14 (32ch) -> 7x7 (64ch) -> 4x4 (128ch)."""
    def __init__(self, in_ch=3):
        super().__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU())
        self.s2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.s3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU())

    def forward(self, x):
        f1 = self.s1(x)   # 14x14x32
        f2 = self.s2(f1)  # 7x7x64
        f3 = self.s3(f2)  # 4x4x128
        return f1, f2, f3


class LatentLevel(nn.Module):
    """One pyramidal level: latent array cross-attending to a feature map,
    with an optional top-down gate modulating the value projection."""
    def __init__(self, n_latents, d_latent, d_feat, n_heads=4, d_gate=0):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_latent) * 0.02)
        self.q_proj = nn.Linear(d_latent, d_latent)
        self.k_proj = nn.Linear(d_feat, d_latent)
        self.v_proj = nn.Linear(d_feat, d_latent)
        self.attn = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_latent)
        self.norm2 = nn.LayerNorm(d_latent)
        self.mlp = nn.Sequential(nn.Linear(d_latent, d_latent * 2), nn.GELU(),
                                 nn.Linear(d_latent * 2, d_latent))
        self.d_gate = d_gate
        if d_gate > 0:
            # top-down gate: coarse context -> per-latent gate in [0,1]
            self.gate = nn.Sequential(nn.Linear(d_gate, d_latent), nn.GELU(),
                                      nn.Linear(d_latent, n_latents), nn.Sigmoid())

    def forward(self, feat, gate_ctx=None):
        # feat: (B, C, H, W)
        B, C, H, W = feat.shape
        kv = feat.flatten(2).transpose(1, 2)          # (B, H*W, C)
        lat = self.latents.expand(B, -1, -1)          # (B, M, D)
        q = self.q_proj(lat)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        out, _ = self.attn(q, k, v, need_weights=False)  # (B, M, D)
        if self.d_gate > 0 and gate_ctx is not None:
            g = self.gate(gate_ctx).unsqueeze(-1)     # (B, M, 1)
            out = out * g  # top-down routing: gate which latents update
        lat = self.norm1(lat + out)
        lat = self.norm2(lat + self.mlp(lat))
        return lat


class PLANNet(nn.Module):
    """Pyramidal Latent Attention Network (Candidate A, novel).

    Coarse latents are updated first; they then GATE the value projection of
    the finer levels' cross-attention (top-down routing). Readout pools all
    three latent levels.
    """
    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()
        self.stem = ConvStem(in_ch)
        # coarse -> fine; gates flow top-down, so build coarse first in fwd
        self.lvl3 = LatentLevel(n_latents=16, d_latent=128, d_feat=128)          # <- F3 4x4
        self.lvl2 = LatentLevel(n_latents=32, d_latent=96, d_feat=64, d_gate=128)  # <- F2 7x7
        self.lvl1 = LatentLevel(n_latents=64, d_latent=64, d_feat=32, d_gate=96)   # <- F1 14x14
        self.head = nn.Sequential(
            nn.Linear(128 + 96 + 64, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes))

    def forward(self, x):
        f1, f2, f3 = self.stem(x)
        B = x.shape[0]
        l3 = self.lvl3(f3)                                   # coarse first
        ctx3 = l3.mean(dim=1)                                # (B,128)
        l2 = self.lvl2(f2, gate_ctx=ctx3)                    # gated by coarse
        ctx2 = l2.mean(dim=1)                                # (B,96)
        l1 = self.lvl1(f1, gate_ctx=ctx2)                    # gated by mid
        rep = torch.cat([l3.mean(1), l2.mean(1), l1.mean(1)], dim=1)
        return self.head(rep)


class GCViTLitePerceiverLite(nn.Module):
    """Faithful SMALL analog of Habib's GCViT+PerceiverIO hybrid.

    GCViT-lite: patch-embedded transformer with a global-context token per
    stage (global query attending to all patch tokens). Perceiver-lite: ONE
    flat latent array cross-attending to the final feature map (the flat
    bottleneck PLAN-Net replaces).
    """
    def __init__(self, in_ch=3, num_classes=2, dim=128, depth=4, heads=4,
                 n_latents=48):
        super().__init__()
        self.patch = nn.Conv2d(in_ch, dim, 4, stride=4)  # 28->7, 49 tokens
        self.cls_global = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, 50, dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True,
                                       activation="gelu")
            for _ in range(depth)
        ])
        # global-context readouts after each block (GCViT-style): reuse cls
        self.latents = nn.Parameter(torch.randn(1, n_latents, dim) * 0.02)
        self.pq = nn.Linear(dim, dim)
        self.pk = nn.Linear(dim, dim)
        self.pv = nn.Linear(dim, dim)
        self.pattn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.pnorm = nn.LayerNorm(dim)
        self.pmlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                  nn.Linear(dim * 2, dim))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        B = x.shape[0]
        t = self.patch(x).flatten(2).transpose(1, 2)          # (B,49,D)
        t = torch.cat([self.cls_global.expand(B, -1, -1), t], dim=1) + self.pos
        for blk in self.blocks:
            t = blk(t)
        lat = self.latents.expand(B, -1, -1)
        out, _ = self.pattn(self.pq(lat), self.pk(t), self.pv(t),
                            need_weights=False)
        lat = self.pnorm(lat + out)
        lat = lat + self.pmlp(lat)
        return self.head(lat.mean(dim=1))


class ResNet18Small(nn.Module):
    """CIFAR-style ResNet-18 (3x3 stride-1 stem) for 28x28."""
    def __init__(self, in_ch=3, num_classes=2, width=32):
        super().__init__()
        def block(ci, co, stride=1):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(),
                nn.Conv2d(co, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co))
        self.stem = nn.Sequential(nn.Conv2d(in_ch, width, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU())
        self.l1 = block(width, width);      self.s1 = nn.Identity()
        self.l2 = block(width, width * 2, 2)
        self.s2 = nn.Sequential(nn.Conv2d(width, width * 2, 1, 2, bias=False),
                                nn.BatchNorm2d(width * 2))
        self.l3 = block(width * 2, width * 4, 2)
        self.s3 = nn.Sequential(nn.Conv2d(width * 2, width * 4, 1, 2, bias=False),
                                nn.BatchNorm2d(width * 4))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = F.relu(self.l1(x) + self.s1(x))
        x = F.relu(self.l2(x) + self.s2(x))
        x = F.relu(self.l3(x) + self.s3(x))
        return self.fc(self.pool(x).flatten(1))


class ViTTiny(nn.Module):
    def __init__(self, in_ch=3, num_classes=2, dim=128, depth=4, heads=4, patch=4):
        super().__init__()
        self.patch = nn.Conv2d(in_ch, dim, patch, stride=patch)
        n = (28 // patch) ** 2
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True,
                                       activation="gelu")
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(B, -1, -1), t], dim=1) + self.pos
        for blk in self.blocks:
            t = blk(t)
        return self.head(self.norm(t[:, 0]))


def build_model(name, in_ch, num_classes):
    if name == "plannet":
        return PLANNet(in_ch, num_classes)
    if name == "gcvit-perceiver-lite":
        return GCViTLitePerceiverLite(in_ch, num_classes)
    if name == "resnet18":
        return ResNet18Small(in_ch, num_classes)
    if name == "vit-tiny":
        return ViTTiny(in_ch, num_classes)
    raise ValueError(name)
