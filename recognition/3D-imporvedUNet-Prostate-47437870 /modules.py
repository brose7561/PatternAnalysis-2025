"""
modules.py — Core network components for the 3D Improved U-Net model.

Includes:
- DyReLU3d: Dynamic activation
- CBAM3d: Channel & spatial attention
- ImprovedResidualBlock3d: Enhanced ResNet-style blocks
- Encoder/Decoder stages and UNet3DImproved model

Author: Benjamin Rose
Date: 2025
Project: 3D Improved U-Net (Prostate MRI Segmentation)
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


class DyReLU3d(nn.Module):
    """
    Dynamic ReLU (Dy-ReLU) for 3D tensors.
    Learns per-channel piecewise-linear activation parameters from global context.
    Uses K=2 segments: y = max_k(a_k * x + b_k).
    """
    def __init__(self, channels: int, reduction: int = 16, K: int = 2):
        super().__init__()
        self.C = channels
        self.K = K
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 2 * K * channels)  # outputs [a1..aK, b1..bK]
        # base parameters to stabilize training
        self.register_parameter("alpha", nn.Parameter(torch.ones(K)))
        self.register_parameter("beta", nn.Parameter(torch.zeros(K)))
        self.register_parameter("lambda_a", nn.Parameter(torch.tensor(0.5)))
        self.register_parameter("lambda_b", nn.Parameter(torch.tensor(0.5)))

    def forward(self, x):
        n, c, d, h, w = x.shape
        g = self.pool(x).view(n, c)
        g = F.relu(self.fc1(g), inplace=True)
        params = torch.sigmoid(self.fc2(g))
        a, b = torch.split(params, self.K * self.C, dim=1)
        a = a.view(n, self.K, self.C, 1, 1, 1)
        b = b.view(n, self.K, self.C, 1, 1, 1)
        # adjust around base alpha/beta
        a = self.alpha.view(1, self.K, 1, 1, 1, 1) + self.lambda_a * (a - 0.5)
        b = self.beta.view(1, self.K, 1, 1, 1, 1) + self.lambda_b * (b - 0.5)
        x_exp = x.unsqueeze(1)
        out = (a * x_exp + b).max(dim=1).values
        return out


class FRN3d(nn.Module):
    """Filter Response Normalization (FRN) for 3D tensors."""
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
    
    def forward(self, x):
        nu2 = torch.mean(x * x, dim=[2, 3, 4], keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps)
        return x * self.weight + self.bias


# TLU3d is removed as it's being replaced by DyReLU3d in the block


class ChannelAttention3d(nn.Module):
    """CBAM channel attention module for 3D inputs."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels)
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = torch.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1, 1)
        return x * scale


class SpatialAttention3d(nn.Module):
    """CBAM spatial attention with a 3D conv filter."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        mmax = x.max(dim=1, keepdim=True).values
        attn = torch.sigmoid(self.conv(torch.cat([mean, mmax], dim=1)))
        return x * attn


class CBAM3d(nn.Module):
    """Full CBAM block: channel then spatial attention."""
    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention3d(channels, reduction=reduction)
        self.sa = SpatialAttention3d(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ImprovedResidualBlock3d(nn.Module):
    """Residual block using DyReLU and BN layers with optional dropout."""
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(in_ch)
        self.act1 = DyReLU3d(in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)

        self.norm2 = FRN3d(out_ch)
        self.act2 = DyReLU3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)

        self.drop = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.proj = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.drop(out)
        out = self.conv2(self.act2(self.norm2(out)))
        
        out = out + identity
        return out


class DownStage3d(nn.Module):
    """Encoder stage: residual block followed by 2× downsampling."""
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.block = ImprovedResidualBlock3d(in_ch, out_ch, dropout_p=dropout_p)
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x):
        feat = self.block(x)
        down = self.pool(feat)
        return feat, down


class UpStage3d(nn.Module):
    """Decoder stage: upsample, CBAM on skip, concat, then residual block."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout_p: float = 0.0, cbam_reduction: int = 8):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2, bias=False)
        self.cbam = CBAM3d(skip_ch, reduction=cbam_reduction)
        self.block = ImprovedResidualBlock3d(out_ch + skip_ch, out_ch, dropout_p=dropout_p)

    def forward(self, x, skip):
        x = self.up(x)
        # adjust for size mismatch
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        if dz != 0 or dy != 0 or dx != 0:
            x = F.pad(x, (0, max(0, dx), 0, max(0, dy), 0, max(0, dz)))
            x = x[:, :, :skip.size(2), :skip.size(3), :skip.size(4)]
        
        skip = self.cbam(skip)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNet3DImproved(nn.Module):
    """
    Improved 3D U-Net with:
    - DyReLU-based residual blocks
    - CBAM attention on skip connections
    - Encoder-decoder symmetry with bridge and 1×1 head
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 6, base_ch: int = 16, depth: int = 4, dropout_p: float = 0.0):
        super().__init__()
        chs = [base_ch * (2 ** i) for i in range(depth)]

        # Encoder
        self.enc = nn.ModuleList()
        prev = in_channels
        for c in chs:
            self.enc.append(DownStage3d(prev, c, dropout_p=dropout_p))
            prev = c

        # Bottleneck
        self.bridge = ImprovedResidualBlock3d(chs[-1], chs[-1] * 2, dropout_p=dropout_p)

        # Decoder
        dec = []
        in_ch = chs[-1] * 2
        for skip_c in reversed(chs):
            dec.append(UpStage3d(in_ch, skip_c, skip_c, dropout_p=dropout_p))
            in_ch = skip_c
        self.dec = nn.ModuleList(dec)

        # Classification head
        self.head = nn.Conv3d(chs[0], num_classes, 1, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Kaiming init for conv/linear layers, unit init for BN."""
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
    
    def forward(self, x):
        """Forward pass through encoder, bridge, and decoder."""
        skips = []
        out = x
        
        for stage in self.enc:
            feat, out = stage(out)
            skips.append(feat)
            
        out = self.bridge(out)
        
        for stage, skip in zip(self.dec, reversed(skips)):
            out = stage(out, skip)
            
        return self.head(out)