import math
import torch
from torch import nn
import torch.nn.functional as F


class FRN3d(nn.Module):
    """
    Filter Response Normalization (FRN) for 3D tensors.
    Normalizes across spatial and channel dimensions within a single sample (batch-size independent).
    """
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Gamma and Beta (learnable scale and shift)
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
    
    def forward(self, x):
        # Compute the mean of the squared feature map (across spatial dimensions)
        # Dimensions are [N, C, D, H, W] -> sum/mean over D, H, W
        nu2 = torch.mean(x * x, dim=[2, 3, 4], keepdim=True)
        # Normalize: x / sqrt(E[x^2] + eps)
        x = x * torch.rsqrt(nu2 + self.eps)
        # Scale and Shift: Gamma * x + Beta
        return x * self.weight + self.bias


class TLU3d(nn.Module):
    """
    Thresholded Linear Unit (TLU) for 3D tensors.
    Used in conjunction with FRN. TLU has a learnable bias (tau) 
    to shift the activation function, preventing the features from vanishing.
    """
    def __init__(self, num_features: int):
        super().__init__()
        # Tau (learnable threshold)
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))

    def forward(self, x):
        # TLU: max(x, tau)
        return torch.max(x, self.tau)



class ChannelAttention3d(nn.Module):
    """CBAM Channel attention for 3D feature maps."""
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
    """CBAM Spatial attention for 3D feature maps with 7x7x7 kernel."""
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
    """Convolutional Block Attention Module (channel -> spatial) for 3D tensors."""
    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention3d(channels, reduction=reduction)
        self.sa = SpatialAttention3d(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ImprovedResidualBlock3d(nn.Module):
    """
    Improved Stage Residual Block: FRN -> TLU -> 3x3 -> FRN -> TLU -> 3x3 with identity mapping.
    This replaces the original BN+DyReLU for batch-size independence.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        
        # FRN and TLU replacement
        self.norm1 = FRN3d(in_ch)
        self.act1 = TLU3d(in_ch) 
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

        # FRN and TLU replacement
        self.norm2 = FRN3d(out_ch)
        self.act2 = TLU3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)

        self.drop = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        
        # Forward pass using FRN -> TLU
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.drop(out)
        out = self.conv2(self.act2(self.norm2(out)))
        
        out = out + identity
        return out



class DownStage3d(nn.Module):
    """Encoder stage: improved residual block followed by 2x downsampling via max pooling."""
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        # Uses the new ImprovedResidualBlock3d with FRN/TLU
        self.block = ImprovedResidualBlock3d(in_ch, out_ch, dropout_p=dropout_p)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        feat = self.block(x)
        down = self.pool(feat)
        return feat, down


class UpStage3d(nn.Module):
    """Decoder stage: transposed conv upsample, CBAM on skip features, concatenate, improved residual block."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout_p: float = 0.0, cbam_reduction: int = 8):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.cbam = CBAM3d(skip_ch, reduction=cbam_reduction)
        # Note: The input to the block is (out_ch + skip_ch) after concatenation
        self.block = ImprovedResidualBlock3d(out_ch + skip_ch, out_ch, dropout_p=dropout_p)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad upsampled feature map (x) to match skip connection size
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
    IResUnet3+ 3D Architecture:
    - Stage Residual Encoder/Decoder Blocks using FRN and TLU
    - CBAM applied to skip features before fusion
    - Standard U-Net skip connections (concatenation)
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 6, base_ch: int = 16, depth: int = 4, dropout_p: float = 0.0):
        super().__init__()
        chs = [base_ch * (2 ** i) for i in range(depth)]
        
        # Encoder (Contracting Path)
        self.enc = nn.ModuleList()
        prev = in_channels
        for c in chs:
            self.enc.append(DownStage3d(prev, c, dropout_p=dropout_p))
            prev = c

        # Bridge (Bottleneck)
        self.bridge = ImprovedResidualBlock3d(chs[-1], chs[-1] * 2, dropout_p=dropout_p)
        
        # Decoder (Expanding Path)
        dec = []
        in_ch = chs[-1] * 2
        for skip_c in reversed(chs):
            out_c = skip_c
            dec.append(UpStage3d(in_ch, skip_c, out_c, dropout_p=dropout_p))
            in_ch = out_c
        self.dec = nn.ModuleList(dec)
        
        # Final Output Head
        self.head = nn.Conv3d(chs[0], num_classes, kernel_size=1, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            # Initializing with Kaiming Normal (He initialization)
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, (FRN3d,)):
            # FRN weight (Gamma) to ones, bias (Beta) to zeros
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (TLU3d,)):
            # TLU bias (Tau) to zeros
            if m.tau is not None:
                nn.init.zeros_(m.tau)
        elif isinstance(m, (nn.Linear,)):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
    
    def forward(self, x):
        skips = []
        out = x
        
        # Encoder
        for stage in self.enc:
            feat, out = stage(out)
            skips.append(feat)
            
        # Bridge
        out = self.bridge(out)
        
        # Decoder
        for stage, skip in zip(self.dec, reversed(skips)):
            out = stage(out, skip)
            
        # Head
        return self.head(out)