# modules.py
import torch
from torch import nn

class ConvBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, norm='instance', act='leaky', dropout_p=0.0):
        super().__init__()
        norm_layer = nn.InstanceNorm3d if norm == 'instance' else nn.BatchNorm3d
        act_layer = nn.LeakyReLU if act == 'leaky' else nn.ReLU
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch),
            act_layer(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch),
            act_layer(inplace=True),
            nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)

class DownBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.conv = ConvBlock3d(in_ch, out_ch, **kwargs)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.pool(x)
        return x, x_down

class UpBlock3d(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, **kwargs):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3d(out_ch + skip_ch, out_ch, **kwargs)

    def forward(self, x, skip):
        x = self.up(x)
        # handle odd tensor sizes
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        if dz != 0 or dy != 0 or dx != 0:
            x = nn.functional.pad(x, (0, max(0, dx), 0, max(0, dy), 0, max(0, dz)))
            x = x[:, :, :skip.size(2), :skip.size(3), :skip.size(4)]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=6, base_ch=16, depth=4, dropout_p=0.0):
        super().__init__()
        chs = [base_ch * (2 ** i) for i in range(depth)]
        self.downs = nn.ModuleList()
        prev = in_channels
        for c in chs:
            self.downs.append(DownBlock3d(prev, c, norm='instance', act='leaky', dropout_p=dropout_p))
            prev = c
        self.bottleneck = ConvBlock3d(chs[-1], chs[-1] * 2, norm='instance', act='leaky', dropout_p=dropout_p)
        ups = []
        in_ch = chs[-1] * 2
        for skip_c in reversed(chs):
            out_c = skip_c
            ups.append(UpBlock3d(in_ch, skip_c, out_c, norm='instance', act='leaky', dropout_p=dropout_p))
            in_ch = out_c
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv3d(chs[0], num_classes, kernel_size=1)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        skips = []
        out = x
        for d in self.downs:
            s, out = d(out)
            skips.append(s)
        out = self.bottleneck(out)
        for u, s in zip(self.ups, reversed(skips)):
            out = u(out, s)
        return self.head(out)
