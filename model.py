import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


# ── Генератор ────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Дилатированный остаточный блок с несколькими уровнями дилатации."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.c1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d,
                                  padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.c2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=1,
                                  padding=get_padding(kernel_size, 1)))
            for _ in dilation
        ])

    def forward(self, x):
        for a, b in zip(self.c1, self.c2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = a(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = b(xt)
            x = x + xt
        return x

    def remove_weight_norm(self):
        for c in list(self.c1) + list(self.c2):
            remove_parametrizations(c, "weight")


class Generator(nn.Module):
    """
    HiFiGAN генератор.
    Вход:  [B, num_mels, T_mel]
    Выход: [B, 1, T_audio]
    """

    def __init__(self, cfg):
        super().__init__()
        self.nk = len(cfg.resblock_kernel_sizes)
        ch = cfg.upsample_initial_channel

        self.pre = weight_norm(nn.Conv1d(cfg.num_mels, ch, 7, padding=3))

        self.ups = nn.ModuleList()
        self.res = nn.ModuleList()

        for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
            out_ch = ch // (2 ** (i + 1))
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(ch // (2 ** i), out_ch, k, stride=u, padding=(k - u) // 2)
            ))
            for kr, dr in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
                self.res.append(ResBlock(out_ch, kr, dr))

        self.post = weight_norm(nn.Conv1d(out_ch, 1, 7, padding=3))

    def forward(self, x):
        x = self.pre(x)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            # MRF: усредняем параллельные ResBlock-и
            x = sum(self.res[i * self.nk + j](x) for j in range(self.nk)) / self.nk
        x = F.leaky_relu(x, LRELU_SLOPE)
        return torch.tanh(self.post(x))

    def remove_weight_norm(self):
        remove_parametrizations(self.pre, "weight")
        for up in self.ups:
            remove_parametrizations(up, "weight")
        for rb in self.res:
            rb.remove_weight_norm()
        remove_parametrizations(self.post, "weight")


# ── Дискриминаторы ────────────────────────────────────────────────────────────

class PeriodDiscriminator(nn.Module):
    """Один суб-дискриминатор в Multi-Period Discriminator."""

    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        norm = weight_norm

        self.convs = nn.ModuleList([
            norm(nn.Conv2d(1,    32,   (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm(nn.Conv2d(32,   128,  (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm(nn.Conv2d(128,  512,  (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm(nn.Conv2d(512,  1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1,           padding=(2, 0))),
        ])
        self.conv_post = norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        # дополняем до кратности периоду
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), "reflect")
            t = t + pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, y_real, y_fake):
        real_outs, fake_outs, real_fmaps, fake_fmaps = [], [], [], []
        for d in self.discriminators:
            r_out, r_fmap = d(y_real)
            f_out, f_fmap = d(y_fake)
            real_outs.append(r_out)
            fake_outs.append(f_out)
            real_fmaps.append(r_fmap)
            fake_fmaps.append(f_fmap)
        return real_outs, fake_outs, real_fmaps, fake_fmaps


class ScaleDiscriminator(nn.Module):
    """Один суб-дискриминатор в Multi-Scale Discriminator."""

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1,    128,  15, 1, padding=7)),
            norm(nn.Conv1d(128,  128,  41, 2, groups=4,  padding=20)),
            norm(nn.Conv1d(128,  256,  41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256,  512,  41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512,  1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5,  1, padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, y_real, y_fake):
        real_outs, fake_outs, real_fmaps, fake_fmaps = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y_real = self.pools[i - 1](y_real)
                y_fake = self.pools[i - 1](y_fake)
            r_out, r_fmap = d(y_real)
            f_out, f_fmap = d(y_fake)
            real_outs.append(r_out)
            fake_outs.append(f_out)
            real_fmaps.append(r_fmap)
            fake_fmaps.append(f_fmap)
        return real_outs, fake_outs, real_fmaps, fake_fmaps
