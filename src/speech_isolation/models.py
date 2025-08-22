# Minimal model placeholders. Replace with real separation models (Conv-TasNet, Demucs, Open-Unmix, Spleeter, etc.)
import torch
import torch.nn as nn

class BaseSeparator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # input: (B, T)
        raise NotImplementedError()

class IdentitySeparator(BaseSeparator):
    def forward(self, x):
        # returns two channels: speech and background (naive split)
        return x.unsqueeze(1), torch.zeros_like(x).unsqueeze(1)


class Conv1dEncoder(nn.Module):
    def __init__(self, in_chan=1, out_chan=256, kernel_size=16, stride=8):
        super().__init__()
        self.conv = nn.Conv1d(in_chan, out_chan, kernel_size, stride=stride, bias=False)

    def forward(self, x):
        # x: (B, 1, T)
        return self.conv(x)


class Conv1dDecoder(nn.Module):
    def __init__(self, in_chan=256, out_chan=1, kernel_size=16, stride=8):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_chan, out_chan, kernel_size, stride=stride, bias=False)

    def forward(self, x):
        return self.deconv(x)


class TemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size-1)//2 * dilation, dilation=dilation)
        self.relu = nn.PReLU()
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        out = self.relu(out)
        return out + x


class SimpleTCN(nn.Module):
    def __init__(self, channels=256, n_blocks=4):
        super().__init__()
        layers = []
        for i in range(n_blocks):
            layers.append(TemporalBlock(channels, kernel_size=3, dilation=2**i))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvTasNetSmall(BaseSeparator):
    """A very small Conv-TasNet-like separator for sanity testing.

    Input: (B, T) float waveform. Output: (B, 2, T) separated sources.
    """
    def __init__(self, enc_channels=256, kernel_size=16, stride=8, n_sources=2):
        super().__init__()
        self.encoder = Conv1dEncoder(1, enc_channels, kernel_size, stride)
        self.separator = SimpleTCN(enc_channels, n_blocks=4)
        # mask conv to produce n_sources masks
        self.mask_conv = nn.Conv1d(enc_channels, enc_channels * n_sources, kernel_size=1)
        self.decoder = Conv1dDecoder(enc_channels, 1, kernel_size, stride)
        self.n_sources = n_sources

    def forward(self, x):
        # x: (B, T) or (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        enc = self.encoder(x)  # (B, C, L)
        sep = self.separator(enc)
        masks = self.mask_conv(sep)  # (B, C*n_sources, L)
        B, Cn, L = masks.shape
        C = Cn // self.n_sources
        masks = masks.view(B, self.n_sources, C, L)
        outputs = []
        for i in range(self.n_sources):
            masked = enc * torch.sigmoid(masks[:, i])
            dec = self.decoder(masked)  # (B, 1, T)
            outputs.append(dec)
        out = torch.cat(outputs, dim=1)  # (B, n_sources, T)
        # squeeze channel dim if needed
        return out
