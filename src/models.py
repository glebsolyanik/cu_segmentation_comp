import torch
import torch.nn as nn


class BasicUnet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        channels = [in_channels, *[base_channels * 2**i for i in range(5)]]
        encoder_channels = channels[:-1]
        decoder_channels = channels[:0:-1]

        self.encoder = nn.ModuleList([
            conv_block(c_in, c_out) for c_in, c_out in zip(encoder_channels, encoder_channels[1:])
        ])
        self.downsamplers = nn.ModuleList([nn.MaxPool2d(kernel_size=2) for _ in self.encoder])

        self.bottleneck = conv_block(channels[-2], channels[-1])

        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)
            for c_in, c_out in zip(decoder_channels, decoder_channels[1:])
        ])
        self.decoder = nn.ModuleList([
            conv_block(c_in, c_out) for c_in, c_out in zip(decoder_channels, decoder_channels[1:])
        ])

        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encode, downsample in zip(self.encoder, self.downsamplers):
            x = encode(x)
            skips.append(x)
            x = downsample(x)

        x = self.bottleneck(x)

        for upsample, decode, skip in zip(self.upsamplers, self.decoder, reversed(skips)):
            x = upsample(x)
            x = decode(torch.cat([skip, x], dim=1))

        return self.out_conv(x)


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        Conv2dBNReLU(in_channels, out_channels),
        Conv2dBNReLU(out_channels, out_channels),
    )


class Conv2dBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
