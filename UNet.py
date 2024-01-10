import torch
from torch import nn
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encode(x)


class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=(2, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.encode = Encoder(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.encode(x)


class UNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # TODO your code here create all the UNET layers
        self.enc_1 = Encoder(in_channels=3, out_channels=64)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_2 = Encoder(in_channels=64, out_channels=128)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_3 = Encoder(in_channels=128, out_channels=256)
        self.max_pool_3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_4 = Encoder(in_channels=256, out_channels=512)
        self.max_pool_4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_5 = Encoder(in_channels=512, out_channels=1024)
        self.dec_1 = Decoder(in_channels=1024, out_channels=512)
        self.dec_2 = Decoder(in_channels=512, out_channels=256)
        self.dec_3 = Decoder(in_channels=256, out_channels=128)
        self.dec_4 = Decoder(in_channels=128, out_channels=64)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(1, 1))

    def forward(self, x):
        x1 = self.enc_1(x)
        x11 = self.max_pool_1(x1)
        x2 = self.enc_2(x11)
        x22 = self.max_pool_2(x2)
        x3 = self.enc_3(x22)
        x33 = self.max_pool_3(x3)
        x4 = self.enc_4(x33)
        x44 = self.max_pool_4(x4)
        x5 = self.enc_5(x44)
        x = self.dec_1(x5, x4)
        x = self.dec_2(x, x3)
        x = self.dec_3(x, x2)
        x = self.dec_4(x, x1)
        logits = self.out_conv(x)
        return logits

