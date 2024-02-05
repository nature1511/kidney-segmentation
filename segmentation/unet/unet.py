import torch
from torch import nn
from ..unet.parts import DoubleConv, Down, Up_conv, Attention_block


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        super(UNet, self).__init__()
        n1 = 64
        size_channels = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.inc = DoubleConv(n_channels, size_channels[0])
        self.down1 = Down(size_channels[0], size_channels[1])
        self.down2 = Down(size_channels[1], size_channels[2])
        self.down3 = Down(size_channels[2], size_channels[3])

        self.down4 = Down(size_channels[3], size_channels[4])

        self.up5 = Up_conv(size_channels[4], size_channels[3], bilinear)
        self.up_conv5 = DoubleConv(size_channels[4], size_channels[3])

        self.up4 = Up_conv(size_channels[3], size_channels[2], bilinear)
        self.up_conv4 = DoubleConv(size_channels[3], size_channels[2])

        self.up3 = Up_conv(size_channels[2], size_channels[1], bilinear)
        self.up_conv3 = DoubleConv(size_channels[2], size_channels[1])

        self.up2 = Up_conv(size_channels[1], size_channels[0], bilinear)
        self.up_conv2 = DoubleConv(size_channels[1], size_channels[0])

        self.conv1x1 = nn.Conv2d(
            size_channels[0], n_classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        # the numbering of variables d corresponds to the numbering of convolutions
        d1 = self.inc(x)  # 64-1

        d2 = self.down1(d1)  # 128-2

        d3 = self.down2(d2)  # 256 - 3
        d4 = self.down3(d3)  # 512 - 4
        d5 = self.down4(d4)  # 1024 - 5

        up5 = self.up5(d5)  # 512
        up5 = torch.cat([d4, up5], dim=1)  # 1024
        up5 = self.up_conv5(up5)  # 512

        up4 = self.up4(up5)  # 256
        up4 = torch.cat([d3, up4], dim=1)  # 512
        up4 = self.up_conv4(up4)  # 256

        up3 = self.up3(up4)  # 128
        up3 = torch.cat([d2, up3], dim=1)  # 256
        up3 = self.up_conv3(up3)  # 128

        up2 = self.up2(up3)  # 64
        up2 = torch.cat([d1, up2], dim=1)  # 128
        up2 = self.up_conv2(up2)  # 64

        out = self.conv1x1(up2)

        return out


class AttU_Net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(AttU_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        n1 = 32
        size_channels = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.inc = DoubleConv(n_channels, size_channels[0])
        self.down1 = Down(size_channels[0], size_channels[1])
        self.down2 = Down(size_channels[1], size_channels[2])
        self.down3 = Down(size_channels[2], size_channels[3])

        self.down4 = Down(size_channels[3], size_channels[4])

        self.att5 = Attention_block(
            size_channels[4], size_channels[3], size_channels[2]
        )
        self.up5 = Up_conv(size_channels[4], size_channels[3], bilinear)
        self.up_conv5 = DoubleConv(size_channels[4], size_channels[3])

        self.att4 = Attention_block(
            size_channels[3], size_channels[2], size_channels[1]
        )
        self.up4 = Up_conv(size_channels[3], size_channels[2], bilinear)
        self.up_conv4 = DoubleConv(size_channels[3], size_channels[2])

        self.att3 = Attention_block(
            size_channels[2], size_channels[1], size_channels[0]
        )
        self.up3 = Up_conv(size_channels[2], size_channels[1], bilinear)
        self.up_conv3 = DoubleConv(size_channels[2], size_channels[1])

        self.att2 = Attention_block(
            size_channels[1], size_channels[0], size_channels[0] // 2
        )
        self.up2 = Up_conv(size_channels[1], size_channels[0], bilinear)
        self.up_conv2 = DoubleConv(size_channels[1], size_channels[0])

        self.conv1x1 = nn.Conv2d(
            size_channels[0], n_classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        # the numbering of variables d corresponds to the numbering of convolutions
        d1 = self.inc(x)  # 64-1

        d2 = self.down1(d1)  # 128-2

        d3 = self.down2(d2)  # 256 - 3
        d4 = self.down3(d3)  # 512 - 4
        d5 = self.down4(d4)  # 1024 - 5

        att5 = self.att5(d5, d4)
        up5 = self.up5(d5)
        up5 = torch.cat([att5, up5], dim=1)
        up5 = self.up_conv5(up5)

        att4 = self.att4(up5, d3)
        up4 = self.up4(up5)
        up4 = torch.cat([att4, up4], dim=1)
        up4 = self.up_conv4(up4)

        att3 = self.att3(up4, d2)
        up3 = self.up3(up4)
        up3 = torch.cat([att3, up3], dim=1)
        up3 = self.up_conv3(up3)

        # print(up3.shape, d1.shape)
        att2 = self.att2(up3, d1)
        up2 = self.up2(up3)  # 64
        up2 = torch.cat([d1, up2], dim=1)  # 128
        up2 = self.up_conv2(up2)  # 64

        out = self.conv1x1(up2)

        return out
