import lighter
import torch
import torch.nn as nn


class ResidualBlock(lighter.Model):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class ResUNet(lighter.Model):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()

        #  encoder
        self.in_conv = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.enc1 = ResidualBlock(features)

        self.down1 = nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = ResidualBlock(features * 2)

        self.down2 = nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1)
        self.enc3 = ResidualBlock(features * 4)

        #  bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(features * 4),
            ResidualBlock(features * 4)
        )

        #  decoder
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(features * 2)

        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(features)

        #  reconstruction
        self.fuse2 = nn.Conv2d(features * 4, features * 2, kernel_size=1)
        self.fuse1 = nn.Conv2d(features * 2, features, kernel_size=1)

        self.out_conv = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        s1 = self.in_conv(x)
        s1 = self.enc1(s1)

        s2 = self.down1(s1)
        s2 = self.enc2(s2)

        s3 = self.down2(s2)
        s3 = self.enc3(s3)

        b = self.bottleneck(s3)

        u2 = self.up2(b)
        u2 = torch.cat([u2, s2], dim=1)
        u2 = self.fuse2(u2)
        u2 = self.dec3(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, s1], dim=1)
        u1 = self.fuse1(u1)
        u1 = self.dec2(u1)

        res_output = self.out_conv(u1)

        return torch.clamp(x + res_output, 0.0, 1.0)


if __name__ == "__main__":
    model = ResUNet()
    input_tensor = torch.randn(1, 3, 128, 128)
    output_tensor = model(input_tensor)
    print(input_tensor.shape)
    print(output_tensor.shape)







