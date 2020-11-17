import torch.nn as nn


class BCLBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BCLBlock, self).__init__()
        self.bcr_block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.bcr_block(x)


class SimpleDiscriminator(nn.Module):
    def __init__(self, in_channel):
        super(SimpleDiscriminator, self).__init__()
        # height, width = input_size
        self.discriminator = nn.Sequential(
            self.build_simple_block(in_channel, 64),
            self.build_simple_block(64, 128),
            self.build_simple_block(128, 256),
            self.build_simple_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.discriminator(x)

    @staticmethod
    def build_simple_block(in_channel, out_channel):
        """
        Build the BRC Block
        Args:
            in_channel:
            out_channel:
        return:
        """
        return BCLBlock(in_channel, out_channel)


class DFactory:
    @staticmethod
    def build_discriminator():
        return SimpleDiscriminator(3)

