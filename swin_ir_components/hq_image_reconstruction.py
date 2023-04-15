import torch
from torch import nn


class HQImageReconstruction(nn.Sequential):
    def __init__(self, input_dim, scale, embed_dim):
        assert (scale & (scale - 1)) != 0, "ERROR: Can only upscale in powers of 2 (2x, 4x, etc...)"

        before_up_sample = nn.Sequential(
            nn.Conv2d(in_channels=input_dim,
                      out_channels=embed_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.LeakyReLU()
        )

        up_sample = []
        for _ in range(torch.log2(scale)):
            up_sample.append(nn.Conv2d(in_channels=embed_dim,
                                       out_channels=4*embed_dim,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1))  # conv layer with same padding, 4x channels
            up_sample.append(nn.PixelShuffle(upscale_factor=2))

        after_up_sample = nn.Conv2d(in_channels=embed_dim,
                                    out_channels=3,  # 3-channel RGB output
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)  # conv layer with same padding

        modules = [before_up_sample] + up_sample + [after_up_sample]
        super().__init__(*modules)
