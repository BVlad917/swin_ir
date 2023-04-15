import torch
from torch import nn


class HQImageReconstruction(nn.Module):
    def __init__(self, input_dim, scale, embed_dim):
        super().__init__()
        assert (scale & (scale - 1)) == 0, "ERROR: Can only upscale in powers of 2 (2x, 4x, etc...)"
        # convolutional layer before applying the up sample
        self.before_up_sample = nn.Sequential(
            nn.Conv2d(in_channels=input_dim,
                      out_channels=embed_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.LeakyReLU()
        )

        # up-sample: sequence of convolutional layers + pixel shuffle layers
        up_sample_layers = []
        for _ in range(int(torch.log2(torch.tensor(scale)))):
            up_sample_layers.append(nn.Conv2d(in_channels=embed_dim,
                                              out_channels=4 * embed_dim,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1))  # conv layer with same padding, 4x channels
            up_sample_layers.append(nn.PixelShuffle(upscale_factor=2))
        self.up_sample = nn.Sequential(*up_sample_layers)

        # convolutional layer after applying the up sample, maps to 3-channel RGB images
        self.after_up_sample = nn.Conv2d(in_channels=embed_dim,
                                         out_channels=3,  # 3-channel RGB output
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)  # conv layer with same padding

    def forward(self, x):
        # input shape: (B, C, H, W)
        x = self.before_up_sample(x)  # (B, EMBED_DIM, H, W)
        x = self.up_sample(x)  # (B, EMBED_DIM, H * SCALE, W * SCALE)
        x = self.after_up_sample(x)  # (B, 3, H * SCALE, W * SCALE)
        return x
