from torch import nn


class ShallowFeatureExtractor(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        # Conv layer with "same" padding
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=embed_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1)

    def forward(self, x):
        # input shape: (B, C, H, W)
        x = self.conv(x)  # (B, EMBED_DIM, H, W)
        return x
