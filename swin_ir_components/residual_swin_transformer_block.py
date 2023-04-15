from torch import nn

from swin_v1_components.swin_v1_transformer_block import SwinTransformerBlock


class ResidualSwinTransformerBlock(nn.Module):
    def __init__(self, num_layers, input_dim, heads, head_dim, window_size, drop_path_probs, attn_drop_prob,
                 proj_drop_prob):
        super().__init__()
        # checks
        assert num_layers % 2 == 0, "Stage layers need to be divisible by 2 for regular and shifted blocks"
        if isinstance(drop_path_probs, list):
            assert len(drop_path_probs) == num_layers, "Must give exactly one DropPath rate to each layer"

        dpp = drop_path_probs  # abbreviation, just to have clean code below
        self.swin_transformer_blocks = nn.Sequential(
            *[SwinTransformerBlock(input_dim=input_dim,
                                   heads=heads,
                                   head_dim=head_dim,
                                   window_size=window_size,
                                   shifted=(i % 2 == 1),
                                   mlp_dim=input_dim * 4,
                                   drop_path_prob=dpp[i] if isinstance(dpp, list) else dpp,
                                   attn_drop_prob=attn_drop_prob,
                                   proj_drop_prob=proj_drop_prob)
              for i in range(num_layers)]
        )
        self.conv = nn.Conv2d(in_channels=input_dim,
                              out_channels=input_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1)

        # custom weight initialization in the normalization layers
        self._init_norms()

    def forward(self, x):
        # input shape: (B, C, H, W)
        x = self.swin_transformer_blocks(x)  # (B, C, H, W)
        x = self.conv(x)  # (B, C, H, W)
        return x

    def _init_norms(self):
        for block in self.swin_transformer_blocks:
            nn.init.zeros_(block.norm1.norm.bias)
            nn.init.zeros_(block.norm1.norm.weight)
            nn.init.zeros_(block.norm2.norm.bias)
            nn.init.zeros_(block.norm2.norm.weight)
