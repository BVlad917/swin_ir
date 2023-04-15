import torch
from torch import nn

from swin_ir_components.residual_swin_transformer_block import ResidualSwinTransformerBlock
from swin_v1_components.residual import Residual


class DeepFeatureExtractor(nn.Module):
    def __init__(self, depths, input_dim, heads, head_dim, window_size, drop_path_rate,
                 attn_drop_prob, proj_drop_prob):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.res_swin_transformer_blocks = nn.Sequential(
            *[ResidualSwinTransformerBlock(num_layers=depths[idx],
                                           input_dim=input_dim,
                                           heads=heads,
                                           head_dim=head_dim,
                                           window_size=window_size,
                                           drop_path_probs=dpr[sum(depths[:idx]):sum(depths[:idx+1])],
                                           attn_drop_prob=attn_drop_prob,
                                           proj_drop_prob=proj_drop_prob)
              for idx in range(len(depths))]
        )
        self.conv = nn.Conv2d(in_channels=input_dim,
                              out_channels=input_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.combined = nn.Sequential(
            self.res_swin_transformer_blocks,
            self.conv
        )
        self.deep_feature_extractor = Residual(fn=self.combined, drop_path_prob=0)

    def forward(self, x):
        # input shape: (B, C, H, W)
        x = self.deep_feature_extractor(x)  # (B, C, H, W)
        return x
