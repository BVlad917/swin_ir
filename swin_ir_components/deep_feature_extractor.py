import torch
from torch import nn

from swin_ir_components.residual_swin_transformer_block import ResidualSwinTransformerBlock
from swin_v1_components.residual import Residual


class DeepFeatureExtractor(nn.Module):
    def __init__(self, input_dim, depths, heads, head_dim, window_size, drop_path_rate,
                 attn_drop_prob, proj_drop_prob):
        super().__init__()
        # input shape: (B, C, H, W)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.res_swin_transformer_blocks = nn.Sequential(
            # k residual swin transformer blocks
            *[ResidualSwinTransformerBlock(num_layers=depths[idx],
                                           input_dim=input_dim,
                                           heads=heads[idx],
                                           head_dim=head_dim,
                                           window_size=window_size,
                                           drop_path_probs=dpr[sum(depths[:idx]):sum(depths[:idx+1])],
                                           attn_drop_prob=attn_drop_prob,
                                           proj_drop_prob=proj_drop_prob)
              for idx in range(len(depths))],
            # (B, C, H, W)

            # one convolutional layer with 3x3 kernel and same padding at the end of the deep feature extractor
            nn.Conv2d(in_channels=input_dim,
                      out_channels=input_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1)
            # (B, C, H, W)
        )
        self.deep_feature_extractor = Residual(fn=self.combined, drop_path_prob=0)

    def forward(self, x):
        # input shape: (B, C, H, W)
        x = self.deep_feature_extractor(x)  # (B, C, H, W)
        return x
