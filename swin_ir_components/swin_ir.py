from torch import nn

from swin_ir_components.deep_feature_extractor import DeepFeatureExtractor
from swin_ir_components.hq_image_reconstruction import HQImageReconstruction
from swin_ir_components.shallow_feature_extractor import ShallowFeatureExtractor


class SwinIR(nn.Module):
    def __init__(self, embed_dim, depths, heads, head_dim, windows_size, drop_path_rate,
                 attn_drop_prob, proj_drop_prob, scale):
        super().__init__()
        self.shallow_feature_extractor = ShallowFeatureExtractor(in_channels=3,
                                                                 embed_dim=embed_dim)
        self.deep_feature_extractor = DeepFeatureExtractor(input_dim=embed_dim,
                                                           depths=depths,
                                                           heads=heads,
                                                           head_dim=head_dim,
                                                           window_size=windows_size,
                                                           drop_path_rate=drop_path_rate,
                                                           attn_drop_prob=attn_drop_prob,
                                                           proj_drop_prob=proj_drop_prob)
        self.hq_image_reconstruction = HQImageReconstruction(input_dim=embed_dim,
                                                             scale=scale,
                                                             embed_dim=embed_dim)

    def forward(self, x):
        # input shape: (B, 3, H, W)
        x = self.shallow_feature_extractor(x)  # (B, EMBED_DIM, H, W)
        x = self.deep_feature_extractor(x)  # (B, EMBED_DIM, H, W)
        x = self.hq_image_reconstruction(x)  # (B, 3, H * SCALE, W * SCALE)
        return x
