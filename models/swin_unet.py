import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer


class SwinUNet(nn.Module):
    """Swin Transformer based network for image segmentation.

    This module repurposes the classification Swin Transformer as an encoder and
    adds a lightweight decoder that upsamples features back to the input
    resolution for dense prediction tasks such as brain tumor segmentation.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=2,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        fused_window_process=False,
    ):
        super().__init__()

        self.backbone = SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            fused_window_process=fused_window_process,
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(self.backbone.num_features, self.backbone.num_features // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.backbone.num_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.backbone.num_features // 2, num_classes, kernel_size=1),
        )

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.backbone.patch_embed(x)
        if self.backbone.ape:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)
        for layer in self.backbone.layers:
            x = layer(x)
        x = self.backbone.norm(x)

        B, L, C = x.shape
        H_feat, W_feat = self.backbone.patch_embed.patches_resolution
        for _ in range(len(self.backbone.layers) - 1):
            H_feat //= 2
            W_feat //= 2
        x = x.view(B, H_feat, W_feat, C).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=self.backbone.patch_embed.patches_resolution, mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        x = self.decoder(x)
        return x
