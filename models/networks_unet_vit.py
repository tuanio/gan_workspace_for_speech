# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import torch
from torch import nn
import functools
import numpy as np
import copy
from .utils import get_activ_layer, get_norm_layer
from .networks_unet import UNet


class PositionWiseFFN(nn.Module):

    def __init__(self, features, ffn_features, activ = 'gelu', **kwargs):
        super().__init__(**kwargs)

        self.net = nn.Sequential(
            nn.Linear(features, ffn_features),
            get_activ_layer(activ),
            nn.Linear(ffn_features, features),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):

    def __init__(
        self, features, ffn_features, n_heads, activ = 'gelu', norm = None,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.norm1 = get_norm_layer(norm, features)
        self.atten = nn.MultiheadAttention(features, n_heads)

        self.norm2 = get_norm_layer(norm, features)
        self.ffn   = PositionWiseFFN(features, ffn_features, activ)

        self.rezero = rezero

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, x):
        # x: (L, N, features)

        # Step 1: Multi-Head Self Attention
        y1 = self.norm1(x)
        y1, _atten_weights = self.atten(y1, y1, y1)

        y  = x + self.re_alpha * y1

        # Step 2: PositionWise Feed Forward Network
        y2 = self.norm2(y)
        y2 = self.ffn(y2)

        y  = y + self.re_alpha * y2

        return y

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

class TransformerEncoder(nn.Module):

    def __init__(
        self, features, ffn_features, n_heads, n_blocks, activ, norm,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.encoder = nn.Sequential(*[
            TransformerBlock(
                features, ffn_features, n_heads, activ, norm, rezero
            ) for _ in range(n_blocks)
        ])

    def forward(self, x):
        # x : (N, L, features)

        # y : (L, N, features)
        y = x.permute((1, 0, 2))
        y = self.encoder(y)

        # result : (N, L, features)
        result = y.permute((1, 0, 2))

        return result

class FourierEmbedding(nn.Module):
    # arXiv: 2011.13775

    def __init__(self, features, height, width, **kwargs):
        super().__init__(**kwargs)
        self.projector = nn.Linear(2, features)
        self._height   = height
        self._width    = width

    def forward(self, y, x):
        # x : (N, L)
        # y : (N, L)
        x_norm = 2 * x / (self._width  - 1) - 1
        y_norm = 2 * y / (self._height - 1) - 1

        # z : (N, L, 2)
        z = torch.cat((x_norm.unsqueeze(2), y_norm.unsqueeze(2)), dim = 2)

        return torch.sin(self.projector(z))

class ViTInput(nn.Module):

    def __init__(
        self, input_features, embed_features, features, height, width,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._height   = height
        self._width    = width

        x = torch.arange(width).to(torch.float32)
        y = torch.arange(height).to(torch.float32)

        x, y   = torch.meshgrid(x, y)
        self.x = x.reshape((1, -1))
        self.y = y.reshape((1, -1))

        self.register_buffer('x_const', self.x)
        self.register_buffer('y_const', self.y)

        self.embed  = FourierEmbedding(embed_features, height, width)
        self.output = nn.Linear(embed_features + input_features, features)

    def forward(self, x):
        # x     : (N, L, input_features)
        # embed : (1, height * width, embed_features)
        #       = (1, L, embed_features)
        embed = self.embed(self.y_const, self.x_const)

        # embed : (1, L, embed_features)
        #      -> (N, L, embed_features)
        embed = embed.expand((x.shape[0], *embed.shape[1:]))

        # result : (N, L, embed_features + input_features)
        result = torch.cat([embed, x], dim = 2)

        # (N, L, features)
        return self.output(result)

class PixelwiseViT(nn.Module):

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, image_shape, rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.image_shape = image_shape

        self.trans_input = ViTInput(
            image_shape[0], embed_features, features,
            image_shape[1], image_shape[2],
        )

        self.encoder = TransformerEncoder(
            features, ffn_features, n_heads, n_blocks, activ, norm, rezero
        )

        self.trans_output = nn.Linear(features, image_shape[0])

    def forward(self, x):
        # x : (N, C, H, W)

        # itokens : (N, C, H * W)
        itokens = x.view(*x.shape[:2], -1)

        # itokens : (N, C,     H * W)
        #        -> (N, H * W, C    )
        #         = (N, L,     C)
        itokens = itokens.permute((0, 2, 1))

        # y : (N, L, features)
        y = self.trans_input(itokens)
        y = self.encoder(y)

        # otokens : (N, L, C)
        otokens = self.trans_output(y)

        # otokens : (N, L, C)
        #        -> (N, C, L)
        #         = (N, C, H * W)
        otokens = otokens.permute((0, 2, 1))

        # result : (N, C, H, W)
        result = otokens.view(*otokens.shape[:2], *self.image_shape[1:])

        return result

class MaskViTUNetGenerator(nn.Module):
    # (2.02, 2) -> conv kernel size 3 will become 129x128
    def __init__(
        self, in_c, out_c, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, image_shape, unet_features_list, unet_activ, unet_norm,
        unet_downsample = 'conv',
        unet_upsample   = 'upsample-conv',
        unet_rezero     = False,
        rezero          = True,
        activ_output    = None,
        up_scale_factors=(2, 2, 2, (2.02, 2)),
        **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.image_shape = image_shape

        self.net = UNet(
            in_c, out_c, unet_features_list, unet_activ, unet_norm, image_shape,
            unet_downsample, unet_upsample, unet_rezero,
            up_scale_factors=up_scale_factors
        )

        bottleneck = PixelwiseViT(
            features, n_heads, n_blocks, ffn_features, embed_features,
            activ, norm,
            image_shape = self.net.get_inner_shape(),
            rezero      = rezero
        )

        self.net.set_bottleneck(bottleneck)

        self.output = get_activ_layer(activ_output)

    def forward(self, x, m):
        # x : (N, C, H, W)
        x = torch.cat([x * m, m], dim=1)
        result = self.net(x)

        return self.output(result)
