# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import torch
from torch import nn
import functools
import numpy as np
import copy

def extract_name_kwargs(obj):
    if isinstance(obj, dict):
        obj    = copy.copy(obj)
        name   = obj.pop('name')
        kwargs = obj
    else:
        name   = obj
        kwargs = {}

    return (name, kwargs)

def get_norm_layer(norm, features):
    name, kwargs = extract_name_kwargs(norm)

    if name is None:
        return nn.Identity(**kwargs)

    if name == 'layer':
        return nn.LayerNorm((features,), **kwargs)

    if name == 'batch':
        return nn.BatchNorm2d(features, **kwargs)

    if name == 'instance':
        return nn.InstanceNorm2d(features, **kwargs)

    raise ValueError("Unknown Layer: '%s'" % name)

def get_norm_layer_fn(norm):
    return lambda features : get_norm_layer(norm, features)

def get_activ_layer(activ):
    name, kwargs = extract_name_kwargs(activ)

    if (name is None) or (name == 'linear'):
        return nn.Identity()

    if name == 'gelu':
        return nn.GELU(**kwargs)

    if name == 'relu':
        return nn.ReLU(**kwargs)

    if name == 'leakyrelu':
        return nn.LeakyReLU(**kwargs)

    if name == 'tanh':
        return nn.Tanh()

    if name == 'sigmoid':
        return nn.Sigmoid()

    raise ValueError("Unknown activation: '%s'" % name)

def select_optimizer(parameters, optimizer):
    name, kwargs = extract_name_kwargs(optimizer)

    if name == 'AdamW':
        return torch.optim.AdamW(parameters, **kwargs)

    if name == 'Adam':
        return torch.optim.Adam(parameters, **kwargs)

    raise ValueError("Unknown optimizer: '%s'" % name)

def select_loss(loss):
    name, kwargs = extract_name_kwargs(loss)

    if name.lower() in [ 'l1', 'mae' ]:
        return nn.L1Loss(**kwargs)

    if name.lower() in [ 'l2', 'mse' ]:
        return nn.MSELoss(**kwargs)

    raise ValueError("Unknown loss: '%s'" % name)

def calc_tokenized_size(image_shape, token_size):
    # image_shape : (C, H, W)
    # token_size  : (H_t, W_t)
    if image_shape[1] % token_size[0] != 0:
        raise ValueError(
            "Token width %d does not divide image width %d" % (
                token_size[0], image_shape[1]
            )
        )

    if image_shape[2] % token_size[1] != 0:
        raise ValueError(
            "Token height %d does not divide image height %d" % (
                token_size[1], image_shape[2]
            )
        )

    # result : (N_h, N_w)
    return (image_shape[1] // token_size[0], image_shape[2] // token_size[1])

def img_to_tokens(image_batch, token_size):
    # image_batch : (N, C, H, W)
    # token_size  : (H_t, W_t)

    # result : (N, C, N_h, H_t, W)
    result = image_batch.view(
        (*image_batch.shape[:2], -1, token_size[0], image_batch.shape[3])
    )

    # result : (N, C, N_h, H_t, W       )
    #       -> (N, C, N_h, H_t, N_w, W_t)
    result = result.view((*result.shape[:4], -1, token_size[1]))

    # result : (N, C, N_h, H_t, N_w, W_t)
    #       -> (N, N_h, N_w, C, H_t, W_t)
    result = result.permute((0, 2, 4, 1, 3, 5))

    return result

def img_from_tokens(tokens):
    # tokens : (N, N_h, N_w, C, H_t, W_t)
    # result : (N, C, N_h, H_t, N_w, W_t)
    result = tokens.permute((0, 3, 1, 4, 2, 5))

    # result : (N, C, N_h, H_t, N_w, W_t)
    #       -> (N, C, N_h, H_t, N_w * W_t)
    #        = (N, C, N_h, H_t, W)
    result = result.reshape((*result.shape[:4], -1))

    # result : (N, C, N_h, H_t, W)
    #       -> (N, C, N_h * H_t, W)
    #        = (N, C, H, W)
    result = result.reshape((*result.shape[:2], -1, result.shape[4]))

    return result

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

def get_downsample_x2_conv2_layer(features, **kwargs):
    return (
        nn.Conv2d(features, features, kernel_size = 2, stride = 2, **kwargs),
        features
    )

def get_downsample_x2_conv3_layer(features, **kwargs):
    return (
        nn.Conv2d(
            features, features, kernel_size = 3, stride = 2, padding = 1,
            **kwargs
        ),
        features
    )

def get_downsample_x2_pixelshuffle_layer(features, **kwargs):
    out_features = 4 * features
    return (nn.PixelUnshuffle(downscale_factor = 2, **kwargs), out_features)

def get_downsample_x2_pixelshuffle_conv_layer(features, **kwargs):
    out_features = features * 4

    layer = nn.Sequential(
        nn.PixelUnshuffle(downscale_factor = 2, **kwargs),
        nn.Conv2d(
            out_features, out_features, kernel_size = 3, padding = 1
        ),
    )

    return (layer, out_features)

def get_upsample_x2_deconv2_layer(features, **kwargs):
    return (
        nn.ConvTranspose2d(
            features, features, kernel_size = 2, stride = 2, **kwargs
        ),
        features
    )

def get_upsample_x2_upconv_layer(features, scale_factor=2, **kwargs):
    layer = nn.Sequential(
        nn.Upsample(scale_factor = scale_factor, **kwargs),
        nn.Conv2d(features, features, kernel_size = 3, padding = 1),
    )

    return (layer, features)

def get_upsample_x2_pixelshuffle_conv_layer(features, **kwargs):
    out_features = features // 4

    layer = nn.Sequential(
        nn.PixelShuffle(upscale_factor = 2, **kwargs),
        nn.Conv2d(out_features, out_features, kernel_size = 3, padding = 1),
    )

    return (layer, out_features)

def get_downsample_x2_layer(layer, features):
    name, kwargs = extract_name_kwargs(layer)

    if name == 'conv':
        return get_downsample_x2_conv2_layer(features, **kwargs)

    if name == 'conv3':
        return get_downsample_x2_conv3_layer(features, **kwargs)

    if name == 'avgpool':
        return (nn.AvgPool2d(kernel_size = 2, stride = 2, **kwargs), features)

    if name == 'maxpool':
        return (nn.MaxPool2d(kernel_size = 2, stride = 2, **kwargs), features)

    if name == 'pixel-unshuffle':
        return get_downsample_x2_pixelshuffle_layer(features, **kwargs)

    if name == 'pixel-unshuffle-conv':
        return get_downsample_x2_pixelshuffle_conv_layer(features, **kwargs)

    raise ValueError("Unknown Downsample Layer: '%s'" % name)

def get_upsample_x2_layer(layer, features, scale_factor=2):
    name, kwargs = extract_name_kwargs(layer)

    if name == 'deconv':
        return get_upsample_x2_deconv2_layer(features, **kwargs)

    if name == 'upsample':
        return (nn.Upsample(scale_factor = 2, **kwargs), features)

    if name == 'upsample-conv':
        return get_upsample_x2_upconv_layer(features, scale_factor=scale_factor, **kwargs)

    if name == 'pixel-shuffle':
        return (nn.PixelShuffle(upscale_factor = 2, **kwargs), features // 4)

    if name == 'pixel-shuffle-conv':
        return get_upsample_x2_pixelshuffle_conv_layer(features, **kwargs)

    raise ValueError("Unknown Upsample Layer: '%s'" % name)

class UnetBasicBlock(nn.Module):

    def __init__(
        self, in_features, out_features, activ, norm, mid_features = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if mid_features is None:
            mid_features = out_features

        self.block = nn.Sequential(
            get_norm_layer(norm, in_features),
            nn.Conv2d(in_features, mid_features, kernel_size = 3, padding = 1),
            get_activ_layer(activ),

            get_norm_layer(norm, mid_features),
            nn.Conv2d(
                mid_features, out_features, kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),
        )

    def forward(self, x):
        return self.block(x)

class UNetEncBlock(nn.Module):

    def __init__(
        self, features, activ, norm, downsample, input_shape, **kwargs
    ):
        super().__init__(**kwargs)

        self.downsample, output_features = \
            get_downsample_x2_layer(downsample, features)

        (C, H, W)  = input_shape
        self.block = UnetBasicBlock(C, features, activ, norm)

        self.output_shape = (output_features, H//2, W//2)

    def get_output_shape(self):
        return self.output_shape

    def forward(self, x):
        r = self.block(x)
        y = self.downsample(r)
        return (y, r)

class UNetDecBlock(nn.Module):

    def __init__(
        self, output_shape, activ, norm, upsample, input_shape,
        rezero = True, scale_factor=2, **kwargs
    ):
        super().__init__(**kwargs)

        self.upsample, input_features = get_upsample_x2_layer(
            upsample, input_shape[0], scale_factor=scale_factor
        )

        self.block = UnetBasicBlock(
            2 * input_features, output_shape[0], activ, norm,
            mid_features = max(input_features, input_shape[0])
        )

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, x, r):
        # x : (N, C, H_in, W_in)
        # r : (N, C, H_out, W_out)

        # x : (N, C_up, H_out, W_out)
        x = self.re_alpha * self.upsample(x)

        # y : (N, C + C_up, H_out, W_out)
        y = torch.cat([x, r], dim = 1)

        # result : (N, C_out, H_out, W_out)
        return self.block(y)

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

class UNetBlock(nn.Module):

    def __init__(
        self, features, activ, norm, image_shape, downsample, upsample,
        rezero = True, up_scale_factor=2, **kwargs
    ):
        super().__init__(**kwargs)

        self.conv = UNetEncBlock(
            features, activ, norm, downsample, image_shape
        )

        self.inner_shape  = self.conv.get_output_shape()
        self.inner_module = None

        self.deconv = UNetDecBlock(
            image_shape, activ, norm, upsample, self.inner_shape, rezero, scale_factor=up_scale_factor
        )

    def get_inner_shape(self):
        return self.inner_shape

    def set_inner_module(self, module):
        self.inner_module = module

    def get_inner_module(self):
        return self.inner_module

    def forward(self, x):
        # x : (N, C, H, W)

        # y : (N, C_inner, H_inner, W_inner)
        # r : (N, C_inner, H, W)
        (y, r) = self.conv(x)

        # y : (N, C_inner, H_inner, W_inner)
        y = self.inner_module(y)

        print("ViT:", y.size())

        # y : (N, C, H, W)
        y = self.deconv(y, r)

        return y

class UNet(nn.Module):

    def __init__(
        self, in_c, out_c, features_list, activ, norm, image_shape, downsample, upsample,
        rezero = True, up_scale_factors=(2, 2, 2, (2.02, 2)), **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.features_list = features_list
        self.image_shape   = image_shape

        self._construct_input_layer(activ, in_c)
        self._construct_output_layer(out_c)

        unet_layers = []
        curr_image_shape = (features_list[0], *image_shape[1:])

        for idx, features in enumerate(features_list):
            layer = UNetBlock(
                features, activ, norm, curr_image_shape, downsample, upsample,
                rezero, up_scale_factor=up_scale_factors[idx]
            )
            curr_image_shape = layer.get_inner_shape()
            unet_layers.append(layer)

        for idx in range(len(unet_layers)-1):
            unet_layers[idx].set_inner_module(unet_layers[idx+1])

        self.unet = unet_layers[0]

    def _construct_input_layer(self, activ, in_c):
        self.layer_input = nn.Sequential(
            nn.Conv2d(
                in_c, self.features_list[0],
                kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),
        )

    def _construct_output_layer(self, out_c):
        self.layer_output = nn.Conv2d(
            self.features_list[0], out_c, kernel_size = 1
        )

    def get_innermost_block(self):
        result = self.unet

        for _ in range(len(self.features_list)-1):
            result = result.get_inner_module()

        return result

    def set_bottleneck(self, module):
        self.get_innermost_block().set_inner_module(module)

    def get_bottleneck(self):
        return self.get_innermost_block().get_inner_module()

    def get_inner_shape(self):
        return self.get_innermost_block().get_inner_shape()

    def forward(self, x):
        # x : (N, C, H, W)

        y = self.layer_input(x)
        y = self.unet(y)
        y = self.layer_output(y)

        return y


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
