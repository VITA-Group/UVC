# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]




""" Vision Transformer (ViT) in PyTorch
"""
import torch
import torch.nn as nn
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from .helpers import load_pretrained
# from .layers import DropPath, to_2tuple, trunc_normal_
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from .resnet import resnet26d, resnet50d
# from .registry import register_model
import pdb
import numpy as np

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


def get_sparsity(module):

    total = module.weight.numel()
    zeros = (module.weight==0).sum()
    return zeros / total


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        macs = []
        # s1 = get_sparsity(self.fc1)
        # s2 = get_sparsity(self.fc2)

        macs.append(self.fc1.weight.shape[0] * x.shape[0]* x.shape[1]* x.shape[2])
        x = self.fc1(x)

        x = self.act(x)
        x = self.drop(x)

        macs.append(self.fc2.weight.shape[0] * x.shape[0]* x.shape[1]* x.shape[2])
        x = self.fc2(x)

        x = self.drop(x)
        return x, macs


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        macs_qk = 0
        macs_v  = 0
        macs_w1 = 0
        # B, N, Embed_Dim
        bs, in1, in2 = x.shape[0], x.shape[1], x.shape[2] # torch.Size([256, 197, 192])
        m1, m2 = self.qkv.weight.shape[0], self.qkv.weight.shape[1] # torch.Size([576, 192])
        # s1 = get_sparsity(self.qkv)
        # s2 = get_sparsity(self.proj)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        macs_qk += bs * m1 * in1 * in2

        q, k, v = qkv[0], qkv[1], qkv[2]   # torch.Size([256, 3, 197, 64])
        attn = (q @ k.transpose(-2, -1)) * self.scale # torch.Size([256, 3, 64, 197])
        macs_qk += k.transpose(-2, -1).shape[-1] * q.shape[0]* q.shape[1]* q.shape[2]* q.shape[3]

        attn = attn.softmax(dim=-1) # torch.Size([256, 3, 197, 197])
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        macs_v += attn.shape[-1] * v.shape[0]* v.shape[1]* v.shape[2]* v.shape[3]

        x = x.reshape(B, N, C)
        x = self.proj(x)
        macs_w1 += x.shape[0]* x.shape[1]* x.shape[2]* self.proj.weight.shape[-1]

        x = self.proj_drop(x)
        return x, [macs_qk, macs_v, macs_w1]


def compute_indicator(input_tensor, how='l1'):

    input_tensor = input_tensor.transpose(0, 2)
    head_num = input_tensor.shape[0]
    indicator_list = []
    for i in range(head_num):
        norm = torch.norm(input_tensor[i], p=1).detach().cpu().item()
        indicator_list.append(norm)
    return indicator_list


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):

        attention, macs1 = self.attn(self.norm1(x))
        x = x + self.drop_path(attention)

        mlp, macs2 = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp)
        return x, macs1 + macs2
    # def forward(self, x):
    #     x = x + self.drop_path(self.attn(self.norm1(x)))
    #     x = x + self.drop_path(self.mlp(self.norm2(x)))
    #     return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, enable_part_gating=0 ,img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, pruning_type='unstructure'):
        super().__init__()
        self.pruning_type = pruning_type
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        total_macs = []

        B = x.shape[0]
        x = self.patch_embed(x)
        print(x.shape)

        kernel_size = self.patch_embed.proj.kernel_size
        in_channels = self.patch_embed.proj.in_channels

        macs = np.prod(x.shape) * np.prod(kernel_size) * in_channels * (1 - get_sparsity(self.patch_embed.proj))
        total_macs.append(float(macs))

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x, macs = blk(x)
            total_macs.append(macs)

        x = self.norm(x)
        print(total_macs)
        return x[:, 0], torch.Tensor(total_macs)

    def forward(self, x):

        x, total_macs = self.forward_features(x)
        x = self.head(x)
        macs = x.shape[0] * x.shape[1] * self.head.weight.shape[-1] * (1 - get_sparsity(self.head))
        total_macs.append(macs)

        return x, total_macs







class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, enable_jumping, enable_block_gating,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        total_macs = []

        B = x.shape[0]
        x = self.patch_embed(x)


        kernel_size = self.patch_embed.proj.kernel_size
        in_channels = self.patch_embed.proj.in_channels

        macs_embed = np.prod(x.shape) * np.prod(kernel_size) * in_channels
        # total_macs.append(float(macs))

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x, macs = blk(x)
            total_macs.append(macs)

        x = self.norm(x)
        # print(total_macs)
        return x[:, 0], x[:, 1], (macs_embed, torch.Tensor(total_macs))

    def forward(self, x):
        x, x_dist, total_macs = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return (x, x_dist), total_macs
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2, total_macs
