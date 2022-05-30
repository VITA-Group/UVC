import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from timm.models.vision_transformer import  _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models.layers.helpers import to_2tuple


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


def scatter(logits, index, k):
    bs = logits.shape[0]
   #print('bs = {}'.format(bs))

    x_index = torch.arange(bs).reshape(-1, 1).expand(bs,k)
    x_index = x_index.reshape(-1).tolist()
    y_index = index.reshape(-1).tolist()

    output = torch.zeros_like(logits).cuda()
    output[x_index, y_index] = 1.0
   #print(output.sum(dim=1))

    return output


def gumbel_softmax(logits, k=0.9, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (torch.Tensor, float, bool, float, int) -> torch.Tensor

    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).cuda().exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.topk(k, dim=dim)[1]
        y_hard = scatter(logits, index, k)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    if torch.isnan(ret).sum():
        import ipdb
        ipdb.set_trace()
        raise OverflowError(f'gumbel softmax output: {ret}')
    return ret

def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
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

        macs.append(self.fc1.weight.shape[0] * x.shape[0]* x.shape[1]* x.shape[2])
        x = self.fc1(x)

        x = self.act(x)
        x = self.drop(x)

        macs.append(self.fc2.weight.shape[0] * x.shape[0]* x.shape[1]* x.shape[2])
        x = self.fc2(x)

        x = self.drop(x)

        return x, macs


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        macs = []
        bs, in1, in2 = x.shape[0], x.shape[1], x.shape[2]
        m1, m2 = self.qkv.weight.shape[0], self.qkv.weight.shape[1]


        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        macs.append(bs * m1 * in1 * in2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        macs.append(k.transpose(-2, -1).shape[-1] * q.shape[0]* q.shape[1]* q.shape[2]* q.shape[3])

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        macs.append(attn.shape[-1] * v.shape[0]* v.shape[1]* v.shape[2]* v.shape[3])

        x = self.proj(x)
        x = self.proj_drop(x)
        macs.append(x.shape[0]* x.shape[1]* x.shape[2]* self.proj.weight.shape[-1])

        return x, macs


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, enable_part_gating=0, gumbel_hard=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.gumbel_hard = gumbel_hard
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Skip gating for attention

        self.enable_part_gating = enable_part_gating
        if self.enable_part_gating:
            print("=====> Part gating enabled <=====")
        self.attn_skip_gating    = nn.Parameter(torch.Tensor([-1, 1]))
        self.mlp_skip_gating     = nn.Parameter(torch.Tensor([-1, 1]))



    def forward(self, x):
        if self.enable_part_gating:
            shortcut = x
            attn_distrib = F.gumbel_softmax(self.attn_skip_gating, tau=0.5, hard=self.gumbel_hard, eps=1e-10, dim=-1)
            x, macs_attn = self.attn(self.norm1(x))

            x = attn_distrib[0]*shortcut + attn_distrib[1]*self.drop_path(x)
            # for i, macs in enumerate(macs_attn):
            #     macs_attn[i] = macs * attn_distrib[1]


            shortcut = x
            mlp_distrib = F.gumbel_softmax(self.mlp_skip_gating, tau=0.5, hard=self.gumbel_hard, eps=1e-10, dim=-1)
            x, macs_mlp = self.mlp(self.norm2(x))
            x = mlp_distrib[0]*shortcut + mlp_distrib[1]*self.drop_path(x)
            # for i, macs in enumerate(macs_mlp):
            #     macs_mlp[i] = macs * mlp_distrib[1]


        else:
            shortcut = x
            x, macs_attn = self.attn(self.norm1(x))
            x = shortcut + self.drop_path(x)

            shortcut = x
            x, macs_mlp = self.mlp(self.norm2(x))
            x = shortcut + self.drop_path(x)


        return x, macs_attn + macs_mlp


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, gumbel_hard=True, enable_part_gating=0 ,img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.gumbel_hard = gumbel_hard

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, enable_part_gating = enable_part_gating, gumbel_hard = self.gumbel_hard)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        kernel_size = self.patch_embed.proj.kernel_size
        in_channels = self.patch_embed.proj.in_channels
        macs_embed = np.prod(x.shape) * np.prod(kernel_size) * in_channels


        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)



        macs_list = []
        for blk in self.blocks:
            x, tmp_macs = blk(x)
            macs_list.append(tmp_macs)



        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return (x[:, 0], x[:, 1]), (macs_embed, macs_list)

    def forward(self, x):
        x, macs_list = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist, macs_list
            else:
                return (x + x_dist) / 2, macs_list
        else:
            x = self.head(x)
        return x, macs_list




class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, enable_dist, enable_jumping=0, enable_block_gating=0, enable_part_gating=0, enable_patch_gating=0, gumbel_hard=True, use_gumbel=False, eps = 0.1, enable_warmup=False, patch_hard=False, *args, **kwargs):
        super().__init__(gumbel_hard, *args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))  if enable_dist else None
        self.num_tokens = 2 if enable_dist else 1
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 and enable_dist else None


        # Skip gating for the whole block
        self.enable_block_gating = enable_block_gating
        # Jumping connections that are jumps to the last layer
        self.enable_jumping = enable_jumping
        # Whether use a uniform patch slimming gating across all layers
        self.enable_patch_gating = enable_patch_gating
        # Distribution function, choices = [gumbel, softmax]
        self.use_gumbel = use_gumbel
        self.eps = eps
        self.gumbel = nn.Linear(self.embed_dim, 1)

        self.enable_warmup = enable_warmup

        if self.enable_block_gating:
            print("=====> Block gating enabled <=====")

        self.block_skip_gating = nn.Parameter(torch.Tensor([-1, 1]).expand(len(self.blocks),2).contiguous())
        self.patch_gating      = nn.Parameter(torch.zeros(1, self.patch_embed.grid_size[0]*self.patch_embed.grid_size[1], 1)) if self.enable_patch_gating==1 else None

        # Gumbel hard is only set to False when the compressed model is on training
        self.gumbel_hard = gumbel_hard
        self.patch_hard = patch_hard

        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        if self.head_dist is not None:
            self.head_dist.apply(self._init_weights)

    def forward_features(self, x, tau=-1, ratio=0.9):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)
        if self.enable_patch_gating==1:
            if self.patch_hard:
                patch_gating_norm = torch.sigmoid(self.patch_gating)
                one     = torch.ones_like(patch_gating_norm)
                ge      = torch.ge(patch_gating_norm, 0.5)
                zero    = torch.zeros_like(patch_gating_norm)
                mask    = torch.where(ge, one, zero)
                mask[:,0] = 1
                x       = x * mask
            else:
                x       = x * torch.sigmoid(self.patch_gating)

        if tau > 0:
            emb_dim = x.shape[2]
            token_number = x.shape[1]
            number = int(ratio * token_number)
            token_scores = self.gumbel(x)
            token_scores = token_scores.reshape(B, -1)
            token_mask = gumbel_softmax(F.log_softmax(token_scores, dim=-1), k=number, tau=tau, hard=True)
            token_mask[:,0] = 1.
            token_mask = token_mask.expand(emb_dim,-1,-1).permute(1,2,0)

            x = x * token_mask

        kernel_size = self.patch_embed.proj.kernel_size
        in_channels = self.patch_embed.proj.in_channels
        macs_embed = np.prod(x.shape) * np.prod(kernel_size) * in_channels

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        if self.dist_token is not None:
            dist_token = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)


        macs_list = []
        # accum is the collection of each layer and accumulate them together, constructing a jumping connnection from every layer to the last layer
        accum = 0
        for i, blk in enumerate(self.blocks):
            tmp_macs = []
            if self.enable_block_gating:
                distrib = torch.zeros(2).cuda()

                if self.enable_warmup:
                    distrib = torch.ones(2).cuda() * 0.5
                elif self.use_gumbel==1:
                    distrib = F.gumbel_softmax(self.block_skip_gating[i], tau=0.5, hard=self.gumbel_hard, eps=1e-10, dim=-1)
                else:
                    distrib[1] = self.block_skip_gating[i][1]**2/(self.block_skip_gating[i][1]**2 + self.eps)
                    distrib[0] = 1 - distrib[1]


                tmp_x, tmp_macs = blk(x)

                x = distrib[1]*tmp_x + distrib[0]*x
                macs_list.append(tmp_macs)

            else:
                # Easily decided only by architecture parameter magnitude
                if self.block_skip_gating[i][1]> self.block_skip_gating[i][0]:
                    x, tmp_macs = blk(x)
                macs_list.append(tmp_macs)

            # This skip connection connects every layer to the last layer
            accum += x

        if self.enable_jumping:
            x = accum
        x = self.norm(x)
        return x[:, 0], x[:, 1], (macs_embed, macs_list)

    def forward(self, x, tau=-1, number=0.9):
        # x, x_dist, macs_list = self.forward_features(x)
        # x = self.head(x)
        # x_dist = self.head_dist(x_dist)
        # if self.training:
        #     return (x, x_dist), macs_list
        # else:
        #     # during inference, return the average of both classifier predictions
        #     return (x + x_dist) / 2, macs_list


        x, x_dist, macs_list = self.forward_features(x, tau, number)
        x = self.head(x)
        if self.head_dist is None:
            x_dist = x
        else:
            x_dist = self.head_dist(x_dist)
        if self.training:
            return (x, x_dist), macs_list
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2, macs_list
