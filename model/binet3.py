import math

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from model.token_performer import Token_performer
import torch.nn.functional as F
from model.PVT_V2 import Block as PVT_Block
from model.PVT_V2 import OverlapPatchEmbed
from functools import partial

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
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                 Block(
                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                        for i in range(depth)])

        self.rgb_norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, rgb_fea):

        for block in self.blocks:
            rgb_fea = block(rgb_fea)

        rgb_fea = self.rgb_norm(rgb_fea)

        return rgb_fea


class Transformer(nn.Module):
    def __init__(self, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(Transformer, self).__init__()

        self.encoderlayer = TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)

    def forward(self, rgb_fea):

        rgb_memory = self.encoderlayer(rgb_fea)

        return rgb_memory


class token_TransformerEncoder(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(token_TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                 Block(
                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                        for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x

def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, add_maxpool=False,
            bias=True, act_layer=nn.ReLU, norm_layer=None, gate_layer='sigmoid'):
        super(SEModule, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=bias)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class tamper_feat_predict(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5


        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

        self.preNorm = nn.BatchNorm2d(dim, momentum=0.1)
        self.expansion_conv = nn.Conv2d(dim, 4 * dim, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(4 * dim, momentum=0.1)
        self.depthwise = nn.Conv2d(4 * dim, 4 * dim, kernel_size=3, stride=1, padding=1,
                                   groups=4 * dim)
        self.norm2 = nn.Sequential(
            nn.BatchNorm2d(4 * dim, momentum=0.1),
            nn.SiLU()
        )
        self.se = SEModule(channels=4 * dim)
        self.projection_conv = nn.Conv2d(4 * dim, dim, kernel_size=1, stride=1, bias=False)
        drop_path = 0
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.project2 = nn.Conv2d(dim, 2, 1)




    def forward(self, feat):
        B, L, C = feat.shape
        feat = self.norm(feat)
        shorcut = feat

        conv_feat = feat.transpose(1,2).reshape(B, C, int(L**0.5), int(L**0.5))
        ##edit

        #shortcut2 = conv_feat
        x = self.preNorm(conv_feat)#4, 192, 320     2, 320
        x = self.expansion_conv(x)
        x = self.norm1(x)
        x = self.depthwise(x)
        x = self.norm2(x)
        x = self.se(x)
        x = self.projection_conv(x)
        x = self.drop_path(x) #+ shortcut2
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.reshape(B, C, -1).transpose(1,2)

        q = self.q(feat).reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, 1, self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        pred_feat = (attn @ v).transpose(1, 2).reshape(B, L, C)
        pred_feat = self.proj(pred_feat)
        pred_feat = self.proj_drop(pred_feat)

        feat = pred_feat + shorcut
        return feat

class token_Transformer(nn.Module):
    def __init__(self, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_Transformer, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_s = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.tamper_feat_predict = tamper_feat_predict(dim=embed_dim, num_heads=1)


    def forward(self, feat):
        B, L, C = feat.shape
        feat = self.mlp_s(self.norm(feat))   # [B, 14*14, 384]
        return self.tamper_feat_predict(feat)



class decoder_module(nn.Module):
    def __init__(self, dim=512, dim2=320, img_size=512, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True):
        super(decoder_module, self).__init__()
        self.fuse = fuse
        self.project = nn.Linear(dim2, dim2 * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio,  img_size // ratio), kernel_size=kernel_size, stride=stride, padding=padding)
        if self.fuse:
            self.concatFuse = nn.Sequential(
                nn.Linear(dim2 * 2, dim2),
                nn.GELU(),
                nn.Linear(dim2, dim2),
            )
            self.att = Token_performer(dim=dim2, in_dim=dim2, kernel_ratio=0.5)

            # project input feature to 64 dim
            self.norm = nn.LayerNorm(dim)

            self.mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim2)
            )
    def forward(self, dec_fea, enc_fea=None):
        if self.fuse:
            dec_fea = self.mlp(self.norm(dec_fea))

        # [1] token upsampling by the proposed reverse T2T module
        dec_fea = self.project(dec_fea)
        # [B, H*W, token_dim*kernel_size*kernel_size]
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)

        if self.fuse:
            dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
            dec_fea = self.att(dec_fea)
        return dec_fea


class token_trans(nn.Module):
    def __init__(self, in_dim=64, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_trans, self).__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        self.norm2_c = nn.LayerNorm(embed_dim)
        self.mlp2_c = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )
        self.tamper_feat_predict = tamper_feat_predict(dim=embed_dim, num_heads=1)

    def forward(self, fea):
        B, _, _ = fea.shape
        fea = self.mlp(self.norm(fea))
        fea = self.tamper_feat_predict(fea)
        fea = self.mlp2(self.norm2(fea))

        return fea

class Decoder(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64, depth=2, img_size=512):

        super(Decoder, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        self.norm_c = nn.LayerNorm(embed_dim)
        self.mlp_c = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        self.img_size = img_size
        # token upsampling and multi-level token fusion
        self.decoder1 = decoder_module(dim=64, token_dim=token_dim, img_size=img_size, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True, fuseing="3")
        self.decoder2 = decoder_module(dim=64, token_dim=token_dim, img_size=img_size, ratio=4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True, fuseing="2")
        self.decoder3 = decoder_module(dim=64, token_dim=token_dim, img_size=img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)

        # token based multi-task predictions
        self.token_trans_08 = token_trans(in_dim=token_dim, embed_dim=64, depth=depth, num_heads=1)
        self.token_trans_04 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)

        # predict saliency maps
        self.predict_16 = nn.Linear(token_dim, 1)
        self.predict_08 = nn.Linear(token_dim, 1)
        self.predict_04 = nn.Linear(token_dim, 1)
        self.predict_01 = nn.Linear(token_dim, 1)


        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feat16,
                               feat_08, feat_04):
        B, _, _, = feat16.size()



        feat16 = self.mlp(self.norm(feat16))

        predict_16 = self.predict_16(feat16)
        predict_16 = predict_16.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        # 1/16 -> 1/8
        # reverse T2T and fuse low-level feature
        feat_08 = self.decoder1(feat16, feat_08)

        # token prediction
        feat_08 = self.token_trans_08(feat_08)

        predict_08 = self.predict_08(feat_08)
        predict_08 = predict_08.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)


        # 1/8 -> 1/4
        feat_04 = self.decoder2(feat_08, feat_04)

        feat_04 = self.token_trans_04(feat_04)

        predict_04 = self.predict_04(feat_04)
        predict_04 = predict_04.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)

        # 1/4 -> 1
        feat_01 = self.decoder3(feat_04)

        predict_01 = self.predict_01(feat_01)
        predict_01 = predict_01.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)

        return [predict_16, predict_08, predict_04, predict_01]

from model.PVT_V2 import pvt_v2_b2

class Fuse16_32(nn.Module):
    def __init__(self, img_size=224):
        super(Fuse16_32, self).__init__()
        self.img_size=img_size
        self.linear = nn.Linear(832, 320)

    def forward(self, feat16, feat32):
        B,L,dim = feat16.shape
        feat16 = feat16.transpose(1, 2).reshape(B, -1, self.img_size//16, self.img_size//16)
        feat32 = feat32.transpose(1, 2).reshape(B, -1, self.img_size//32, self.img_size//32)
        x = torch.cat([feat16, F.interpolate(feat32, scale_factor=2)], dim=1)
        x = x.reshape(B,L, -1)
        x = self.linear(x)
        return x

def label_to_onehot(gt, num_classes, ignore_index=-1):
    '''
    gt: ground truth with size (N, H, W)
    num_classes: the number of classes of different label
    '''
    N, H, W = gt.size()
    x = gt
    x[x == ignore_index] = num_classes
    # convert label into onehot format
    onehot = torch.zeros(N, x.size(1), x.size(2), num_classes + 1).cuda()
    onehot = onehot.scatter_(-1, x.unsqueeze(-1), 1)

    return onehot.permute(0, 3, 1, 2)

class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        use_gt            : whether use the ground truth label map to compute the similarity map
        fetch_attention   : whether return the estimated similarity map
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 use_gt=False,
                 use_bg=False,
                 fetch_attention=False,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.fetch_attention = fetch_attention
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy, gt_label=None):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        if self.use_gt and gt_label is not None:
            gt_label = label_to_onehot(gt_label.squeeze(1).type(torch.cuda.LongTensor), proxy.size(2)-1)
            sim_map = gt_label[:, :, :, :].permute(0, 2, 3, 1).view(batch_size, h*w, -1)
            if self.use_bg:
                bg_sim_map = 1.0 - sim_map
                bg_sim_map = F.normalize(bg_sim_map, p=1, dim=-1)
            sim_map = F.normalize(sim_map, p=1, dim=-1)
        else:
            sim_map = torch.matmul(query, key)
            sim_map = (self.key_channels**-.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value) # hw x k x k x c
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        if self.use_bg:
            bg_context = torch.matmul(bg_sim_map, value)
            bg_context = bg_context.permute(0, 2, 1).contiguous()
            bg_context = bg_context.view(batch_size, self.key_channels, *x.size()[2:])
            bg_context = self.f_up(bg_context)
            bg_context = F.interpolate(input=bg_context, size=(h, w), mode='bilinear', align_corners=True)
            return context, bg_context
        else:
            if self.fetch_attention:
                return context, sim_map
            else:
                return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 use_gt=False,
                 use_bg=False,
                 fetch_attention=False,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     use_gt,
                                                     use_bg,
                                                     fetch_attention,
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.

    use_gt=True: whether use the ground-truth label to compute the ideal object contextual representations.
    use_bg=True: use the ground-truth label to compute the ideal background context to augment the representations.
    use_oc=True: use object context or not.
    """
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 use_gt=False,
                 use_bg=False,
                 use_oc=True,
                 fetch_attention=False,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.use_oc = use_oc
        self.fetch_attention = fetch_attention
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           use_gt,
                                                           use_bg,
                                                           fetch_attention,
                                                           bn_type)
        if self.use_bg:
            if self.use_oc:
                _in_channels = 3 * in_channels
            else:
                _in_channels = 2 * in_channels
        else:
            _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats, gt_label=None):
        if self.use_gt and gt_label is not None:
            if self.use_bg:
                context, bg_context = self.object_context_block(feats, proxy_feats, gt_label)
            else:
                context = self.object_context_block(feats, proxy_feats, gt_label)
        else:
            if self.fetch_attention:
                context, sim_map = self.object_context_block(feats, proxy_feats)
            else:
                context = self.object_context_block(feats, proxy_feats)

            output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output



class ContextAwareModule(nn.Module):
    def __init__(self, in_channels, img_size=512):
        self.project = nn.Conv2d(in_channels, 2, 1)
        self.spatial_context_head = SpatialGather_Module(2)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=in_channels,
                                                     key_channels=64,
                                                     out_channels=in_channels,
                                                     scale=1,
                                                     dropout=0.05,
                                                     bn_type='torchbn')
        self.head = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.img_size = img_size

    def forward(self, x):
        prob = self.project(x)
        context = self.spatial_context_head(x, prob)
        x = self.spatial_ocr_head(x, context)
        x = self.head(x)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=True)
        return x



def label_to_onehot(gt, num_classes, ignore_index=-1):
    '''
    gt: ground truth with size (N, H, W)
    num_classes: the number of classes of different label
    '''
    N, H, W = gt.size()
    x = gt
    x[x == ignore_index] = num_classes
    # convert label into onehot format
    onehot = torch.zeros(N, x.size(1), x.size(2), num_classes + 1).cuda()
    onehot = onehot.scatter_(-1, x.unsqueeze(-1), 1)

    return onehot.permute(0, 3, 1, 2)

class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        use_gt            : whether use the ground truth label map to compute the similarity map
        fetch_attention   : whether return the estimated similarity map
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 use_gt=False,
                 use_bg=False,
                 fetch_attention=False,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.fetch_attention = fetch_attention
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy, gt_label=None):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        if self.use_gt and gt_label is not None:
            gt_label = label_to_onehot(gt_label.squeeze(1).type(torch.cuda.LongTensor), proxy.size(2)-1)
            sim_map = gt_label[:, :, :, :].permute(0, 2, 3, 1).view(batch_size, h*w, -1)
            if self.use_bg:
                bg_sim_map = 1.0 - sim_map
                bg_sim_map = F.normalize(bg_sim_map, p=1, dim=-1)
            sim_map = F.normalize(sim_map, p=1, dim=-1)
        else:
            sim_map = torch.matmul(query, key)
            sim_map = (self.key_channels**-.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value) # hw x k x k x c
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        if self.use_bg:
            bg_context = torch.matmul(bg_sim_map, value)
            bg_context = bg_context.permute(0, 2, 1).contiguous()
            bg_context = bg_context.view(batch_size, self.key_channels, *x.size()[2:])
            bg_context = self.f_up(bg_context)
            bg_context = F.interpolate(input=bg_context, size=(h, w), mode='bilinear', align_corners=True)
            return context, bg_context
        else:
            if self.fetch_attention:
                return context, sim_map
            else:
                return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 use_gt=False,
                 use_bg=False,
                 fetch_attention=False,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     use_gt,
                                                     use_bg,
                                                     fetch_attention,
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.

    use_gt=True: whether use the ground-truth label to compute the ideal object contextual representations.
    use_bg=True: use the ground-truth label to compute the ideal background context to augment the representations.
    use_oc=True: use object context or not.
    """
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 use_gt=False,
                 use_bg=False,
                 use_oc=True,
                 fetch_attention=False,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.use_oc = use_oc
        self.fetch_attention = fetch_attention
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           use_gt,
                                                           use_bg,
                                                           fetch_attention,
                                                           bn_type)
        if self.use_bg:
            if self.use_oc:
                _in_channels = 3 * in_channels
            else:
                _in_channels = 2 * in_channels
        else:
            _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats, gt_label=None):
        if self.use_gt and gt_label is not None:
            if self.use_bg:
                context, bg_context = self.object_context_block(feats, proxy_feats, gt_label)
            else:
                context = self.object_context_block(feats, proxy_feats, gt_label)
        else:
            if self.fetch_attention:
                context, sim_map = self.object_context_block(feats, proxy_feats)
            else:
                context = self.object_context_block(feats, proxy_feats)

            output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output



class ContextAwareModule(nn.Module):
    def __init__(self, in_channels, img_size=224):
        super(ContextAwareModule, self).__init__()

        self.spatial_context_head = SpatialGather_Module(2)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=in_channels,
                                                     key_channels=64,
                                                     out_channels=in_channels,
                                                     scale=1,
                                                     dropout=0.05,
                                                     bn_type='torchbn')
        self.head = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.img_size = img_size

    def forward(self, x, prob):
        context = self.spatial_context_head(x, prob)
        x = self.spatial_ocr_head(x, context)
        x = self.head(x)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=True)
        return x


class Up(nn.Module):
    def __init__(self, dim, kernel_size, output_size, stride, padding):
        super(Up, self).__init__()
        self.output_size = output_size
        self.expand = nn.Linear(dim, dim * kernel_size * kernel_size)
        self.fold = nn.Fold(output_size=(output_size, output_size), kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        B,_,_ = x.shape
        x = self.expand(x)
        x = self.fold(x.transpose(1, 2))
        x = x.reshape(B, -1, (self.output_size)*(self.output_size)).transpose(1, 2)
        return x


class Fuse_Edge(nn.Module):
    def __init__(self, img_size):
        super(Fuse_Edge, self).__init__()
        self.img_size = img_size
        self.concatFuse04 = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )

        self.concatFuse08 = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )

        self.concatFuse16 = nn.Sequential(
            nn.Linear(320 + 64, 320),
            nn.GELU(),
            nn.Linear(320, 320)
        )

        self.concatFuse32 = nn.Sequential(
            nn.Linear(512 + 64, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )



    def forward(self, img_edge, feat_32, feat_16, feat_08, feat_04):
        B,L,D = img_edge.shape
        img_edge = img_edge.transpose(1, 2).reshape(B, D, self.img_size, self.img_size)
        img_edge04 = F.interpolate(img_edge, size=(self.img_size//4, self.img_size//4), mode="bilinear", align_corners=True).reshape(B, D, -1).transpose(1, 2)
        img_edge08 = F.interpolate(img_edge, size=(self.img_size // 8, self.img_size // 8), mode="bilinear",
                                   align_corners=True).reshape(B, D, -1).transpose(1, 2)
        img_edge16 = F.interpolate(img_edge, size=(self.img_size // 16, self.img_size // 16), mode="bilinear",
                                   align_corners=True).reshape(B, D, -1).transpose(1, 2)
        img_edge32 = F.interpolate(img_edge, size=(self.img_size // 32, self.img_size // 32), mode="bilinear",
                                   align_corners=True).reshape(B, D, -1).transpose(1, 2)
        feat_04 = self.concatFuse04(torch.cat([img_edge04, feat_04], dim=2))
        feat_08 = self.concatFuse08(torch.cat([img_edge08, feat_08], dim=2))
        feat_16 = self.concatFuse16(torch.cat([img_edge16, feat_16], dim=2))
        feat_32 = self.concatFuse32(torch.cat([img_edge32, feat_32], dim=2))
        return feat_32, feat_16, feat_08, feat_04



def gen_cons_conv_weight(shape):
  center = int(shape / 2)
  accumulation = 0
  for i in range(shape):
    for j in range(shape):
      if i != center or j != center:
        dis = math.sqrt((i - center) * (i - center) + (j - center) * (j - center))
        accumulation += 1 / dis

  base = 1 / accumulation
  arr = torch.zeros((shape, shape), requires_grad=False)
  for i in range(shape):
    for j in range(shape):
      if i != center or j != center:
        dis = math.sqrt((i - center) * (i - center) + (j - center) * (j - center))
        arr[i][j] = base / dis
  arr[center][center] = -1

  return arr.unsqueeze(0).unsqueeze(0).repeat(3, 3, 1, 1)


class EEB(nn.Module):
  def __init__(self, in_channels, out_channels, inter_scale=4):
    super(EEB, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, int(in_channels / inter_scale), kernel_size=1, stride=1, padding=0)

    self.conv2 = nn.Conv2d(int(in_channels / inter_scale), int(in_channels / inter_scale), kernel_size=3, stride=1,
                           padding=1)
    self.relu = nn.ReLU()
    self.bn = nn.BatchNorm2d(int(in_channels / inter_scale))
    self.conv3 = nn.Conv2d(int(in_channels / inter_scale), int(in_channels / inter_scale), kernel_size=3, stride=1,
                           padding=1)

    self.conv4 = nn.Conv2d(int(in_channels / inter_scale), out_channels, kernel_size=1, stride=1, padding=0)
    self.bn2 = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    B,L,D = x.shape
    x = x.transpose(1,2).reshape(B,D, int(L**0.5), int(L**0.5))
    x = self.conv1(x)

    res = self.conv2(x)
    res = self.bn(res)
    res = self.relu(res)
    res = self.conv3(res)
    res = self.bn(res)
    res = self.relu(res)

    x = self.conv4(x + res)
    x = self.relu(self.bn2(x))
    return x


class ConvBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()

  def forward(self, input):
    x = self.conv1(input)
    return self.relu(self.bn(x))


class FeatureFusionModule(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.concatFuse = nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.GELU(),
        nn.Linear(out_channels, out_channels),
    )
    self.att = Token_performer(dim=out_channels, in_dim=out_channels, kernel_ratio=0.5)

  def forward(self, input_1, input_2):
    x = self.concatFuse(torch.cat([input_1, input_2], dim=2))
    x = self.att(x)
    return x


class Net(nn.Module):
    def __init__(self, training=True):
        self.training = training
        super(Net, self).__init__()
        self.cons_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, bias=False, padding=2)
        with torch.no_grad():
            self.cons_conv.weight.copy_(gen_cons_conv_weight(5))

        self.encoder = pvt_v2_b2()
        self.img_size = 512

        self.encoder.load_state_dict(torch.load('model/pretrain/pvt_v2_b2.pth', map_location='cpu'), strict=False)

        self.fuse_img_edge = Fuse_Edge(self.img_size)

        self.fuse32_16 = decoder_module(dim=512, dim2=320, img_size=self.img_size, ratio=16,
                                        kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.fuse16_08 = decoder_module(dim=320, dim2=128, img_size=self.img_size, ratio=8, kernel_size=(3, 3),
                                        stride=(2, 2),
                                        padding=(1, 1))
        self.fuse08_04 = decoder_module(dim=128, dim2=64, img_size=self.img_size, ratio=4, kernel_size=(3, 3),
                                        stride=(2, 2),
                                        padding=(1, 1))

        self.decoder_to_1 = decoder_module(dim=64, dim2=64, img_size=self.img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4),
                                           padding=(2, 2), fuse=False)


        self.linear_01 = nn.Linear(64, 1)

        depths = [3,4,6,3]
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
        cur = depths[0]+depths[1]
        self.block3 = nn.ModuleList([PVT_Block(
                dim=128, num_heads=2, mlp_ratio=4, qkv_bias=True,
                qk_scale=None,
                drop=0.0, attn_drop=0.0, drop_path=dpr[cur + j], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                sr_ratio=1, linear=False)
                for j in range(9)])

        self.img_decoder_to_1 = decoder_module(dim=64, dim2=64, img_size=self.img_size, ratio=1, kernel_size=(7, 7),
                                           stride=(4, 4),
                                           padding=(2, 2), fuse=False)
        self.linear_img_edge = nn.Linear(64, 1)

        self.up4 = decoder_module(dim=512, dim2=512, img_size=self.img_size//8, ratio=1, kernel_size=(7, 7),
                                           stride=(4, 4),
                                           padding=(2, 2), fuse=False)
        self.mlp_to_32 = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 512)
        )
        self.up2 = decoder_module(dim=320, dim2=320, img_size=self.img_size//8, ratio=1, kernel_size=(3, 3),
                                           stride=(2, 2),
                                           padding=(1, 1), fuse=False)
        self.mlp_to_16 = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 320)
        )

        self.ffm = FeatureFusionModule(in_channels=512+320, out_channels=128)


    def forward(self, input):
        B, _, _, _ = input.shape

        x = self.encoder(input)
        x = x[::-1]
        feat_32 = x[0]
        feat_16 = x[1]
        feat_08 = x[2]
        feat_04 = x[3]

        high_resolution_branch_feat = feat_08
        for blk in self.block3:
            high_resolution_branch_feat = blk(high_resolution_branch_feat, self.img_size//8, self.img_size//8)
        up4_feat32 = self.up4(feat_32)
        up4_feat32 = (self.mlp_to_32(high_resolution_branch_feat) + up4_feat32)
        up2_feat16 = self.up2(feat_16)
        up2_feat16 = (self.mlp_to_16(high_resolution_branch_feat) + up2_feat16)

        ffm = self.ffm(up4_feat32, up2_feat16)
        ffm = ffm + feat_08
        fusion_04 = self.fuse08_04(ffm, feat_04)
        out_01 = self.decoder_to_1(fusion_04)
        pred_01 = self.linear_01(out_01).transpose(1, 2).reshape(B, 1, self.img_size, self.img_size)

        outputs = [
            pred_01]
        if self.training:
            return outputs
        else:
            return outputs[-1]

if __name__ == '__main__':
    net = Net()
    y = net(torch.randn(2,3,512,512))



