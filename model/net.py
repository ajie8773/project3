import math

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from model.token_performer import Token_performer
import torch.nn.functional as F

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

    def forward(self, feat):
        B, L, C = feat.shape
        feat = self.norm(feat)
        shorcut = feat

        conv_feat = feat.transpose(1,2).reshape(B, C, int(L**0.5), int(L**0.5))
        pool_feat = F.adaptive_avg_pool2d(conv_feat, 1).reshape(B, C, -1).transpose(1,2)

        q = self.q(feat).reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(pool_feat).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(pool_feat).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

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
    def __init__(self, dim=384, token_dim=64, img_size=224, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True, fuseing="3"):
        super(decoder_module, self).__init__()

        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio,  img_size // ratio), kernel_size=kernel_size, stride=stride, padding=padding)
        self.fuse = fuse
        if self.fuse:
            if fuseing == "3":
                self.concatFuse = nn.Sequential(
                    nn.Linear(token_dim*3, token_dim),
                    nn.GELU(),
                    nn.Linear(token_dim, token_dim),
                )
            else:
                self.concatFuse = nn.Sequential(
                    nn.Linear(token_dim * 2, token_dim),
                    nn.GELU(),
                    nn.Linear(token_dim, token_dim),
                )
            self.att = Token_performer(dim=token_dim, in_dim=token_dim, kernel_ratio=0.5)

            # project input feature to 64 dim
            self.norm = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )

    def forward(self, dec_fea, enc_fea=None):

        if self.fuse:
            # from 384 to 64
            dec_fea = self.mlp(self.norm(dec_fea))

        # [1] token upsampling by the proposed reverse T2T module
        dec_fea = self.project(dec_fea)
        # [B, H*W, token_dim*kernel_size*kernel_size]
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)
        # [B, HW, C]

        if self.fuse:
            # [2] fuse encoder fea and decoder fea
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
    def __init__(self, embed_dim=384, token_dim=64, depth=2, img_size=224):

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

class Net(nn.Module):
    def __init__(self, training=True):
        self.training = training
        super(Net, self).__init__()
        self.encoder = pvt_v2_b2()
        #self.encoder.load_state_dict(torch.load('model/pretrain/pvt_v2_b2.pth', map_location='cpu'), strict=False)
        self.transformer = Transformer(embed_dim=320, depth=4, num_heads=4, mlp_ratio=3.)
        self.token_trans = token_Transformer(embed_dim=320, depth=4, num_heads=4, mlp_ratio=3.)

        self.decoder = Decoder(embed_dim=320, token_dim=64, depth=2, img_size=224)

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.encoder(x)
        x = x[::-1]
        feat_32 = x[0]
        feat_16 = x[1]
        feat_08 = x[2]
        feat_04 = x[3]

        feat_16 = self.transformer(feat_16)

        feat16 = self.token_trans(feat_16)
        outputs = self.decoder(feat16, feat_08, feat_04)

        out_16 = nn.functional.interpolate(outputs[0], scale_factor=16)
        out_08 = nn.functional.interpolate(outputs[1], scale_factor=8)
        out_04 = nn.functional.interpolate(outputs[2], scale_factor=4)
        out_01 = outputs[3]
        outputs = [out_16, out_08, out_04, out_01]
        if self.training:
            return outputs
        else:
            return outputs[-1]

if __name__ == '__main__':
    net = Net()
    y = net(torch.randn(4,3,224,224))



