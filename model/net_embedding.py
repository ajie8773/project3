import math

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from model.token_performer import Token_performer
import torch.nn.functional as F
from model.PVT_V2 import pvt_v2_b2
from model.transformer import TransformerDecoder,TransformerDecoderLayer


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





class Gate(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

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

    def forward(self, fea, embedding):
        B, N, C = fea.shape

        q = self.q(fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(embedding).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(embedding).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        feat = (attn @ v).transpose(1, 2).reshape(B, N, C)
        feat = self.proj(feat)
        feat = self.proj_drop(feat)

        feat = feat + fea
        return feat

class EmbbdeingCrossFormer(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super(EmbbdeingCrossFormer, self).__init__()
        decoder_layer = TransformerDecoderLayer(in_dim, nhead=8, dim_feedforward=4*in_dim,
                                                dropout=0.0, activation=nn.LeakyReLU, normalize_before=False)
        decoder_norm = nn.LayerNorm(in_dim)
        self.decoder = TransformerDecoder(decoder_layer, 2, decoder_norm,
                                          return_intermediate=False)

        self.linear_embedding = nn.Linear(embed_dim, in_dim)

    def forward(self, x, query_embed):
        x = x.transpose(0, 1)
        query_embed = self.linear_embedding(query_embed)
        hs = self.decoder(torch.zeros_like(query_embed), x, memory_key_padding_mask=None, pos=None, query_pos=query_embed)
        return hs



class Net(nn.Module):
    def __init__(self, training=True):
        self.training = training
        super(Net, self).__init__()
        self.encoder = pvt_v2_b2()
        self.img_size = 512
        self.encoder.load_state_dict(torch.load('model/pretrain/pvt_v2_b2.pth', map_location='cpu'), strict=False)
        self.fuse32_16 = decoder_module(dim=512, dim2=320, img_size=self.img_size, ratio=16,
                                        kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.fuse16_08 = decoder_module(dim=320, dim2=128, img_size=self.img_size, ratio=8, kernel_size=(3, 3), stride=(2, 2),
                                        padding=(1, 1))
        self.fuse08_04 = decoder_module(dim=128, dim2=64, img_size=self.img_size, ratio=4, kernel_size=(3, 3), stride=(2, 2),
                                        padding=(1, 1))
        self.decoder_to_1 = decoder_module(dim=64, dim2=64, img_size=self.img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4),
                                           padding=(2, 2), fuse=False)

        self.linear_32 = nn.Linear(512, 1)
        self.linear_16 = nn.Linear(320, 1)
        self.linear_08 = nn.Linear(128, 1)
        self.linear_04 = nn.Linear(64, 1)
        self.linear_01 = nn.Linear(64, 1)

        embed_dim = 512
        self.query_embedding = nn.Parameter(
            data=torch.randn(1, embed_dim, dtype=torch.float),
            requires_grad=True)

        self.embbdeing0 = EmbbdeingCrossFormer(in_dim=512, embed_dim=embed_dim)
        self.embbdeing1 = EmbbdeingCrossFormer(in_dim=320, embed_dim=512)
        self.gate1 = Gate(dim=320)
        self.embbdeing2 = EmbbdeingCrossFormer(in_dim=128, embed_dim=320)
        self.gate2 = Gate(dim=128)
        self.embbdeing3 = EmbbdeingCrossFormer(in_dim=64, embed_dim=128)
        self.gate3 = Gate(dim=64)


    def forward(self, input):
        B, _, _, _ = input.shape
        x = self.encoder(input)
        x = x[::-1]
        feat_32 = x[0]
        feat_16 = x[1]
        feat_08 = x[2]
        feat_04 = x[3]



        out_32 = self.linear_32(feat_32)
        pred_32 = F.interpolate(out_32.transpose(1, 2).reshape(B, 1, self.img_size // 32, self.img_size // 32),
                                size=(input.size(2), input.size(3)), mode="bilinear", align_corners=True)

        query_embed = self.query_embedding.unsqueeze(1).repeat(1, B, 1)
        query_embed = self.embbdeing0(feat_32, query_embed)

        fusion_16 = self.fuse32_16(feat_32, feat_16)
        query_embed = self.embbdeing1(fusion_16, query_embed)
        fusion_16 = self.gate1(fusion_16, query_embed)
        out_16 = self.linear_16(fusion_16)
        pred_16 = F.interpolate(out_16.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16), size=(input.size(2), input.size(3)), mode="bilinear", align_corners=True)

        fusion_08 = self.fuse16_08(fusion_16, feat_08)
        query_embed = self.embbdeing2(fusion_08, query_embed)
        fusion_08 = self.gate2(fusion_08, query_embed)
        out_08 = self.linear_08(fusion_08)
        pred_08 = F.interpolate(out_08.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8), size=(input.size(2), input.size(3)), mode="bilinear", align_corners=True)

        fusion_04 = self.fuse08_04(fusion_08, feat_04)
        query_embed = self.embbdeing3(fusion_04, query_embed)
        fusion_04 = self.gate3(fusion_04, query_embed)
        out_04 = self.linear_04(fusion_04)
        pred_04 = F.interpolate(out_04.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4), size=(input.size(2), input.size(3)), mode="bilinear", align_corners=True)

        out_01 = self.decoder_to_1(fusion_04)
        pred_01 = self.linear_01(out_01).transpose(1, 2).reshape(B, 1, self.img_size, self.img_size)






        outputs = [
            pred_32,
            pred_16,
            pred_08,
            pred_04,
            pred_01
                   ]
        if self.training:
            return outputs
        else:
            return outputs[-1]

if __name__ == '__main__':
    net = Net()
    y = net(torch.randn(4,3,512,512))



