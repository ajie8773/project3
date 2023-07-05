import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import timm
from model.aspp import build_aspp
from model.srm import setup_srm_layer


class Net(nn.Module):
    def __init__(self, training=True):
        super(Net, self).__init__()
        self.training=training
        cfg = {'global_params': {'with_srm': True, 'with_con': True}, 'model_params': {'encoder': 'resnet50', 'aspp_outplane': 512, 'num_class': 2, 'optimizer': 'adam', 'lr': 0.0001, 'epoch': 200, 'con_alpha': 1}, 'dataset_params': {'base_dir': '', 'batch_size': 2, 'patch_size': 4, 'im_size': 256, 'contrast_temperature': 0.1, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'imbalance_weight': [0.0892, 0.9108]}}
        inplanes = 2048
        self.cfg = cfg
        self.encoder = timm.create_model("resnet50", pretrained=False, features_only=True,
                                         out_indices=[4])
        self.encoder.load_state_dict(torch.load('model/pretrain/resnet50.pth', map_location='cpu'), strict=False)
        self.conv_srm = setup_srm_layer()
        self.encoder_srm = timm.create_model(self.cfg['model_params']['encoder'], pretrained=False, features_only=True,
                                             out_indices=[4])
        self.encoder_srm.load_state_dict(torch.load('model/pretrain/resnet50.pth', map_location='cpu'), strict=False)
        if self.cfg['global_params']['with_srm'] == True:
            self.aspp = build_aspp(inplanes=inplanes * 2, outplanes=self.cfg['model_params']['aspp_outplane'])
        else:
            self.aspp = build_aspp(inplanes=inplanes, outplanes=self.cfg['model_params']['aspp_outplane'])

        self.decoder = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,
                                               padding=1, stride=1, bias=False),
                                     nn.BatchNorm2d(512),
                                     nn.Conv2d(512, 1, kernel_size=1,
                                               stride=1, bias=True))
        self.projection = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=1)
        )

    def forward(self, inp):
        x = self.encoder(inp)[0]

        if self.cfg['global_params']['with_srm'] == True:
            x_srm = self.conv_srm(inp)
            x_srm = self.encoder_srm(x_srm)[0]
            x = torch.cat([x, x_srm], dim=1)

        x = self.aspp(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        out = self.decoder(x)
        proj = self.projection(x)
        outputs = [
            proj,
            out
        ]
        if self.training:
            return outputs
        else:
            return out