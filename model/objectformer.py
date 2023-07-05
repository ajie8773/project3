import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch

feature_dim = 4


class BCIM(nn.Module):
    def __init__(self, height, width, h_patches, w_patches):
        super().__init__()
        self.height = height
        self.width = width
        self.h_patches = h_patches
        self.w_patches = w_patches
        self.patch_height = int(height / h_patches)
        self.patch_width = int(width / w_patches)
        self.window_size = 3
        self.weight = nn.Parameter(
            torch.zeros(size=(feature_dim * self.patch_height * self.patch_width * 2,
                              feature_dim * self.patch_height * self.patch_width * 2,
                              self.window_size,
                              self.window_size)),
            requires_grad=False)
        for i in range(feature_dim * self.patch_height * self.patch_width * 2):
            self.weight[i, i, :, :] = 1
        self.bias = nn.Parameter(torch.zeros(feature_dim * self.patch_height * self.patch_width * 2),
                                 requires_grad=False)

    def forward(self, p_vector):
        batch_size = p_vector.size()[0]
        # p_vector 10, 128*8*8*2, 28, 38
        norm2_p_vector = torch.norm(p_vector, p=2, dim=1, keepdim=True)  # p_vector 10, 1, 28, 38
        unit_p_vector = p_vector / norm2_p_vector  # p_vector 10, 128*8*8*2, 28, 38
        window_sum_unit_p_vector = F.conv2d(unit_p_vector, weight=self.weight, bias=self.bias, padding=int((self.window_size-1)/2)) / self.window_size / self.window_size
        # p_vector 10, 128*8*8*2, 28, 38
        Sim = torch.mul(unit_p_vector, window_sum_unit_p_vector).sum(dim=1, keepdim=True)  # 10, 1, 28, 38
        p_vector = torch.mul(p_vector, Sim)  # p_vector 10, 128*8*8*2, 28, 38
        p_vector = p_vector.permute(0, 2, 3, 1).contiguous()  # 10, 28, 38, 128*8*8*2
        p_vector = p_vector.view(batch_size, self.h_patches, self.w_patches,
                                 feature_dim * self.patch_height * self.patch_width, 2)
        # 10, 28, 38, 128*8*8, 2
        p_vector = p_vector.permute(0, 4, 1, 2, 3).contiguous()     # 10， 2， 28， 38， 128*8*8
        p_vector = p_vector.view(batch_size, 2 * self.h_patches * self.w_patches,
                                 feature_dim * self.patch_height * self.patch_width)
        # 10, 2*28*38, 128*8*8
        return p_vector


class ObjectDecoder(nn.Module):
    def __init__(self, height, width, h_patches, w_patches):
        super().__init__()
        self.height = height
        self.width = width
        self.h_patches = h_patches
        self.w_patches = w_patches
        patch_height = int(height / h_patches)
        patch_width = int(width / w_patches)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.p_layernorm = nn.LayerNorm(feature_dim * patch_height * patch_width)
        self.object_layernorm = nn.LayerNorm(feature_dim * patch_height * patch_width)
        self.query_embedding_Matrix = nn.Linear(in_features=feature_dim * patch_height * patch_width,
                                                out_features=feature_dim * patch_height * patch_width)
        self.key_embedding_Matrix = nn.Linear(in_features=feature_dim * patch_height * patch_width,
                                              out_features=feature_dim * patch_height * patch_width)
        self.value_embedding_Matrix = nn.Linear(in_features=feature_dim * patch_height * patch_width,
                                                out_features=feature_dim * patch_height * patch_width)
        self.softmax_layer = nn.Softmax(dim=2)
        self.projection_layer = nn.Sequential(nn.Linear(in_features=feature_dim * patch_height * patch_width,
                                                        out_features=feature_dim * patch_height * patch_width),
                                              nn.LayerNorm(feature_dim * patch_height * patch_width),
                                              nn.GELU(),
                                              nn.Linear(in_features=feature_dim * patch_height * patch_width,
                                                        out_features=feature_dim * patch_height * patch_width),
                                              nn.LayerNorm(feature_dim * patch_height * patch_width),
                                              nn.GELU())
        self.BCIM = BCIM(height, width, h_patches, w_patches)

    def forward(self, p_vector, object_vector):
        batch_size = p_vector.size()[0]
        p_vector = self.p_layernorm(p_vector)  # 10, 2*28*38, 128*8*8
        object_vector = self.object_layernorm(object_vector)  # N, 128*8*8
        query = self.query_embedding_Matrix(p_vector)  # 10, 2*28*38, 128*8*8
        key = self.key_embedding_Matrix(object_vector)  # N, 128*8*8
        value = self.value_embedding_Matrix(object_vector)  # N, 128*8*8

        key = key.permute(1, 0).contiguous()  # 128*8*8, N
        A = torch.matmul(query, key)  # 10, 2*28*38, N
        A = self.softmax_layer(A)  # 10, 2*28*38, N
        res_p_vector = torch.matmul(A, value)  # 10, 2*28*38, 128*8*8
        p_vector = p_vector + res_p_vector  # 10, 2*28*38, 128*8*8

        projection_p_vector = self.projection_layer(p_vector)
        p_vector = p_vector + projection_p_vector  # 10, 2*28*38, 128*8*8

        p_vector = p_vector.permute(0, 2, 1).contiguous()  # 10, 128*8*8, 2*28*38
        p_vector = p_vector.view(batch_size, feature_dim * self.patch_height * self.patch_width * 2,
                                 self.h_patches, self.w_patches)
        #   10, 128*8*8*2, 28, 38

        p_vector = self.BCIM(p_vector)  # 10, 2*28*38, 128*8*8
        return p_vector


class ObjectEncoder(nn.Module):
    def __init__(self, height, width, h_patches, w_patches):
        super().__init__()
        patch_height = int(height / h_patches)
        patch_width = int(width / w_patches)
        self.object_layernorm = nn.LayerNorm(feature_dim * patch_height * patch_width)
        self.p_layernorm = nn.LayerNorm(feature_dim * patch_height * patch_width)
        self.object_embedding_Matrix = nn.Sequential(nn.Linear(in_features=feature_dim * patch_width * patch_height,
                                                               out_features=feature_dim * patch_width * patch_height),
                                                     nn.LayerNorm(feature_dim * patch_width * patch_height))
        self.key_embedding_Matrix = nn.Sequential(nn.Linear(in_features=feature_dim * patch_width * patch_height,
                                                            out_features=feature_dim * patch_width * patch_height),
                                                  nn.LayerNorm(feature_dim * patch_width * patch_height))
        self.value_embedding_Matrix = nn.Sequential(nn.Linear(in_features=feature_dim * patch_width * patch_height,
                                                              out_features=feature_dim * patch_width * patch_height),
                                                    nn.LayerNorm(feature_dim * patch_width * patch_height))
        self.softmax_layer = nn.Softmax(dim=2)
        self.interaction_Matrix = nn.Sequential(nn.Linear(in_features=feature_dim * patch_height * patch_width,
                                                          out_features=feature_dim * patch_height * patch_width))
        self.linear_projection_layer = nn.Sequential(nn.Linear(in_features=feature_dim * patch_height * patch_width,
                                                               out_features=feature_dim * patch_height * patch_width),
                                                     nn.GELU(),
                                                     nn.Linear(in_features=feature_dim * patch_height * patch_width,
                                                               out_features=feature_dim * patch_height * patch_width))

    def forward(self, objector_vector, p_vector):
        batch_size = p_vector.size()[0]
        objector_vector = self.object_layernorm(objector_vector)  # N, 128*8*8
        p_vector = self.p_layernorm(p_vector)  # 10, 2*28*38, 128*8*8
        object_embedding = self.object_embedding_Matrix(objector_vector)  # N, 128*8*8
        key_embedding = self.key_embedding_Matrix(p_vector)  # 10, 2*28*38, 128*8*8
        value_embedding = self.value_embedding_Matrix(p_vector)  # 10, 2*28*38, 128*8*8

        key_embedding = key_embedding.permute(0, 2, 1).contiguous()  # 10, 128*8*8, 2*28*38

        A = torch.matmul(object_embedding, key_embedding)  # 10, N, 2*28*38
        A = self.softmax_layer(A)  # 10, N, 2*28*38

        res_object_vector = torch.matmul(A, value_embedding)  # 10, N, 128*8*8
        res_object_vector = res_object_vector.sum(dim=0) / batch_size  # N, 128*8*8
        objector_vector = objector_vector + res_object_vector  # N, 128*8*8

        interaction_object_vector = self.interaction_Matrix(objector_vector)
        objector_vector = objector_vector + interaction_object_vector  # N, 128*8*8

        linear_projection = self.linear_projection_layer(objector_vector)
        objector_vector = objector_vector + linear_projection

        return objector_vector  # N, 128*8*8


class EnlargeModule(nn.Module):
    def __init__(self, height, width, h_patches, w_patches):
        super().__init__()
        patch_height = int(height / h_patches)
        patch_width = int(width / w_patches)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=feature_dim * patch_height * patch_width * 2, out_channels=64,
                                             kernel_size=(3, 3), stride=(1, 1), padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128,
                                             kernel_size=(3, 3), stride=(1, 1), padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256,
                                             kernel_size=(3, 3), stride=(1, 1), padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())

    def forward(self, x):
        # x 10, 128*8*8*2, 28, 38
        out = self.conv1(x)  # 10, 64, 28, 38
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)  # 10, 64, 56, 76
        out = self.conv2(out)  # 10, 128, 56, 76
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)  # 10, 128, 112, 152
        out = self.conv3(out)  # 10, 256, 112, 152
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)  # 10, 256, 224, 304
        return out


class Encoder_Decoder_Block(nn.Module):
    def __init__(self, height, width, h_patches, w_patches):
        super().__init__()
        self.Encoder = ObjectEncoder(height, width, h_patches, w_patches)
        self.Decoder = ObjectDecoder(height, width, h_patches, w_patches)

    def forward(self, object_vector, p_vector):
        updated_object_vector = self.Encoder(object_vector, p_vector)
        updated_p_vector = self.Decoder(p_vector, updated_object_vector)
        return updated_object_vector, updated_p_vector



class Net(nn.Module):
    def __init__(self, training=True):
        super(Net, self).__init__()
        self.training = training
        height=512
        width=512
        h_patches=64
        w_patches=64
        self.height = height
        self.width = width
        self.h_patches = h_patches
        self.w_patches = w_patches
        patch_height = int(height / h_patches)
        patch_width = int(width / w_patches)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.Gr_Extractor = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
                                                    padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                                    stride=(1, 1),
                                                    padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                                    stride=(1, 1),
                                                    padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(inplace=True))
        self.Gf_Extractor = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
                                                    padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                                    stride=(1, 1),
                                                    padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                                    stride=(1, 1),
                                                    padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(inplace=True))
        self.Gr_Splicer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=feature_dim * patch_height * patch_width,
                      kernel_size=(patch_height, patch_width),
                      stride=(patch_height, patch_width),
                      padding=0))

        self.Gf_Splicer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=feature_dim * patch_height * patch_width,
                      kernel_size=(patch_height, patch_width),
                      stride=(patch_height, patch_width),
                      padding=0))

        self.object_vector = nn.Parameter(
            torch.normal(mean=0, std=1e-5, size=(16, feature_dim * patch_height * patch_width)))
        self.encoder_decoder_block1 = Encoder_Decoder_Block(height, width, h_patches, w_patches)
        self.encoder_decoder_block2 = Encoder_Decoder_Block(height, width, h_patches, w_patches)

        self.localization_layer = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256,
                                                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                nn.BatchNorm2d(256),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(in_channels=256, out_channels=256,
                                                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                nn.BatchNorm2d(256),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(in_channels=256, out_channels=128,
                                                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                nn.BatchNorm2d(128),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(in_channels=128, out_channels=1,
                                                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                nn.Sigmoid())
        self.classifier_layer = nn.Sequential(nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
                                              nn.Flatten(start_dim=1),
                                              nn.Linear(in_features=int(feature_dim * 2 * self.height * self.width / 4),
                                                        out_features=1),
                                              nn.Sigmoid())
        self.enlarge_layer = EnlargeModule(height, width, h_patches, w_patches)

    def forward(self, x):
        y = x
        batch_size = x.size()[0]
        Gr = self.Gr_Extractor(x)  # 10, 256, 224, 304
        Gf = self.Gf_Extractor(y)  # 10, 256, 224, 304

        Gr_p = self.Gr_Splicer(Gr)  # 10, 128*8*8, 28, 38
        Gf_p = self.Gf_Splicer(Gf)  # 10, 128*8*8, 28, 38

        Gr_p = Gr_p.view(batch_size, feature_dim * self.patch_height * self.patch_width,
                         self.h_patches * self.w_patches)
        # 10, 128*8*8, 28*38
        Gf_p = Gf_p.view(batch_size, feature_dim * self.patch_height * self.patch_width,
                         self.h_patches * self.w_patches)
        # 10, 128*8*8, 28*38

        p_vector = torch.cat([Gr_p, Gf_p], dim=2)  # 10, 128*8*8, 2*28*38
        p_vector = p_vector.permute(0, 2, 1).contiguous()  # 10, 2*28*38, 128*8*8

        updated_object_vector, p_vector = self.encoder_decoder_block1(self.object_vector, p_vector)
        updated_object_vector, p_vector = self.encoder_decoder_block2(updated_object_vector, p_vector)
        self.object_vector.data = updated_object_vector.data

        p_vector = p_vector.permute(0, 2, 1).contiguous()  # 10, 128*8*8, 2*28*38
        p_vector = p_vector.view(batch_size, feature_dim * self.patch_height * self.patch_width, 2, self.h_patches,
                                 self.w_patches)
        # 10, 128*8*8, 2, 28, 38
        p_vector = p_vector.view(batch_size, feature_dim * self.patch_height * self.patch_width * 2, self.h_patches,
                                 self.w_patches)
        # 10, 128*8*8*2, 28, 38
        p_vector = self.enlarge_layer(p_vector)  # 10, 256, 224, 304
        pre_mask = self.localization_layer(p_vector)
        # pre_class = self.classifier_layer(p_vector)
        # return pre_mask, pre_class
        if self.training:
            return pre_mask
        else:
            return pre_mask
