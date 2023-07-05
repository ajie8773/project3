import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
import skimage.io as io


def square_patch_contrast_loss(feat, mask, device, temperature=0.6):
    # feat shape should be (Batch, Total_Patch_number, Feature_dimension)
    # mask should be (Batch, H, W)

    mem_mask = torch.eq(mask, mask.transpose(1, 2)).float()
    mem_mask_neg = torch.add(torch.negative(mem_mask), 1)

    feat_logits = torch.div(torch.matmul(feat, feat.transpose(1, 2)), temperature)
    identity = torch.eye(feat_logits.shape[-1]).to(device)
    neg_identity = torch.add(torch.negative(identity), 1).detach()
    del identity
    torch.cuda.empty_cache()

    feat_logits = torch.mul(feat_logits, neg_identity)

    feat_logits_max, _ = torch.max(feat_logits, dim=1, keepdim=True)
    feat_logits = feat_logits - feat_logits_max.detach()

    feat_logits = torch.exp(feat_logits)

    neg_sum = torch.sum(torch.mul(feat_logits, mem_mask_neg), dim=-1)
    del mem_mask_neg
    torch.cuda.empty_cache()
    denominator = torch.add(feat_logits, neg_sum.unsqueeze(dim=-1))
    division = torch.div(feat_logits, denominator + 1e-18)

    del feat_logits
    torch.cuda.empty_cache()

    loss_matrix = -torch.log(division + 1e-18)
    loss_matrix = torch.mul(loss_matrix, mem_mask)
    loss_matrix = torch.mul(loss_matrix, neg_identity)
    loss = torch.sum(loss_matrix, dim=-1)


    del neg_identity, loss_matrix
    torch.cuda.empty_cache()

    loss = torch.div(loss, mem_mask.sum(dim=-1) - 1 + 1e-18)

    del mem_mask
    torch.cuda.empty_cache()

    return loss


def flat(mask):
    batch_size = mask.shape[0]
    h = 16
    mask = F.interpolate(mask,size=(int(h),int(h)), mode='bilinear')
    x = mask.view(batch_size, 1, -1).permute(0, 2, 1)
    # print(x.shape)  b 28*28 1
    g = x @ x.transpose(-2,-1) # b 28*28 28*28
    g = g.unsqueeze(1) # b 1 28*28 28*28
    return g


def square_patch_contrast_loss(feat, mask, device, temperature=0.6):
    # feat shape should be (Batch, Total_Patch_number, Feature_dimension)
    # mask should be (Batch, H, W)

    mem_mask = torch.eq(mask, mask.transpose(1, 2)).float()
    mem_mask_neg = torch.add(torch.negative(mem_mask), 1)

    feat_logits = torch.div(torch.matmul(feat, feat.transpose(1, 2)), temperature)
    identity = torch.eye(feat_logits.shape[-1]).to(device)
    neg_identity = torch.add(torch.negative(identity), 1).detach()

    feat_logits = torch.mul(feat_logits, neg_identity)

    feat_logits_max, _ = torch.max(feat_logits, dim=1, keepdim=True)
    feat_logits = feat_logits - feat_logits_max.detach()

    feat_logits = torch.exp(feat_logits)

    neg_sum = torch.sum(torch.mul(feat_logits, mem_mask_neg), dim=-1)
    denominator = torch.add(feat_logits, neg_sum.unsqueeze(dim=-1))
    division = torch.div(feat_logits, denominator + 1e-18)

    loss_matrix = -torch.log(division + 1e-18)
    loss_matrix = torch.mul(loss_matrix, mem_mask)
    loss_matrix = torch.mul(loss_matrix, neg_identity)
    loss = torch.sum(loss_matrix, dim=-1)

    loss = torch.div(loss, mem_mask.sum(dim=-1) - 1 + 1e-18)

    return loss



class ConfidentLoss:
    def __init__(self, lmbd=3):
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.weight = [1.0]
        self.lmbda = float(int(lmbd) / 10)

    def weighted_bce(self, pred, gt):
        weit = 1 + 4 * torch.abs(F.avg_pool2d(gt, kernel_size=31, stride=1, padding=15) - gt)
        wbce = (self.bce(pred, gt) * weit).sum(dim=[2, 3]) / weit.sum(dim=[2, 3])

        smooth = 1e-8
        pred = torch.sigmoid(pred)
        inter = ((pred * gt) * weit).sum(dim=(2, 3))
        union = ((pred + gt) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter+smooth) / (union - inter+smooth)

        loss = (wbce+wiou).mean()
        return loss


    def confident_loss(self, pred, gt, beta=2):
        y = torch.sigmoid(pred)
        weight = beta * y * (1 - y)
        weight = weight.detach()
        loss = (self.bce(pred, gt) * weight).mean()
        loss2 = self.lmbda * beta * (y * (1 - y)).mean()
        return loss + loss2


    def rrunet_loss(self, pred, label):
        creition = nn.BCELoss()
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        label = label.view(-1)
        loss = creition(pred, label)
        return loss

    def cflnet_loss(self, proj, out, label):
        pred = F.interpolate(out, (256, 256), mode='bilinear', align_corners=True)
        proj = F.interpolate(proj, (256, 256), mode='bilinear', align_corners=True)
        p_len = 4


        cfeature = F.avg_pool2d(proj, kernel_size=p_len, stride=p_len)
        Ba, Ch, _, _ = cfeature.shape
        cfeature = cfeature.view(Ba, Ch, -1)
        cfeature = torch.transpose(cfeature, 1, 2)
        cfeature = F.normalize(cfeature, dim=-1)

        mask_con = label.detach()
        mask_con = F.avg_pool2d(mask_con, kernel_size=p_len, stride=p_len)
        mask_con = (mask_con > 0.5).int().float()
        mask_con = mask_con.view(Ba, -1)
        mask_con = mask_con.unsqueeze(dim=1)

        device = torch.device("cuda")
        contrast_temperature = 0.1
        c_loss = square_patch_contrast_loss(cfeature, mask_con, device, contrast_temperature)
        c_loss = c_loss.mean(dim=-1)
        c_loss = c_loss.mean()
        imbalance_weight = torch.tensor([0.0892, 0.9108]).to(device)
        target = label.gt(0.5).float()
        loss = self.weighted_bce(pred, target).mean()
        total_loss = loss + c_loss

        return total_loss

    def get_value(self, pred, label):
        target = label.gt(0.5).float()
        return self.weighted_bce(pred, target).mean()


    def MSE_loss(self, pred, label):
        weit = 1 + 4 * torch.abs(F.avg_pool2d(label, kernel_size=31, stride=1, padding=15) - label)
        loss = ((F.mse_loss(pred, label) * weit).sum(dim=[2, 3]) / weit.sum(dim=[2, 3])).mean()
        return loss

    def L1_loss(self, pred, label):
        weit = 1 + 4 * torch.abs(F.avg_pool2d(label, kernel_size=31, stride=1, padding=15) - label)
        l1loss = ((F.l1_loss(pred, label, None) * weit).sum(dim=[2, 3]) / weit.sum(dim=[2, 3])).mean()
        return l1loss

    def single_value(self, pred, label):
        target = label.gt(0.5).float()
        loss = self.weighted_bce(pred, target).mean()
        return loss


    #feat = torch.randn(2,256,64,64)
    #tar = torch.randn(2,1,64,64)
    def contrastive_loss(self, dev, feat, target):
        B,C,H,W = feat.shape
        patch_size = 1
        contrast_temperature = 0.1

        # target = F.avg_pool2d(target, kernel_size=patch_size, stride=patch_size).detach()
        target = target.gt(0.5).float().detach()
        target = target.reshape(B, 1, -1)

        # feat = F.avg_pool2d(feat, kernel_size=patch_size, stride=patch_size)
        feat = feat.reshape(B, C, -1).transpose(1, 2)
        feat = F.normalize(feat, dim=-1)

        c_loss = square_patch_contrast_loss(feat, target, dev, contrast_temperature)
        c_loss = c_loss.mean(dim=-1)
        c_loss = c_loss.mean()
        return c_loss

    def my_loss(self, pred, label):
        B,_,_,_ = pred.shape
        pred = pred.squeeze(1)
        label = label.squeeze(1).detach()
        target = label.gt(0.5).float()
        temper = (pred * target).view(B, -1)
        not_target = (1-target)
        no_temper = (pred * not_target).view(B, -1)
        cs = torch.cosine_similarity(temper, no_temper, dim=-1).mean()
        return cs

    def att_loss(self, attn_pred, label, pred_32):
        attn_label = flat(label)
        attn_pred_32 = flat(torch.sigmoid(pred_32.detach()))
        w = torch.abs(attn_label - attn_pred_32) + 1
        loss = F.binary_cross_entropy_with_logits(attn_pred, attn_label, weight=w, reduction='mean')#2,1,784,784
        return loss


