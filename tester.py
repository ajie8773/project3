import os
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import generate_loader
from tqdm import tqdm
from utils import calculate_mae
from sklearn import metrics
import cv2
import numpy as np

def sklearn_calprecise(pred, img_gt):
    gt = img_gt.squeeze(0)
    gt = gt.squeeze(0)
    gt = gt.detach().cpu().numpy().astype('uint8').flatten()
    pred_numpy = pred.detach().cpu().numpy().astype('uint8').flatten()
    precision = metrics.precision_score(gt, pred_numpy)
    recall = metrics.recall_score(gt, pred_numpy)
    f1 = metrics.f1_score(gt, pred_numpy)
    #acc = metrics.accuracy_score(gt, pred_numpy)

    return precision, recall, f1

def calprecise(pred, img_gt):
    acc_mask = torch.mul(pred.float(), img_gt)
    acc_sum = acc_mask.sum()

    pred_sum = pred.sum()
    gt_sum = img_gt.sum()

    acc = acc_sum / (pred_sum + 0.0001)
    recall = acc_sum / (gt_sum + 0.0001)
    f1 = 2 * acc * recall / (acc + recall + 0.0001)
    return acc, recall, f1

def calIOU(Mask, pred):
    ComArea = np.sum(Mask*pred)
    iou = ComArea/(np.sum(Mask)+np.sum(pred)-ComArea+1e-8)
    return iou

class Tester():
    def __init__(self, module, opt):
        self.opt = opt

        self.dev = torch.device("cuda:{}".format(opt.GPU_ID) if torch.cuda.is_available() else "cpu")
        self.net = module.Net(opt)
        self.net = self.net.to(self.dev)

        msg = "# params:{}\n".format(
            sum(map(lambda x: x.numel(), self.net.parameters())))
        print(msg)

        self.test_loader = generate_loader("test", opt)

    @torch.no_grad()
    def evaluate(self, path):
        opt = self.opt

        try:
            print('loading model from: {}'.format(path))
            self.load(path)
        except Exception as e:
            print(e)

        self.net.eval()

        if opt.save_result:
            save_root = os.path.join(opt.save_root, opt.save_msg)
            os.makedirs(save_root, exist_ok=True)

        precision = 0
        recall = 0
        f1 = 0
        test_iou = []
        test_auc = []
        
        for i, inputs in enumerate(tqdm(self.test_loader)):
            MASK = inputs[0].to(self.dev)
            IMG = inputs[1].to(self.dev)
            NAME = inputs[2][0]

            b, c, h, w = MASK.shape
            
            pred = self.net(IMG)

            MASK[MASK > 0] = 1


            #pred_sal = F.pixel_shuffle(pred, 4)
            pred = F.interpolate(pred, (h,w), mode='bilinear', align_corners=False)
            pred = torch.sigmoid(pred).squeeze()

            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            img_precision, img_recall, img_f1 = sklearn_calprecise(pred, MASK)
            test_iou.append(calIOU(MASK.squeeze(0).squeeze(0).cpu().numpy(), pred.cpu().numpy()))
            test_auc.append(max(metrics.roc_auc_score(MASK.squeeze(0).squeeze(0).cpu().numpy().astype('uint8').ravel(), pred.cpu().numpy().astype('uint8').ravel()), metrics.roc_auc_score(MASK.squeeze(0).squeeze(0).cpu().numpy().astype('uint8').ravel(), (1-pred).cpu().numpy().astype('uint8').ravel())))

            mask = (MASK * 255.).squeeze().detach().cpu().numpy().astype('uint8')
            pred = (pred * 255.).detach().cpu().numpy().astype('uint8')

            # matt_img = pred[0].repeat(1,256,1,1)
            # matt_img = F.pixel_shuffle(matt_img, 16)
            # matt_img = F.interpolate(matt_img, (h,w), mode='bilinear', align_corners=False)
            # matt_img = torch.sigmoid(matt_img)
            # matt_img = (matt_img*255.).squeeze().detach().cpu().numpy().astype('uint8')



            if opt.save_result:
                save_path_msk = os.path.join(save_root, "{}_msk.png".format(NAME))
                #save_path_matt = os.path.join(save_root, "{}_matt.png".format(NAME))
                origal_img_path = os.path.join("E:/data/IMD2020_wjh/test/images/", NAME + '.tif')#[:-3]
                origal_img = cv2.imread(origal_img_path)#Tp_D_CND_M_N_art00076_art00077_10289_gt
                if origal_img is None:
                    origal_img_path = os.path.join("E:/data/IMD2020_wjh/test/images/", NAME + '.jpg')
                    origal_img = cv2.imread(origal_img_path)  #
                origal_img_save_path = os.path.join(save_root, "{}_img.png".format(NAME))
                cv2.imwrite(origal_img_save_path, origal_img)
                io.imsave(save_path_msk, mask)
                #io.imsave(save_path_matt, matt_img)
                
                if opt.save_all:
                    for idx, sal in enumerate(pred[1:]):
                        scale=224//(sal.shape[-1])
                        sal_img = F.pixel_shuffle(sal,scale)
                        sal_img = F.interpolate(sal_img, (h,w), mode='bilinear', align_corners=False)
                        sal_img = torch.sigmoid(sal_img)
                        sal_path = os.path.join(save_root, "{}_sal_{}.png".format(NAME, idx))
                        sal_img = sal_img.squeeze().detach().cpu().numpy()
                        sal_img = (sal_img * 255).astype('uint8')
                        io.imsave(sal_path, sal_img)
                else:
                    # save pred image
                    save_path_pred = os.path.join(save_root, "{}_pre.png".format(NAME))
                    io.imsave(save_path_pred, pred)

            precision += img_precision
            recall += img_recall
            f1 += img_f1

        print("total outcome-> precision:%.4f recall:%.4f f1:%.4f" % (
            precision / len(self.test_loader), recall / len(self.test_loader),
            f1 / len(self.test_loader)))
        print("iou ", np.mean(test_iou))
        print("auc ", np.mean(test_auc))

    def load(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state_dict)
        return

