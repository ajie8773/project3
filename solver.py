import os
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LogWritter, calculate_mae
from data import generate_loader
from loss_fn import ConfidentLoss
from tqdm import tqdm
from sklearn import metrics
import numpy as np


def calIOU(Mask, pred):
    ComArea = np.sum(Mask*pred)
    iou = ComArea/(np.sum(Mask)+np.sum(pred)-ComArea+1e-8)
    return iou

class Solver():
    def __init__(self, module, opt):
        self.opt = opt
        self.logger = LogWritter(opt)
        
        self.dev = torch.device("cuda:{}".format(opt.GPU_ID) if torch.cuda.is_available() else "cpu")
        self.net = module.Net(opt)
        self.net = self.net.to(self.dev)
            
        msg = "# params:{}\n".format(sum(map(lambda x: x.numel(), self.net.parameters())))
        print(msg)
        self.logger.update_txt(msg)

        self.loss_fn = ConfidentLoss(lmbd=opt.lmbda)
        
        # gather parameters
        base, head = [], []
        for name, param in self.net.named_parameters():
            if "encoder" in name:
                base.append(param)
            else:
                head.append(param)
        #assert base!=[], 'encoder is empty'
        self.optim = torch.optim.Adam([{'params':base},{'params':head}], opt.lr,betas=(0.9, 0.999), eps=1e-8)

        self.train_loader = generate_loader("train", opt)
        self.eval_loader = generate_loader("val", opt)

        self.best_f1, self.best_f1_step = 0, 0
        self.best_auc, self.best_auc_step = 0, 0
        self.best_iou, self.best_iou_step = 0,0

    def fit(self):
        opt = self.opt
        
        for step in range(self.opt.max_epoch):
            #  assign different learning rate
            power = (step+1)//opt.decay_step
            self.optim.param_groups[0]['lr'] = opt.lr * (0.5 ** power) * 0.1    # for base
            self.optim.param_groups[1]['lr'] = opt.lr * (0.5 ** power)         # for head
            print('LR base: {}, LR head: {}'.format(self.optim.param_groups[0]['lr'],
                                                    self.optim.param_groups[1]['lr']))


            for i, inputs in enumerate(tqdm(self.train_loader)):
                label, img, img_edge, label_edge = inputs
                self.optim.zero_grad()

                img = img.to(self.dev)
                label = label.to(self.dev)
                img_edge = img_edge.to(self.dev)
                label_edge = label_edge.to(self.dev)

                B,_,H,W = img.shape
                label_32 = F.interpolate(label, size=(H//32, W//32), mode="bilinear",
                                         align_corners=True)
                label_16 = F.interpolate(label, size=(H//16, W//16), mode="bilinear",
                                         align_corners=True)
                label_08 = F.interpolate(label, size=(H//8, W//8), mode="bilinear",
                                         align_corners=True)
                label_04 = F.interpolate(label, size=(H//4, W//4), mode="bilinear",
                                         align_corners=True)

                [pred_01] = self.net(img)

                loss = self.loss_fn.get_value(pred_01, label)

                loss.backward()

                if opt.gclip > 0:
                    torch.nn.utils.clip_grad_value_(self.net.parameters(), opt.gclip)

                self.optim.step()
            # eval
            print("[{}/{}]".format(step+1, self.opt.max_epoch))
            self.summary_and_save(step)
            

    def summary_and_save(self, step):
        print('evaluate...')
        val_f1, val_auc, val_iou = self.evaluate()

        if val_f1 >= self.best_f1:
            self.best_f1, self.best_f1_step = val_f1, step + 1
            self.save(step)
        if val_auc >= self.best_auc:
            self.best_auc, self.best_auc_step = val_auc, step + 1
        if val_iou >= self.best_iou:
            self.best_iou, self.best_iou_step = val_iou, step + 1
        else:
            if self.opt.save_every_ckpt:
                self.save(step)

        msg1 = "[{}/{}] f1: {:.3f} (Best F1: {:.3f} @ {}step)\n".format(step+1, self.opt.max_epoch, val_f1, self.best_f1, self.best_f1_step) \
               + "[{}/{}] auc: {:.3f} (Best auc: {:.3f} @ {}step)\n".format(step + 1, self.opt.max_epoch, val_auc, self.best_auc, self.best_auc_step) \
               +"[{}/{}] iou: {:.3f} (Best iou: {:.3f} @ {}step)\n".format(step+1, self.opt.max_epoch, val_iou, self.best_iou, self.best_iou_step)
        print(msg1)
        self.logger.update_txt(msg1)

    @torch.no_grad()
    def evaluate(self):
        opt = self.opt
        self.net.eval()

        if opt.save_result:
            save_root = os.path.join(opt.save_root, opt.dataset)
            os.makedirs(save_root, exist_ok=True)

        f1 = 0
        auc = []
        iou = []

        for i, inputs in enumerate(tqdm(self.eval_loader)):
            MASK = inputs[0].to(self.dev)
            IMG = inputs[1].to(self.dev)
            NAME = inputs[2][0]


            b,c,h,w = MASK.shape

            pred = self.net(IMG)

            MASK = MASK.squeeze().detach().cpu().numpy()
            #pred_sal = F.pixel_shuffle(pred[-1], 4) # from 56 to 224
            pred = F.interpolate(pred, (h,w), mode='bilinear', align_corners=False)

            pred = torch.sigmoid(pred).squeeze().detach().cpu().numpy()

            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            MASK[MASK > 0] = 1

            if opt.save_result:
                pred = (pred * 255.).astype('uint8')
                save_path = os.path.join(save_root, "{}_sal_eval.png".format(NAME))
                io.imsave(save_path, pred)


            auc.append(max(metrics.roc_auc_score(MASK.astype(int).ravel(), pred.ravel()), metrics.roc_auc_score(MASK.astype(int).ravel(), (1-pred).ravel())))

            iou.append(calIOU(MASK, pred))

            pred = pred.astype('uint8').flatten()
            MASK = MASK.astype('uint8').flatten()
            img_f1 = metrics.f1_score(MASK, pred)
            f1 += img_f1





        self.net.train()
        return f1 / len(self.eval_loader), np.mean(auc), np.mean(iou)

    def load(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state_dict)
        return

    def save(self, step):
        os.makedirs(self.opt.ckpt_root, exist_ok=True)
        save_path = os.path.join(self.opt.ckpt_root, str(step)+".pt")
        torch.save(self.net.state_dict(), save_path)
