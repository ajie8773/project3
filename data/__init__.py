import importlib
import numpy as np
import skimage.io as io
import skimage.color as color
import torch
import torch_dct

from augments import Augment
from tqdm import tqdm
import os
import cv2
from torchvision import transforms

def generate_loader(phase, opt):
    kwargs = {
        "batch_size": opt.batch_size if phase == "train" else 1,
        "num_workers": opt.num_workers if phase == "train" else 1,
        "shuffle": phase == "train",
        "drop_last": phase == "train",
    }

    dataset = getattr(importlib.import_module("data.casiadata"), "casiaData")(phase, opt)
    return torch.utils.data.DataLoader(dataset, **kwargs)




class BaseDataset(torch.utils.data.Dataset):

    def cv_imread(self, filePath, color=cv2.IMREAD_COLOR):
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), color)
        return cv_img

    def toedge(self, gray):

        edge = cv2.erode(gray, self.kernel)
        edge = gray - edge
        return edge

    def __init__(self, phase, opt):
        self.phase = phase
        self.opt = opt
        self.aug = Augment(self.opt)
        self.size = 512
        self.kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    def __getitem__(self, index):
        if self.phase == "train":

            img = self.cv_imread(self.IMG_paths[index])
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR).copy()


            label = self.cv_imread(self.MASK_paths[index], cv2.IMREAD_GRAYSCALE)
            label = label // 255

            # img_edge = cv2.Canny(img,10,100)
            # img_edge = img_edge//255
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_edge = self.toedge(gray)
            img_edge = img_edge.astype(np.float32) / 255

            img = img.astype(np.float32) / 255.
            img = img.transpose([2, 0, 1])
            label_edge = self.toedge(label)

            # label = label == 255
            # label = label.astype(np.uint8)

            img_edge = cv2.resize(img_edge, (self.size, self.size), interpolation=cv2.INTER_LINEAR).copy()
            label = cv2.resize(label, (self.size, self.size), interpolation=cv2.INTER_LINEAR).copy()
            label = torch.from_numpy(label).float().unsqueeze(0)
            label_edge = cv2.resize(label_edge, (self.size, self.size), interpolation=cv2.INTER_LINEAR).copy()
            label_edge = torch.from_numpy(label_edge).float().unsqueeze(0)
            img_edge = torch.from_numpy(img_edge).float().unsqueeze(0)
            return label, img, img_edge, label_edge

        else:
            img = self.cv_imread(self.IMG_paths[index])
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR).copy()

            label = self.cv_imread(self.MASK_paths[index], cv2.IMREAD_GRAYSCALE)
            label = label // 255

            NAME = (os.path.split(self.MASK_paths[index])[1]).split('.')[0]

            img = img.astype(np.float32) / 255.
            img = img.transpose([2, 0, 1])

            label = torch.from_numpy(label).float().unsqueeze(0)

            return label, img, NAME

        if len(IMG.shape) < 3:
            IMG = color.gray2rgb(IMG)


    def __len__(self):
        return len(self.IMG_paths)