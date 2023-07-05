import os
import glob
import data



class casiaData(data.BaseDataset):
    def __init__(self, phase, opt):
        root = opt.dataset_root

        if phase == "val":
            dir_MASK, dir_IMG = self.get_valsubdir()
        elif phase == "test":
            dir_MASK, dir_IMG = self.get_testsubdir()
        else:
            dir_MASK, dir_IMG = self.get_subdir()
        self.MASK_paths = sorted(glob.glob(os.path.join(root, dir_MASK, "*.png")))
        self.IMG_paths = sorted(glob.glob(os.path.join(root, dir_IMG, "*.*")))


        super().__init__(phase, opt)

    def get_subdir(self):
        dir_MASK = "train/masks"
        dir_IMG = "train/images"
        return dir_MASK, dir_IMG

    def get_valsubdir(self):
        dir_MASK = "val/masks"
        dir_IMG = "val/images"
        return dir_MASK, dir_IMG

    def get_testsubdir(self):
        dir_MASK = "test/masks"
        dir_IMG = "test/images"
        return dir_MASK, dir_IMG