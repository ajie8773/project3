from PIL import Image, ImageFilter
# import ImageFilter
import os
from skimage import util, data, io, transform
from skimage.filters import unsharp_mask
import numpy as np
import random
import cv2


from tqdm import tqdm

input_dir = r'E:\data\casia_data\test\images'
output_dir = r'E:\data\casia_data\test\imagesBlur11'

for filename in tqdm(os.listdir(input_dir)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    inpath = input_dir + "/" + filename  # 获取文件路径
    outpath = output_dir + "/" + filename  # 保存文件路径
    img = cv2.imdecode(np.fromfile(inpath), cv2.IMREAD_COLOR)  # cv2.imread不支持中文路径
    out = cv2.GaussianBlur(img, (11, 11), 0)
    cv2.imencode('.jpg', out)[1].tofile(outpath)

