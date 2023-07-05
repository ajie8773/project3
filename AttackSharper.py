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
output_dir = r'E:\data\casia_data\test\imagesSharper20'

for filename in tqdm(os.listdir(input_dir)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    inpath = input_dir + "/" + filename  # 获取文件路径
    outpath = output_dir + "/" + filename  # 保存文件路径
    img = cv2.imdecode(np.fromfile(inpath), cv2.IMREAD_COLOR)  # cv2.imread不支持中文路径
    img = unsharp_mask(img, radius=8.0, amount=2.0)  # 此处调整amount 从0.5到2.0
    img = img * 255  # 返回值范围在[0,1]区间，因为rgb范围在0-255，所以”*255“是为了从[0,1]区间转到[0,255]区间
    img = img.astype(np.int_)
    cv2.imwrite(outpath, img)## cv2.imencode('.jpg', img)[1].tofile(outpath)     #cv2.imwrite不支持中文路径


