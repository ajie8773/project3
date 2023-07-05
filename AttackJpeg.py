from PIL import Image
import os
import cv2
from tqdm import tqdm
#wjh\Columbia  #wjh\NIST16  #IMD2020_wjh
input_dir1 = r'E:\data\IMD2020_wjh\test\images'
output_dir1 = r'E:\data\IMD2020_wjh\test\imagesJPEGquality100'

input_dir2 = r'E:\data\wjh\Columbia\test\images'
output_dir2 = r'E:\data\wjh\Columbia\test\imagesJPEGquality100'

input_dir3 = r'E:\data\wjh\NIST16\test\images'
output_dir3 = r'E:\data\wjh\NIST16\test\imagesJPEGquality100'

for filename in tqdm(os.listdir(input_dir1)):
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    inpath = input_dir1 + "/" + filename  # 获取文件路径
    img = Image.open(inpath)  # 读取图片
    outpath = output_dir1 + "/" + filename  # 获取文件路径
    img.save(outpath, "JPEG", quality=95)

for filename in tqdm(os.listdir(input_dir2)):
    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2)
    inpath = input_dir2 + "/" + filename  # 获取文件路径
    img = Image.open(inpath)  # 读取图片
    outpath = output_dir2 + "/" + filename  # 获取文件路径
    img.save(outpath, "JPEG", quality=95)

for filename in tqdm(os.listdir(input_dir3)):
    if not os.path.exists(output_dir3):
        os.makedirs(output_dir3)
    inpath = input_dir3 + "/" + filename  # 获取文件路径
    img = Image.open(inpath)  # 读取图片
    outpath = output_dir3 + "/" + filename  # 获取文件路径
    img.save(outpath, "JPEG", quality=95)


    # img_noise = gaussian_noise(noise_img, 0, 0.12)  # 高斯噪声
    # img_noise = sp_noise(noise_img,0.025)# 椒盐噪声 salt pepper noise
    # img_noise  = random_noise(noise_img,500)# 随机噪声




