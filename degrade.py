import os
import cv2
import os
from utils.degradation import random_mixed_kernels,random_add_gaussian_noise,random_add_jpg_compression
import math
import numpy as np
import  random
# random.seed(1234)#exp use
random.seed(1234)

def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # 限制坐标区域不超过样本大小
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


if __name__ == '__main__':
    kernel_list= ['iso', 'aniso']
    kernel_prob= [0.5, 0.5]
    blur_sigma=[0.1, 10]
    downsample_range= [1, 12]
    noise_range=[0, 15]
    jpeg_range=[30, 100]
    blur_kernel_size= 31
    kernel = random_mixed_kernels(
                kernel_list,
                kernel_prob,
                blur_kernel_size,
                blur_sigma,
                blur_sigma,
                [-math.pi, math.pi],
                noise_range=None
            )
    input_dir = './data/NUDT_mosaic/images/'
    mask_dir ='./data/NUDT_mosaic/masks/'
    img_list =os.listdir(input_dir)
    w_cut = './add_noise/NUDT_mosaic/'
    os.makedirs(w_cut,exist_ok=True)
    for data in img_list:
        img_gt =cv2.imread(os.path.join(input_dir,data))
        h, w, _ = img_gt.shape
        img_mask = cv2.imread(os.path.join(mask_dir, data))
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        lam = np.random.beta(0.25, 0.25)
        scale = np.random.uniform(downsample_range[0],downsample_range[1])
        chooses = [[0,0,250,250],[0,266,0,512],[266,0,512,0],[266,266,512,512]]
        c = random.choices(chooses)
        bbx1, bby1, bbx2, bby2 =c[0][0],c[0][1],c[0][2],c[0][3]
        img_lq[bbx1:bbx2, bby1:bby2, :] = img_gt[bbx1:bbx2, bby1:bby2, :]
        img_lq[img_mask>0] = img_gt[img_mask>0]
        cv2.imwrite(os.path.join(w_cut, data), img_lq)