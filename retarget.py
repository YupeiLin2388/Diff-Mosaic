import os

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

img_path = './results/NUDT_moc/'
mask_dir ='./data/NUDT_mosaic/masks/'
gt_dir ='./data/NUDT_mosaic/images/'
data_list =os.listdir(img_path)
print(data_list)
retarget = "./results/retarget/NUDT_test/"
os.makedirs(retarget,exist_ok=True)
need_traintxt = True
if need_traintxt:
    with open(retarget+'train.txt','a') as f:
        for data in data_list:
            img_gt =cv2.imread(os.path.join(gt_dir,data.split('_0')[0]+'.png'))
            mask = cv2.imread(os.path.join(mask_dir, data.split('_0')[0] + '.png'))
            try:
                im = cv2.imread(os.path.join(img_path, data))
            except:
                continue

            os.makedirs(os.path.join(retarget, 'images'), exist_ok=True)
            os.makedirs(os.path.join(retarget, 'masks'), exist_ok=True)
            im[mask > 0] = img_gt[mask > 0]
            cv2.imwrite(os.path.join(retarget + 'images/', data), im)
            cv2.imwrite(os.path.join(retarget + 'masks/', data), mask)
            f.write(data.split('.')[0]+ '\n')


else:
    for data in data_list:
        img_gt = cv2.imread(os.path.join(gt_dir, data.split('_0')[0] + '.png'))
        mask = cv2.imread(os.path.join(mask_dir, data.split('_0')[0] + '.png'))
        im = cv2.imread(os.path.join(img_path, data))
        print(img_gt is None,mask is None,im is None)
        im[mask>0] =img_gt[mask>0]
        cv2.imwrite(os.path.join(retarget,data),im)