import cv2
import os
import numpy as np
import random
from tqdm import tqdm


import numpy as np

def random_crop(image, mask,crop_size):
    height, width = image.shape[:2]
    target_height, target_width = crop_size
    start_y = np.random.randint(0, height - target_height + 1)
    start_x = np.random.randint(0, width - target_width + 1)
    cropped_image = image[start_y:start_y + target_height, start_x:start_x + target_width]
    crop_mask = mask[start_y:start_y + target_height, start_x:start_x + target_width]
    return cropped_image,crop_mask



if __name__ == '__main__':

    a = []
    random.seed(1234)
    np.random.seed(1234)
    save_path = './data/NUDT_mosaic/'
    imgs_path = './data/NUDT-SIRST/images/'
    masks = './data//NUDT-SIRST/masks/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'masks'), exist_ok=True)

    with open('./data/NUDT-SIRST/50_50/train.txt','r') as f:
        res = f.readlines()
        a= [b[:-1]+'.png' for b in res]
    random.shuffle(a)
    b = a.copy()
    lens = len(a)
    height = 256
    width = 256
    lam = np.random.beta(0.4, 0.4)
    with open(os.path.join(save_path,'train.txt'),'w') as f:
            for i,img_name in enumerate(a):
                # print(i)
                if i >= 400:
                    break
                flag =0
                lam = np.random.beta(0.25, 0.25)

                c = b.copy()
                c.remove(img_name)
                random.shuffle(c)
                img1 = cv2.imread(os.path.join(imgs_path,img_name))
                mask1 = cv2.imread(os.path.join(masks, img_name))

                img2 = cv2.imread(os.path.join(imgs_path, c[1]))
                mask2 = cv2.imread(os.path.join(masks, c[1]))
                img3 = cv2.imread(os.path.join(imgs_path,c[2]))
                mask3 = cv2.imread(os.path.join(masks, c[2]))

                img4 = cv2.imread(os.path.join(imgs_path, c[3]))
                mask4 = cv2.imread(os.path.join(masks, c[3]))
                if img4 is None or img3 is None or img2 is None or img1 is None or mask1 is None or mask2 is None or mask3 is None or mask4 is None:
                    print(img4 is None , img3 is None , img2 is None , img1 is None , mask1 is None , mask2 is None , mask3 is None ,mask4 is None)
                    continue
                img_list = [img1,img2, img3, img4]
                mask_list = [mask1, mask2, mask3, mask4]

                for j,(img,mask) in enumerate(zip(img_list,mask_list)):
                    if img.shape[0]<256 or img.shape[1]<256:
                        img =cv2.resize(img,(256,256))
                        mask = cv2.resize(mask,(256,256))
                    elif img.shape[0]>256 or img.shape[1]>256:
                        crop_size =(256,256)
                        cropped_image, crop_mask = random_crop(img,mask,crop_size)
                        while len(crop_mask[crop_mask>0])<0:
                            cropped_image, crop_mask = random_crop(img, mask, crop_size)
                        img,mask = cropped_image, crop_mask
                    img_list[j] = img
                    mask_list[j] = mask
                img_1 = np.vstack((img_list[0].copy(), img_list[1].copy()))
                img_2 = np.vstack((img_list[2].copy(), img_list[3].copy()))
                img_new = np.hstack((img_1, img_2))
                mask_1 = np.vstack((mask_list[0].copy(), mask_list[1].copy()))
                mask_2 = np.vstack((mask_list[2].copy(), mask_list[3].copy()))
                mask_new = np.hstack((mask_1, mask_2))
                cv2.imwrite(os.path.join(save_path+'images/',img_name.split('.')[0]+'_moc.png'),img_new)
                cv2.imwrite( os.path.join(save_path+'masks/',img_name.split('.')[0]+'_moc.png'),mask_new)
                f.write(img_name.split('.')[0]+'_moc'+'\n')