from PIL import Image
import os
import numpy as np
import os.path as osp
import cv2

base_dir = '/home/fashionteam/underwear_512/'

for folder in os.listdir(base_dir):
    image = np.array(Image.open(osp.join(base_dir, folder, 'crop.png')))
    # print(image.shape)
    # print(np.mean(image[:,:,3]))
    sum_image = image[:,:,0:1]+image[:,:,1:2]+image[:,:,2:3]
    # mask = (image[:,:,1:2] < 253).astype(np.float32) * 255
    mask = image[:,:,3:4]

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    # print(image[:,:,0:1])
    # print(mask)
    # print(mask.shape)
    cv2.imwrite(osp.join(base_dir, folder, 'image_mask.jpg'), mask)