import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image

def imsave(nparray,fname):
    Image.fromarray(nparray).save(fname)

TARGET_SIZE = (512,512)
base_dir = "/home/fashionteam/dataset_MVC_bottoms/test/"
folder_list = os.listdir(base_dir)

t = 0
for i, folder in enumerate(folder_list):
    image_list = os.listdir(os.path.join(base_dir, folder))
    image_list = [image for image in image_list if os.path.isfile(os.path.join(base_dir, folder, image)) and '4x' in image and not ('resize' in image)]
    for image in image_list:
        deep_folder = osp.join(base_dir, folder, image.split('-')[1])
        print(deep_folder)

        if osp.isfile(osp.join(deep_folder, 'segment_vis_resize.jpg')):
            os.remove(osp.join(deep_folder, 'segment_vis_resize.jpg'))

        if osp.isfile(osp.join(deep_folder, 'segment_resize.jpg')):
            os.remove(osp.join(deep_folder, 'segment_resize.jpg'))

        if osp.isfile(osp.join(deep_folder, 'pose.pkl')):
            os.remove(osp.join(deep_folder, 'pose.pkl'))
        
        if osp.isfile(osp.join(deep_folder, 'pose_resize.pkl')):
            os.remove(osp.join(deep_folder, 'pose_resize.pkl'))