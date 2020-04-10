import argparse
import torch
import os
from PIL import Image
import sys
import cv2
sys.path.append("/home/fashionteam/pose/pytorch_Realtime_Multi-Person_Pose_Estimation/")
from a_pose import *
sys.path.append("/home/fashionteam/CIHP_PGN/")
from seg import *

def action(dir, first):
	pose(dir)
	segmentation(dir, first)

# change to the directory that contains file each with image.png inside
BASE_DIR = "/home/fashionteam/body_face"

# action(BASE_DIR,True)

data_path = BASE_DIR 
data_list = os.listdir(data_path)
data_list.sort()
first = True


# path_image = osp.join("/home/fashionteam/dataset_test/2/F_korea_2.png")
# target = Image.open(path_image)

# seg = seg_img(target, True)
# seg[0].save('segment_vis.png')
# cv2.imwrite('segment.png',seg[1])
# print(' [*] SAVED')

segmentation(BASE_DIR, first)