import argparse
import torch
import os
from PIL import Image
import sys
import cv2
sys.path.append("/home/fashionteam/pose/pytorch_Realtime_Multi-Person_Pose_Estimation/")
from a_pose import *
sys.path.append("/home/fashionteam/LIP_JPPNet/")
from seg import *

def action(dir, first):
	pose(dir)
	segmentation(dir, first)

# change to the directory that contains file each with images.png inside
BASE_DIR = "/home/fashionteam/mvc_test/"

data_path = BASE_DIR 
data_list = os.listdir(data_path)
data_list.sort()
print(data_list)
first = True

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

for elem in data_list:
	dir = os.path.join(data_path, elem)
	
	_seg = os.path.exists(os.path.join(dir, "segment.png"))
	_pose = os.path.exists(os.path.join(dir, "pose.pkl"))

	if (_seg and _pose):
		continue
	elif (_seg): pose(dir)
	elif (_pose): 
		segmentation(dir, first)
		first = False
	else: 
		action(dir, first)
		first = False
	
	print("{} DONE".format(elem))

