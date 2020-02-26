import sys
sys.path.append("/home/fashionteam/pose/pytorch_Realtime_Multi-Person_Pose_Estimation/")
from a_pose import *
sys.path.append("/home/fashionteam/LIP_JPPNet/")
from a_parse import *

def action(dir, first):
	pose(dir)
	segmentation(dir, first)

data_path = "/home/fashionteam/final_dataset_white/"
data_list = os.listdir(data_path)
data_list.sort()
first = True

for elem in data_list:
	dir = os.path.join(data_path, elem)
	action(dir, first)
	first = False
	print("{} DONE".format(elem))

