import random
import os
import shutil


mvc_root = "/home/fashionteam/dataset_MVC_tops/test"
mvc_list = os.listdir(mvc_root)
random.shuffle(mvc_list)
tar_root = "/home/fashionteam/mvc_test"
if not os.path.exists(tar_root): os.mkdir(tar_root)

"""										
for elem in os.listdir(tar_root):
	save_root = os.path.join(tar_root, elem)
	cloth_path = os.path.join(mvc_root, elem)
	shutil.copy(os.path.join(cloth_path, "crop.png"), os.path.join(save_root, "cloth.png"))
	shutil.copy(os.path.join(cloth_path, "mask.png"), os.path.join(save_root, "cloth_mask.png"))
"""

for i in range(5):
	name = mvc_list[i]
	cloth_name = mvc_list[10+i]
	cloth_root = os.path.join(mvc_root, cloth_name)
	data_root = os.path.join(mvc_root, name)
	save_root = os.path.join(tar_root, name)
	if not os.path.exists(save_root): os.makedirs(save_root)
	shutil.copy(os.path.join(cloth_root, "crop.png"), os.path.join(save_root, "cloth.png"))
	shutil.copy(os.path.join(cloth_root, "mask.png"), os.path.join(save_root, "cloth_mask.png"))
	img_root = os.path.join(data_root, "{}-p-4x_resize.jpg".format(name))
	shutil.copy(img_root, os.path.join(save_root, "image.jpg"))


