from PIL import Image
import os
import numpy as np
import os.path as osp
import cv2
import pickle

base_dir = '/home/fashionteam/viton_512/train/cloth-mask/'
def load_pkl(name):
    with open(name,"rb") as f:
        data = pickle.load(f)
    return data

def save_pkl(path,data):
    with open(path, "wb") as f:
        pickle.dump(data,f)
threshold_h = 160
threshold_l = 80
list = []
c=0
for image_ in os.listdir(base_dir):
    image = np.array(Image.open(osp.join(base_dir, image_)))
    if np.mean(image) > threshold_h or np.mean(image) < threshold_l:
        list.append(image_[:-6])
        print(image_[:-6])
        c+=1
print(c, len(list))

image_files = load_pkl(osp.join("/home/fashionteam/ClothFlow/","viton_neck_train.pkl"))
print(image_files[0])
for i in list:
    if i in image_files:
        image_files.remove(i)
        print(i)

print(len(image_files))

save_pkl('/home/fashionteam/ClothFlow/viton_real_train.pkl', image_files)