from PIL import Image
import os
import numpy as np
import os.path as osp
import cv2
import pickle

base_dir = '/home/fashionteam/viton_512/train/'
def load_pkl(name):
    with open(name,"rb") as f:
        data = pickle.load(f)
    return data

def save_pkl(path,data):
    with open(path, "wb") as f:
        pickle.dump(data,f)

a = load_pkl('/home/fashionteam/ClothFlow/viton_real_train_fullface.pkl')

print(len(a))
asdf = [0 for i in range(20)]
cp_list = []
for image_ in a:
    path_pose = osp.join(base_dir, 'pose_pkl', image_+'_0.pkl')
    with open(path_pose, 'rb') as f:
        pose_label = pickle.load(f)
        pose_data = pose_label

        if 0 in pose_data.keys():
            cp_list.append(image_)
        else:
            print(image_)
        # asdf[len(pose_data)] += 1

        # if len(pose_data) >= 12:
        #     cp_list.append(image_)
        
        # if len(pose_data) == 12:
        #     print(image_)
print(asdf)
# image_files = load_pkl(osp.join("/home/fashionteam/ClothFlow/","viton_neck_train.pkl"))
# print(image_files[0])
# for i in list:
#     if i in image_files:
#         image_files.remove(i)
#         print(i)

# print(len(image_files))
print(len(cp_list))
save_pkl('/home/fashionteam/ClothFlow/viton_ful1face.pkl', cp_list)