from PIL import Image
import os
import numpy as np
import os.path as osp
import cv2
import pickle
import torchvision.transforms as transforms

base_dir = '/home/fashionteam/viton_512/train/'
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

print(len(os.listdir(osp.join(base_dir, 'image-seg'))))
for image_ in os.listdir(osp.join(base_dir, 'image-seg')):
    path_seg = osp.join(base_dir, 'image-seg', image_)
    seg = Image.open(path_seg)
    transform_1ch = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
    parse = transform_1ch(seg)
    parse_array = np.array(seg)
    head = (parse_array == 1).astype(np.float32) + \
            (parse_array == 2).astype(np.float32) + \
            (parse_array == 4).astype(np.float32) + \
            (parse_array == 13).astype(np.float32)
    np.mean(head[0])
    if np.mean(head[0]) > 0:
        list.append(image_[:-6])

print(len(list))
image_files = load_pkl(osp.join("/home/fashionteam/ClothFlow/","viton_real_train.pkl"))
answer = []
for i in image_files:
    if not i in list:
        answer.append(i)

# print(len(image_files))
print(len(answer))
save_pkl('/home/fashionteam/ClothFlow/viton_real_train_fullface.pkl', answer)