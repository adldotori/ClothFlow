import os
import os.path as osp
import shutil
import pickle

base_dir = '/home/fashionteam/viton_512/train/image/'
target_dir = '/home/fashionteam/bulk'
pkl_path = '/home/fashionteam/ClothFlow/viton_real_train.pkl'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(data)

for i in data:
    i = i+'_0.jpg'
    shutil.copy2(osp.join(base_dir, i), osp.join(target_dir, i))
