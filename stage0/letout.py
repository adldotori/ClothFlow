from PIL import Image
from PIL import ImageDraw
from pathlib import Path
import os
import os.path as osp
import numpy as np
import json
import pickle

base_dir = '/home/fashionteam/viton_resize/train/'
src_folder = 'image-parse/'
tar_folder = 'image-l-parse/'

file_list = os.listdir(osp.join(base_dir, src_folder))

# print(file_list)

for file in file_list:
    print(file)
    path_image = osp.join(base_dir, src_folder, file)
    seg = Image.open(path_image)
    parse_array = np.array(seg)
    print(parse_array.shape)
    print(parse_array[parse_array != 0])    
    print(parse_array)
    result = Image.fromarray((parse_array*255).astype(np.uint8))
    if seg == result:
        print(1)
    result.save('sdf.png')
    break