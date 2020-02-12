import os
import os.path as osp
import numpy as np
import  cv2
from PIL import Image

base_dir = "/home/fashionteam/dataset_MVC_down/"

os.system('mkdir '+base_dir+'train')
os.system('mkdir '+base_dir+'test')
folder_list = [i for i in os.listdir(base_dir) if i[0] != 't']

for i, folder in enumerate(folder_list):
    
    if folder[-2] == '3' or folder[-2] == '9':
        os.system('mv '+base_dir+folder+' '+base_dir+'test/'+folder)
    else:
        os.system('mv '+base_dir+folder+' '+base_dir+'train/'+folder)