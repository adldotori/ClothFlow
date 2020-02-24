import os
import os.path as osp
import numpy as np
import  cv2
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageDraw
import pickle

def imsave(nparray,fname):
    i = Image.fromarray(nparray, mode='L')
    i.save(fname)

def line(c_pose_map, c_pointx, c_pointy, i, j):
    if (c_pointx[i], c_pointy[i]) != (-1, -1) and (c_pointx[j], c_pointy[j]) != (-1, -1):
        cv2.line(c_pose_map, (c_pointx[i], c_pointy[i]), (c_pointx[j], c_pointy[j]),255,5)

H, W = (512, 512)
base_dir = "/home/fashionteam/dataset_MVC_tops/train/"
# -------------- VITON ---------------
'''
pkls = os.listdir(base_dir)
t = 0
for i, pkl in enumerate(pkls):
        with open(os.path.join(base_dir, folder, image, pkl), 'rb') as f:
            pose_label = pickle.load(f)        
            c_pose_data = pose_label

        point_num = 18
        c_pose_map = np.zeros((H, W)).astype(np.uint8)
        r = 5
        c_pose = Image.new('L', (W, H))
        c_pose_draw = ImageDraw.Draw(c_pose)
        c_pointx = [0 for i in range(point_num)]
        c_pointy = [0 for i in range(point_num)]
        # print('    '+str(c_pose_data.keys()))
        if not c_pose_data.keys():
            print(folder+'/'+image+':'+'Key Error')
        for i in range(point_num):
            one_map = Image.new('L', (W, H))
            draw = ImageDraw.Draw(one_map)
            if i in c_pose_data.keys():
                c_pointx[i] = c_pose_data[i][0]
                c_pointy[i] = c_pose_data[i][1]
            else:
                c_pointx[i] = -1
                c_pointy[i] = -1

            #c_pointx = c_pointx * 192 / 762
            #c_pointy = c_pointy * 256 / 1000
            if c_pointx[i] > 1 and c_pointy[i] > 1:
                draw.rectangle((c_pointx[i] - r, c_pointy[i] - r, c_pointx[i] + r, c_pointy[i]+ r), 'white', 'white')
            # one_map = np.expand_dims(one_map, axis=2)
            c_pose_map = c_pose_map + one_map
        line(c_pose_map,c_pointx, c_pointy,0,1)
        line(c_pose_map,c_pointx, c_pointy,0,14)
        line(c_pose_map,c_pointx, c_pointy,0,15)
        line(c_pose_map,c_pointx, c_pointy,1,2)
        line(c_pose_map,c_pointx, c_pointy,1,5)
        line(c_pose_map,c_pointx, c_pointy,1,8)
        line(c_pose_map,c_pointx, c_pointy,1,11)
        line(c_pose_map,c_pointx, c_pointy,2,3)
        line(c_pose_map,c_pointx, c_pointy,3,4)
        line(c_pose_map,c_pointx, c_pointy,5,6)
        line(c_pose_map,c_pointx, c_pointy,6,7)
        line(c_pose_map,c_pointx, c_pointy,8,9)
        line(c_pose_map,c_pointx, c_pointy,9,10)
        line(c_pose_map,c_pointx, c_pointy,11,12)
        line(c_pose_map,c_pointx, c_pointy,12,13)
        line(c_pose_map,c_pointx, c_pointy,14,16)
        line(c_pose_map,c_pointx, c_pointy,15,17)
        # print(os.path.join(base_dir, folder, image, 'pose.png'))
        imsave(c_pose_map, os.path.join('/home/fashionteam/viton_resize/image_pose/', pkl+'.png'))
'''
# -------------- MVC -----------------

folder_list = os.listdir(base_dir)
t = 0
for i, folder in enumerate(folder_list):
    image_list = os.listdir(os.path.join(base_dir, folder))
    image_list = [image for image in image_list if os.path.isdir(os.path.join(base_dir, folder, image))]
    print('[{}/{}]'.format(i, len(folder_list)))
    for image in image_list:
        with open(os.path.join(base_dir, folder, image, 'pose_resize.pkl'), 'rb') as f:
            pose_label = pickle.load(f)        
            c_pose_data = pose_label
        ###

        point_num = 18
        c_pose_map = np.zeros((H, W)).astype(np.uint8)
        r = 5
        c_pose = Image.new('L', (W, H))
        c_pose_draw = ImageDraw.Draw(c_pose)
        c_pointx = [0 for i in range(point_num)]
        c_pointy = [0 for i in range(point_num)]
        # print('    '+str(c_pose_data.keys()))
        if not c_pose_data.keys():
            print(folder+'/'+image+':'+'Key Error')
        print(c_pose_data.keys())
        for i in range(point_num):
            one_map = Image.new('L', (W, H))
            draw = ImageDraw.Draw(one_map)
            if i in c_pose_data.keys():
                c_pointx[i] = c_pose_data[i][0]
                c_pointy[i] = c_pose_data[i][1]
            else:
                c_pointx[i] = -1
                c_pointy[i] = -1

            #c_pointx = c_pointx * 192 / 762
            #c_pointy = c_pointy * 256 / 1000
            if c_pointx[i] > 1 and c_pointy[i] > 1:
                draw.rectangle((c_pointx[i] - r, c_pointy[i] - r, c_pointx[i] + r, c_pointy[i]+ r), 'white', 'white')
            # one_map = np.expand_dims(one_map, axis=2)
            c_pose_map = c_pose_map + one_map
        line(c_pose_map,c_pointx, c_pointy,0,1)
        line(c_pose_map,c_pointx, c_pointy,0,14)
        line(c_pose_map,c_pointx, c_pointy,0,15)
        line(c_pose_map,c_pointx, c_pointy,1,2)
        line(c_pose_map,c_pointx, c_pointy,1,5)
        line(c_pose_map,c_pointx, c_pointy,1,8)
        line(c_pose_map,c_pointx, c_pointy,1,11)
        line(c_pose_map,c_pointx, c_pointy,2,3)
        line(c_pose_map,c_pointx, c_pointy,3,4)
        line(c_pose_map,c_pointx, c_pointy,5,6)
        line(c_pose_map,c_pointx, c_pointy,6,7)
        line(c_pose_map,c_pointx, c_pointy,8,9)
        line(c_pose_map,c_pointx, c_pointy,9,10)
        line(c_pose_map,c_pointx, c_pointy,11,12)
        line(c_pose_map,c_pointx, c_pointy,12,13)
        line(c_pose_map,c_pointx, c_pointy,14,16)
        line(c_pose_map,c_pointx, c_pointy,15,17)
        # print(os.path.join(base_dir, folder, image, 'pose.png'))
        imsave(c_pose_map, os.path.join(base_dir, folder, image, 'pose.png'))
