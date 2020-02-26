import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import shutil
import cv2

dataroot = "/home/fashionteam/dataset_MVC_down/"

def get_vis(pair,position):
    img = osp.join(dataroot,pair,position,"segment_vis_resize.jpg")
    image = Image.open(img)
    np_img = np.array(image)
    return np_img


def imsave(nparray,fname):
    Image.fromarray(nparray).save(fname)


def check_pixel(nparray,a,b):
    return nparray[a,b,:]

def existColor(nparray,color):
    R = nparray[:,:,0]
    G = nparray[:,:,1]
    B = nparray[:,:,2]
    R = (R == color[0])
    G = (G == color[1])
    B = (B == color[2])
    comb = logical_and([R,G,B])
    return comb


def logical_and(li):
    if len(li) == 2:
        return np.logical_and(li[0],li[1])
    else:
        return np.logical_and(li[0],logical_and(li[1:]))

def logical_or(li):
    if len(li) == 2:
        return np.logical_or(li[0],li[1])
    else:
        return np.logical_or(li[0],logical_or(li[1:]))

def filter(np2darray):
    a,b = np2darray.shape
    np2darray_ = np.reshape(np2darray,(a,b,1))
    return np.concatenate((np2darray_,np2darray_,np2darray_),axis=2)


# Color = [[255,85,0],[0,119,221],[0,0,85]] # top
# orange, lightblue,navy
Color = [[85,85,0]] # bottom

lidi = os.listdir(dataroot)
make_mask = True
make_crop = True


for i in range(len(lidi)):
    print(i, lidi[i])
    
    for j in [i for i in os.listdir(os.path.join(dataroot, lidi[i])) if i in ['p','1','3']]:
        try:
            seg = get_vis(lidi[i],j)
            img = get_vis(lidi[i],j)
        except:
            print('>>> {}{}'.format(i, lidi[i]))
            continue

        H, W, C = img.shape
        temp = []
        for colour in Color:
            temp.append(existColor(seg,colour))
        mask = logical_or(temp)
        if make_crop:
            filt = filter(mask)
            img = filt * img
            imsave(img, osp.join(dataroot,lidi[i],j,"segment_crop.png"))
            exit()