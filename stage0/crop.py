import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import shutil
import cv2

TARGET_SIZE = (512,512)
IS_TOPS = True
dataroot = "/home/fashionteam/dataset_MVC_tops/train/"
print(IS_TOPS)
def naming(pair,position):
    return pair + "-" + position + "-4x.jpg"

def get_vis(pair,position):
    img = osp.join(dataroot,pair,position,"segment_vis.png")
    image = Image.open(img)
    np_img = np.array(image)
    return np_img

def get_img(pair,position):
    img = osp.join(dataroot,pair,naming(pair,position))
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
    if len(li) == 1:
        return li[0]
    elif len(li) == 2:
        return np.logical_or(li[0],li[1])
    else:
        return np.logical_or(li[0],logical_or(li[1:]))

def filter(np2darray):
    a,b = np2darray.shape
    np2darray_ = np.reshape(np2darray,(a,b,1))
    return np.concatenate((np2darray_,np2darray_,np2darray_),axis=2)


if IS_TOPS:
    Color = [[255,85,0],[0,119,221],[0,0,85]]
    # orange, lightblue,navy
else:
    Color = [[0,85,85]]


lidi = os.listdir(dataroot)
make_mask = True
make_crop = True

for i in range(len(lidi)):
    try:
        seg = get_vis(lidi[i],"p")
        img = get_img(lidi[i],"p")
    except OSError:
        print('>>> {}{}'.format(i, lidi[i]))
        continue

    H, W, C = img.shape
    temp = []
    for colour in Color:
        temp.append(existColor(seg,colour))
    # print(temp)

    check = True
    for j in range(len(temp)):
        if not len(temp[j][temp[j]==True]) == 0:
            check = False
    if check:
        f = open('delete.txt','a')
        print(i, lidi[i])
        f.write(lidi[i]+'\n')
        f.close()
    mask = logical_or(temp)
    if make_crop:
        filt = filter(mask)
        img = filt * img
        img_ori = np.pad(img, ((0,0),(160,160),(0,0)), 'constant', constant_values=(0))
        img_ori = cv2.resize(img_ori, TARGET_SIZE)
        imsave(img_ori, osp.join(dataroot,lidi[i],"crop.png"))
        # img = cv2.resize(img, (192, 256))
        # imsave(img, osp.join(dataroot,lidi[i],"crop_min.png"))
    if make_mask:
        A = mask.astype(np.uint8) * 25
        A_ori = np.pad(A, ((0,0),(160,160)), 'constant', constant_values=(0))
        A_ori = cv2.resize(A_ori, TARGET_SIZE)
        imsave(A_ori, osp.join(dataroot,lidi[i],"mask.png"))
        # A = cv2.resize(A, (192, 256))
        # imsave(A , osp.join(dataroot,lidi[i],"mask_min.png"))